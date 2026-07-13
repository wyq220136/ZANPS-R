# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')
from Utils import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from functools import partial
from network_modules import *
from Utils import *



class RefineNet(nn.Module):
  def __init__(self, cfg=None, c_in=4, n_view=1):
    super().__init__()
    self.cfg = cfg
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
      norm_layer1d = nn.BatchNorm1d
    else:
      norm_layer = None
      norm_layer1d = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )
    self.encodeAB_adapters = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0),
      ),
      nn.Sequential(
        nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0),
      ),
    ])
    for adapter in self.encodeAB_adapters:
      nn.init.zeros_(adapter[-1].weight)
      nn.init.zeros_(adapter[-1].bias)

    embed_dim = 512
    num_heads = 4
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, 3),
    )

    if self.cfg['rot_rep']=='axis_angle':
      rot_out_dim = 3
    elif self.cfg['rot_rep']=='6d':
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, rot_out_dim),
    )

    # Shared high-resolution trunk.
    self.mask_decoder = nn.Sequential(
        ConvBNReLU(C_in=512 + 256, C_out=256, kernel_size=3, stride=1, norm_layer=norm_layer),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        ConvBNReLU(C_in=256, C_out=128, kernel_size=3, stride=1, norm_layer=norm_layer),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        ConvBNReLU(C_in=128, C_out=64, kernel_size=3, stride=1, norm_layer=norm_layer),
    )
    self.geom_head = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
    self.pose_head = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
    self.enable_ref_coord = bool(self.cfg.get('enable_ref_coord', False))
    if self.enable_ref_coord:
      self.coord_head = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
      self.coord_conf_head = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)


  def forward(self, A, B):
    """
    @A: (B,C,H,W)
    """
    bs = len(A)
    output = {}

    x = torch.cat([A,B], dim=0)
    x = self.encodeA(x)
    a = x[:bs]
    b = x[bs:]

    skip_feat = torch.cat((a,b),1).contiguous()
    ab_feat = skip_feat
    for idx, layer in enumerate(self.encodeAB):
      ab_feat = layer(ab_feat)
      # Insert lightweight adapters after the last two ResNet blocks.
      if idx == 3:
        ab_feat = ab_feat + self.encodeAB_adapters[0](ab_feat)
      elif idx == 4:
        ab_feat = ab_feat + self.encodeAB_adapters[1](ab_feat)

    ab = self.pos_embed(ab_feat.reshape(bs, ab_feat.shape[1], -1).permute(0,2,1))

    output['trans'] = self.trans_head(ab).mean(dim=1)
    output['rot'] = self.rot_head(ab).mean(dim=1)
    
    ab_feat_upsampled = F.interpolate(ab_feat, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
    combined_feat = torch.cat([ab_feat_upsampled, skip_feat], dim=1)
    mask_feat = self.mask_decoder(combined_feat)
    geom_logits = self.geom_head(mask_feat)
    pose_logits = self.pose_head(mask_feat)
    geom_logits = F.interpolate(geom_logits, size=A.shape[-2:], mode='bilinear', align_corners=False)
    pose_logits = F.interpolate(pose_logits, size=A.shape[-2:], mode='bilinear', align_corners=False)
    geom_validity = torch.sigmoid(geom_logits)
    pose_utility = torch.sigmoid(pose_logits)
    output['geom_logits'] = geom_logits
    output['pose_logits'] = pose_logits
    output['geom_validity'] = geom_validity
    output['pose_utility'] = pose_utility
    # Backward-compatible key used by existing callers.
    output['validity_mask'] = geom_validity * pose_utility
    if self.enable_ref_coord:
      ref_coord = self.coord_head(mask_feat)
      coord_conf_logits = self.coord_conf_head(mask_feat)
      ref_coord = F.interpolate(ref_coord, size=A.shape[-2:], mode='bilinear', align_corners=False)
      coord_conf_logits = F.interpolate(coord_conf_logits, size=A.shape[-2:], mode='bilinear', align_corners=False)
      output['ref_coord'] = ref_coord
      output['coord_conf_logits'] = coord_conf_logits
      output['coord_conf'] = torch.sigmoid(coord_conf_logits)

    return output
