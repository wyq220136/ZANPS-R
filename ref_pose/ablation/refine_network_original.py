# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os
import sys

import torch
import torch.nn as nn

code_dir = os.path.dirname(os.path.realpath(__file__))
ref_pose_dir = os.path.normpath(os.path.join(code_dir, ".."))
learning_models_dir = os.path.join(ref_pose_dir, "learning", "models")
if ref_pose_dir not in sys.path:
    sys.path.append(ref_pose_dir)
if learning_models_dir not in sys.path:
    sys.path.append(learning_models_dir)

from Utils import *  # noqa
from network_modules import *  # noqa


class RefineNetOriginal(nn.Module):
  """
  Original FoundationPose refiner architecture used for ablation.

  This intentionally excludes the validity-mask decoder, adapter layers, and
  any learned mask gating added by the main experiment.
  """

  def __init__(self, cfg=None, c_in=4, n_view=1):
    super().__init__()
    self.cfg = cfg
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
    else:
      norm_layer = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in, C_out=64, kernel_size=7, stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64, C_out=128, kernel_size=3, stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
      ConvBNReLU(256, 512, kernel_size=3, stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
      nn.Linear(512, 3),
    )

    if self.cfg["rot_rep"] == "axis_angle":
      rot_out_dim = 3
    elif self.cfg["rot_rep"] == "6d":
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
      nn.Linear(512, rot_out_dim),
    )

  def forward(self, A, B):
    bs = len(A)
    output = {}

    x = torch.cat([A, B], dim=0)
    x = self.encodeA(x)
    a = x[:bs]
    b = x[bs:]

    ab = torch.cat((a, b), 1).contiguous()
    ab = self.encodeAB(ab)
    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0, 2, 1))

    output["trans"] = self.trans_head(ab).mean(dim=1)
    output["rot"] = self.rot_head(ab).mean(dim=1)
    return output
