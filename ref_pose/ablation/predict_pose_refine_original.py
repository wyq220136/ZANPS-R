import logging
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
ref_pose_dir = os.path.normpath(os.path.join(code_dir, ".."))
if ref_pose_dir not in sys.path:
  sys.path.append(ref_pose_dir)

from Utils import *  # noqa
from learning.datasets.h5_dataset import PoseRefinePairH5Dataset
from learning.training.predict_pose_refine import make_crop_data_batch
from ref_pose.ablation.refine_network_original import RefineNetOriginal


class OriginalPoseRefinePredictor:
  """
  Original FoundationPose pose refiner for ablation.

  It loads the vanilla refiner checkpoint and predicts only trans/rot updates.
  No validity-mask decoder, no mask gating, and no main-experiment adapter layers
  are used in this path.
  """

  def __init__(self):
    logging.info("welcome original FoundationPose refiner ablation")
    self.amp = True
    self.run_name = "2023-10-28-18-33-37"
    model_name = "model_best.pth"
    ckpt_dir = os.path.join(ref_pose_dir, "weights", self.run_name, model_name)

    self.cfg = OmegaConf.load(os.path.join(ref_pose_dir, "weights", self.run_name, "config.yml"))
    self.cfg["ckpt_dir"] = ckpt_dir
    self.cfg["enable_amp"] = True

    if "use_normal" not in self.cfg:
      self.cfg["use_normal"] = False
    if "use_mask" not in self.cfg:
      self.cfg["use_mask"] = False
    if "use_BN" not in self.cfg:
      self.cfg["use_BN"] = False
    if "c_in" not in self.cfg:
      self.cfg["c_in"] = 4
    if "crop_ratio" not in self.cfg or self.cfg["crop_ratio"] is None:
      self.cfg["crop_ratio"] = 1.2
    if "n_view" not in self.cfg:
      self.cfg["n_view"] = 1
    if "trans_rep" not in self.cfg:
      self.cfg["trans_rep"] = "tracknet"
    if "rot_rep" not in self.cfg:
      self.cfg["rot_rep"] = "axis_angle"
    if "zfar" not in self.cfg:
      self.cfg["zfar"] = 3
    if "normalize_xyz" not in self.cfg:
      self.cfg["normalize_xyz"] = False
    if isinstance(self.cfg["zfar"], str) and "inf" in self.cfg["zfar"].lower():
      self.cfg["zfar"] = np.inf
    if "normal_uint8" not in self.cfg:
      self.cfg["normal_uint8"] = False

    self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file="", mode="test")
    self.model = RefineNetOriginal(cfg=self.cfg, c_in=self.cfg["c_in"]).cuda()

    logging.info(f"Using original pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if "model" in ckpt:
      ckpt = ckpt["model"]
    missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
    unexpected = [k for k in unexpected if k.startswith(("mask_decoder.", "geom_head.", "pose_head.", "encodeAB_adapters."))]
    if missing:
      logging.warning(f"Original ablation refiner missing checkpoint keys: {len(missing)}")
    if unexpected:
      logging.info(f"Ignored main-experiment-only keys in checkpoint: {len(unexpected)}")

    self.model.cuda().eval()
    self.last_trans_update = None
    self.last_rot_update = None
    self.last_validity_mask = None
    logging.info("original ablation refiner init done")

  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, iteration=5):
    logging.info(f"ob_in_cams:{ob_in_cams.shape}")
    tf_to_center = np.eye(4)
    ob_centered_in_cams = ob_in_cams
    mesh_centered = mesh

    if not self.cfg.use_normal:
      normal_map = None

    crop_ratio = self.cfg["crop_ratio"]
    bs = 1

    B_in_cams = torch.as_tensor(ob_centered_in_cams, device="cuda", dtype=torch.float)

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh_centered)

    rgb_tensor = torch.as_tensor(rgb, device="cuda", dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device="cuda", dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device="cuda", dtype=torch.float)
    trans_normalizer = self.cfg["trans_normalizer"]
    if not isinstance(trans_normalizer, float):
      trans_normalizer = torch.as_tensor(list(trans_normalizer), device="cuda", dtype=torch.float).reshape(1, 3)

    trans_delta = None
    rot_mat_delta = None
    for _ in range(iteration):
      pose_data = make_crop_data_batch(
        self.cfg.input_resize,
        B_in_cams,
        mesh_centered,
        rgb_tensor,
        depth_tensor,
        K,
        crop_ratio=crop_ratio,
        normal_map=normal_map,
        xyz_map=xyz_map_tensor,
        cfg=self.cfg,
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        dataset=self.dataset,
        mesh_diameter=mesh_diameter,
      )
      B_in_cams = []
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        with torch.cuda.amp.autocast(enabled=self.amp):
          output = self.model(A, B)
        for k in output:
          output[k] = output[k].float()

        if self.cfg["trans_rep"] == "tracknet":
          if not self.cfg["normalize_xyz"]:
            trans_delta = torch.tanh(output["trans"]) * trans_normalizer
          else:
            trans_delta = output["trans"]
        elif self.cfg["trans_rep"] == "deepim":
          def project_and_transform_to_crop(centers):
            uvs = (pose_data.Ks[b:b+bs] @ centers.reshape(-1, 3, 1)).reshape(-1, 3)
            uvs = uvs / uvs[:, 2:3]
            uvs = (pose_data.tf_to_crops[b:b+bs] @ uvs.reshape(-1, 3, 1)).reshape(-1, 3)
            return uvs[:, :2]

          rot_delta = output["rot"]
          z_pred = output["trans"][:, 2] * pose_data.poseA[b:b+bs][..., 2, 3]
          uvA_crop = project_and_transform_to_crop(pose_data.poseA[b:b+bs][..., :3, 3])
          uv_pred_crop = uvA_crop + output["trans"][:, :2] * self.cfg["input_resize"][0]
          uv_pred = transform_pts(uv_pred_crop, pose_data.tf_to_crops[b:b+bs].inverse().cuda())
          center_pred = torch.cat([uv_pred, torch.ones((len(rot_delta), 1), dtype=torch.float, device="cuda")], dim=-1)
          center_pred = (pose_data.Ks[b:b+bs].inverse().cuda() @ center_pred.reshape(len(rot_delta), 3, 1)).reshape(len(rot_delta), 3) * z_pred.reshape(len(rot_delta), 1)
          trans_delta = center_pred - pose_data.poseA[b:b+bs][..., :3, 3]
        else:
          trans_delta = output["trans"]

        if self.cfg["rot_rep"] == "axis_angle":
          rot_mat_delta = torch.tanh(output["rot"]) * self.cfg["rot_normalizer"]
          rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)
        elif self.cfg["rot_rep"] == "6d":
          rot_mat_delta = rotation_6d_to_matrix(output["rot"]).permute(0, 2, 1)
        else:
          raise RuntimeError

        if self.cfg["normalize_xyz"]:
          trans_delta *= (mesh_diameter / 2)

        B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
        B_in_cams.append(B_in_cam)

      B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)

    B_in_cams_out = B_in_cams @ torch.tensor(tf_to_center[None], device="cuda", dtype=torch.float)
    torch.cuda.empty_cache()
    self.last_trans_update = trans_delta
    self.last_rot_update = rot_mat_delta
    self.last_validity_mask = None

    if get_vis:
      return B_in_cams_out, None
    return B_in_cams_out, None
