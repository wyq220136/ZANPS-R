from Utils import *
from dataloader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
import sys



class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/', load_render_models=True):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    # Rotation neighborhood sampling around coarse init pose.
    self.rotation_neighborhood = os.getenv("FP_ROTATION_NEIGHBORHOOD", "narrow").strip().lower()  # hemisphere / quarter / narrow / full
    self.min_local_hypotheses = 64

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(min_n_views=40, inplane_step=60)

    self.glctx = glctx

    if load_render_models:
      if scorer is not None:
        self.scorer = scorer
      else:
        self.scorer = ScorePredictor()

      if refiner is not None:
        self.refiner = refiner
      else:
        self.refiner = PoseRefinePredictor()
    else:
      self.scorer = None
      self.refiner = None

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)



  def make_rotation_grid(self, min_n_views=40, inplane_step=60, init_pose=None):
    # init_pose here is expected to be cam_in_ob (camera pose in object frame).
    ag_region = 15.0
    if self.rotation_neighborhood == "quarter":
      ag_region = 45.0
    elif self.rotation_neighborhood == "hemisphere":
      ag_region = 90.0
    elif self.rotation_neighborhood == "narrow":
      ag_region = 15.0
    elif self.rotation_neighborhood == "full":
      init_pose = None
      ag_region = 180.0

    cam_in_obs = sample_views_icosphere(n_views=min_n_views, init_pose=init_pose, ag_region=ag_region)
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def _rotation_geodesic_deg(self, rot_batch, rot_ref):
    # rot_batch: (N,3,3), rot_ref: (3,3)
    rel = rot_batch @ rot_ref.t()
    tr = rel[:,0,0] + rel[:,1,1] + rel[:,2,2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_theta))


  def _select_local_rotation_grid(self, init_rot):
    if self.rotation_neighborhood == "full":
      return self.rot_grid.clone()

    ang = self._rotation_geodesic_deg(self.rot_grid[:, :3, :3], init_rot)
    if self.rotation_neighborhood == "quarter":
      max_ang = 45.0
    elif self.rotation_neighborhood == "narrow":
      max_ang = 15.0
    else:
      max_ang = 90.0  # hemisphere

    keep = ang <= max_ang
    local_grid = self.rot_grid[keep]
    if local_grid.shape[0] < self.min_local_hypotheses:
      # Ensure enough hypotheses even for sparse grids.
      topk = min(self.min_local_hypotheses, self.rot_grid.shape[0])
      ids = torch.argsort(ang)[:topk]
      local_grid = self.rot_grid[ids]
    return local_grid.clone()


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None, init_pose=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    if init_pose is not None:
      init_pose_t = torch.as_tensor(init_pose, device='cuda', dtype=torch.float).reshape(4, 4)
      # Build local rotation grid around coarse pose orientation.
      # make_rotation_grid expects cam_in_ob.
      cam_in_ob = torch.linalg.inv(init_pose_t).data.cpu().numpy()
      self.make_rotation_grid(min_n_views=40, inplane_step=60, init_pose=cam_in_ob)
      ob_in_cams = self.rot_grid.clone()
      center = init_pose_t[:3, 3]
      # Force-include the coarse init itself as one hypothesis.
      ob_in_cams = torch.cat([init_pose_t[None], ob_in_cams], dim=0)
      ob_in_cams[:, :3, 3] = center.reshape(1, 3)
    else:
      ob_in_cams = self.rot_grid.clone()
      center = self.guess_translation(depth=depth, mask=mask, K=K)
      ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.001)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)


  @torch.inference_mode()
  def sample_local_pose_candidates(self, K, rgb, depth, ob_mask, init_pose=None, max_candidates=8):
    """Generate object-frame pose candidates around an input pose without refinement."""
    if ob_mask.ndim == 3:
      ob_mask = ob_mask[..., 0]
    init_pose_centered = None
    if init_pose is not None:
      try:
        init_pose = np.asarray(init_pose, dtype=np.float32).reshape(4, 4)
        tf_to_center = self.get_tf_to_centered_mesh().data.cpu().numpy()
        init_pose_centered = init_pose @ np.linalg.inv(tf_to_center)
      except Exception as e:
        logging.warning(f"Invalid init_pose for local candidate sampling, fallback to global sampling: {e}")
        init_pose_centered = None

    poses = self.generate_random_pose_hypo(
      K=K,
      rgb=rgb,
      depth=depth,
      mask=ob_mask,
      scene_pts=None,
      init_pose=init_pose_centered,
    )
    if init_pose_centered is None:
      center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda', dtype=torch.float)

    poses = poses @ self.get_tf_to_centered_mesh()
    poses_np = poses.data.cpu().numpy().reshape(-1, 4, 4)
    if max_candidates is not None and int(max_candidates) > 0 and len(poses_np) > int(max_candidates):
      # Keep the coarse init and spread the rest across the local rotation grid.
      keep = [0]
      if int(max_candidates) > 1:
        rest = np.linspace(1, len(poses_np) - 1, int(max_candidates) - 1).astype(np.int64).tolist()
        keep.extend(rest)
      poses_np = poses_np[keep]
    return poses_np.astype(np.float32)


  @torch.inference_mode()
  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=8, use_nvdiffrast=True, init_pose=None, return_candidates=False):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if use_nvdiffrast and self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    normal_map = None
    if ob_mask.ndim == 3:
      ob_mask = ob_mask[..., 0]
    valid = (depth>=0.001) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      if return_candidates:
        return pose, [{"pose": pose, "score": 0.0, "rank": 0}]
      return pose

    init_pose_centered = None
    if init_pose is not None:
      try:
        init_pose = np.asarray(init_pose, dtype=np.float32).reshape(4, 4)
        tf_to_center = self.get_tf_to_centered_mesh().data.cpu().numpy()
        init_pose_centered = init_pose @ np.linalg.inv(tf_to_center)
      except Exception as e:
        logging.warning(f"Invalid init_pose provided, fallback to global sampling: {e}")
        init_pose_centered = None

    poses = self.generate_random_pose_hypo(
      K=K,
      rgb=rgb,
      depth=depth,
      mask=ob_mask,
      scene_pts=None,
      init_pose=init_pose_centered,
    )
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    if init_pose_centered is None:
      center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    if not use_nvdiffrast:
      # NvDiff-disabled fallback: skip render-based refine/score and keep a deterministic coarse pose.
      best_pose = poses[0] @ self.get_tf_to_centered_mesh()
      if return_candidates:
        return best_pose.data.cpu().numpy(), [{
          "pose": best_pose.data.cpu().numpy(),
          "score": 0.0,
          "rank": 0,
        }]
      return best_pose.data.cpu().numpy()

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    if self.refiner is None or self.scorer is None:
      self.refiner = PoseRefinePredictor()
      self.scorer = ScorePredictor()

    xyz_map = depth2xyzmap(depth, K)
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]

    logging.info(f'sorted scores:{scores}')

    centered_tf = self.get_tf_to_centered_mesh()
    best_pose = poses[0]@centered_tf

    if return_candidates:
      cand = []
      for rank in range(len(poses)):
        cand.append({
          "pose": (poses[rank] @ centered_tf).data.cpu().numpy(),
          "score": float(scores[rank].data.cpu().item()) if torch.is_tensor(scores[rank]) else float(scores[rank]),
          "rank": int(rank),
        })
      return best_pose.data.cpu().numpy(), cand

    return best_pose.data.cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  @torch.inference_mode()
  def track_one(self, rgb, depth, K, iteration, extra={}, use_nvdiffrast=True):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    if not use_nvdiffrast:
      return (self.pose_last@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
