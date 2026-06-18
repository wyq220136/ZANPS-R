import trimesh
import cv2
import numpy as np

mesh = trimesh.load(r"D:\research\PartNet\scripts\scheme_b_prior_tsdf\Door_8897\view_0\model_0000\fused_mesh.obj", force='mesh')

mesh.show()

mesh = trimesh.load(r"D:\research\PartNet\scripts\scheme_b_prior_tsdf\Door_8897\view_0\model_0000\prior_mesh.obj", force='mesh')

mesh.show()