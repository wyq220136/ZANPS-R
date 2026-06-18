import cv2
import numpy as np
import json
import os

def to_homo(pts):
    return np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)

def project_3d_to_2d(pts_3d, K, ob_in_cam):
    if pts_3d.shape[0] == 3:
        pts_3d = np.append(pts_3d, 1)
    pt_cam = ob_in_cam @ pts_3d.T
    if pt_cam[2] <= 0: 
        return None
    proj = K @ pt_cam[:3]
    return (int(proj[0] / proj[2]), int(proj[1] / proj[2]))

def draw_xyz_axis(img, ob_in_cam, K, scale=0.1, thickness=3):
    oo = np.array([0, 0, 0, 1])
    xx = np.array([scale, 0, 0, 1])
    yy = np.array([0, scale, 0, 1])
    zz = np.array([0, 0, scale, 1])
    p_o = project_3d_to_2d(oo, K, ob_in_cam)
    p_x = project_3d_to_2d(xx, K, ob_in_cam)
    p_y = project_3d_to_2d(yy, K, ob_in_cam)
    p_z = project_3d_to_2d(zz, K, ob_in_cam)
    if p_o is None: return img
    if p_x: cv2.arrowedLine(img, p_o, p_x, (0, 0, 255), thickness, cv2.LINE_AA, tipLength=0.2)
    if p_y: cv2.arrowedLine(img, p_o, p_y, (0, 255, 0), thickness, cv2.LINE_AA, tipLength=0.2)
    if p_z: cv2.arrowedLine(img, p_o, p_z, (255, 0, 0), thickness, cv2.LINE_AA, tipLength=0.2)
    return img

def draw_posed_3d_box(K, img, ob_in_cam, size, line_color=(0, 255, 255), linewidth=2):
    """ 绘制 3D 边界框 """
    hs = size / 2.0
    # 8 个顶点定义
    vertices = np.array([
        [-hs[0], -hs[1], -hs[2]], [hs[0], -hs[1], -hs[2]],
        [hs[0], hs[1], -hs[2]], [-hs[0], hs[1], -hs[2]],
        [-hs[0], -hs[1], hs[2]], [hs[0], -hs[1], hs[2]],
        [hs[0], hs[1], hs[2]], [-hs[0], hs[1], hs[2]]
    ])
    
    lines = [
        (0,1), (1,2), (2,3), (3,0), # 底面
        (4,5), (5,6), (6,7), (7,4), # 顶面
        (0,4), (1,5), (2,6), (3,7)  # 侧棱
    ]

    pts_2d = []
    for v in vertices:
        pts_2d.append(project_3d_to_2d(v, K, ob_in_cam))

    for start, end in lines:
        if pts_2d[start] is not None and pts_2d[end] is not None:
            cv2.line(img, pts_2d[start], pts_2d[end], line_color, linewidth, cv2.LINE_AA)
    return img

def get_size_from_scale_file(scale_path):
    """ 加载 scale 文件夹下的 4x1 txt 文件 """
    try:
        # 加载 4x1 矩阵，取前 3 个值作为 x, y, z 的 size
        scale_data = np.loadtxt(scale_path).flatten()
        return scale_data[:3]
    except Exception as e:
        print(f"Error loading scale file {scale_path}: {e}")
        return np.array([0.3, 0.3, 0.3]) # 加载失败时的默认值

def visualize_frame(frame_id, base_dir, out_dir):
    # 1. 路径拼接
    obj_name = "_".join(frame_id.split('_')[:2])
    rgb_path = os.path.join(base_dir, "objs", obj_name, "rgb", f"{frame_id}.png")
    meta_path = os.path.join(base_dir, "metafile", f"{frame_id}.json")
    # 新增 scale 路径
    scale_path = os.path.join(base_dir, "scale", f"{frame_id}.txt")

    if not os.path.exists(meta_path) or not os.path.exists(rgb_path):
        print(f"Skipping {frame_id}: File not found.")
        return

    img = cv2.imread(rgb_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # 2. 解析位姿与内参 (方案1: 转置旋转矩阵)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)
    R_c2w = np.array(meta['world2camera_rotation']).reshape(3, 3)
    C_world = np.array(meta['camera_pos'])

    R_w2c = R_c2w.T 
    T_gt_cam = -R_w2c @ C_world

    ob_in_cam = np.eye(4)
    ob_in_cam[:3, :3] = R_w2c
    ob_in_cam[:3, 3] = T_gt_cam

    # 3. 加载 Scale 尺寸
    if os.path.exists(scale_path):
        size = get_size_from_scale_file(scale_path)
        print(f"Loaded size from scale file: {size}")
    else:
        print(f"Scale file not found at {scale_path}, using default size.")
        size = np.array([0.4, 0.4, 0.7]) 

    # 4. 绘制并保存
    img = draw_posed_3d_box(K, img, ob_in_cam, size)
    img = draw_xyz_axis(img, ob_in_cam, K, scale=0.15)
    
    os.makedirs(out_dir, exist_ok=True)
    save_name = os.path.join(out_dir, f"scale_check_{frame_id}.jpg")
    cv2.imwrite(save_name, img)
    print(f"Success! Output saved to: {save_name}")

if __name__ == "__main__":
    DATA_ROOT = "/data1/yuquan/CAPNet/data/test_inter"
    # 请确保该 frame_id 在 scale 文件夹下有对应的 txt 文件
    TEST_FRAME = "TrashCan_4108_0_19" 
    OUTPUT_DIR = "/data1/yuquan/PartNet/ref_pose/vis_results/"
    
    visualize_frame(TEST_FRAME, DATA_ROOT, OUTPUT_DIR)