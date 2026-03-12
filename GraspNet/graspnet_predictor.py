"""
抓取姿态预测模块
输入RGBD图像，输出抓取姿态（相机坐标系下）
基于 GraspNet baseline demo.py 实现，使用整个图片作为 workspace
"""

import os
import sys
import numpy as np
import open3d as o3d
import torch
from PIL import Image
import cv2

from graspnetAPI import GraspGroup

# 添加必要的路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


def get_net(checkpoint_path, num_view=300):
    """初始化并加载网络模型"""
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], 
                   is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if 'epoch' in checkpoint:
        print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, checkpoint['epoch']))
    else:
        print("-> loaded checkpoint %s" % checkpoint_path)
    net.eval()
    return net


def get_and_process_data(rgb_image, depth_image, intrinsic, factor_depth, num_point=20000):
    """
    根据给定的 RGB 图、深度图，生成输入点云及其它必要数据
    使用整个图片作为 workspace（参考 demo.py 实现）
    
    参数:
        rgb_image: RGB图像，可以是字符串路径或numpy数组
        depth_image: 深度图像，可以是字符串路径或numpy数组
        intrinsic: 相机内参矩阵 (3, 3)
        factor_depth: 深度因子
        num_point: 采样点数，默认20000（参考 demo.py）
    
    返回:
        end_points: 包含点云数据的字典
        cloud: Open3D点云对象
    """
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(rgb_image, str):
        color = np.array(Image.open(rgb_image), dtype=np.float32) / 255.0
    elif isinstance(rgb_image, np.ndarray):
        # 如果是BGR格式（OpenCV），转换为RGB
        if len(rgb_image.shape) == 3:
            color = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            color = rgb_image.astype(np.float32) / 255.0
    else:
        raise TypeError("rgb_image 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_image, str):
        depth = np.array(Image.open(depth_image))
    elif isinstance(depth_image, np.ndarray):
        depth = depth_image.copy()
    else:
        raise TypeError("depth_image 既不是字符串路径也不是 NumPy 数组！")

    # 3. 生成点云（参考 demo.py）
    width = color.shape[1]
    height = color.shape[0]
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], 
                       intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # 4. 获取有效点（使用整个图片作为 workspace，参考 demo.py 的 mask 方式）
    mask = (depth > 0)  # 只过滤无效深度，使用整个图片
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # 5. 采样点（参考 demo.py）
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # 6. 转换为 Open3D 点云和 torch tensor（参考 demo.py）
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d


def get_grasps(net, end_points):
    """前向推理获取抓取预测（参考 demo.py）"""
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud, voxel_size=0.01, collision_thresh=0.01):
    """碰撞检测（参考 demo.py）"""
    mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg


def predict_grasp_pose(rgb_image, depth_image, intrinsic, factor_depth, 
                       checkpoint_path=None, num_point=20000, num_view=300,
                       collision_thresh=0.01, voxel_size=0.01, visual=True):
    """
    根据RGB图像和深度图像预测抓取姿态（参考 demo.py 结构）
    
    参数:
        rgb_image: RGB图像，可以是字符串路径或numpy数组
        depth_image: 深度图像，可以是字符串路径或numpy数组
        intrinsic: 相机内参矩阵 (3, 3)
        factor_depth: 深度因子
        checkpoint_path: 模型checkpoint路径，默认使用 logs/log_rs/checkpoint-rs.tar
        num_point: 采样点数，默认20000（参考 demo.py）
        num_view: 视角数量，默认300（参考 demo.py）
        collision_thresh: 碰撞检测阈值，默认0.01
        voxel_size: 体素大小，默认0.01
        visual: bool, 是否可视化抓取结果（默认False）
    
    返回:
        dict: 包含以下键的字典
            - 'translation': numpy数组 (3,)，抓取位置，单位米，在相机坐标系下
            - 'rotation_matrix': numpy数组 (3, 3)，抓取旋转矩阵，在相机坐标系下
            - 'width': float，抓取宽度，单位米
            - 'score': float，抓取得分
            - 'grasp_group': GraspGroup对象（可选，用于进一步处理）
    
    注意:
        - 输出的抓取姿态在相机坐标系下
        - 使用整个图片作为 workspace（纯 GraspNet 实现）
    """
    # 1. 设置默认 checkpoint 路径
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ROOT_DIR, 'logs/log_rs/checkpoint-rs.tar')
    
    # 2. 加载网络
    net = get_net(checkpoint_path, num_view=num_view)
    
    # 3. 获取和处理数据（使用整个图片作为 workspace）
    end_points, cloud = get_and_process_data(rgb_image, depth_image, intrinsic, 
                                            factor_depth, num_point=num_point)
    
    # 4. 获取抓取预测
    gg = get_grasps(net, end_points)
    
    # 5. 碰撞检测
    if collision_thresh > 0:
        gg = collision_detection(gg, cloud, voxel_size=voxel_size, 
                                collision_thresh=collision_thresh)
    
    # 6. NMS 去重 + 按得分排序（参考 demo.py）
    gg.nms()
    gg.sort_by_score()
    
    # 7. 提取最佳抓取信息
    if len(gg) == 0:
        raise RuntimeError("未能生成有效的抓取预测")
    
    best_grasp = gg[0]  # 获取得分最高的抓取
    
    # 8. 可视化（可选）
    if visual:
        gg_vis = gg[:1]  # 显示前50个抓取（参考 demo.py）
        grippers = gg_vis.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    
    # 9. 构建返回结果
    result = {
        'translation': [g.translation for g in [gg[0]]],  # (3,) numpy数组，单位米，相机坐标系
        'rotation_matrix': [g.rotation_matrix for g in [gg[0]]],  # (3, 3) numpy数组，相机坐标系
        'width': [float(g.width) for g in [gg[0]]],  # 抓取宽度，单位米
        'score': [float(g.score) for g in [gg[0]]],  # 抓取得分
        'grasp_group': [gg[0]]  # 完整的GraspGroup对象（可选使用）
    }
    print("抓取姿态预测成功")
    
    return result


if __name__ == '__main__':
    # 使用示例（参考 demo.py）
    color_img_path = os.path.join(ROOT_DIR, 'graspnet-baseline/doc/example/color.png')
    depth_img_path = os.path.join(ROOT_DIR, 'graspnet-baseline/doc/example/depth.png')
    
    # 相机内参示例（需要根据实际相机调整）
    intrinsic = np.array([[570.0, 0, 320.0],
                          [0, 570.0, 240.0],
                          [0, 0, 1]])
    factor_depth = 1000.0  # 深度因子，根据实际深度单位调整（毫米转米）
    
    # 调用预测函数（使用整个图片作为 workspace）
    grasp_pose = predict_grasp_pose(
        color_img_path, 
        depth_img_path, 
        intrinsic, 
        factor_depth,
        num_point=20000,  # 参考 demo.py 默认值
        visual=True
    )
    
    print("\n=== 抓取姿态预测结果 ===")
    print(f"抓取位置 (相机坐标系): {grasp_pose['translation']}")
    print(f"抓取旋转矩阵 (相机坐标系):\n{grasp_pose['rotation_matrix']}")
    print(f"抓取宽度: {grasp_pose['width']:.4f} 米")
    print(f"抓取得分: {grasp_pose['score']:.4f}")

