import zmq
import numpy as np
import cv2
import pickle  # 用 pickle 序列化对象，或者直接发 bytes
import torch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GRASPNET_DIR = os.path.join(PROJECT_ROOT, 'GraspNet')
sys.path.append(GRASPNET_DIR)

# from grasp_predictor import predict_grasp_pose
from graspnet_predictor import predict_grasp_pose
# 初始化 ZMQ
context = zmq.Context()
socket = context.socket(zmq.REP) # REP = Reply (应答端)
socket.bind("tcp://0.0.0.0:5555") # 监听 5555 端口

print("GraspNet Server (ZMQ) 已启动，等待连接...")

def recv_numpy_array(sock):
    """接收一个 NumPy 数组的元数据和数据。"""
    # 1. 接收元数据字符串
    meta_data_str = sock.recv_string()
    # 2. 接收数据字节
    data_bytes = sock.recv()

    # 3. 解析元数据
    parts = meta_data_str.split(',')
    name = parts[0]
    dtype_name = parts[-1]
    shape = tuple(map(int, parts[1:-1])) # 形状 (高, 宽, 通道)
    
    # 4. 从字节缓冲区重构数组
    array = np.frombuffer(data_bytes, dtype=dtype_name).reshape(shape)
    return name, array

while True:
    try:
        # 1. 等待接收消息 (阻塞式)
        # --- 接收所有数据的逻辑 ---
        received_data = {}
        message_count = 0 
        while message_count < 4: # 我们发送了 4 组数据 (RGB, Depth, Intrinsic, Factor)
            try:
                name, array = recv_numpy_array(socket)
                received_data[name] = array
                message_count += 1
            except zmq.Again: # 处理非阻塞模式下的情况
                continue

        # 提取 factor_depth 的标量值
        factor_depth = received_data['depth_scale'][0]
        rgb_img = received_data['rgb']     
        depth_img = received_data['depth'] 
        intrinsic = received_data['intrinsic']
        
        print(f"收到图片: {rgb_img.shape}")

        # 2. 跑模型
        fifty_poses = predict_grasp_pose(rgb_img, depth_img, intrinsic, factor_depth, visual=True)
        # print(fifty_poses)
        
        # 3. 发送结果回去
        result = {
            "status": "ok",
            "translation": fifty_poses['translation'],
            "rotation": fifty_poses['rotation_matrix'],
            "width": fifty_poses['width'],
            "score": fifty_poses['score']
        }
        print(result)
        socket.send_pyobj(result)
        
    except KeyboardInterrupt:
        break
