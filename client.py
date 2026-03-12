import cv2
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image
import os
import sys
import json
import time
import zmq

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

sys.path.append(os.path.join(os.path.dirname(__file__), 'arx5-sdk', 'python'))
import arx5_interface as arx5

# 导入抓取预测模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, 'YOLO_World-SAM-GraspNet'))
# from grasp_predictor import predict_grasp_pose

def rpy2rotm(rpy):
    """将RPY角度转换为旋转矩阵"""
    roll, pitch, yaw = rpy
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy, cy*sx, cy*cx]
    ])


def pose_6d_to_matrix(pose_6d):
    """将6D位姿转换为4x4齐次变换矩阵"""
    x, y, z, roll, pitch, yaw = pose_6d
    T = np.eye(4)
    T[:3, :3] = rpy2rotm([roll, pitch, yaw])
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose_6d(T):
    """将4x4齐次变换矩阵转换为6D位姿"""
    x, y, z = T[:3, 3]
    R = T[:3, :3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([x, y, z, roll, pitch, yaw])


class OrbbecCamera:
    def __init__(self):
        config = Config()  # Initialize the config for the pipeline
        self.pipeline = Pipeline()  # Create the pipeline object

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
        except Exception as e:
            print(f"错误：无法获取相机配置: {e}")
            raise RuntimeError(f"相机初始化失败：无法获取流配置 - {e}")

        try:
            self.pipeline.enable_frame_sync()
        except Exception as e:
            print(f"警告：无法启用帧同步: {e}")

        try:
            self.pipeline.start(config)
        except Exception as e:
            print(f"错误：无法启动 pipeline: {e}")
            raise RuntimeError(f"相机初始化失败：无法启动 pipeline - {e}") 

        self.color_intrinsic = color_profile.get_intrinsic()
        self.color_distortion = color_profile.get_distortion()
        self.depth_intrinsic = depth_profile.get_intrinsic()
        self.depth_distortion = depth_profile.get_distortion()
        self.depth_scale = 1000.0 

        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)


    def stop(self):
        self.pipeline.stop()


    def collect_data(self):
        print("=================collecting data=====================")
        while True:
            # ============================== 相机采集数据 ================================
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                print('frames is None')
                time.sleep(0.1)
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                print('color_frame or depth_frame is None')
                continue
            frames = self.align_filter.process(frames)
            if not frames:
                continue
            frames  = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("Failed to convert frame to image")
                continue
            # cv2.imshow("Yellow Object Detection", color_image)
            try:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width()))
            except ValueError:
                print("Failed to reshape depth data")
                continue
            depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            # ============================== 相机采集数据结束 得到frame ================================
            display_color = color_image.copy()
            display_depth = depth_data.copy()
            cv2.imshow("Color Image", display_color)
            cv2.imshow("Depth Image", display_depth)
            key = cv2.waitKey(1)
            if key == ord('g'):
                cv2.destroyAllWindows()
                return color_image, depth_data


class EyeInHandPickPlace:
    def __init__(self, model='X5', interface='can0'):
        self.camera = OrbbecCamera()
        self.controller = arx5.Arx5CartesianController(model, interface)
        self.controller.set_log_level(arx5.LogLevel.INFO)
        self.robot_config = self.controller.get_robot_config()
        # load eye in hand calibration result
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        calib_file_path = os.path.join(current_script_dir, 'eye_in_hand_calibration.json')
        with open(calib_file_path, 'r') as f:
            calibration_data = json.load(f)
            T_camera_to_eef = np.eye(4)
            T_camera_to_eef[:3, :3] = np.array(calibration_data['camera_to_eef']['rotation_matrix'])
            T_camera_to_eef[:3, 3] = np.array(calibration_data['camera_to_eef']['translation'])
        self.T_camera_to_eef = T_camera_to_eef
    

    def move_to_pose_smooth(self, controller, target_pose_6d, target_gripper_pos, duration=3.0):
        """
        平滑移动到目标位姿
        
        Args:
            controller: 控制器实例
            target_pose_6d: 目标位姿 [x, y, z, roll, pitch, yaw]
            target_gripper_pos: 目标夹爪位置
            duration: 移动持续时间（秒）
        """
        current_state = controller.get_eef_state()
        current_pose = current_state.pose_6d()
        current_gripper = current_state.gripper_pos
        
        # 获取控制器配置
        controller_config = controller.get_controller_config()
        dt = controller_config.controller_dt
        # 使用一个合适的间隔时间，确保时间戳足够大
        interpolate_interval_s = max(dt, 0.05)  # 至少50ms
        
        # 计算插值步数
        num_steps = int(duration / interpolate_interval_s)
        
        print(f"正在移动到目标位姿，预计耗时 {duration:.1f} 秒...")
        print(f"current_pose: {current_pose}")
        print(f"target_pose: {target_pose_6d}")
        
        for i in range(num_steps):
            alpha = float(i + 1) / num_steps
            # 使用平滑插值（easeInOutQuad）
            if alpha < 0.5:
                t = 2 * alpha * alpha
            else:
                t = -2 * (1 - alpha) * (1 - alpha) + 1
            
            # 插值位姿
            interpolated_pose = current_pose + (target_pose_6d - current_pose) * t
            interpolated_gripper = current_gripper + (target_gripper_pos - current_gripper) * t
            
            # 获取当前状态以确保时间戳是最新的
            eef_state = controller.get_eef_state()
            eef_cmd = arx5.EEFState(interpolated_pose, interpolated_gripper)
            eef_cmd.timestamp = eef_state.timestamp + interpolate_interval_s
            controller.set_eef_cmd(eef_cmd)
            
            # 等待到下一个时间点
            current_time = time.time()
            while time.time() < current_time + interpolate_interval_s:
                time.sleep(0.01)  # 小步等待，避免CPU占用过高
        
        # 等待到达目标
        time.sleep(0.5)
        print("已到达目标位姿")
        current_state = controller.get_eef_state()
        current_pose = current_state.pose_6d()
        print(f"到达的位姿: {current_pose}")


    def collect_data(self):
        # 1. move to search pose
        self.controller.reset_to_home()
        time.sleep(1.0)
        self.move_to_pose_smooth(self.controller, np.array([0.18, 0, 0.2, 0, 1.35, 0]), self.robot_config.gripper_width, 3.0)
        search_pose = self.controller.get_eef_state().pose_6d().copy()
        try:
            color_image, depth_data = self.camera.collect_data()
        finally:
            self.controller.reset_to_home()
            time.sleep(1.0)
        return color_image, depth_data, search_pose


    def execute_trajectory(self, trajectory):
        """执行轨迹"""
        approach_pose, grasp_pose, lift_pose, place_pose = trajectory
        move_duration = 3.0  # 移动持续时间（秒）
        try:
            self.controller.reset_to_home()
            time.sleep(1.0)
            # 步骤1: 移动到抓取位姿上方
            print("=" * 50)
            print("步骤 1/6: 移动到抓取位姿上方")
            print("=" * 50)
            self.move_to_pose_smooth(self.controller, approach_pose, self.robot_config.gripper_width, move_duration)
            time.sleep(0.5)
            
            # 步骤2: 移动到抓取位姿
            print("=" * 50)
            print("步骤 2/6: 移动到抓取位姿")
            print("=" * 50)
            self.move_to_pose_smooth(self.controller, grasp_pose, self.robot_config.gripper_width, move_duration)
            time.sleep(0.5)
            
            # 步骤3: 关闭夹爪（抓取物体）
            print("=" * 50)
            print("步骤 3/6: 关闭夹爪（抓取物体）")
            print("=" * 50)
            print("正在关闭夹爪...")
            self.move_to_pose_smooth(self.controller, grasp_pose, 0.0, 1.0)  # 关闭夹爪
            time.sleep(0.5)
            print("夹爪已关闭，物体已抓取")
            
            # 步骤4: 抬起物体
            print("=" * 50)
            print("步骤 4/6: 抬起物体")
            print("=" * 50)
            self.move_to_pose_smooth(self.controller, lift_pose, 0.0, move_duration)
            time.sleep(0.5)
            
            # 步骤5: 移动到放置位置
            print("=" * 50)
            print("步骤 5/6: 移动到放置位置")
            print("=" * 50)
            self.move_to_pose_smooth(self.controller, place_pose, 0.0, move_duration)
            time.sleep(0.5)
            
            # 步骤6: 打开夹爪（放下物体）
            print("=" * 50)
            print("步骤 6/6: 打开夹爪（放下物体）")
            print("=" * 50)
            print("正在打开夹爪...")
            self.move_to_pose_smooth(self.controller, place_pose, self.robot_config.gripper_width, 1.0)
            time.sleep(0.5)
            print("夹爪已打开，物体已放下")
            
            # 步骤7: 回到初始状态
            print("=" * 50)
            print("步骤 7/7: 回到初始状态")
            print("=" * 50)
            print("正在返回初始位置...")
            self.controller.reset_to_home()
            time.sleep(1.0)
            
            print("\n" + "=" * 50)
            print("任务完成！")
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n任务被用户中断")
            print("正在返回初始位置...")
            self.controller.reset_to_home()
            time.sleep(1.0)
        except Exception as e:
            print(f"\n\n发生错误: {e}")
            print("正在返回初始位置...")
            self.controller.reset_to_home()
            time.sleep(1.0)
            raise
        finally:
            print("正在设置控制器为阻尼模式...")
            self.controller.set_to_damping()


    def convert_grasp_coordinate(self, matrix_in_camera, search_pose):
        """将相机坐标系下的抓取位姿转换为基座坐标系下的抓取位姿"""
        # get base^T_eef
        # eef_pose_6d = self.controller.get_eef_state().pose_6d().copy()  # eef pose in base frame
        T_eef_to_base = pose_6d_to_matrix(search_pose)
        print(f"search_pose: {search_pose}")
        print(f"eef_pose_6d: {self.controller.get_eef_state().pose_6d().copy()}")
        
        # get base^T_camera
        T_camera_to_base = T_eef_to_base @ self.T_camera_to_eef  # base^T_camera = base^T_eef * eef^T_camera
        
        # get base^T_point
        matrix_in_base = T_camera_to_base @ matrix_in_camera
        pose_in_base = matrix_to_pose_6d(matrix_in_base)
        print(f"pose_in_base: {pose_in_base}")
        
        # safty protection
        pose_in_base[0] = max(pose_in_base[0], 0.15)
        pose_in_base[0] = min(pose_in_base[0], 0.35)
        pose_in_base[1] = max(pose_in_base[1], -0.25) if pose_in_base[1] < 0 else min(pose_in_base[1], 0.25)
        pose_in_base[2] = min(pose_in_base[2], 0.1)
        pose_in_base[2] = max(pose_in_base[2]-0.04, 0.02)
        pose_in_base[3] = pose_in_base[3] if pose_in_base[3] < 1.57 else pose_in_base[3] - 3.14
        pose_in_base[3] = pose_in_base[3] if pose_in_base[3] > -1.57 else 3.14 + pose_in_base[3]
        return pose_in_base


    def plan_trajectory(self, grasp_pose):
        """规划轨迹"""
        approach_pose = np.array([grasp_pose[0], grasp_pose[1], grasp_pose[2] + 0.09, 0, 1.57, 0])
        lift_pose = np.array([0.2, 0.2, 0.2, 0, 1.57, 0])
        place_pose = np.array([0.2, 0.2, 0.1, 0, 1.57, 0])
        trajectory = [approach_pose, grasp_pose, lift_pose, place_pose]
        return trajectory


def send_numpy_array(sock, name, array, send_more=True):
    """发送一个 NumPy 数组的元数据和数据。"""
    # 1. 创建元数据字符串: [名称],[高],[宽],[通道],[数据类型]
    meta_data = f"{name},{','.join(map(str, array.shape))},{array.dtype.name}"
    
    # 2. 发送元数据 (SNDMORE 表示后面还有消息)
    sock.send_string(meta_data, zmq.SNDMORE)
    
    # 3. 发送数据字节
    sock.send(array.tobytes(), zmq.SNDMORE if send_more else 0)


def main():
    # 初始化 ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REQ) # REQ = Request (请求端)
    socket.connect("tcp://localhost:5555") # 连接服务端
    eye_in_hand_pick_place = EyeInHandPickPlace(model='X5', interface='can0')
    while True:
        # point_in_camera = eye_in_hand_pick_place.search_yellow_object()
        color_image, depth_data, search_pose = eye_in_hand_pick_place.collect_data()

        if color_image is None or depth_data is None:
            continue
        fx = eye_in_hand_pick_place.camera.depth_intrinsic.fx
        fy = eye_in_hand_pick_place.camera.depth_intrinsic.fy
        cx = eye_in_hand_pick_place.camera.depth_intrinsic.cx
        cy = eye_in_hand_pick_place.camera.depth_intrinsic.cy
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
        factor_depth = eye_in_hand_pick_place.camera.depth_scale
        
        # 发送rgb, depth, intrinsic, depth_scale到server
        print("发送请求中...")
        send_numpy_array(socket, "rgb", color_image)
        send_numpy_array(socket, "depth", depth_data)
        send_numpy_array(socket, "intrinsic", intrinsic)
        send_numpy_array(socket, "depth_scale", np.array([factor_depth], dtype=np.float32), False)
        reply = socket.recv_pyobj()
        print(f"grasp_poses_in_camera: {reply}")
        grasp_poses = [{'rotation_matrix': reply['rotation'][i], 'translation': reply['translation'][i]} for i in range(len(reply['rotation']))]

        # 将相机坐标系下的抓取位姿转换为基座坐标系下的抓取位姿,并规划轨迹并执行
        for grasp_pose in grasp_poses:
            matrix_in_camera = np.eye(4)
            matrix_in_camera[:3, :3] = grasp_pose['rotation_matrix']
            matrix_in_camera[:3, 3] = grasp_pose['translation']
            grasp_pose = eye_in_hand_pick_place.convert_grasp_coordinate(matrix_in_camera, search_pose)
            print(f"grasp_pose_in_base: {grasp_pose}")
            trajectory = eye_in_hand_pick_place.plan_trajectory(grasp_pose)
            eye_in_hand_pick_place.execute_trajectory(trajectory)

    # stop camera
    eye_in_hand_pick_place.camera.stop()

if __name__ == "__main__":
    main()