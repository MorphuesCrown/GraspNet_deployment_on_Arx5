import os
import sys
import time
import click
import numpy as np
import cv2
import json
from pathlib import Path

try:
    from pyorbbecsdk import Pipeline
except ImportError:
    print("错误: 请安装 pyorbbecsdk")
    sys.exit(1)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
from utils import frame_to_bgr_image


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


def rvec2rotm(rvec):
    """将旋转向量转换为旋转矩阵"""
    R, _ = cv2.Rodrigues(rvec)
    return R


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

def inv_matrix(T):
    """将4x4齐次变换矩阵求逆"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


class OrbbecCamera:
    """Orbbec相机封装类"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        初始化Orbbec相机
        
        Args:
            camera_matrix: 相机内参矩阵 (3x3)，如果为None则使用默认值或需要标定
            dist_coeffs: 畸变系数，如果为None则使用默认值
        """
        print("正在初始化Orbbec相机...")
        self.pipeline = Pipeline()
        self.pipeline.start()
        
        # 等待相机稳定
        print("等待相机稳定...")
        for _ in range(10):
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                time.sleep(0.1)
        
        # 获取第一帧以确定分辨率
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            raise RuntimeError("无法从相机获取帧！")
        
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("无法获取彩色帧！")
        
        self.width = color_frame.get_width()
        self.height = color_frame.get_height()
        print(f"相机初始化成功: {self.width}x{self.height}")
        
        # 设置相机内参和畸变系数
        camera_param = None
        if camera_matrix is None or dist_coeffs is None:
            # 需要从SDK获取参数
            try:
                camera_param = self.pipeline.get_camera_param()
            except Exception as e:
                print(f"从SDK获取相机参数失败: {e}")
        
        # 设置相机内参
        if camera_matrix is not None:
            self.camera_matrix = np.array(camera_matrix)
            print("使用用户提供的相机内参")
        elif camera_param is not None and hasattr(camera_param, 'rgb_intrinsic'):
            # 从SDK获取RGB相机内参
            rgb_intrinsic = camera_param.rgb_intrinsic
            fx = rgb_intrinsic.fx
            fy = rgb_intrinsic.fy
            cx = rgb_intrinsic.cx
            cy = rgb_intrinsic.cy
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            print(f"成功从SDK获取RGB相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        else:
            # 使用默认内参
            fx = fy = self.width * 0.7
            cx = self.width / 2.0
            cy = self.height / 2.0
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            print("使用默认内参")
        
        # 设置畸变系数
        if dist_coeffs is not None:
            self.dist_coeffs = np.array(dist_coeffs)
        elif camera_param is not None and hasattr(camera_param, 'rgb_distortion'):
            # 从SDK获取RGB相机畸变系数
            rgb_distortion = camera_param.rgb_distortion
            rgb_intrinsic = camera_param.rgb_intrinsic
            fx = rgb_intrinsic.fx
            
            # SDK返回的畸变系数异常大，需要归一化
            # 根据测试，除以fx（而不是fx²）得到的结果最合理
            k1_raw = rgb_distortion.k1
            k2_raw = rgb_distortion.k2
            k3_raw = rgb_distortion.k3 if hasattr(rgb_distortion, 'k3') else 0.0
            p1_raw = rgb_distortion.p1
            p2_raw = rgb_distortion.p2
            
            # 归一化畸变系数（除以焦距）
            # 注意：这是根据实际测试得出的归一化方法
            # 如果标定误差仍然很大，可能需要调整归一化方式
            k1 = k1_raw / fx
            k2 = k2_raw / (fx * fx)
            k3 = k3_raw / (fx * fx * fx)
            p1 = p1_raw / fx
            p2 = p2_raw / fx
            
            # OpenCV使用5个畸变系数: [k1, k2, p1, p2, k3]
            self.dist_coeffs = np.array([[
                k1, k2, p1, p2, k3
            ]], dtype=np.float32)
            
            print(f"畸变系数归一化:")
            print(f"  原始值: k1={k1_raw:.2f}, k2={k2_raw:.2f}, k3={k3_raw:.2f}")
            print(f"  归一化后: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}, p1={p1:.6f}, p2={p2:.6f}")
            # print(f"  正常范围: k1,k2应在±0.1~±1.0之间")
        else:
            # 默认无畸变
            self.dist_coeffs = np.zeros((1, 5), dtype=np.float32)
        # # 用于测试：暂时使用零畸变
        # self.dist_coeffs = np.zeros((1, 5), dtype=np.float32)
        
        print(f"相机内参矩阵:\n{self.camera_matrix}")
        print(f"畸变系数: {self.dist_coeffs.flatten()}")
    
    def get_frames(self):
        """获取彩色图"""
        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                return None
            
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return None
            
            color_image = frame_to_bgr_image(color_frame)
            return color_image
        except Exception as e:
            print(f"获取帧时出错: {e}")
            return None
    
    def get_camera_matrix(self):
        """获取相机内参矩阵"""
        return self.camera_matrix.copy()
    
    def get_dist_coeffs(self):
        """获取畸变系数"""
        return self.dist_coeffs.copy()
    
    def stop(self):
        """停止相机"""
        self.pipeline.stop()


class HandEyeCalibrator:
    """手眼标定类"""
    
    def __init__(self, model, interface, board_size=(9, 6), square_size=0.025, 
                 camera_matrix=None, dist_coeffs=None):
        """
        初始化手眼标定器
        
        Args:
            model: 机器人型号 (X5 or L5)
            interface: CAN接口名称 (can0等)
            board_size: 棋盘格尺寸 (内角点数)
            square_size: 棋盘格方格大小（米）
            camera_matrix: 相机内参矩阵（可选）
            dist_coeffs: 畸变系数（可选）
        """
        self.model = model
        self.interface = interface
        self.board_size = board_size
        self.square_size = square_size
        
        print("正在初始化相机...")
        self.camera = OrbbecCamera(camera_matrix, dist_coeffs)
        self.camera_matrix = self.camera.get_camera_matrix()
        self.dist_coeffs = self.camera.get_dist_coeffs()
        
        print("正在初始化机器人控制器...")
        self.controller = arx5.Arx5CartesianController(model, interface)
        self.controller.set_log_level(arx5.LogLevel.INFO)
        
        # 设置为阻尼模式，允许手动移动机械臂
        print("正在设置机械臂为阻尼模式（允许手动移动）...")
        self.controller.set_to_damping()
        print("机械臂已设置为阻尼模式，现在可以手动移动机械臂")
        
        # 准备棋盘格3D点（物体坐标系）
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 存储标定数据
        self.camera_poses = []  # 标定板在相机坐标系中的位姿
        self.robot_poses = []   # 机器人末端在基坐标系中的位姿
        
        print("\n手眼标定器初始化完成")
        print(f"棋盘格尺寸: {board_size[0]}x{board_size[1]} 内角点")
        print(f"方格大小: {square_size*1000:.1f}mm")
    
    def detect_chessboard(self, image):
        """检测棋盘格"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, self.board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret, rvec, tvec = cv2.solvePnP(
                self.objp, corners, self.camera_matrix, self.dist_coeffs
            )
            if ret:
                print(f"rvec: {rvec}")
                print(f"tvec: {tvec}")
                return True, corners, rvec, tvec
        return False, None, None, None
    
    def collect_data(self, min_samples=15):
        """收集标定数据"""
        print(f"\n开始收集标定数据，至少需要 {min_samples} 个样本")
        print("操作说明:")
        print("  - 手动移动机械臂到不同位姿（机械臂已设置为阻尼模式）")
        print("  - 确保相机能够清晰看到标定板")
        print("  - 按 's' 保存当前位姿")
        print("  - 按 'q' 完成收集")
        
        sample_count = 0
        while sample_count < min_samples:
            color_image = self.camera.get_frames()
            if color_image is None:
                time.sleep(0.1)
                continue
            
            ret, corners, rvec, tvec = self.detect_chessboard(color_image)
            display_image = color_image.copy()
            
            if ret:
                cv2.drawChessboardCorners(display_image, self.board_size, corners, ret)
                # 绘制坐标轴
                axis_points, _ = cv2.projectPoints(
                    np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, -0.05]], dtype=np.float32),
                    rvec, tvec, self.camera_matrix, self.dist_coeffs
                )
                axis_points = np.int32(axis_points).reshape(-1, 2)
                cv2.line(display_image, tuple(axis_points[0]), tuple(axis_points[1]), (0, 0, 255), 3) # red
                cv2.line(display_image, tuple(axis_points[0]), tuple(axis_points[2]), (0, 255, 0), 3) # green
                cv2.line(display_image, tuple(axis_points[0]), tuple(axis_points[3]), (255, 0, 0), 3) # blue
                status_text = f"Board detected - Samples: {sample_count}/{min_samples} - Press 's' to save"
            else:
                status_text = f"Board not detected - Samples: {sample_count}/{min_samples}"
            
            cv2.putText(display_image, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, "Press 'q' to finish", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Hand-Eye Calibration', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and ret:
                # 计算标定板在相机坐标系中的位姿
                # 注意：solvePnP 返回的 rvec 和 tvec 表示"从标定板到相机"的变换
                # 即：P_camera = R @ P_board + t，或 T_camera_from_board = [R t; 0 1]
                # 但手眼标定需要的是"从相机到标定板"的变换，所以需要求逆
                T_camera_from_board = np.eye(4)
                T_camera_from_board[:3, :3] = rvec2rotm(rvec.flatten())
                T_camera_from_board[:3, 3] = tvec.flatten() # camera^T_board
                # 求逆得到从相机到标定板的变换
                T_camera_to_board = inv_matrix(T_camera_from_board) # board^T_camera

                
                
                # 获取机器人末端在基座坐标系中的位姿
                # 使用.copy()避免引用问题
                eef_state = self.controller.get_eef_state()
                eef_pose_6d = eef_state.pose_6d().copy()
                
                # 检查eef状态是否有效
                if np.allclose(eef_pose_6d, 0, atol=1e-6):
                    print(f"警告: EEF状态全为0，请确保机械臂已正确初始化并处于阻尼模式")
                    continue
                
                T_eef_to_base = pose_6d_to_matrix(eef_pose_6d) # base^T_eef
                
                # 保存数据对
                self.camera_poses.append(T_camera_to_board) # board^T_camera
                self.robot_poses.append(T_eef_to_base) # base^T_eef
                sample_count += 1
                print(f"样本 {sample_count} 已保存 - EEF位置: [{eef_pose_6d[0]:.3f}, {eef_pose_6d[1]:.3f}, {eef_pose_6d[2]:.3f}]")
            
            elif key == ord('q'):
                if sample_count >= min_samples or input(f"只有 {sample_count} 个样本，是否继续? (y/n): ").lower() == 'y':
                    break
        
        cv2.destroyAllWindows()
        print(f"\n数据收集完成，共收集 {len(self.camera_poses)} 个样本")
        return len(self.camera_poses)
    
    def calibrate(self):
        T_eef2base = [self.robot_poses[i] for i in range(len(self.robot_poses))]    # base^T_eef
        R_eef2base = [T_eef2base[i][:3, :3] for i in range(len(T_eef2base))]
        t_eef2base = [T_eef2base[i][:3, 3] for i in range(len(T_eef2base))]

        T_target2cam = [inv_matrix(self.camera_poses[i]) for i in range(len(self.camera_poses))]    # camera^T_board
        R_target2cam = [T_target2cam[i][:3, :3] for i in range(len(T_target2cam))]
        t_target2cam = [T_target2cam[i][:3, 3] for i in range(len(T_target2cam))]

        R_cam2eef, t_cam2eef = cv2.calibrateHandEye(
            R_eef2base, t_eef2base, 
            R_target2cam, t_target2cam,      
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        print(f"R_cam2eef: {R_cam2eef}")
        print(f"t_cam2eef: {t_cam2eef}")

        T_eef_to_camera = np.eye(4)
        T_eef_to_camera[:3, :3] = R_cam2eef
        T_eef_to_camera[:3, 3] = t_cam2eef[:3, 0]

        return T_eef_to_camera
        
    
    def save_calibration(self, T_eef_to_camera, output_file="eye_in_hand_calibration.json"):
        """保存标定结果"""
        pose_6d = matrix_to_pose_6d(T_eef_to_camera)
        
        result = {
            "model": self.model,
            "calibration_type": "eye-to-hand",
            "camera": "Orbbec",
            "camera_to_eef": {
                "translation": T_eef_to_camera[:3, 3].tolist(),
                "rotation_matrix": T_eef_to_camera[:3, :3].tolist(),
                "pose_6d": pose_6d.tolist(),
            },
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.flatten().tolist(),
            "num_samples": len(self.camera_poses),
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n标定结果已保存到: {output_path.absolute()}")
        print(f"相机位姿 (pose_6d): [{pose_6d[0]:.4f}, {pose_6d[1]:.4f}, {pose_6d[2]:.4f}, "
              f"{np.degrees(pose_6d[3]):.2f}, {np.degrees(pose_6d[4]):.2f}, {np.degrees(pose_6d[5]):.2f}] deg")
    
    def verify_calibration(self, T_base_to_camera):
        """验证标定结果"""
        print("\n开始验证标定结果，按'q'退出")
        
        while True:
            color_image = self.camera.get_frames()
            if color_image is None:
                continue
            
            ret, corners, rvec, tvec = self.detect_chessboard(color_image)
            display_image = color_image.copy()
            
            if ret:
                T_camera_from_board = np.eye(4)
                T_camera_from_board[:3, :3] = rvec2rotm(rvec.flatten())
                T_camera_from_board[:3, 3] =  tvec.flatten()
                T_camera_to_board = inv_matrix(T_camera_from_board)                
                
                T_base_to_board = T_camera_to_board @ T_base_to_camera
                board_pose_6d = matrix_to_pose_6d(inv_matrix(T_base_to_board))
                
                # 重要：使用.copy()避免引用问题
                # pose_6d()返回的可能是内部缓冲区的引用，需要复制
                eef_pose_6d = self.controller.get_eef_state().pose_6d().copy()
                position_error = np.linalg.norm(board_pose_6d[:3] - eef_pose_6d[:3])
                
                cv2.drawChessboardCorners(display_image, self.board_size, corners, ret)
                info_text = [
                    f"Board in base: [{board_pose_6d[0]:.3f}, {board_pose_6d[1]:.3f}, {board_pose_6d[2]:.3f}]",
                    f"Board in camera: [{T_camera_from_board[0,3]:.3f}, {T_camera_from_board[1,3]:.3f}, {T_camera_from_board[2,3]:.3f}]",
                    f"EEF pos: [{eef_pose_6d[0]:.3f}, {eef_pose_6d[1]:.3f}, {eef_pose_6d[2]:.3f}]",
                    f"Error: {position_error*1000:.1f} mm"
                ]
                for i, text in enumerate(info_text):
                    cv2.putText(display_image, text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "Board not detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Calibration Verification', display_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """清理资源"""
        self.camera.stop()

    def save_samples(self):
        """保存样本数据到Numpy文件，未压缩"""
        np.save('samples.npy', self.camera_poses)
        np.save('robot_poses.npy', self.robot_poses)
        print(f"样本数据已保存到samples.npy和robot_poses.npy")

    def load_samples(self):
        """从Numpy文件加载样本数据"""
        self.camera_poses = np.load('samples.npy') # board^T_camera
        self.robot_poses = np.load('robot_poses.npy') # base^T_eef
        print(f"样本数据已加载到samples.npy和robot_poses.npy")


def load_camera_matrix(file_path):
    """从JSON文件加载相机内参"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            camera_matrix = np.array(data.get('camera_matrix', []))
            dist_coeffs = np.array(data.get('distortion_coefficients', []))
            if len(dist_coeffs.shape) == 1:
                dist_coeffs = dist_coeffs.reshape(1, -1)
            return camera_matrix, dist_coeffs
    except Exception as e:
        print(f"加载相机内参文件失败: {e}")
        return None, None
        

def main():
    calibrator = None
    verify = False
    collect = True
    min_samples = 15
    try:
        # 初始化标定器
        calibrator = HandEyeCalibrator(
            model="X5",
            interface="can0",
            board_size=(7, 6),
            square_size=0.024,
            camera_matrix=None,
            dist_coeffs=None
        )
        
        # 收集数据
        if collect:
            num_samples = calibrator.collect_data(min_samples=min_samples)
            if num_samples < 3:
                print("错误: 样本数量不足")
                return
        
        else:
            calibrator.load_samples()

        # 执行标定
        T_eef_to_camera = calibrator.calibrate()

        # 保存标定结果
        calibrator.save_calibration(T_eef_to_camera)
        
        # 验证（可选）
        if verify:
            calibrator.verify_calibration(T_eef_to_camera)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if calibrator:
            calibrator.cleanup()


if __name__ == "__main__":
    main()
