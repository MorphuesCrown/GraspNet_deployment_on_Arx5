# GraspNet_deployment_on_Arx5

## 1. Introduction
This repository provides the deployment solution for a robotic grasping system using **GraspNet** and the **ARX5 robotic arm**.

## 2. Environment Setup
Since the dependencies for GraspNet and the ARX5 SDK conflict, it is recommended to use **two separate environments** communicating via **ZMQ**:

### Environment 1: Control (arx5-sdk & orbbec)
This environment handles camera data collection and robotic arm control. 
For detailed information, please check the official repositories:
* **ARX5 SDK**: [real-stanford/arx5-sdk](https://github.com/real-stanford/arx5-sdk)
* **Orbbec SDK**: [orbbec/pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
```bash
conda env create -f environment_control.yml
conda activate control_env

# After activating, you may need to build the arx5-sdk python bindings:
cd arx5-sdk && mkdir build && cd build && cmake .. && make -j
```

### Environment 2: GraspNet
This environment handles graspnet computation.
* **GraspNet Baseline**: [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
```bash
conda env create -f environment_grasp.yml
conda activate grasp_env
```

## 3. Usage
1.  In the **GraspNet environment**, run the server:
    ```bash
    python grasp_server.py
    ```
2.  In the **ARX5 & Orbbec environment**, run the client:
    ```bash
    python client.py
    ```

**Note:** If the provided calibration data is inaccurate, a calibration script is included for re-calibration.
