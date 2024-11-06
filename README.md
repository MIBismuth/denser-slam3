# 🎯 DenSer-SLAM

⚠️ **IMPORTANT NOTICE**: This project is currently under heavy development and is **NOT YET FUNCTIONAL**. Basically jerry-rigged it in two days to get minimal functionality.

## 💡 Overview

DenSer-SLAM aims to densify traditional SLAM output by combining sparse SLAM tracking with AI-powered depth estimation. The project integrates [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) for robust camera tracking and sparse mapping with [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) for dense, per-pixel depth estimation. By fusing these technologies, we work towards creating a more complete and detailed 3D reconstruction of the environment.

The system utilizes shared memory for real-time communication between C++ (ORB-SLAM3) and Python (Depth-Anything) components, enabling efficient data transfer with minimal latency.

## 🚀 Features

- Real-time monocular SLAM using ORB-SLAM3
- Deep learning-based depth estimation with Depth-Anything
- High-performance shared memory communication between C++ and Python
- Python integration for advanced data processing
- Point cloud generation capabilities (🚧 Under maintenance)

## 📋 Prerequisites

- Python 3.11 (recommended)
- ORB-SLAM3 dependencies
- Environment variable `ORB_SLAM3_ROOT_DIR` set to your ORB-SLAM3 installation path

## 🛠️ Installation

### 1. Setting up ORB-SLAM3

```bash
# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)
```

Copy the following files from your ORB-SLAM3 installation:
- Copy `Vocabulary/` directory → `./Vocabulary/`
- Copy `Monocular/` directory → `./Monocular/`
- Add your datasets to `Datasets/`

### 2. Setting up Depth-Anything

```bash
# Install Python dependencies
pip install -r requirements.txt

# Copy the Depth-Anything model
# Download and place the TorchHub model in the specified directory
```

## 🎮 Usage

### Running ORB-SLAM3 Component

```bash
./build/orbimp \
    Vocabulary/ORBvoc.txt \
    Monocular/EuRoC.yaml \
    Datasets/EuRoc/MH01 \
    Monocular/EuRoC_TimeStamps/MH01.txt \
    dataset-MH01_mono
```

### Running Depth Estimation

```bash
python main.py  # Run while running the ORB_SLam3
```

## 🚧 Known Issues

- Point cloud generation is currently under maintenance
- Please check the issues tab for updates or to report new issues

## 📝 To-Do

- [ ] Fix point cloud generation functionality (that's the whole point after all)
- [ ] Add more documentation


