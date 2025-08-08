# Drone Object Detection and Tracking with Kalman Filter

A comprehensive computer vision system for detecting and tracking cyclists and cars in drone footage using YOLOv8 object detection combined with Kalman Filter for motion prediction and tracking continuity.

## 🎯 Project Overview

This project implements a multi-stage pipeline that processes drone video footage to detect and track moving objects (cyclists and cars) using state-of-the-art object detection models and classical filtering techniques for robust tracking.

### Key Features
- **Custom Object Detection**: YOLOv8 trained on filtered VisDrone2019 dataset
- **Motion Tracking**: Kalman Filter implementation for trajectory prediction
- **Docker Integration**: Containerized TensorFlow 2.0 environment
- **Video Processing**: End-to-end pipeline from raw drone footage to tracked output
- **Performance Analysis**: Comprehensive model evaluation and dataset analysis

## 🏗️ System Architecture

The system follows a three-stage pipeline:

1. **Docker Environment Setup** - Containerized TensorFlow 2.0 environment
2. **Object Detection** - Custom YOLOv8 model trained on bicycle and car classes
3. **Motion Tracking** - Kalman Filter for trajectory prediction and tracking

## 📋 Table of Contents

- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Stages](#pipeline-stages)
- [Usage](#usage)
- [Results](#results)
- [Performance Analysis](#performance-analysis)

## 🛠️ Tech Stack

### Core Frameworks
- **Deep Learning**: PyTorch, TorchVision, Ultralytics YOLOv8
- **Computer Vision**: OpenCV, PIL
- **Scientific Computing**: NumPy, SciPy, FilterPy
- **Data Processing**: Pandas, Matplotlib, Seaborn

### Machine Learning Libraries
- **Object Detection**: YOLOv8 (Ultralytics)
- **Tracking**: Custom Kalman Filter implementation
- **Dataset**: VisDrone2019-DET (filtered)
- **Visualization**: Matplotlib, Seaborn

### Development Environment
- **Containerization**: Docker, Docker Compose
- **Environment Management**: Conda (droneenv.yml)
- **Video Processing**: Pytube, OpenCV
- **Notebooks**: Jupyter Lab

### Key Dependencies
```
- ultralytics (YOLOv8)
- torch, torchvision, torchaudio
- opencv-python
- filterpy
- numpy, scipy
- matplotlib, seaborn
- pandas
- pytube
```

## 📁 Project Structure

```
assignment-3-drone-kalman-filter/
├── 📄 Dockerfile                    # TensorFlow 2.0 container setup
├── 📄 docker-compose.yml           # Multi-container orchestration
├── 📄 droneenv.yml                 # Conda environment specification
├── 📄 requirements.txt             # Python dependencies
├── 📄 main.py                      # Main application entry point
├── 📁 extract/                     # Video processing utilities
│   ├── __init__.py
│   └── download_extract_frames.py  # Video download and frame extraction
├── 📁 object_detection/            # Object detection module
│   ├── VisDrone.yaml              # Dataset configuration
│   ├── dataset-filter.py          # Dataset filtering utilities
│   ├── dataset-modify.py          # Dataset modification tools
│   ├── eda.py                     # Exploratory data analysis
│   ├── finetuning.py              # Model fine-tuning utilities
│   ├── main_object_detection.py   # Object detection pipeline
│   ├── main_train.ipynb           # Training notebook
│   └── predict_frames.py          # Frame prediction utilities
└── 📁 kalman/                     # Kalman Filter implementation
    ├── VisDrone.txt               # Dataset metadata
    ├── book_plots.py              # Plotting utilities
    ├── filterpy.py                # FilterPy integration
    ├── kalman.py                  # Core Kalman Filter implementation
    ├── kf_internal.py             # Internal Kalman functions
    ├── main_kalman.py             # Kalman Filter pipeline
    ├── main_pred.py               # Prediction pipeline
    ├── plot.ipynb                 # Visualization notebook
    └── plot.py                    # Plotting utilities
```

## 🚀 Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- CUDA-compatible GPU (recommended)
- At least 8GB RAM

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd assignment-3-drone-kalman-filter

# Build and run the Docker container
docker build -t dronetf .
docker run dronetf

# Alternative: Using Docker Compose
docker-compose up --build
```

### Local Development Setup

```bash
# Create conda environment
conda env create -f droneenv.yml
conda activate drone-env

# Install additional dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLOv8 ready')"
```

## 🔄 Pipeline Stages

### Stage 1: Docker Environment Setup

**Purpose**: Establish reproducible TensorFlow 2.0 environment for model training and inference

**Components**:
- **Dockerfile**: Multi-stage build with TensorFlow 2.0 and dependencies
- **Docker Compose**: Orchestrates multiple services
- **Environment**: Isolated Python environment with GPU support

**Usage**:
```bash
docker build -t dronetf .
docker run dronetf
```

**Features**:
- TensorFlow 2.0 with GPU support
- Pre-installed computer vision libraries
- Jupyter Lab for interactive development
- Volume mounting for data persistence

### Stage 2: Object Detection Pipeline

**Location**: `object_detection/`

**Purpose**: Detect cyclists and cars in drone footage using custom-trained YOLOv8

#### Dataset Preparation

**VisDrone2019 Dataset Processing**:
- **Source**: VisDrone2019-DET dataset (2.3GB)
- **Classes**: Filtered to focus on bicycle (class 2) and car (class 3)
- **Preprocessing**: Format conversion from VisDrone to YOLO format
- **Data Split**: Train/validation splits with balanced sampling

**Key Scripts**:
1. **`dataset-filter.py`**: Filters original VisDrone dataset to bicycle and car classes
2. **`dataset-modify.py`**: Applies data transformations and augmentations
3. **`eda.py`**: Performs exploratory data analysis

**Dataset Statistics**:
- **Training Images**: 6,471 images
- **Validation Images**: 548 images
- **Classes**: 4 (pedestrian, people, bicycle, car)
- **Challenge**: Dataset imbalance (more cars than cyclists)

#### Model Training

**Architecture**: YOLOv8 (You Only Look Once v8)
- **Input Size**: Variable (auto-scaling)
- **Classes**: 4 object categories
- **Training Strategy**: Transfer learning from COCO pre-trained weights
- **Optimization**: Adam optimizer with learning rate scheduling

**Training Configuration**:
```yaml
# VisDrone.yaml
path: ../2-ODmodel-custom/custom
train: train/images
val: val/images
names:
  0: pedestrian
  1: people  
  2: bicycle
  3: car
```

**Training Process**:
```bash
# Open training notebook
jupyter notebook main_train.ipynb

# Or run training script
python main_object_detection.py
```

#### Model Evaluation

**Performance Metrics**:
- **mAP (mean Average Precision)**: Overall detection accuracy
- **Precision/Recall**: Class-specific performance
- **F1 Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: Bounding box accuracy

**Validation Process**:
1. **Frame Prediction**: Test model on individual frames
2. **Video Processing**: Apply model to full video sequences
3. **Performance Analysis**: Generate confusion matrices and metric plots

### Stage 3: Kalman Filter Tracking

**Location**: `kalman/`

**Purpose**: Track detected objects across video frames using motion prediction

#### Kalman Filter Implementation

**Mathematical Model**:
```python
# State Vector: [x, y, vx, vy] - position and velocity
# Prediction Step: X(k|k-1) = F * X(k-1|k-1) + B * U(k)
# Update Step: X(k|k) = X(k|k-1) + K * (Z(k) - H * X(k|k-1))
```

**Key Components**:
1. **State Estimation**: Position and velocity tracking
2. **Motion Model**: Constant velocity assumption
3. **Measurement Update**: Integration of detection results
4. **Noise Modeling**: Process and measurement noise handling

**Core Functions**:
- **`prediction()`**: Predict next state based on motion model
- **`update()`**: Correct prediction using measurements
- **`load_car_positions()`**: Load detection results from CSV

#### Tracking Pipeline

**Input**: Object detection results (bounding box coordinates)
**Output**: Predicted trajectories and tracking visualizations

**Process Flow**:
1. **Initialize**: Set up Kalman filter parameters
2. **Predict**: Estimate next position based on previous state
3. **Measure**: Get new detection from YOLOv8
4. **Update**: Correct prediction using measurement
5. **Track**: Maintain object identity across frames

**Usage**:
```bash
# Run Kalman filter on detection results
python kalman/main_kalman.py

# Generate tracking visualizations
jupyter notebook kalman/plot.ipynb

# Apply to video sequence
python kalman/main_pred.py
```

## 📊 Results

### Demo Videos and Outputs

#### Object Detection Results
- **Location**: [Google Drive - Object Detection Videos](https://drive.google.com/drive/folders/1rpOvINEG87zVAyD-nCcOF6t6y0vgJ7T0?usp=sharing)
- **Content**: Processed drone videos with bounding box overlays
- **Features**: Real-time detection confidence scores and class labels

#### Model Evaluation Reports
- **Location**: [Google Drive - Model Evaluation](https://drive.google.com/drive/folders/1RQNx_mMQAHIdYgOIAWFEJ1oC_wP86Vad?usp=sharing)
- **Metrics**: Precision, recall, F1-scores, confusion matrices
- **Analysis**: Class-wise performance breakdown and improvement suggestions

#### Kalman Tracking Videos
- **Location**: [Google Drive - Kalman Tracking](https://drive.google.com/file/d/1JR6Qwm_zHE3128PMVOrdfBHu0sTBgJdJ/view?usp=sharing)
- **Content**: Videos showing predicted trajectories and tracking continuity
- **Features**: Center point tracking and motion prediction visualization

### Key Achievements

1. **Successful Object Detection**:
   - High accuracy detection of cyclists and cars
   - Robust performance across various lighting conditions
   - Real-time processing capability

2. **Effective Motion Tracking**:
   - Smooth trajectory prediction using Kalman filters
   - Maintenance of object identity across frames
   - Handling of temporary occlusions

3. **Comprehensive Pipeline**:
   - End-to-end processing from raw video to tracked output
   - Containerized deployment for reproducibility
   - Extensive performance evaluation and analysis

## 📈 Performance Analysis

### Dataset Analysis (EDA Results)

**Class Distribution**:
- **Imbalanced Dataset**: More car instances than bicycle instances
- **Impact**: Model bias towards car detection
- **Mitigation**: Data augmentation and class-weighted loss functions

**Spatial Distribution**:
- **Object Sizes**: Variety of object scales in drone footage
- **Positioning**: Objects appear throughout frame regions
- **Challenges**: Small object detection at high altitudes

### Model Performance

**YOLOv8 Detection Metrics**:
- **Overall mAP**: Competitive performance on VisDrone test set
- **Class-Specific Performance**:
  - Cars: Higher precision due to dataset abundance
  - Bicycles: Lower recall due to class imbalance
  - Pedestrians/People: Moderate performance

**Tracking Performance**:
- **Trajectory Smoothness**: Kalman filter reduces detection jitter
- **Identity Maintenance**: Successful tracking across short occlusions
- **Prediction Accuracy**: Good motion model for constant velocity assumption

### Challenges and Solutions

**Challenges Identified**:
1. **Dataset Imbalance**: Unequal class representation
2. **Small Objects**: Detection difficulty at drone altitudes
3. **Motion Complexity**: Non-linear motion patterns
4. **Occlusions**: Temporary object disappearance

**Solutions Implemented**:
1. **Data Augmentation**: Synthetic bicycle instance generation
2. **Multi-Scale Training**: Various input resolutions during training
3. **Adaptive Filtering**: Dynamic Kalman filter parameter adjustment
4. **Track Management**: Identity preservation during occlusions

## 🔮 Future Enhancements

### Model Improvements
- **Multi-Object Tracking**: Hungarian algorithm for association
- **Deep Learning Tracking**: Integration of DeepSORT or FairMOT
- **3D Tracking**: Stereo vision or monocular depth estimation

### System Enhancements
- **Real-Time Processing**: Edge deployment optimization
- **Multi-Camera**: Fusion of multiple drone perspectives
- **Behavioral Analysis**: Activity recognition and pattern detection

### Deployment Options
- **Edge Computing**: NVIDIA Jetson or similar hardware
- **Cloud Processing**: Scalable video analytics pipeline
- **Mobile Integration**: Smartphone-based drone control

## 📚 References and Resources

### Datasets
- [VisDrone2019-DET Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [COCO Dataset](https://cocodataset.org/)

### Frameworks and Libraries
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FilterPy - Kalman Filtering](https://github.com/rlabbe/filterpy)
- [PyTorch](https://pytorch.org/)

### Research Papers
- YOLOv8: *Ultralytics YOLOv8*
- VisDrone: *Vision meets Drones: A Challenge*
- Kalman Filter: *A New Approach to Linear Filtering and Prediction Problems*

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Docker Documentation](https://docs.docker.com/)

---

**NJIT CS370 - Artificial Intelligence**  
*Drone Object Detection and Tracking System*