# Video Retrieval via Object Detection

A comprehensive computer vision pipeline for intelligent video content retrieval using object detection and similarity search. This project implements an end-to-end system that downloads videos, extracts frames, detects objects using custom YOLO models, generates embeddings with autoencoders, and enables semantic video search through vector databases.

## 🏗️ Architecture Overview

The system follows a four-stage pipeline:

1. **Video Acquisition & Preprocessing** - YouTube video download and frame extraction
2. **Object Detection** - Custom YOLOv8 model trained on extended COCO dataset
3. **Feature Extraction** - Convolutional autoencoder for embedding generation
4. **Vector Search** - Azure Cosmos DB with pgvector for similarity retrieval

## 📋 Table of Contents

- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Stages](#pipeline-stages)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)

## 🛠️ Tech Stack

### Core Frameworks
- **Deep Learning**: TensorFlow/Keras, PyTorch, Ultralytics YOLOv8
- **Computer Vision**: OpenCV, PIL, Matplotlib
- **GUI Framework**: CustomTkinter
- **Video Processing**: Pytube, YouTube-Transcript-API

### Machine Learning Libraries
- **Object Detection**: YOLOv8 (Ultralytics)
- **Embeddings**: Custom Convolutional Autoencoder
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, TensorBoard

### Cloud & Database
- **Vector Database**: Azure Cosmos DB for PostgreSQL with pgvector
- **Dimensionality Reduction**: PCA
- **Deployment**: Azure Cloud Services

### Development Tools
- **Data Annotation**: LabelImg
- **Dataset**: COCO + Custom classes (bridges, buildings, plants)
- **Model Training**: Jupyter Notebooks
- **Performance Monitoring**: TensorBoard

## 📁 Project Structure

```
2-video-retrieval-via-object-detection/
├── 1-youtube-downloader/          # Video acquisition module
│   ├── app-YouTube-Downloader.py   # GUI-based YouTube downloader
│   ├── app-extract-frames.py       # Frame extraction utility
│   ├── app-extract-subtitles.py    # Subtitle extraction
│   ├── requirements.txt            # Dependencies
│   └── video-names.txt            # Video metadata
├── 2-detect-objects/              # Object detection module
│   └── Yolov8-custom/             # Custom YOLO implementation
│       ├── YOLO_V8_main.ipynb     # Training notebook
│       ├── classes.txt            # Custom class definitions
│       ├── coco-custom.yaml       # Model configuration
│       ├── results/               # Training results & metrics
│       └── weights/               # Trained model weights
├── 3-embedding-model/             # Feature extraction module
│   ├── ROI_bbox_compile.py        # Bounding box compilation
│   ├── ROI_crop.py               # Region of Interest cropping
│   ├── autoencoder-main.py        # Autoencoder architecture
│   ├── autoencoder-similarity-search.ipynb  # Similarity search demo
│   ├── embedding_model.py         # Embedding utilities
│   ├── autoencoder_model.keras    # Trained autoencoder model
│   └── image-cropped-rois/        # Extracted ROI images
└── 4-index-embedding-azure/       # Vector database integration
    ├── azurecv.ipynb             # Azure integration demo
    └── azure-demo.png            # Results visualization
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Azure account (for vector database)

### Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd 2-video-retrieval-via-object-detection

# Install YouTube downloader dependencies
cd 1-youtube-downloader
pip install -r requirements.txt

# Install additional dependencies for the complete pipeline
pip install ultralytics tensorflow keras torch torchvision
pip install opencv-python pillow matplotlib pandas
pip install azure-cosmos psycopg2-binary
```

## 🔄 Pipeline Stages

### Stage 1: Video Acquisition & Preprocessing

**Location**: `1-youtube-downloader/`

**Purpose**: Download YouTube videos and extract frames for analysis

**Key Components**:
- **YouTube Downloader**: GUI-based tool using Pytube for video acquisition
- **Frame Extractor**: Converts videos to individual frames for processing
- **Subtitle Extractor**: Downloads accompanying text data

**Technologies**: Pytube, CustomTkinter, YouTube-Transcript-API

**Usage**:
```bash
cd 1-youtube-downloader
python app-YouTube-Downloader.py
```

### Stage 2: Object Detection

**Location**: `2-detect-objects/Yolov8-custom/`

**Purpose**: Detect and classify objects in video frames using custom-trained YOLO model

**Key Features**:
- **Custom Dataset**: Extended COCO dataset with additional classes (bridges, buildings, plants, water, mountains, roads)
- **YOLOv8 Architecture**: State-of-the-art object detection
- **Training Pipeline**: Complete training workflow with performance monitoring

**Model Configuration**:
- **Classes**: 8 custom classes (bridge, ocean, mountain, tree, road, buildings, water)
- **Training**: 100 epochs with comprehensive metrics
- **Performance**: Precision/Recall curves, F1 scores, confusion matrices

**Usage**:
```bash
# Open training notebook
jupyter notebook YOLO_V8_main.ipynb
```

### Stage 3: Feature Extraction & Embedding

**Location**: `3-embedding-model/`

**Purpose**: Generate semantic embeddings from detected objects for similarity search

**Architecture**:
- **Convolutional Autoencoder**: Custom encoder-decoder architecture
- **Input Processing**: 128x128 grayscale images, MNIST-like preprocessing
- **Embedding Dimension**: High-dimensional feature vectors (50K+)
- **Training Monitoring**: TensorBoard integration for loss tracking

**Key Components**:
1. **ROI Extraction**: Crop regions of interest based on YOLO detections
2. **Image Preprocessing**: Normalize and resize for autoencoder input
3. **Model Training**: Custom CNN autoencoder with batch normalization
4. **Similarity Search**: Cosine similarity-based retrieval

**Model Architecture**:
```python
# Encoder: Conv2D -> BatchNorm -> LeakyReLU -> Downsample
# Decoder: Conv2DTranspose -> BatchNorm -> LeakyReLU -> Upsample
# Latent space: Compressed representation for similarity search
```

**Usage**:
```bash
cd 3-embedding-model
python autoencoder-main.py
jupyter notebook autoencoder-similarity-search.ipynb
```

### Stage 4: Vector Database & Retrieval

**Location**: `4-index-embedding-azure/`

**Purpose**: Index embeddings in vector database for fast similarity search

**Implementation**:
- **Database**: Azure Cosmos DB for PostgreSQL with pgvector extension
- **Dimensionality Reduction**: PCA to reduce embeddings to <16K dimensions
- **Query System**: Top-K similarity search using cosine distance
- **Visualization**: Search results with similarity scores

**Features**:
- **Scalable Storage**: Cloud-based vector database
- **Fast Retrieval**: Optimized similarity search
- **Visual Results**: Top-9 similar images with confidence scores

**Usage**:
```bash
cd 4-index-embedding-azure
jupyter notebook azurecv.ipynb
```

## 📊 Model Performance

### Object Detection (YOLOv8)
- **Training Duration**: 100 epochs
- **Dataset**: COCO + Custom annotations
- **Classes**: 8 object categories
- **Metrics**: Precision/Recall curves, F1 scores, mAP

### Autoencoder Embedding Model
- **Architecture**: Convolutional Encoder-Decoder
- **Input Size**: 128×128×1 (grayscale)
- **Training**: TensorBoard monitoring with epoch/iteration loss tracking
- **Performance**: Image reconstruction quality and embedding separability

### Similarity Search Results
- **Top-K Retrieval**: Configurable number of similar images
- **Similarity Metric**: Cosine similarity
- **Response Time**: Optimized for real-time queries

## 🎯 Results

### Visual Outputs

1. **Object Detection Results**:
   - Bounding box predictions with confidence scores
   - Class labels for detected objects
   - Performance metrics visualization

2. **Embedding Model Performance**:
   ![Training Progress](./3-embedding-model/autoencoder-performance/autoencoder-train-epoch.png)
   ![Training Iterations](./3-embedding-model/autoencoder-performance/autoencoder-train-epoch-iteration.png)

3. **Similarity Search Examples**:
   ![Similarity Search 1](./3-embedding-model/autoencoder-performance/similarity-search-output-1.png)
   ![Similarity Search 2](./3-embedding-model/autoencoder-performance/similarity-search-output-2.png)

4. **Azure Vector Database Demo**:
   ![Azure Integration](./4-index-embedding-azure/azure-demo.png)

### Key Achievements
- **End-to-End Pipeline**: Complete video-to-retrieval system
- **Custom Object Detection**: Extended COCO dataset with domain-specific classes
- **Semantic Search**: Content-based video frame retrieval
- **Cloud Integration**: Scalable vector database deployment
- **Performance Monitoring**: Comprehensive metrics and visualization

## 🔮 Future Enhancements

- **Multi-modal Search**: Integrate text and audio features
- **Real-time Processing**: Stream processing capabilities
- **Advanced Embeddings**: Transformer-based feature extraction
- **Mobile Deployment**: Edge computing optimization
- **Interactive Interface**: Web-based search platform

## 📚 References

- [COCO Dataset](https://cocodataset.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/)
- [pgvector Extension](https://github.com/pgvector/pgvector)

---

**NJIT CS370 - Artificial Intelligence**  
*Video Indexing and Retrieval Pipeline*