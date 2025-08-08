# Segment Anything Model for Remote Sensing Applications

A comprehensive computer vision project implementing Meta AI's Segment Anything Model (SAM) for satellite imagery segmentation, with a focus on sidewalk and infrastructure detection from remote sensing data.

## 🎯 Project Overview

This project explores the application of foundational models in computer vision, specifically leveraging the Segment Anything Model (SAM) to segment sidewalks and other urban features from satellite imagery. The work demonstrates the adaptation of large-scale foundation models for specialized remote sensing applications.

### Key Objectives
- **Foundation Model Application**: Implement SAM for remote sensing use cases
- **Sidewalk Segmentation**: Specialized detection of sidewalks from satellite imagery
- **Model Fine-tuning**: Adapt pre-trained SAM for specific geospatial features
- **Production Deployment**: Deploy model via Hugging Face Spaces and web applications

## 🏗️ System Architecture

The project follows a four-milestone development approach:

1. **Environment Setup** - Docker containerization with PyTorch
2. **SAM Implementation** - Base model integration and testing
3. **Model Fine-tuning** - Custom training on sidewalk datasets
4. **Production Deployment** - Web application and demonstration

## 📋 Table of Contents

- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Project Milestones](#project-milestones)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [References](#references)

## 🛠️ Tech Stack

### Core Frameworks
- **Deep Learning**: PyTorch 2.2.2, TorchVision 0.17.2
- **Computer Vision**: Segment Anything Model (SAM), OpenCV
- **Geospatial**: GDAL, Rasterio, Geopandas, Leafmap
- **Interactive Computing**: Jupyter Notebooks, IPyWidgets

### Model & Data Processing
- **Foundation Model**: segment-anything, segment-anything-hq
- **Geospatial Processing**: segment-geospatial, pystac-client
- **Data Handling**: NumPy, Pandas, Pillow
- **Visualization**: Matplotlib, Folium, IPyLeaflet

### Deployment & Infrastructure
- **Containerization**: Docker, Docker Compose
- **Web Framework**: Flask, Django (components)
- **Cloud Storage**: Azure Blob Storage, Google Cloud Storage
- **Model Hosting**: Hugging Face Spaces, Transformers

### Development Environment
- **Notebooks**: Jupyter with geospatial extensions
- **Data Visualization**: Interactive maps with Folium and Leafmap
- **Model Management**: Hugging Face Hub integration
- **Package Management**: Comprehensive requirements with 240+ dependencies

## 🚀 Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- CUDA-compatible GPU (recommended for training)
- At least 16GB RAM (for large model inference)

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd MetaAI-segment-anything-model

# Build and run the container
cd meta-ai-segment-anything-model
docker build -t working-sam .
docker-compose up --build
```

### Access Jupyter Environment

After building the container:
1. Look for the server prompt: `http://127.0.0.1:8888/tree/notebooks...`
2. Use this link to access your notebooks
3. For local network access, replace `127.0.0.1` with your computer's IP address

### Local Development Setup

```bash
# Create virtual environment
python -m venv sam-env
source sam-env/bin/activate  # On Windows: sam-env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r meta-ai-segment-anything-model/requirements.txt

# Launch Jupyter
jupyter notebook
```

## 🔄 Project Milestones

### Milestone 1: Docker Environment Setup

**Objective**: Establish reproducible PyTorch environment for SAM development

**Components**:
- **Dockerfile**: Multi-stage build with Python 3.9 and PyTorch 2.2.2
- **Docker Compose**: Orchestrated Jupyter notebook service
- **Requirements**: Comprehensive dependency management with geospatial libraries

**Features**:
- PyTorch 2.2.2 with TorchVision 0.17.2
- GDAL 3.6.2 for geospatial data processing
- Jupyter notebook server on port 1128
- Volume mounting for persistent development

**Usage**:
```bash
cd meta-ai-segment-anything-model
docker build -t working-sam .
docker-compose up --build
```

### Milestone 2: SAM Implementation & Replication

**Objective**: Implement and demonstrate core SAM functionality on satellite imagery

**Key Notebooks**:
- **`sam-reimplementation.ipynb`**: Core SAM implementation and testing
- **Interactive Colab Demo**: Comprehensive satellite imagery segmentation

**Features**:
- Complete SAM model integration
- Satellite imagery preprocessing pipeline
- Interactive segmentation demonstrations
- Geospatial visualization with interactive maps

**Demo Access**:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IZ-54GI-5cp5oTfhc8_NKHkIZRp8Yw_A?usp=sharing)

**Capabilities**:
- Automatic mask generation for satellite images
- Point and box prompt-based segmentation
- Multi-scale object detection and segmentation
- Geospatial coordinate integration

### Milestone 3: Model Fine-tuning for Sidewalk Detection

**Objective**: Specialize SAM for sidewalk and infrastructure segmentation

**Key Components**:
- **Training Pipeline**: Custom fine-tuning on sidewalk datasets
- **Data Processing**: Specialized preprocessing for urban imagery
- **Model Optimization**: Adaptation for remote sensing characteristics

**Notebooks**:
- **`main.ipynb`**: Complete training workflow and experimentation
- **`predict.ipynb`**: Inference demonstration on Google Earth screenshots
- **`retrieve-parquet.ipynb`**: Dataset processing and management

**Training Features**:
- Custom dataset preparation for sidewalk segmentation
- Transfer learning from pre-trained SAM weights
- Performance monitoring and validation
- Model export for production deployment

**Evaluation Metrics**:
- Intersection over Union (IoU) for segmentation accuracy
- Precision and recall for sidewalk detection
- Visual quality assessment on diverse urban scenes
- Computational efficiency analysis

### Milestone 4: Production Deployment & Demonstration

**Objective**: Deploy trained model for public access and demonstration

**Deployment Platforms**:
- **Hugging Face Spaces**: Interactive web application
- **Video Demonstration**: Comprehensive project walkthrough

**Production Features**:
- Real-time image upload and segmentation
- Interactive result visualization
- Model performance metrics display
- User-friendly web interface

**Access Links**:
- **Live Application**: [Hugging Face Spaces](https://huggingface.co/spaces/chloecodes/sam)
- **Code Walkthrough**: [Demo Video](https://drive.google.com/file/d/1pfNURxCghOVTSUYtp3h53hsBx2Jq6jhZ/view?usp=sharing)

**Web Application Features**:
- Drag-and-drop image upload
- Real-time segmentation processing
- Interactive mask visualization
- Downloadable results
- Model performance statistics

## 📊 Model Performance

### SAM Base Model Capabilities

**Architecture**: Vision Transformer (ViT) based encoder-decoder
- **Model Size**: 2.4B parameters (ViT-H backbone)
- **Training Data**: [SA-1B dataset](https://ai.meta.com/datasets/segment-anything/) (11 million images, 1.1 billion masks)
- **Zero-shot Transfer**: Strong performance on unseen domains

### Fine-tuned Model Performance

**Sidewalk Detection Metrics**:
- **IoU Score**: 0.8
- **Processing Speed**: Real-time inference capability

### Technical Specifications

**Inference Performance**:
- **GPU Memory**: ~8GB VRAM for ViT-H model
- **Processing Time**: ~2-5 seconds per image (depending on resolution)
- **Input Resolution**: Flexible, optimized for satellite imagery scales
- **Output Quality**: High-fidelity segmentation masks

## 🌐 Deployment

### Hugging Face Spaces Application

The production model is deployed as an interactive web application on Hugging Face Spaces, providing:

- **Real-time Segmentation**: Upload satellite images for instant processing
- **Interactive Interface**: User-friendly web UI for non-technical users
- **Result Visualization**: High-quality mask overlays and downloadable outputs
- **Model Information**: Performance metrics and usage guidelines

### Local Deployment Options

```bash
# Run Flask application locally
cd meta-ai-segment-anything-model
python app.py

# Or use Docker for consistent environment
docker-compose up --build
```

## 🎓 Academic Context

**Course**: Artificial Intelligence  
**Supervisor**: [Prof. Pantelis Monogioudis](https://www.linkedin.com/in/pantelis/)  
**Author**: Uyen Nguyen  
**Institution**: New Jersery Institute of Technology

### Research Background

This project represents the computer vision equivalent of large language models like [GPT-3](https://arxiv.org/abs/2005.14165), demonstrating how foundation models can be adapted for specialized applications. The work explores:

- **Foundation Model Transfer**: Adapting general-purpose models for domain-specific tasks
- **Geospatial AI**: Application of deep learning to remote sensing data
- **Model Specialization**: Fine-tuning strategies for infrastructure detection
- **Production AI**: End-to-end deployment of research models

## 📚 References and Resources

### Research Papers
- **SAM Paper**: [Segment Anything](https://arxiv.org/abs/2304.02643) - Meta AI
- **Foundation Models**: [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)
- **Remote Sensing AI**: Computer Vision Applications in Geospatial Analysis

### Technical Resources
- **SAM Official**: [Segment Anything Model](https://segment-anything.com/)
- **Meta AI Repository**: [segment-anything](https://github.com/facebookresearch/segment-anything)
- **Geospatial SAM**: [segment-geospatial](https://github.com/opengeos/segment-geospatial)

### Datasets and Benchmarks
- **SA-1B Dataset**: Meta's 11M image segmentation dataset
- **Satellite Imagery**: Various urban and suburban scenes
- **Custom Sidewalk Dataset**: Curated for infrastructure segmentation

### Development Tools
- **PyTorch**: [Deep Learning Framework](https://pytorch.org/)
- **Hugging Face**: [Model Hub and Deployment](https://huggingface.co/)
- **Leafmap**: [Interactive Geospatial Analysis](https://leafmap.org/)

---

**Winter Internship Project**  
*Computer Vision & Remote Sensing Applications*  
*NJIT Artificial Intelligence Course*