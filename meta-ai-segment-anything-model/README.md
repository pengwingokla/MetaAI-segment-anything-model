# Segment Anything Model

## 🚵‍♀️ Milestone 1: Torch Docker Container

### Environment Setup & Containerization
This milestone establishes a reproducible PyTorch development environment using Docker containerization. The setup ensures consistent dependency management and cross-platform compatibility for SAM model development.

For testing the docker containers, sample codes are added into main.py files. The container includes a comprehensive Python environment with all necessary dependencies for computer vision and geospatial processing.

### Docker Build & Deployment
```bash
# Build the Docker image with PyTorch and GDAL support
docker build -t working-sam .

# Launch the containerized Jupyter environment
docker-compose up --build
```

### Jupyter Notebook Access
For Jupyter notebooks, after building the container you should see a prompt saying that server is online at http://127.0.0.1:8888/tree/notebooks... Use that link to access your notebook. The Jupyter server runs on port 1128 within the container and is accessible through the Docker port mapping.

If you want to use your notebook in your local network, replace 127.0.0.1 with your computer's ip address. Then, you should be able to access it in your local network. This enables collaborative development and remote access to the notebook environment.

### Core Dependencies & Requirements
The `requirements.txt` ensures the following core frameworks are installed along with 240+ additional packages:

**Primary Deep Learning Stack:**
```
torch==2.2.2          # PyTorch deep learning framework
torchvision==0.17.2   # Computer vision utilities and models
```

**Additional Key Dependencies:**
- **Geospatial Processing**: GDAL==3.6.2, rasterio, geopandas, leafmap
- **SAM Models**: segment-anything==1.0, segment-anything-hq==0.3, segment-geospatial==0.10.4
- **Interactive Computing**: ipywidgets, ipyleaflet, folium
- **Data Science**: numpy, pandas, matplotlib, opencv-python
- **Cloud Integration**: azure-storage-blob, google-cloud-storage
- **Web Framework**: flask, django components

### Container Architecture
The Docker container is built on Python 3.9 with the following layers:
- **Base Layer**: Python 3.9 runtime environment
- **System Dependencies**: GDAL library installation for geospatial data processing
- **Python Dependencies**: Comprehensive package installation from requirements.txt
- **Jupyter Configuration**: Notebook server setup with custom port (1128) and network access
- **Volume Mounting**: Persistent storage for notebooks and data

## 🚵‍♀️ Milestone 2: Replicate SAM Implementation

### SAM Model Integration & Satellite Imagery Processing
This milestone focuses on implementing and demonstrating the core Segment Anything Model functionality specifically adapted for satellite imagery segmentation. The implementation showcases the model's zero-shot capabilities on remote sensing data.

The `sam-reimplementation.ipynb` notebook shows how to use segment satellite imagery using the Segment Anything Model (SAM). The notebook demonstrates:

**Core SAM Capabilities:**
- **Automatic Mask Generation**: Generate comprehensive segmentation masks for entire satellite images
- **Prompt-Based Segmentation**: Interactive segmentation using point prompts, bounding boxes, and text descriptions  
- **Multi-Scale Processing**: Handle various satellite image resolutions and zoom levels
- **Geospatial Integration**: Coordinate system handling and geographic reference preservation

**Technical Implementation Features:**
- **Model Loading**: Integration of pre-trained SAM weights (ViT-H, ViT-L, ViT-B variants)
- **Image Preprocessing**: Satellite imagery normalization and format conversion
- **Inference Pipeline**: Efficient batch processing and memory management
- **Visualization**: Interactive mapping with Folium and Leafmap integration
- **Export Capabilities**: Mask output in various formats (GeoJSON, shapefile, raster)

### Interactive Colab Demonstration
The content inside notebook is too big to display so please refer to this Colab environment for comprehensive demonstrations and interactive visualizations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IZ-54GI-5cp5oTfhc8_NKHkIZRp8Yw_A?usp=sharing)

**Colab Features:**
- **Interactive Maps**: Leaflet-based visualization with zoom and pan capabilities
- **Real-time Processing**: Live segmentation with immediate visual feedback  
- **Multiple Examples**: Various satellite scenes including urban, rural, and coastal areas
- **Performance Metrics**: Processing time and accuracy measurements
- **Comparison Studies**: Different SAM model variants and parameter settings

### Notebook Content Overview
The demonstration includes:
1. **Model Architecture Exploration**: Understanding SAM's encoder-decoder structure
2. **Data Loading Pipeline**: Satellite imagery acquisition and preprocessing
3. **Segmentation Workflows**: Step-by-step segmentation processes
4. **Result Analysis**: Quantitative and qualitative evaluation of outputs
5. **Interactive Widgets**: User controls for parameters and visualization options

## 🚵‍♀️ Milestone 3: Finetune the SAM model for the sidewalks dataset

### Custom Model Training & Specialization
This milestone involves adapting the pre-trained Segment Anything Model for specialized sidewalk and urban infrastructure detection from satellite imagery. The fine-tuning process optimizes the model for specific geospatial features and improves accuracy on urban planning applications.

**Training Pipeline Architecture:**
- **Dataset Preparation**: Custom sidewalk annotation dataset creation and augmentation
- **Transfer Learning**: Leveraging pre-trained SAM weights as initialization
- **Loss Function Design**: Custom loss functions optimized for infrastructure segmentation
- **Validation Strategy**: Geographic cross-validation to ensure model generalization

### Comprehensive Training Workflow
**`main.ipynb`** - Complete training processes and experimentation:

### Inference & Evaluation Pipeline  
**`predict.ipynb`** - Demonstrates the prediction capabilities on Google Earth screenshots:
- **Real-world Testing**: Evaluation on actual Google Earth satellite imagery
- **Sidewalk Detection**: Specialized detection and segmentation of pedestrian pathways
- **Performance Benchmarking**: Intersection over Union (IoU) scores and pixel accuracy metrics

## 🚵‍♀️ Milestone 4: Hugging Face App and Video Demo

### Production Deployment & Public Access
This milestone focuses on deploying the trained SAM model for public access and demonstration through web applications and comprehensive documentation. The deployment showcases the practical applications of the fine-tuned model in real-world scenarios.

### Interactive Web Application
**[Hugging Face Spaces Application](https://huggingface.co/spaces/chloecodes/sam)**

**Application Features:**
- **Real-time Image Upload**: Drag-and-drop interface for satellite image processing
- **Interactive Segmentation**: Point-click segmentation with immediate visual feedback
- **Multiple Model Options**: Selection between different SAM variants (base, fine-tuned, HQ)
- **Customizable Parameters**: Adjustable confidence thresholds and processing options
- **Download Capabilities**: Export segmentation masks in multiple formats
- **Performance Metrics**: Real-time processing time and accuracy statistics
- **Mobile Responsive**: Optimized interface for various device sizes

**Technical Implementation:**
- **Gradio Framework**: User-friendly web interface with minimal setup
- **Model Loading**: Efficient pre-loading and caching of SAM models
- **Image Processing**: Server-side processing with optimized inference pipelines
- **Result Visualization**: Interactive mask overlays and transparency controls
- **API Integration**: RESTful endpoints for programmatic access
- **Scalability**: Auto-scaling deployment on Hugging Face infrastructure

### Comprehensive Video Demonstration
**[Video Presentation](https://drive.google.com/file/d/1pfNURxCghOVTSUYtp3h53hsBx2Jq6jhZ/view?usp=sharing)**

**Video Content Overview:**
- **Project Introduction**: Background, objectives, and methodology explanation
- **Technical Walkthrough**: Step-by-step demonstration of each milestone
- **Model Architecture**: Detailed explanation of SAM components and modifications
- **Training Process**: Fine-tuning procedure and dataset preparation
- **Results Showcase**: Performance comparisons and real-world applications
- **Web App Demo**: Live demonstration of the Hugging Face application
- **Future Work**: Discussion of potential improvements and extensions