# VisionID DeepStream

## Project Description

This project implements a real-time computer vision pipeline using NVIDIA DeepStream that:

1. **Detects persons** in video streams using YOLO11
2. **Tracks detected persons** using DeepSORT algorithm for consistent object tracking across frames
3. **Detects faces in parallel** for identification and analysis

The system leverages DeepStream's GPU-accelerated processing capabilities to perform these tasks efficiently in real-time.

## Key Components

- **Person Detection**: YOLO11-based person detection model
- **Object Tracking**: NvDeepSORT tracker for multi-object tracking
- **Face Detection**: Parallel face detection pipeline using YOLO11
- **Video Processing**: DeepStream-based video analytics pipeline

## Configuration Files

- `config_infer_primary_yolo11.txt` - YOLO11 person detection configuration
- `config_infer_primary_yolo11_face.txt` - YOLO11 face detection configuration
- `config_tracker_NvDeepSORT.yml` - DeepSORT tracker configuration
- `app.py` - Main application pipeline

## Requirements

- NVIDIA DeepStream SDK
- CUDA-capable GPU
- Python 3.x
- PyGObject and GStreamer bindings

