# Exercise Correction System Overview

## Project Description
This project is a real-time exercise form analysis system that uses computer vision and deep learning to detect and correct exercise form. It combines MediaPipe Pose detection with custom deep learning models to provide real-time feedback on exercise technique.

## Core Components

### 1. Pose Detection Module (`PoseModule2.py`)
- Wrapper around MediaPipe Pose for detecting human body landmarks
- Key features:
  - Real-time pose detection
  - Landmark coordinate extraction
  - Angle calculation between joints
  - Drawing utilities for visualization
- Uses MediaPipe's pose detection model with configurable parameters:
  - `static_mode`: For batch processing vs. video stream
  - `model_complexity`: 0, 1, or 2 (higher = more accurate but slower)
  - `smooth_landmarks`: Reduces jitter in landmark detection
  - `min_detection_confidence`: Threshold for pose detection
  - `min_tracking_confidence`: Threshold for landmark tracking

### 2. Base Exercise Class (`Exercise.py`)
- Abstract base class for all exercise analyzers
- Common functionality:
  - Video capture and processing
  - Pose detection integration
  - Frame processing pipeline
  - Angle calculation utilities
- Subclasses implement specific exercise analysis logic

### 3. Exercise-Specific Analyzers
Each exercise has its own analyzer class inheriting from `Exercise`:

#### Bicep Curl (`BicepCurl.py`)
- Tracks arm angles for curl form
- Monitors:
  - Elbow angle (up/down position)
  - Shoulder stability
  - Rep counting
- Configurable for left/right arm tracking

#### Squat (`Squat.py`)
- Analyzes squat form
- Monitors:
  - Knee angles
  - Hip depth
  - Back alignment
  - Feet positioning

#### Push-up (`Pushup.py`)
- Tracks push-up form
- Monitors:
  - Elbow angles
  - Body straightness
  - Hip alignment
  - Rep counting

#### Plank (`Plank.py`)
- Analyzes plank form
- Monitors:
  - Body straightness
  - Hip and knee angles
  - Core engagement

#### Overhead Press (`OverheadPress.py`)
- Tracks overhead press form
- Monitors:
  - Shoulder angles
  - Elbow alignment
  - Rep counting
- Configurable for left/right arm tracking

### 4. Advanced Analysis (`SignatureExerciseAnalyzer.py`)
- Combines temporal and signature-based analysis
- Features:
  - Real-time form classification
  - Technique error detection
  - Attention-based feedback
  - Correction suggestions
- Uses two models:
  1. Temporal Model (LSTM):
     - Input: 33 landmarks × 3 coordinates
     - Output: Form correctness (3 classes)
  2. Signature Model (GCN):
     - Input: 25 normalized joints
     - Output: 12 exercise classes

### 5. Pose Normalization (`pose_normalization.py`)
- Converts MediaPipe 33-point format to OpenPose 25-point format
- Implements canonical pose normalization
- Handles coordinate transformations

## Model Architecture

### Signature Model (GCN)
- Graph Convolutional Network
- Input: 25 joints × 3 coordinates
- Hidden layers: 256 features
- Output: 12 exercise classes
- Features:
  - DCT transformation
  - Attention mechanism
  - Correction generation

### Temporal Model (LSTM)
- Bidirectional LSTM
- Input: 33 landmarks × 3 coordinates
- Hidden size: 128
- 2 layers
- Output: 3 classes (correct/incorrect/neutral)

## Usage

### Basic Exercise Analysis
```bash
# Bicep Curl
python BicepCurl.py --video path/to/video.mp4 --side left

# Squat
python Squat.py --video path/to/video.mp4

# Push-up
python Pushup.py --video path/to/video.mp4

# Plank
python Plank.py --video path/to/video.mp4

# Overhead Press
python OverheadPress.py --video path/to/video.mp4 --side right
```

### Advanced Analysis
```bash
python SignatureExerciseAnalyzer.py --video path/to/video.mp4 --model path/to/model.pth
```

## Dependencies
- OpenCV
- MediaPipe
- PyTorch
- NumPy
- TensorFlow (for MediaPipe)

## Common Issues and Solutions

### MediaPipe Warnings
- TensorFlow Lite XNNPACK delegate warning: Normal, indicates CPU optimization
- Feedback manager warning: Can be ignored, doesn't affect functionality
- Landmark projection warning: Handled internally by MediaPipe

### CUDA/GPU Usage
- Models automatically use CUDA if available
- Falls back to CPU if CUDA not available
- No manual configuration needed

### Performance Considerations
- Higher model complexity = better accuracy but slower performance
- Smooth landmarks reduces jitter but adds latency
- Adjust confidence thresholds based on needs

## Future Improvements
1. Add more exercise types
2. Implement real-time correction suggestions
3. Add exercise progression tracking
4. Improve model accuracy with more training data
5. Add support for multiple people in frame
6. Implement exercise-specific metrics and goals 