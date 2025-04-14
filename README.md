# Exercise Correction using MediaPipe Pose

This project provides tools for real-time exercise analysis and correction using computer vision. It utilizes the MediaPipe Pose library to detect human body landmarks and calculates joint angles to provide feedback on exercise form for various exercises like Bicep Curls, Overhead Press, and Squats.

## Project Structure

-   `PoseModule2.py`: Contains the `PoseDetector` class, which wraps the MediaPipe Pose functionality for detecting landmarks and calculating angles.
-   `Exercise.py`: Defines an abstract base class `Exercise` that provides common video processing and analysis structure.
-   `BicepCurl.py`: Implements exercise analysis logic specifically for Bicep Curls, inheriting from `Exercise`.
-   `OverheadPress.py`: Implements exercise analysis logic specifically for Overhead Press, inheriting from `Exercise`.
-   `Squat.py`: Implements exercise analysis logic specifically for Squats, inheriting from `Exercise`.
-   `PointExtractor.py`: A simple script to demonstrate basic pose landmark detection using `PoseModule2.py`.
-   `requirements.txt`: Lists the necessary Python packages.
-   `samples/`: Contains sample video files for testing (you might need to create this directory and add videos).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Exercise-Correction
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have Python 3.x installed.)*

## Usage

The analysis scripts can be run using either a live webcam feed or a pre-recorded video file.

### Bicep Curl Analysis

```bash
# Using webcam (tracks left arm by default)
python BicepCurl.py

# Using a video file (tracks left arm)
python BicepCurl.py --video path/to/your/bicep_curl_video.mp4

# Using a video file and tracking the right arm
python BicepCurl.py --video samples/bicep_curl/1.mp4 --side right
```

### Overhead Press Analysis

```bash
# Using webcam (tracks left arm by default)
python OverheadPress.py

# Using a video file (tracks left arm)
python OverheadPress.py --video path/to/your/overhead_press_video.mp4

# Using a video file and tracking the right arm
python OverheadPress.py --video samples/overhead_press/1.mp4 --side right
```

### Squat Analysis

```bash
# Using webcam
python Squat.py

# Using a video file
python Squat.py --video path/to/your/squat_video.mp4
# Example:
python Squat.py --video samples/squat/1.mp4
```
*(Note: The `--side` argument is not used for the Squat analysis as it primarily focuses on knee angles.)*

### Basic Landmark Extraction Demo

This script just shows the detected pose landmarks without exercise-specific analysis.

```bash
# Using webcam
python PointExtractor.py

# Using a video file and showing FPS
python PointExtractor.py --video path/to/any_video.mp4 --show_fps
```

Press 'q' in the display window to quit any of the running scripts.

## Extending

To add a new exercise:
1.  Create a new Python file (e.g., `PushUp.py`).
2.  Define a class that inherits from `Exercise` (e.g., `class PushUp(Exercise):`).
3.  Implement the `__init__` method (calling `super().__init__`) and the `analyze_frame(self, img)` method with the specific logic for the new exercise, calculating relevant angles using `self.get_angle()` and updating feedback/state.
4.  Add argument parsing and a `if __name__ == '__main__':` block similar to the existing exercise scripts.
