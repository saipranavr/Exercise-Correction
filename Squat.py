import cv2
import numpy as np
import argparse
from Exercise import Exercise

# Constants for MediaPipe Pose landmarks
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28

# Constants for Squat analysis
# Note: These thresholds might need adjustment based on desired squat depth
ANGLE_THRESHOLD_UP = 165   # Angle considered "up" (legs nearly straight)
ANGLE_THRESHOLD_DOWN = 100 # Angle considered "down" (thighs parallel or lower)
STAGE_UP = "up"
STAGE_DOWN = "down"

class Squat(Exercise):
    """
    Analyzes Squat exercises using pose detection.
    Counts repetitions and provides feedback based on knee angle.
    """
    def __init__(self, video_source=0):
        """
        Initializes the Squat analyzer.

        Args:
            video_source (int or str): Path to video file or camera index.
        """
        super().__init__(video_source)
        self.rep_count = 0
        self.stage = STAGE_UP # Start in the up position
        self.feedback = "Start squatting"
        # Using left leg landmarks for angle calculation by default
        self.hip_idx = LEFT_HIP
        self.knee_idx = LEFT_KNEE
        self.ankle_idx = LEFT_ANKLE
        # Could potentially calculate both sides and average or check consistency

    def analyze_frame(self, img):
        """
        Analyzes the current frame for squat state and provides feedback.

        Args:
            img: The current video frame with pose drawn.

        Returns:
            img: The frame with squat analysis overlay.
        """
        # Calculate knee angle (using left leg landmarks)
        angle = self.get_angle(img, self.hip_idx, self.knee_idx, self.ankle_idx, draw=True)

        # Optional: Calculate right knee angle as well
        # right_angle = self.get_angle(img, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, draw=True)
        # angle = (angle + right_angle) / 2 if angle is not None and right_angle is not None else angle # Example: Average

        if angle is None:
            self.feedback = "Ensure legs are fully visible"
        else:
            # Repetition counting logic based on knee angle
            if angle < ANGLE_THRESHOLD_DOWN: # Squat down
                if self.stage == STAGE_UP:
                    self.stage = STAGE_DOWN
                    self.feedback = "Stand Up"
            elif angle > ANGLE_THRESHOLD_UP: # Stand up
                if self.stage == STAGE_DOWN:
                    self.stage = STAGE_UP
                    self.rep_count += 1
                    self.feedback = "Squat Down"

        self._display_status(img)
        return img

    def _display_status(self, img):
        """Helper method to draw the status box and text on the image."""
        box_width, box_height = 250, 75
        box_color = (0, 0, 139) # Dark Red color for variety
        text_color_label = (200, 200, 200) # Light Gray
        text_color_value = (255, 255, 255) # White
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale, value_scale = 0.6, 1.2
        thickness = 2

        # Draw background box
        cv2.rectangle(img, (0, 0), (box_width, box_height), box_color, -1)

        # Display Reps
        cv2.putText(img, 'REPS', (10, 20), font, label_scale, text_color_label, 1, cv2.LINE_AA)
        cv2.putText(img, str(self.rep_count), (15, 60), font, value_scale, text_color_value, thickness, cv2.LINE_AA)

        # Display Stage
        cv2.putText(img, 'STAGE', (100, 20), font, label_scale, text_color_label, 1, cv2.LINE_AA)
        cv2.putText(img, self.stage.upper(), (95, 60), font, value_scale, text_color_value, thickness, cv2.LINE_AA)

        # Display Feedback below the box
        feedback_y_pos = box_height + 30
        cv2.putText(img, 'FEEDBACK:', (10, feedback_y_pos), font, 0.7, text_color_value, 1, cv2.LINE_AA)
        cv2.putText(img, self.feedback, (10, feedback_y_pos + 25), font, 0.7, text_color_value, thickness, cv2.LINE_AA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Squat Exercise Correction using MediaPipe Pose.')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file. Uses webcam if not specified.')
    # Note: Removed '--side' argument as it's less relevant for squats
    args = parser.parse_args()

    video_input = 0 if args.video is None else args.video
    video_source_msg = "webcam" if args.video is None else f"video file: {args.video}"

    print(f"Starting Squat Correction using {video_source_msg}.")
    print("Press 'q' to quit.")

    try:
        analyzer = Squat(video_source=video_input)
        analyzer.run()
    except IOError as e:
        print(f"ERROR: Could not open video source. {e}")
    except Exception as e:
         print(f"ERROR: An unexpected error occurred: {e}")
