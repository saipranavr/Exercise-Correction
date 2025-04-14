import cv2
import numpy as np
import argparse
from Exercise import Exercise

# Constants for MediaPipe Pose landmarks
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16

# Constants for Overhead Press analysis
ANGLE_THRESHOLD_UP = 160   # Angle considered "up" (arms extended)
ANGLE_THRESHOLD_DOWN = 90  # Angle considered "down" (arms bent)
STAGE_UP = "up"
STAGE_DOWN = "down"

class OverheadPress(Exercise):
    """
    Analyzes Overhead Press exercises using pose detection.
    Counts repetitions and provides feedback based on elbow angle.
    """
    def __init__(self, video_source=0, side='left'):
        """
        Initializes the OverheadPress analyzer.

        Args:
            video_source (int or str): Path to video file or camera index.
            side (str): Arm to track ('left' or 'right'). Default is 'left'.
        """
        super().__init__(video_source)
        self.rep_count = 0
        self.stage = STAGE_DOWN # Start in the down position
        self.feedback = "Start pressing"
        self.side = side.lower()

        if self.side == 'left':
            self.shoulder_idx = LEFT_SHOULDER
            self.elbow_idx = LEFT_ELBOW
            self.wrist_idx = LEFT_WRIST
        elif self.side == 'right':
            self.shoulder_idx = RIGHT_SHOULDER
            self.elbow_idx = RIGHT_ELBOW
            self.wrist_idx = RIGHT_WRIST
        else:
            raise ValueError("Side must be 'left' or 'right'")

    def analyze_frame(self, img):
        """
        Analyzes the current frame for overhead press state and provides feedback.

        Args:
            img: The current video frame with pose drawn.

        Returns:
            img: The frame with overhead press analysis overlay.
        """
        # Calculate elbow angle
        angle = self.get_angle(img, self.shoulder_idx, self.elbow_idx, self.wrist_idx, draw=True)

        if angle is None:
            self.feedback = "Ensure arm is fully visible"
        else:
            # Repetition counting logic based on elbow angle
            if angle > ANGLE_THRESHOLD_UP: # Arms extended up
                if self.stage == STAGE_DOWN:
                    self.stage = STAGE_UP
                    self.feedback = "Lower Down"
            elif angle < ANGLE_THRESHOLD_DOWN: # Arms bent down
                if self.stage == STAGE_UP:
                    self.stage = STAGE_DOWN
                    self.rep_count += 1
                    self.feedback = "Press Up"

        self._display_status(img)
        return img

    def _display_status(self, img):
        """Helper method to draw the status box and text on the image."""
        # Reusing the display logic structure from BicepCurl
        box_width, box_height = 250, 75
        box_color = (0, 128, 0) # Greenish color for variety
        text_color_label = (0, 0, 0) # Black
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
    parser = argparse.ArgumentParser(description='Overhead Press Exercise Correction using MediaPipe Pose.')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file. Uses webcam if not specified.')
    parser.add_argument('--side', type=str, default='left', choices=['left', 'right'],
                        help='Arm to track (left or right). Default: left.')
    args = parser.parse_args()

    video_input = 0 if args.video is None else args.video
    video_source_msg = "webcam" if args.video is None else f"video file: {args.video}"

    print(f"Starting Overhead Press Correction ({args.side.capitalize()} Arm) using {video_source_msg}.")
    print("Press 'q' to quit.")

    try:
        analyzer = OverheadPress(video_source=video_input, side=args.side)
        analyzer.run()
    except IOError as e:
        print(f"ERROR: Could not open video source. {e}")
    except ValueError as e:
        print(f"ERROR: Configuration error. {e}")
    except Exception as e:
         print(f"ERROR: An unexpected error occurred: {e}")
