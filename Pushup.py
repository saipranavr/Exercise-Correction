import cv2
import numpy as np
import argparse
from Exercise import Exercise

# Constants for MediaPipe Pose landmarks
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16
LEFT_HIP, LEFT_KNEE = 23, 25
RIGHT_HIP, RIGHT_KNEE = 24, 26

# Constants for Push-up analysis
ELBOW_ANGLE_THRESHOLD_UP = 160  # Angle considered "up"
ELBOW_ANGLE_THRESHOLD_DOWN = 90   # Angle considered "down"
HIP_ANGLE_THRESHOLD_STRAIGHT_MIN = 150 # Min angle for straight body
HIP_ANGLE_THRESHOLD_STRAIGHT_MAX = 190 # Max angle for straight body (allows slight arch)

STAGE_UP = "up"
STAGE_DOWN = "down"

class Pushup(Exercise):
    """
    Analyzes Push-up exercises using pose detection.
    Counts repetitions and provides feedback based on elbow and hip angles.
    """
    def __init__(self, video_source=0):
        """
        Initializes the Pushup analyzer.

        Args:
            video_source (int or str): Path to video file or camera index.
        """
        super().__init__(video_source)
        self.rep_count = 0
        self.stage = STAGE_UP # Start in the 'up' position
        self.feedback = "Lower body"
        # Use average of left/right side for more robust tracking if one side is obscured
        self.elbow_angle = None
        self.hip_angle = None

    def _calculate_average_angle(self, img, p1_l, p2_l, p3_l, p1_r, p2_r, p3_r, draw=True):
        """Calculates the average angle of left and right sides."""
        angle_l = self.get_angle(img, p1_l, p2_l, p3_l, draw=draw)
        angle_r = self.get_angle(img, p1_r, p2_r, p3_r, draw=draw)

        if angle_l is not None and angle_r is not None:
            return (angle_l + angle_r) / 2
        elif angle_l is not None:
            return angle_l
        elif angle_r is not None:
            return angle_r
        else:
            return None

    def analyze_frame(self, img):
        """
        Analyzes the current frame for push-up state and provides feedback.

        Args:
            img: The current video frame with pose drawn.

        Returns:
            img: The frame with push-up analysis overlay.
        """
        # Calculate average elbow angle (more robust)
        self.elbow_angle = self._calculate_average_angle(
            img,
            LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
            RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
            draw=True # Draw both sides if visible
        )

        # Calculate average hip angle (shoulder-hip-knee) for body straightness
        self.hip_angle = self._calculate_average_angle(
            img,
            LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE,
            RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE,
            draw=False # Don't draw hip angle lines by default
        )

        # Feedback logic
        if self.elbow_angle is None or self.hip_angle is None:
            self.feedback = "Ensure body is fully visible"
        else:
            # Check body straightness first
            if not (HIP_ANGLE_THRESHOLD_STRAIGHT_MIN < self.hip_angle < HIP_ANGLE_THRESHOLD_STRAIGHT_MAX):
                if self.hip_angle < HIP_ANGLE_THRESHOLD_STRAIGHT_MIN:
                    self.feedback = "Raise hips - Keep body straight"
                else: # hip_angle > HIP_ANGLE_THRESHOLD_STRAIGHT_MAX
                    self.feedback = "Lower hips - Keep body straight"
            else:
                # Repetition counting logic based on elbow angle
                if self.elbow_angle < ELBOW_ANGLE_THRESHOLD_DOWN:
                    if self.stage == STAGE_UP:
                        self.stage = STAGE_DOWN
                        self.feedback = "Push up"
                elif self.elbow_angle > ELBOW_ANGLE_THRESHOLD_UP:
                    if self.stage == STAGE_DOWN:
                        self.stage = STAGE_UP
                        self.rep_count += 1
                        self.feedback = "Lower body"
                # Provide guidance during movement if body is straight
                elif self.stage == STAGE_UP:
                    self.feedback = "Lower body"
                elif self.stage == STAGE_DOWN:
                    self.feedback = "Push up"


        self._display_status(img)
        return img

    def _display_status(self, img):
        """Helper method to draw the status box and text on the image."""
        box_width, box_height = 250, 75
        box_color = (0, 128, 0) # Green color for pushups
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
        # Split feedback if too long
        feedback_lines = self.feedback.split(' - ')
        for i, line in enumerate(feedback_lines):
             cv2.putText(img, line, (10, feedback_y_pos + 25 + i*20), font, 0.7, text_color_value, thickness, cv2.LINE_AA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Push-up Exercise Correction using MediaPipe Pose.')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file. Uses webcam if not specified.')
    args = parser.parse_args()

    video_input = 0 if args.video is None else args.video
    video_source_msg = "webcam" if args.video is None else f"video file: {args.video}"

    print(f"Starting Push-up Correction using {video_source_msg}.")
    print("Press 'q' to quit.")

    try:
        analyzer = Pushup(video_source=video_input)
        analyzer.run()
    except IOError as e:
        print(f"ERROR: Could not open video source. {e}")
    except Exception as e:
         print(f"ERROR: An unexpected error occurred: {e}")
