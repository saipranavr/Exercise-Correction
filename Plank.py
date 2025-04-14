import cv2
import numpy as np
import argparse
from Exercise import Exercise

# Constants for MediaPipe Pose landmarks
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Constants for Plank analysis (degrees)
HIP_ANGLE_STRAIGHT_MIN = 160  # Minimum angle for straight body at hips
HIP_ANGLE_STRAIGHT_MAX = 190  # Maximum angle for straight body at hips
KNEE_ANGLE_STRAIGHT_MIN = 165 # Minimum angle for straight legs
KNEE_ANGLE_STRAIGHT_MAX = 195 # Maximum angle for straight legs

class Plank(Exercise):
    """
    Analyzes Plank exercises using pose detection.
    Provides feedback based on hip and knee angles to maintain form.
    """
    def __init__(self, video_source=0):
        """
        Initializes the Plank analyzer.

        Args:
            video_source (int or str): Path to video file or camera index.
        """
        super().__init__(video_source)
        self.feedback = "Align body"
        self.hip_angle = None
        self.knee_angle = None

    def _calculate_average_angle(self, img, p1_l, p2_l, p3_l, p1_r, p2_r, p3_r, draw=False):
        """Calculates the average angle of left and right sides."""
        # Note: Drawing is disabled by default for plank angles to reduce clutter
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
        Analyzes the current frame for plank form and provides feedback.

        Args:
            img: The current video frame with pose drawn.

        Returns:
            img: The frame with plank analysis overlay.
        """
        # Calculate average hip angle (shoulder-hip-knee) for body straightness
        self.hip_angle = self._calculate_average_angle(
            img,
            LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE,
            RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE
        )

        # Calculate average knee angle (hip-knee-ankle) for leg straightness
        self.knee_angle = self._calculate_average_angle(
            img,
            LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
            RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        )

        # Feedback logic
        if self.hip_angle is None or self.knee_angle is None:
            self.feedback = "Ensure body is fully visible"
        else:
            hip_ok = HIP_ANGLE_STRAIGHT_MIN < self.hip_angle < HIP_ANGLE_STRAIGHT_MAX
            knee_ok = KNEE_ANGLE_STRAIGHT_MIN < self.knee_angle < KNEE_ANGLE_STRAIGHT_MAX

            if hip_ok and knee_ok:
                self.feedback = "Hold position - Body aligned"
            elif not hip_ok:
                if self.hip_angle < HIP_ANGLE_STRAIGHT_MIN:
                    self.feedback = "Raise hips"
                else: # hip_angle > HIP_ANGLE_STRAIGHT_MAX
                    self.feedback = "Lower hips"
            elif not knee_ok: # Prioritize hip feedback if both are off
                 if self.knee_angle < KNEE_ANGLE_STRAIGHT_MIN:
                    self.feedback = "Straighten knees"
                 # else: # knee_angle > KNEE_ANGLE_STRAIGHT_MAX (less common issue)
                 #    self.feedback = "Avoid locking knees" # Optional

        self._display_status(img)
        return img

    def _display_status(self, img):
        """Helper method to draw the feedback text on the image."""
        box_height = 50 # Smaller box just for feedback
        box_color = (128, 0, 128) # Purple color for plank
        text_color = (255, 255, 255) # White
        font = cv2.FONT_HERSHEY_SIMPLEX
        feedback_scale = 0.8
        thickness = 2

        # Get image dimensions
        img_h, img_w, _ = img.shape

        # Calculate text size to center it
        (text_width, text_height), _ = cv2.getTextSize(self.feedback, font, feedback_scale, thickness)

        # Position box at the bottom center
        box_y1 = img_h - box_height
        box_y2 = img_h
        box_x1 = (img_w - text_width) // 2 - 20 # Add padding
        box_x2 = (img_w + text_width) // 2 + 20

        # Ensure box coordinates are within image bounds
        box_x1 = max(0, box_x1)
        box_x2 = min(img_w, box_x2)

        # Draw background box
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), box_color, -1)

        # Display Feedback centered in the box
        text_x = (img_w - text_width) // 2
        text_y = img_h - (box_height - text_height) // 2 - 5 # Adjust vertical centering

        cv2.putText(img, self.feedback, (text_x, text_y), font, feedback_scale, text_color, thickness, cv2.LINE_AA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plank Exercise Correction using MediaPipe Pose.')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file. Uses webcam if not specified.')
    args = parser.parse_args()

    video_input = 0 if args.video is None else args.video
    video_source_msg = "webcam" if args.video is None else f"video file: {args.video}"

    print(f"Starting Plank Correction using {video_source_msg}.")
    print("Press 'q' to quit.")

    try:
        analyzer = Plank(video_source=video_input)
        analyzer.run()
    except IOError as e:
        print(f"ERROR: Could not open video source. {e}")
    except Exception as e:
         print(f"ERROR: An unexpected error occurred: {e}")
