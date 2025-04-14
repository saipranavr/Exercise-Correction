import cv2
import abc
from PoseModule2 import PoseDetector # Updated import

class Exercise(abc.ABC):
    """
    Abstract base class for exercise analysis using pose detection.
    """
    def __init__(self, video_source=0):
        """
        Initializes the exercise analyzer.

        Args:
            video_source (int or str): Path to video file or camera index (default: 0).
        """
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")

        # Initialize PoseDetector with default settings
        self.detector = PoseDetector()
        self.landmark_list = []

    def _process_frame(self, frame):
        """Internal method to process a single frame for pose."""
        img = self.detector.find_pose(frame, draw=True) # Draw overall pose
        self.landmark_list = self.detector.find_landmarks(img, draw=False) # Get landmarks without drawing them individually here
        return img

    def get_angle(self, img, p1, p2, p3, draw=True):
        """Calculates angle between three landmarks using PoseDetector."""
        return self.detector.find_angle(img, p1, p2, p3, draw=draw)

    @abc.abstractmethod
    def analyze_frame(self, img):
        """
        Abstract method for exercise-specific analysis and feedback.
        Must be implemented by subclasses.

        Args:
            img: The current video frame with pose drawn.

        Returns:
            img: The frame, potentially modified with exercise feedback/drawing.
        """
        pass

    def run(self):
        """Runs the main video processing loop."""
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Video stream ended or failed to grab frame.")
                break

            # Process frame for pose
            img = self._process_frame(frame)

            # Perform exercise-specific analysis if landmarks are found
            if self.landmark_list:
                img = self.analyze_frame(img)

            cv2.imshow("Exercise Analysis", img)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._cleanup()

    def _cleanup(self):
        """Releases resources."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

    def __del__(self):
        """Ensures cleanup happens if the object is deleted unexpectedly."""
        self._cleanup()

if __name__ == '__main__':
    print("Exercise.py defines a base class and should not be run directly.")
    print("Implement a subclass (e.g., BicepCurl) and run that script.")
