import mediapipe as mp
import math
import cv2

class PoseDetector:
    """
    Uses MediaPipe Pose to detect human pose landmarks in an image.
    """
    def __init__(self, static_mode=False, model_complexity=1, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initializes the PoseDetector.

        Args:
            static_mode: Whether to treat the input images as a batch of static
                         and possibly unrelated images, or a video stream.
            model_complexity: Complexity of the pose landmark model: 0, 1 or 2.
            smooth_landmarks: Whether to filter landmarks across different input
                              images to reduce jitter.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for
                                      pose detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for
                                     landmark tracking to be considered successful.
        """
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_mode,
                                     model_complexity=model_complexity,
                                     smooth_landmarks=smooth_landmarks,
                                     min_detection_confidence=min_detection_confidence,
                                     min_tracking_confidence=min_tracking_confidence)
        # Removed custom drawing specs to use default MediaPipe style

        self.results = None
        self.landmark_list = []

    def find_pose(self, img, draw=True):
        """Detects pose landmarks in the image.

        Args:
            img: The input image (BGR format).
            draw: Whether to draw the landmarks and connections on the image.

        Returns:
            The image with landmarks drawn (if draw=True).
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            # Use default MediaPipe drawing style
            self.mp_draw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return img

    def find_landmarks(self, img, draw=True):
        """Extracts landmark coordinates from the detected pose.

        Args:
            img: The input image (used for coordinate scaling).
            draw: Whether to draw circles on the landmark positions.

        Returns:
            A list of landmarks, where each element is [id, x, y].
            Returns an empty list if no landmarks were detected.
        """
        self.landmark_list = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append([id, cx, cy])
                # Removed drawing from here, find_pose handles it
        return self.landmark_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        """Calculates the angle formed by three landmarks.

        Args:
            img: The image to draw on (if draw=True).
            p1: Index of the first landmark.
            p2: Index of the second landmark (vertex).
            p3: Index of the third landmark.
            draw: Whether to draw the angle visualization.

        Returns:
            The calculated angle in degrees (0-360). Returns None if landmarks
            are not available.
        """
        if not self.landmark_list or not (len(self.landmark_list) > p1 and \
                                          len(self.landmark_list) > p2 and \
                                          len(self.landmark_list) > p3):
            return None

        # Get the landmark coordinates
        _, x1, y1 = self.landmark_list[p1]
        _, x2, y2 = self.landmark_list[p2]
        _, x3, y3 = self.landmark_list[p3]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        # Ensure angle is positive (0-360)
        if angle < 0:
            angle += 360

        # Draw angle value if requested
        if draw:
            # Landmarks and connections are drawn by find_pose using the default style
            # Only display angle value near the vertex
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) # Magenta text
        return angle

if __name__ == "__main__":
    # This script is intended to be used as a module.
    # Example usage can be found in scripts like BicepCurl.py or PointExtractor.py
    print("PoseModule2.py is a module and should be imported, not run directly.")

    # Minimal example for testing the module itself (optional)
    # cap = cv2.VideoCapture(0)
    # detector = PoseDetector()
    # while True:
    #     success, img = cap.read()
    #     if not success: break
    #     img = detector.find_pose(img)
    #     landmarks = detector.find_landmarks(img, draw=False)
    #     if landmarks:
    #         # Example: Calculate left elbow angle
    #         angle = detector.find_angle(img, 11, 13, 15) # LSh, LElb, LWr
    #     cv2.imshow("Pose Test", img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'): break
    # cap.release()
    # cv2.destroyAllWindows()
