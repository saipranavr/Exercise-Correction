import cv2
import time
import argparse
from PoseModule2 import PoseDetector # Use the refactored class

def main(video_source, show_fps):
    """Runs the pose landmark extraction and display."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    detector = PoseDetector()
    prev_time = 0

    print("Starting pose landmark extraction. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Video stream ended or failed to grab frame.")
            break

        # Find pose and draw landmarks
        frame = detector.find_pose(frame, draw=True)
        # Optionally get landmark data (not drawn again here)
        landmark_list = detector.find_landmarks(frame, draw=False)

        # Calculate and display FPS if requested
        if show_fps:
            current_time = time.time()
            if prev_time > 0:
                fps = 1 / (current_time - prev_time)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            prev_time = current_time

        cv2.imshow("Pose Landmark Extractor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and display pose landmarks using MediaPipe Pose.')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file. Uses webcam (index 0) if not specified.')
    parser.add_argument('--show_fps', action='store_true',
                        help='Display Frames Per Second on the output window.')
    args = parser.parse_args()

    video_input = 0 if args.video is None else args.video
    main(video_source=video_input, show_fps=args.show_fps)
