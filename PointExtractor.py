import cv2
import PoseModule2 as pm # Ensure PoseModule2.py is in the same directory
import time # To calculate FPS

# --- Configuration ---
VIDEO_SOURCE = 0  # Use 0 for webcam, or provide a path like "your_video.mp4"
SHOW_FPS = True   # Set to True to display Frames Per Second

# --- Initialization ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

# Initialize the custom Pose Detector from PoseModule2
# You might need to adjust arguments based on the PoseModule2 definition
detector = pm.posture_detector()

prev_frame_time = 0
new_frame_time = 0

print("Starting pose detection. Press 'q' to quit.")

# --- Main Loop ---
while True:
    # 1. Read a frame from the video source
    success, frame = cap.read()
    if not success:
        # If reading from a file, it might be the end of the file
        if isinstance(VIDEO_SOURCE, str):
            print("End of video file.")
        else:
            print("Error: Failed to grab frame from webcam.")
        break

    # 2. Detect the pose and draw landmarks using PoseModule2
    # The find_person method might preprocess or find the main subject
    frame = detector.find_person(frame, draw=True) # Set draw=True/False based on PoseModule2 capability
    
    # The find_landmarks method detects joints and draws them if draw=True
    # It also returns the list of landmarks (we don't use the list here)
    landmark_list = detector.find_landmarks(frame, draw=True) 

    # (Optional) Access landmark data if needed, e.g.:
    # if landmark_list:
    #    # Example: Get coordinates of the nose (landmark 0)
    #    nose_x, nose_y = landmark_list[0][1], landmark_list[0][2]
    #    # print(f"Nose position: ({nose_x}, {nose_y})")
    #    pass

    # 3. Calculate and Display FPS (Optional)
    if SHOW_FPS:
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 4. Display the frame with the pose drawn on it
    cv2.imshow("Pose Detection Output", frame)

    # 5. Check for exit key ('q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# --- Cleanup ---
cap.release()          # Release the video capture object
cv2.destroyAllWindows() # Close all OpenCV windows
print("Resources released.")