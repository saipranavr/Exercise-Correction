import numpy as np
import cv2
import torch
import torch.nn as nn
from PoseModule2 import PoseDetector
import sys
import os
import argparse

# Add the signature model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'signature_model'))
from models import GCN_corr_class_ours
from utils import dct_2d, idct_2d
import mediapipe as mp

class SignatureExerciseAnalyzer:
    def __init__(self, model_path: str = None):
        self.pose_detector = PoseDetector()
        self.sequence_length = 30  # Number of frames to consider for temporal analysis
        self.current_sequence = []
        
        # Updated: 11-class mapping for technique/mistake labels
        self.exercise_classes = {
            0: "Squats: Correct",
            1: "Squats: Feet too wide",
            2: "Squats: Knees inward",
            3: "Squats: Not low enough",
            4: "Squats: Front bent",
            5: "Lunges: Correct",
            6: "Lunges: Not low enough",
            7: "Lunges: Knee passes toe",
            8: "Planks: Correct",
            9: "Planks: Arched back",
            10: "Planks: Hunch back"
        }
        
        # Initialize the signature model
        self.signature_model = GCN_corr_class_ours(
            input_feature=25,
            hidden_feature=256,
            p_dropout=0.5,
            num_stage=2,
            node_n=57,
            classes=12
        )
        
        if model_path:
            self.signature_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
        self.signature_model.eval()
        self.signature_model = self.signature_model.cuda()  # Move to CUDA
        
        # Initialize temporal model
        self.temporal_model = nn.LSTM(
            input_size=33 * 3,  # 33 landmarks with x,y,z coordinates
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        ).cuda()  # Move to CUDA
        
        self.temporal_fc = nn.Linear(128 * 2, 3).cuda()  # Move to CUDA
        
    def normalize_pose(self, landmarks):
        """Normalize pose landmarks to be centered and scaled"""
        # Convert landmarks to numpy array
        landmarks = np.array(landmarks)
        
        # Center the pose around the hip center
        hip_center = np.mean(landmarks[[23, 24]], axis=0)  # Average of left and right hip
        landmarks = landmarks - hip_center
        
        # Scale the pose to have a consistent size
        scale = np.max(np.abs(landmarks))
        if scale > 0:
            landmarks = landmarks / scale
            
        return landmarks
        
    def rotate_pose(self, landmarks):
        """Rotate pose to align with a standard orientation"""
        # Convert landmarks to numpy array
        landmarks = np.array(landmarks)
        
        # Get shoulder and hip vectors
        shoulder_vector = landmarks[12] - landmarks[11]  # Right shoulder - Left shoulder
        hip_vector = landmarks[24] - landmarks[23]  # Right hip - Left hip
        
        # Calculate rotation angle to align shoulders horizontally
        angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Apply rotation to x,y coordinates
        landmarks[:, :2] = np.dot(landmarks[:, :2], rotation_matrix.T)
        
        return landmarks
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract pose landmarks from a single frame and draw the pose on the frame"""
        self.pose_detector.find_pose(frame, draw=True)  # Draw pose skeleton
        landmarks = self.pose_detector.find_landmarks(frame, draw=False)
        if not landmarks:
            return None
        # Convert landmarks to a flat array of coordinates
        pose_data = []
        for landmark in landmarks:
            pose_data.extend([landmark[1], landmark[2], 0])  # x, y, z (z=0 for 2D)
        # Reshape to [33, 3] for processing
        pose_data = np.array(pose_data).reshape(-1, 3)
        # Normalize and rotate the pose
        pose_data = self.normalize_pose(pose_data)
        pose_data = self.rotate_pose(pose_data)
        # Flatten back to 1D array
        return pose_data.flatten()
    
    def process_video(self, video_path: str) -> dict[str, list[float]]:
        """Process a video file and return predictions from both models"""
        cap = cv2.VideoCapture(video_path)
        temporal_predictions = []
        signature_predictions = []
        frame_predictions = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Process frame
            pose_data = self.preprocess_frame(frame)
            if pose_data is not None:
                self.current_sequence.append(pose_data)
                
                # Keep only the last sequence_length frames
                if len(self.current_sequence) > self.sequence_length:
                    self.current_sequence.pop(0)
                
                # Make predictions if we have enough frames
                if len(self.current_sequence) == self.sequence_length:
                    sequence = np.array(self.current_sequence)
                    
                    # Temporal model prediction
                    with torch.no_grad():
                        temporal_input = torch.FloatTensor(sequence).unsqueeze(0).cuda()
                        temporal_out, _ = self.temporal_model(temporal_input)
                        temporal_pred = self.temporal_fc(temporal_out[:, -1, :])
                        temporal_pred = torch.softmax(temporal_pred, dim=1).cpu().numpy()[0]
                        temporal_predictions.append(temporal_pred)
                    
                    # Signature model prediction
                    with torch.no_grad():
                        # Prepare input for signature model
                        signature_input = torch.FloatTensor(sequence).unsqueeze(0).cuda()
                        signature_input = signature_input.reshape(1, 30, 33, 3)
                        
                        # Map MediaPipe joints to model's 57-node structure
                        mapped_input = torch.zeros((1, 30, 57, 3), device='cuda')
                        mapped_input[:, :, :33, :] = signature_input
                        
                        # Reshape to [batch, nodes, features]
                        mapped_input = mapped_input.reshape(1, 57, 30*3)
                        mapped_input = mapped_input[:, :, :25]
                        
                        # Convert to DCT domain
                        dct_input = dct_2d(mapped_input)
                        
                        # Get predictions
                        deltas, att, outputs = self.signature_model(dct_input, None, False)
                        
                        # Get classification (which incorrect technique)
                        _, predicted_class = torch.max(outputs, 1)
                        predicted_class = predicted_class.cpu().numpy()[0]
                        
                        # Get correction deltas
                        corrected_sequence = idct_2d(deltas)
                        
                        # Get attention scores
                        attention_scores = att.squeeze().cpu().numpy()
                        signature_predictions.append({
                            'class': predicted_class,
                            'attention': attention_scores,
                            'correction': corrected_sequence.cpu().numpy()
                        })
                    
                    # Combine predictions
                    frame_prediction = {
                        'temporal': temporal_pred,
                        'signature': {
                            'class': predicted_class,
                            'attention': attention_scores,
                            'correction': corrected_sequence.cpu().numpy()
                        }
                    }
                    frame_predictions.append(frame_prediction)
            
            # Display frame with predictions
            if len(frame_predictions) > 0:
                current_pred = frame_predictions[-1]
                cv2.putText(frame, f"Temporal - Correct: {current_pred['temporal'][0]:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                exercise_name = self.exercise_classes.get(current_pred['signature']['class'], f"Unknown Class {current_pred['signature']['class']}")
                cv2.putText(frame, f"Technique: {exercise_name}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Attention: {np.mean(current_pred['signature']['attention']):.2f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Exercise Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return {
            'frame_predictions': frame_predictions,
            'temporal_predictions': temporal_predictions,
            'signature_predictions': signature_predictions
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signature Exercise Analyzer")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index (default: 0 for webcam)')
    parser.add_argument('--model', type=str, default="signature_model/pretrained_weights/ours.pt", help='Path to model weights')
    args = parser.parse_args()

    analyzer = SignatureExerciseAnalyzer(model_path=args.model)

    # Determine if input is int (camera) or file path
    try:
        video_input = int(args.video)
    except ValueError:
        video_input = args.video

    results = analyzer.process_video(video_input)

    # Print overall predictions
    if results['temporal_predictions']:
        print("\nOverall Temporal Analysis:")
        avg_temporal = np.mean(results['temporal_predictions'], axis=0)
        print(f"Correct: {avg_temporal[0]:.2f}")
        print(f"Incorrect: {avg_temporal[1]:.2f}")
        print(f"Neutral: {avg_temporal[2]:.2f}")

    if results['signature_predictions']:
        print("\nOverall Signature Analysis:")
        # Get most common technique class
        classes = [pred['class'] for pred in results['signature_predictions']]
        most_common_class = max(set(classes), key=classes.count)
        exercise_name = analyzer.exercise_classes.get(most_common_class, f"Unknown Class {most_common_class}")
        print(f"Detected Technique: {exercise_name}")
        print(f"Average Attention Score: {np.mean([np.mean(pred['attention']) for pred in results['signature_predictions']]):.2f}") 