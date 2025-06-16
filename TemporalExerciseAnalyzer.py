import numpy as np
import cv2
from PoseModule2 import PoseDetector
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import mediapipe as mp

class TemporalPoseDataset(Dataset):
    """Dataset for temporal pose sequences"""
    def __init__(self, sequences: List[np.ndarray], labels: List[int], sequence_length: int = 30):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class TemporalPoseModel(nn.Module):
    """Neural network for temporal pose analysis"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 3):
        super(TemporalPoseModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

class TemporalExerciseAnalyzer:
    def __init__(self, model_path: str = None):
        self.pose_detector = PoseDetector()
        self.sequence_length = 30  # Number of frames to consider for temporal analysis
        self.current_sequence = []
        
        # Initialize the model
        self.model = TemporalPoseModel(
            input_size=33 * 3,  # 33 landmarks with x,y,z coordinates
            hidden_size=128,
            num_layers=2,
            num_classes=3  # [correct, incorrect, neutral]
        )
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract pose landmarks from a single frame"""
        self.pose_detector.find_pose(frame, draw=False)
        landmarks = self.pose_detector.find_landmarks(frame, draw=False)
        
        if not landmarks:
            return None
            
        # Convert landmarks to a flat array of coordinates
        pose_data = []
        for landmark in landmarks:
            pose_data.extend([landmark[1], landmark[2], 0])  # x, y, z (z=0 for 2D)
        
        return np.array(pose_data)
    
    def process_video(self, video_path: str) -> Dict[str, List[float]]:
        """Process a video file and return predictions for each frame"""
        cap = cv2.VideoCapture(video_path)
        predictions = []
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
                
                # Make prediction if we have enough frames
                if len(self.current_sequence) == self.sequence_length:
                    sequence = np.array(self.current_sequence)
                    with torch.no_grad():
                        prediction = self.model(torch.FloatTensor(sequence).unsqueeze(0))
                        frame_prediction = torch.softmax(prediction, dim=1).numpy()[0]
                        frame_predictions.append(frame_prediction)
            
            # Display frame with predictions
            if len(frame_predictions) > 0:
                current_pred = frame_predictions[-1]
                cv2.putText(frame, f"Correct: {current_pred[0]:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Incorrect: {current_pred[1]:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Exercise Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return {
            'frame_predictions': frame_predictions,
            'overall_prediction': np.mean(frame_predictions, axis=0) if frame_predictions else None
        }
    
    def train(self, train_data: List[np.ndarray], train_labels: List[int],
              epochs: int = 10, batch_size: int = 32):
        """Train the temporal model on pose sequences"""
        dataset = TemporalPoseDataset(train_data, train_labels, self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    # Example usage
    analyzer = TemporalExerciseAnalyzer()
    
    # Process a video file
    results = analyzer.process_video("path_to_your_video.mp4")
    
    # Print overall prediction
    if results['overall_prediction'] is not None:
        print("Overall exercise form prediction:")
        print(f"Correct: {results['overall_prediction'][0]:.2f}")
        print(f"Incorrect: {results['overall_prediction'][1]:.2f}")
        print(f"Neutral: {results['overall_prediction'][2]:.2f}") 