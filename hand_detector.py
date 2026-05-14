"""
Hand detection and feature extraction using MediaPipe
Handles real-time hand landmark detection and skeleton drawing
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Optional, Tuple, List

from config import (
    MP_HANDS_MAX_HANDS, MP_HANDS_MIN_CONFIDENCE,
    NUM_LANDMARKS, FEATURES_PER_LANDMARK,
    HAND_CONNECTIONS, SMOOTHING_WINDOW
)


class HandDetector:
    """Real-time hand detection and landmark extraction"""
    
    def __init__(self, max_hands: int = MP_HANDS_MAX_HANDS,
                 min_confidence: float = MP_HANDS_MIN_CONFIDENCE):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Temporal smoothing buffers
        self.landmarks_history = deque(maxlen=SMOOTHING_WINDOW)
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List], bool]:
        """
        Detect hand landmarks in frame
        
        Args:
            frame: Input BGR image
            
        Returns:
            Tuple of (frame, landmarks_list, hand_detected)
            landmarks_list: List of 21 landmarks with x, y, z coordinates
        """
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_detected = results.multi_hand_landmarks is not None
        landmarks_list = None
        
        if hand_detected and results.multi_hand_landmarks:
            # Extract first hand (max_num_hands=1)
            landmarks = results.multi_hand_landmarks[0].landmark
            landmarks_list = [(lm.x, lm.y, lm.z) for lm in landmarks]
            
            # Apply temporal smoothing
            self.landmarks_history.append(landmarks_list)
            
        return frame, landmarks_list, hand_detected
    
    def landmarks_to_features(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Convert landmark list to feature vector (63 values: 21 points × 3 coords)
        
        Args:
            landmarks: List of (x, y, z) tuples
            
        Returns:
            1D numpy array of shape (63,)
        """
        if landmarks is None or len(landmarks) != NUM_LANDMARKS:
            return None
        
        features = np.array(landmarks).flatten()  # Shape: (63,)
        return features
    
    def draw_skeleton(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]],
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2, point_radius: int = 5) -> np.ndarray:
        """
        Draw hand skeleton (bones and joints) on frame
        
        Args:
            frame: Input frame
            landmarks: List of normalized (x, y, z) coordinates
            color: BGR color for skeleton
            thickness: Line thickness
            point_radius: Radius of joint circles
            
        Returns:
            Frame with skeleton drawn
        """
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Convert normalized coords to pixel coords
        coords_pixel = []
        for x, y, z in landmarks:
            px = int(x * w)
            py = int(y * h)
            # Clamp to frame bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            coords_pixel.append((px, py))
        
        # Draw bones (connections)
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = coords_pixel[start_idx]
            pt2 = coords_pixel[end_idx]
            cv2.line(frame, pt1, pt2, color, thickness)
        
        # Draw joints (circles)
        for px, py in coords_pixel:
            cv2.circle(frame, (px, py), point_radius, color, -1)
            # Draw outline
            cv2.circle(frame, (px, py), point_radius, (255, 255, 255), 1)
        
        return frame
    
    def get_smoothed_landmarks(self) -> Optional[List[Tuple[float, float, float]]]:
        """
        Get temporally smoothed landmarks (average of recent frames)
        Reduces jitter in skeleton visualization
        
        Returns:
            Smoothed landmarks or None if no history
        """
        if not self.landmarks_history:
            return None
        
        # Average across the deque
        history_array = np.array(list(self.landmarks_history))  # (N, 21, 3)
        smoothed = np.mean(history_array, axis=0)  # (21, 3)
        
        return [(float(x), float(y), float(z)) for x, y, z in smoothed]
    
    def reset(self):
        """Clear history buffers"""
        self.landmarks_history.clear()
    
    def close(self):
        """Release MediaPipe resources"""
        self.hands.close()
