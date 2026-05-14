"""
Gesture classification using LightGBM
Classifies hand landmarks into one of 5 gestures: Rock, Paper, Scissors, Lizard, Spock
"""

import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from typing import Optional, Tuple

from config import (
    Gesture, NUM_FEATURES, LGBM_PARAMS, LGBM_CV_FOLDS,
    GESTURE_MODEL_PATH, GESTURE_SCALER_PATH
)


class GestureClassifier:
    """LightGBM-based gesture classifier"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, features_list: np.ndarray, labels_list: np.ndarray,
              verbose: bool = True) -> dict:
        """
        Train LightGBM classifier with cross-validation
        
        Args:
            features_list: Array of shape (n_samples, 63)
            labels_list: Array of shape (n_samples,) with gesture indices
            verbose: Print training info
            
        Returns:
            Dict with training metrics
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(features_list)
        y = labels_list
        
        # Stratified K-Fold for balanced evaluation
        skf = StratifiedKFold(n_splits=LGBM_CV_FOLDS, shuffle=True, random_state=42)
        fold_scores = []
        
        self.model = lgb.LGBMClassifier(**LGBM_PARAMS)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(10)])
            
            score = self.model.score(X_val, y_val)
            fold_scores.append(score)
            
            if verbose:
                print(f"  Fold {fold+1}/{LGBM_CV_FOLDS}: Accuracy = {score:.4f}")
        
        self.is_trained = True
        
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        metrics = {
            'mean_accuracy': avg_score,
            'std_accuracy': std_score,
            'fold_scores': fold_scores
        }
        
        if verbose:
            print(f"\n✓ Training complete. Mean CV Accuracy: {avg_score:.4f} ± {std_score:.4f}")
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Tuple[Optional[Gesture], Optional[float]]:
        """
        Classify a single gesture
        
        Args:
            features: Array of shape (63,)
            
        Returns:
            Tuple of (Gesture enum, confidence score) or (None, None) if not trained
        """
        if not self.is_trained or self.model is None:
            return None, None
        
        if features.shape != (NUM_FEATURES,):
            return None, None
        
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get class prediction and probability
        pred_class = self.model.predict(X_scaled)[0]
        pred_proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(np.max(pred_proba))
        
        gesture = Gesture(int(pred_class))
        return gesture, confidence
    
    def predict_proba(self, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Get probability distribution over all gestures
        
        Args:
            features: Array of shape (63,)
            
        Returns:
            Array of shape (5,) with probabilities or None
        """
        if not self.is_trained or self.model is None:
            return None
        
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        
        return proba
    
    def save(self, model_path: str = GESTURE_MODEL_PATH,
             scaler_path: str = GESTURE_SCALER_PATH):
        """Save model and scaler to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")
    
    def load(self, model_path: str = GESTURE_MODEL_PATH,
             scaler_path: str = GESTURE_SCALER_PATH) -> bool:
        """Load model and scaler from disk"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print(f"✓ Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"✗ Model files not found. Training required.")
            return False
    
    def get_feature_importance(self, top_n: int = 10) -> list:
        """Get top N important features"""
        if not self.is_trained:
            return []
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return [(idx, importances[idx]) for idx in indices]


class DataAugmentor:
    """Generate augmented training data to improve model robustness"""
    
    @staticmethod
    def add_gaussian_noise(features: np.ndarray, std: float = 0.02) -> np.ndarray:
        """Add Gaussian noise to features"""
        noise = np.random.normal(0, std, features.shape)
        return features + noise
    
    @staticmethod
    def rotate_landmarks(landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate hand landmarks around their centroid
        
        Args:
            landmarks: Array of shape (21, 3) with x, y, z
            angle_deg: Rotation angle in degrees
            
        Returns:
            Rotated landmarks
        """
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        # Rotation matrix (in x-y plane, keep z)
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Center landmarks
        center = np.mean(landmarks, axis=0)
        centered = landmarks - center
        
        # Apply rotation
        rotated = centered @ rot_matrix.T
        
        # Recenter
        return rotated + center
    
    @staticmethod
    def scale_landmarks(landmarks: np.ndarray, scale: float) -> np.ndarray:
        """Scale landmarks around centroid"""
        center = np.mean(landmarks, axis=0)
        return center + scale * (landmarks - center)
    
    @staticmethod
    def augment_sample(features: np.ndarray, num_augmentations: int = 5,
                      noise_std: float = 0.02, rotation_range: float = 10,
                      scale_range: float = 0.1) -> np.ndarray:
        """
        Generate multiple augmented versions of a single sample
        
        Args:
            features: Original feature vector (63,)
            num_augmentations: Number of versions to generate
            noise_std: Gaussian noise std dev
            rotation_range: Max rotation angle in degrees
            scale_range: Max scale factor deviation
            
        Returns:
            Array of shape (num_augmentations, 63)
        """
        augmented = []
        
        for _ in range(num_augmentations):
            # Convert features back to landmarks (21, 3)
            landmarks = features.reshape(21, 3)
            
            # Apply transformations
            if np.random.rand() > 0.3:
                angle = np.random.uniform(-rotation_range, rotation_range)
                landmarks = DataAugmentor.rotate_landmarks(landmarks, angle)
            
            if np.random.rand() > 0.3:
                scale = 1 + np.random.uniform(-scale_range, scale_range)
                landmarks = DataAugmentor.scale_landmarks(landmarks, scale)
            
            # Flatten and add noise
            aug_features = landmarks.flatten()
            aug_features = DataAugmentor.add_gaussian_noise(aug_features, noise_std)
            
            # Clip to valid range [0, 1]
            aug_features = np.clip(aug_features, 0, 1)
            
            augmented.append(aug_features)
        
        return np.array(augmented)
