"""
Gesture classification using XGBoost (or LightGBM fallback)
Classifies hand landmarks into one of 5 gestures: Rock, Paper, Scissors, Lizard, Spock
"""

import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Tuple

from config import (
    Gesture, NUM_FEATURES, CLASSIFIER_TYPE,
    XGB_PARAMS, LGBM_PARAMS, CV_FOLDS,
    GESTURE_MODEL_PATH, GESTURE_SCALER_PATH
)
from logger import app_logger

# Try importing the chosen classifier
if CLASSIFIER_TYPE == 'xgboost':
    try:
        import xgboost as xgb
        app_logger.info("Using XGBoost classifier")
    except ImportError:
        app_logger.warning("XGBoost not installed, falling back to LightGBM")
        import lightgbm as lgb
        CLASSIFIER_TYPE = 'lightgbm'
else:
    import lightgbm as lgb
    app_logger.info("Using LightGBM classifier")


class GestureClassifier:
    """XGBoost or LightGBM based gesture classifier"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classifier_type = CLASSIFIER_TYPE
        
    def train(self, features_list: np.ndarray, labels_list: np.ndarray,
              verbose: bool = True) -> dict:
        """
        Train classifier with cross-validation
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(features_list)
        y = labels_list
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        fold_scores = []
        
        if self.classifier_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBClassifier(**XGB_PARAMS)
        else:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(**LGBM_PARAMS)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if self.classifier_type == 'xgboost':
                self.model.fit(X_train, y_train,
                               eval_set=[(X_val, y_val)],
                               verbose=False)
            else:
                self.model.fit(X_train, y_train,
                               eval_set=[(X_val, y_val)],
                               callbacks=[lgb.early_stopping(10)],
                               verbose=False)
            
            score = self.model.score(X_val, y_val)
            fold_scores.append(score)
            
            if verbose:
                app_logger.debug(f"Fold {fold+1}/{CV_FOLDS}: Accuracy = {score:.4f}")
        
        self.is_trained = True
        
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        metrics = {
            'mean_accuracy': avg_score,
            'std_accuracy': std_score,
            'fold_scores': fold_scores
        }
        
        if verbose:
            app_logger.info(f"Training complete. Mean CV Accuracy: {avg_score:.4f} ± {std_score:.4f}")
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Tuple[Optional[Gesture], Optional[float]]:
        if not self.is_trained or self.model is None:
            return None, None
        
        if features.shape != (NUM_FEATURES,):
            return None, None
        
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        pred_class = self.model.predict(X_scaled)[0]
        pred_proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(np.max(pred_proba))
        
        gesture = Gesture(int(pred_class))
        return gesture, confidence
    
    def predict_proba(self, features: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_trained or self.model is None:
            return None
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        return proba
    
    def save(self, model_path: str = GESTURE_MODEL_PATH,
             scaler_path: str = GESTURE_SCALER_PATH):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        app_logger.info(f"Model saved to {model_path}")
        app_logger.info(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path: str = GESTURE_MODEL_PATH,
             scaler_path: str = GESTURE_SCALER_PATH) -> bool:
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            app_logger.info(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            app_logger.error("Model files not found. Training required.")
            return False
    
    def get_feature_importance(self, top_n: int = 10) -> list:
        if not self.is_trained:
            return []
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return [(idx, importances[idx]) for idx in indices]


class DataAugmentor:
    """Generate augmented training data"""
    
    @staticmethod
    def add_gaussian_noise(features: np.ndarray, std: float = 0.02) -> np.ndarray:
        noise = np.random.normal(0, std, features.shape)
        return features + noise
    
    @staticmethod
    def rotate_landmarks(landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        center = np.mean(landmarks, axis=0)
        centered = landmarks - center
        rotated = centered @ rot_matrix.T
        return rotated + center
    
    @staticmethod
    def scale_landmarks(landmarks: np.ndarray, scale: float) -> np.ndarray:
        center = np.mean(landmarks, axis=0)
        return center + scale * (landmarks - center)
    
    @staticmethod
    def augment_sample(features: np.ndarray, num_augmentations: int = 5,
                      noise_std: float = 0.02, rotation_range: float = 10,
                      scale_range: float = 0.1) -> np.ndarray:
        augmented = []
        for _ in range(num_augmentations):
            landmarks = features.reshape(21, 3)
            if np.random.rand() > 0.3:
                angle = np.random.uniform(-rotation_range, rotation_range)
                landmarks = DataAugmentor.rotate_landmarks(landmarks, angle)
            if np.random.rand() > 0.3:
                scale = 1 + np.random.uniform(-scale_range, scale_range)
                landmarks = DataAugmentor.scale_landmarks(landmarks, scale)
            aug_features = landmarks.flatten()
            aug_features = DataAugmentor.add_gaussian_noise(aug_features, noise_std)
            aug_features = np.clip(aug_features, 0, 1)
            augmented.append(aug_features)
        return np.array(augmented)