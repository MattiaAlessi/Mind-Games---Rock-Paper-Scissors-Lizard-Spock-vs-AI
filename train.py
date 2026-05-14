"""
Data collection and training script
Collect hand gesture samples and train both LightGBM and LSTM models
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

from config import (
    Gesture, GESTURE_NAMES, GESTURE_EMOJIS,
    DATA_DIR, MIN_CONFIDENCE_TO_CLASSIFY, LGBM_MIN_SAMPLES_PER_CLASS
)

from hand_detector import HandDetector
from gesture_classifier import GestureClassifier, DataAugmentor
from sequence_predictor import SequencePredictor, SyntheticDataGenerator


class DataCollector:
    """Collect hand gesture training data"""
    
    def __init__(self, output_dir: str = DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.hand_detector = HandDetector()
        self.data = defaultdict(list)  # gesture -> list of feature vectors
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def collect_gesture(self, gesture: Gesture, num_samples: int = 100):
        """
        Collect training samples for a gesture
        
        Args:
            gesture: Gesture enum to collect
            num_samples: Number of samples to collect
        """
        gesture_name = GESTURE_NAMES[gesture]
        gesture_emoji = GESTURE_EMOJIS[gesture]
        
        print(f"\n📷 Collecting {gesture_name} {gesture_emoji}")
        print(f"Press SPACE to capture, 'N' to skip to next gesture")
        print(f"Target: {num_samples} samples\n")
        
        collected = 0
        frame_count = 0
        
        while collected < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Detect hand every 2 frames
            if frame_count % 2 == 0:
                _, landmarks, hand_detected = self.hand_detector.detect(frame)
                
                if hand_detected and landmarks:
                    # Draw skeleton
                    frame = self.hand_detector.draw_skeleton(
                        frame, landmarks, color=(0, 255, 0)
                    )
            
            # Draw UI
            h, w = frame.shape[:2]
            
            # Progress bar
            progress = int((collected / num_samples) * 30)
            progress_bar = "█" * progress + "░" * (30 - progress)
            cv2.putText(frame, f"[{progress_bar}] {collected}/{num_samples}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            
            # Instructions
            cv2.putText(frame, "SPACE: Cattura | N: Prossimo",
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            cv2.imshow(f"Collecting {gesture_emoji} {gesture_name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                # Capture
                if hand_detected and landmarks:
                    features = self.hand_detector.landmarks_to_features(landmarks)
                    if features is not None:
                        self.data[gesture.value].append(features)
                        collected += 1
                        print(f"  ✓ Catturato {collected}/{num_samples}")
            
            elif key == ord('n'):
                print(f"  ⊘ Saltato {gesture_name}")
                break
            
            elif key == ord('q'):
                print("  Abort")
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def collect_all_gestures(self, samples_per_gesture: int = 100) -> dict:
        """
        Collect samples for all gestures
        
        Args:
            samples_per_gesture: Samples per gesture
            
        Returns:
            Dict with collection statistics
        """
        print("\n🎬 === DATA COLLECTION ===\n")
        
        stats = {}
        for gesture in Gesture:
            success = self.collect_gesture(gesture, samples_per_gesture)
            stats[gesture.name] = len(self.data[gesture.value])
            
            if not success:
                break
        
        return stats
    
    def get_data_arrays(self) -> tuple:
        """
        Get feature and label arrays
        
        Returns:
            Tuple of (features_array, labels_array)
        """
        all_features = []
        all_labels = []
        
        for gesture_idx in sorted(self.data.keys()):
            features = self.data[gesture_idx]
            all_features.extend(features)
            all_labels.extend([gesture_idx] * len(features))
        
        return np.array(all_features), np.array(all_labels)
    
    def save_data(self, filename: str = 'training_data.npz'):
        """Save collected data to disk"""
        X, y = self.get_data_arrays()
        filepath = self.output_dir / filename
        
        np.savez(filepath, X=X, y=y)
        print(f"\n✓ Data saved to {filepath}")
        print(f"  Samples: {len(X)}")
        print(f"  Features per sample: {X.shape[1]}")
        
        return filepath
    
    def close(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.close()


def train_classifier(data_path: str):
    """
    Train LightGBM gesture classifier
    
    Args:
        data_path: Path to training data .npz file
    """
    print("\n🤖 === TRAINING GESTURE CLASSIFIER ===\n")
    
    # Load data
    data = np.load(data_path)
    X, y = data['X'], data['y']
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Check class distribution
    for gesture in Gesture:
        count = np.sum(y == gesture.value)
        print(f"  {GESTURE_NAMES[gesture]:12} : {count:3} samples")
    
    # Check minimum samples
    min_samples = min([np.sum(y == g.value) for g in Gesture])
    if min_samples < LGBM_MIN_SAMPLES_PER_CLASS:
        print(f"\n⚠ Warning: Minimum samples per class is {min_samples}")
        print(f"  Recommend at least {LGBM_MIN_SAMPLES_PER_CLASS}")
    
    # Data augmentation
    print("\nApplying data augmentation...")
    augmentor = DataAugmentor()
    X_augmented, y_augmented = [X], [y]
    
    for gesture in Gesture:
        mask = y == gesture.value
        gesture_samples = X[mask]
        
        if len(gesture_samples) > 0:
            # Augment each sample
            for sample in gesture_samples:
                aug = augmentor.augment_sample(sample, num_augmentations=2)
                X_augmented.append(aug)
                y_augmented.append(np.full(len(aug), gesture.value))
    
    X_train = np.vstack(X_augmented)
    y_train = np.concatenate(y_augmented)
    
    print(f"After augmentation: {len(X_train)} samples")
    
    # Train classifier
    print("\nTraining LightGBM classifier...")
    classifier = GestureClassifier()
    metrics = classifier.train(X_train, y_train, verbose=True)
    
    # Save model
    classifier.save()
    
    print(f"\n✓ Classifier training complete")
    return classifier


def train_lstm(gesture_sequence_length: int = 100, num_sequences: int = 20):
    """
    Train LSTM sequence predictor
    
    Args:
        gesture_sequence_length: Length of each synthetic sequence
        num_sequences: Number of synthetic sequences to generate
    """
    print("\n🔮 === TRAINING LSTM SEQUENCE PREDICTOR ===\n")
    
    # Generate synthetic training data
    print(f"Generating {num_sequences} synthetic gesture sequences...")
    generator = SyntheticDataGenerator()
    sequences = generator.generate_patterns(
        num_sequences=num_sequences,
        seq_length=gesture_sequence_length
    )
    
    # Train LSTM
    predictor = SequencePredictor()
    print("\nTraining LSTM...")
    history = predictor.train(sequences, verbose=True)
    
    # Save model
    predictor.save()
    
    print(f"\n✓ LSTM training complete")
    return predictor


def main():
    parser = argparse.ArgumentParser(description="Train Mind Games models")
    parser.add_argument('--collect', action='store_true',
                       help='Collect training data')
    parser.add_argument('--train-classifier', action='store_true',
                       help='Train gesture classifier')
    parser.add_argument('--train-lstm', action='store_true',
                       help='Train LSTM predictor')
    parser.add_argument('--train-all', action='store_true',
                       help='Collect data and train all models')
    parser.add_argument('--samples-per-gesture', type=int, default=100,
                       help='Samples per gesture (default: 100)')
    parser.add_argument('--data-file', type=str, default='training_data.npz',
                       help='Training data file')
    
    args = parser.parse_args()
    
    data_file = None
    
    # Collect data
    if args.collect or args.train_all:
        collector = DataCollector()
        try:
            collector.collect_all_gestures(args.samples_per_gesture)
            data_file = collector.save_data(args.data_file)
        finally:
            collector.close()
    
    # Train classifier
    if args.train_classifier or args.train_all:
        if data_file is None:
            data_file = Path(DATA_DIR) / args.data_file
        
        if data_file.exists():
            train_classifier(str(data_file))
        else:
            print(f"✗ Data file not found: {data_file}")
    
    # Train LSTM
    if args.train_lstm or args.train_all:
        train_lstm()
    
    print("\n✓ All training complete!")
    print("You can now run: python game_ui.py\n")


if __name__ == '__main__':
    main()
