"""
Data collection and training script
Collect hand gesture samples and train XGBoost/LightGBM model
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

from config import (
    Gesture, GESTURE_NAMES, GESTURE_EMOJIS,
    DATA_DIR, MIN_CONFIDENCE_TO_CLASSIFY, MIN_SAMPLES_PER_CLASS,
    CLASSIFIER_TYPE
)
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier, DataAugmentor
from logger import app_logger


class DataCollector:
    def __init__(self, output_dir: str = DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.hand_detector = HandDetector()
        self.data = defaultdict(list)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def collect_gesture(self, gesture: Gesture, num_samples: int = 100):
        gesture_name = GESTURE_NAMES[gesture]
        gesture_emoji = GESTURE_EMOJIS[gesture]
        print(f"\n📷 Collecting {gesture_name} {gesture_emoji}")
        print(f"Press SPACE to capture, 'N' to skip, Q to quit")
        print(f"Target: {num_samples} samples\n")
        collected = 0
        frame_count = 0
        while collected < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_count += 1
            if frame_count % 2 == 0:
                _, landmarks, hand_detected = self.hand_detector.detect(frame)
                if hand_detected and landmarks:
                    frame = self.hand_detector.draw_skeleton(frame, landmarks, color=(0, 255, 0))
            h, w = frame.shape[:2]
            progress = int((collected / num_samples) * 30)
            progress_bar = "█" * progress + "░" * (30 - progress)
            cv2.putText(frame, f"[{progress_bar}] {collected}/{num_samples}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.putText(frame, "SPACE: Cattura | N: Prossimo | Q: Esci",
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.imshow(f"Collecting {gesture_emoji} {gesture_name}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if hand_detected and landmarks:
                    features = self.hand_detector.landmarks_to_features(landmarks)
                    if features is not None:
                        self.data[gesture.value].append(features)
                        collected += 1
                        print(f"  ✓ Captured {collected}/{num_samples}")
            elif key == ord('n'):
                print(f"  ⊘ Skipped {gesture_name}")
                break
            elif key == ord('q'):
                print("  Aborted")
                return False
        cv2.destroyAllWindows()
        return True

    def collect_all_gestures(self, samples_per_gesture: int = 100) -> dict:
        print("\n🎬 === DATA COLLECTION ===\n")
        stats = {}
        for gesture in Gesture:
            success = self.collect_gesture(gesture, samples_per_gesture)
            stats[gesture.name] = len(self.data[gesture.value])
            if not success:
                break
        return stats

    def get_data_arrays(self) -> tuple:
        all_features = []
        all_labels = []
        for gesture_idx in sorted(self.data.keys()):
            features = self.data[gesture_idx]
            all_features.extend(features)
            all_labels.extend([gesture_idx] * len(features))
        return np.array(all_features), np.array(all_labels)

    def save_data(self, filename: str = 'training_data.npz'):
        X, y = self.get_data_arrays()
        filepath = self.output_dir / filename
        np.savez(filepath, X=X, y=y)
        app_logger.info(f"Data saved to {filepath} - {len(X)} samples")
        return filepath

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.close()


def train_classifier(data_path: str):
    print(f"\n🤖 === TRAINING {CLASSIFIER_TYPE.upper()} GESTURE CLASSIFIER ===\n")
    data = np.load(data_path)
    X, y = data['X'], data['y']
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    for gesture in Gesture:
        count = np.sum(y == gesture.value)
        print(f"  {GESTURE_NAMES[gesture]:12} : {count:3} samples")
    min_samples = min([np.sum(y == g.value) for g in Gesture])
    if min_samples < MIN_SAMPLES_PER_CLASS:
        print(f"\n⚠ Warning: Minimum samples per class is {min_samples} (recommended {MIN_SAMPLES_PER_CLASS})")
    print("\nApplying data augmentation...")
    augmentor = DataAugmentor()
    X_augmented, y_augmented = [X], [y]
    for gesture in Gesture:
        mask = y == gesture.value
        gesture_samples = X[mask]
        for sample in gesture_samples:
            aug = augmentor.augment_sample(sample, num_augmentations=3)
            X_augmented.append(aug)
            y_augmented.append(np.full(len(aug), gesture.value))
    X_train = np.vstack(X_augmented)
    y_train = np.concatenate(y_augmented)
    print(f"After augmentation: {len(X_train)} samples")
    print(f"\nTraining {CLASSIFIER_TYPE.upper()} classifier...")
    classifier = GestureClassifier()
    metrics = classifier.train(X_train, y_train, verbose=True)
    classifier.save()
    print(f"\n✓ Classifier training complete. Accuracy: {metrics['mean_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Mind Games gesture classifier")
    parser.add_argument('--collect', action='store_true', help='Collect training data')
    parser.add_argument('--train-classifier', action='store_true', help='Train gesture classifier')
    parser.add_argument('--train-all', action='store_true', help='Collect data and train')
    parser.add_argument('--samples-per-gesture', type=int, default=100)
    parser.add_argument('--data-file', type=str, default='training_data.npz')
    args = parser.parse_args()
    data_file = None
    if args.collect or args.train_all:
        collector = DataCollector()
        try:
            collector.collect_all_gestures(args.samples_per_gesture)
            data_file = collector.save_data(args.data_file)
        finally:
            collector.close()
    if args.train_classifier or args.train_all:
        if data_file is None:
            data_file = Path(DATA_DIR) / args.data_file
        if data_file.exists():
            train_classifier(str(data_file))
        else:
            print(f"✗ Data file not found: {data_file}")
    print("\n✓ Training complete! Run: python game.py\n")


if __name__ == '__main__':
    main()