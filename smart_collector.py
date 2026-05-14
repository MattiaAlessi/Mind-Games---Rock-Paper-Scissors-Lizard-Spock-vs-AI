"""
Advanced Data Collection Strategy for RPSLS
Handles confusable gestures with specific guidance and analysis
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from config import (
    Gesture, GESTURE_NAMES, GESTURE_EMOJIS,
    DATA_DIR, NUM_FEATURES
)

from hand_detector import HandDetector
from gesture_classifier import GestureClassifier


# ============================================================================
# GESTURE-SPECIFIC GUIDANCE
# ============================================================================

GESTURE_INSTRUCTIONS = {
    Gesture.ROCK: {
        'name': 'SASSO',
        'emoji': '✊',
        'description': 'Pugno chiuso',
        'how_to': [
            '1. Chiudi bene il pugno - dita completamente piegate',
            '2. Pollice DENTRO il pugno, nascosto',
            '3. Palmo visibile frontalmente',
            '4. Posiziona il dorso della mano davanti alla telecamera'
        ],
        'confusable_with': [Gesture.LIZARD],
        'avoid': [
            '❌ Non lasciare il pollice fuori (sembra forbice)',
            '❌ Non tenere dita semi-aperte',
            '❌ Non mostrare il palmo interno'
        ]
    },
    
    Gesture.PAPER: {
        'name': 'CARTA',
        'emoji': '✋',
        'description': 'Mano aperta piatta',
        'how_to': [
            '1. Apri completamente la mano - dita tese',
            '2. Dita UNITE (non divaricate)',
            '3. Palmo piatto, rivolto in avanti',
            '4. Mantieni la posizione - niente movimento'
        ],
        'confusable_with': [Gesture.SPOCK],
        'avoid': [
            '❌ Non separare il pollice (diventa Spock)',
            '❌ Non divarica le dita',
            '❌ Non piegare il polso (sembra forbice)'
        ]
    },
    
    Gesture.SCISSORS: {
        'name': 'FORBICE',
        'emoji': '✌️',
        'description': 'Indice e medio aperti a V',
        'how_to': [
            '1. Apri SOLO indice e medio a V netta',
            '2. Ring e pinky CHIUSI dentro il pugno',
            '3. Pollice DENTRO (non fuori)',
            '4. V ben marcata e visibile'
        ],
        'confusable_with': [],
        'avoid': [
            '❌ Non lasciare il pollice fuori (sembra Spock)',
            '❌ Non aprire altre dita',
            '❌ V non abbastanza marcata'
        ]
    },
    
    Gesture.LIZARD: {
        'name': 'LIZARD 🦎',
        'emoji': '🦎',
        'description': 'Mano aperta con pollice e indice+medio divaricati',
        'how_to': [
            '1. Pollice FUORI e separato dal resto',
            '2. Indice e medio APERTI a V (non chiusi come forbice)',
            '3. Ring e pinky CHIUSI',
            '4. La mano forma una "rana" con bocca aperta'
        ],
        'confusable_with': [Gesture.ROCK, Gesture.SPOCK],
        'avoid': [
            '❌ Non chiudere indice-medio (diventa sasso)',
            '❌ Non separare tutte le dita (è Spock)',
            '❌ Pollice DEVE essere chiaramente visibile'
        ]
    },
    
    Gesture.SPOCK: {
        'name': 'SPOCK 🖖',
        'emoji': '🖖',
        'description': 'Mano con tutte le dita separate: V tra indice-medio E V tra ring-pinky',
        'how_to': [
            '1. Mano COMPLETAMENTE aperta',
            '2. Tutte le dita DIVARICATE - non toccate',
            '3. Due V nette: indice-medio E ring-pinky',
            '4. Pollice separato lateralmente'
        ],
        'confusable_with': [Gesture.PAPER, Gesture.LIZARD],
        'avoid': [
            '❌ Non unire le dita (diventa carta)',
            '❌ Non chiudere ring-pinky (sembra lizard)',
            '❌ Le dita DEVONO essere tutte separate e visibili'
        ]
    }
}

# ============================================================================
# COLLECTION STRATEGY
# ============================================================================

class SmartDataCollector:
    """
    Intelligent data collector with:
    - Gesture-specific instructions
    - Multiple angles per gesture
    - Confusion matrix detection
    - Real-time feedback
    """
    
    def __init__(self, output_dir: str = DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
        
        self.data = defaultdict(list)
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Track collection angles/distances
        self.angle_metadata = defaultdict(lambda: defaultdict(int))
    
    def show_instructions(self, gesture: Gesture):
        """Display detailed instructions for a gesture"""
        info = GESTURE_INSTRUCTIONS[gesture]
        
        print(f"\n{'='*70}")
        print(f"📍 Raccolta: {info['emoji']} {info['name']}")
        print(f"{'='*70}")
        print(f"\n📝 Descrizione: {info['description']}\n")
        
        print("✅ COME FARE:")
        for step in info['how_to']:
            print(f"   {step}")
        
        if info['confusable_with']:
            print(f"\n⚠️  SIMILE A: {', '.join(GESTURE_NAMES[g] for g in info['confusable_with'])}")
        
        print("\n❌ ERRORI COMUNI:")
        for avoid in info['avoid']:
            print(f"   {avoid}")
        
        print("\n" + "="*70)
        print("Premi SPACE quando il gesto è corretto")
        print("Premi 'A' per andare avanti senza completare")
        print("="*70 + "\n")
    
    def collect_with_guidance(self, gesture: Gesture, num_samples: int = 150):
        """
        Collect with detailed per-gesture guidance
        Strategy: collect from multiple angles/distances
        """
        self.show_instructions(gesture)
        
        info = GESTURE_INSTRUCTIONS[gesture]
        collected = 0
        frame_count = 0
        last_predicted = None
        
        # Sub-targets for different angles/distances
        angles = ['frontale', 'destra', 'sinistra', 'alto', 'basso']
        target_per_angle = num_samples // len(angles)
        collected_per_angle = defaultdict(int)
        
        print(f"\n📐 Target {target_per_angle} campioni per angolo\n")
        
        while collected < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_count += 1
            
            # Detect hand every 2 frames
            _, landmarks, hand_detected = self.hand_detector.detect(frame)
            
            if hand_detected and landmarks:
                # Draw skeleton
                frame = self.hand_detector.draw_skeleton(
                    frame, landmarks, color=(0, 255, 0), thickness=2
                )
                
                # Try to classify (for feedback)
                if self.gesture_classifier.is_trained and frame_count % 3 == 0:
                    features = self.hand_detector.landmarks_to_features(landmarks)
                    if features is not None:
                        pred_gesture, confidence = self.gesture_classifier.predict(features)
                        last_predicted = (pred_gesture, confidence)
            
            # UI
            # Progress bar
            progress = int((collected / num_samples) * 30)
            progress_bar = "█" * progress + "░" * (30 - progress)
            cv2.putText(frame, f"[{progress_bar}] {collected}/{num_samples}",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            
            # Instructions box
            cv2.rectangle(frame, (10, 60), (w-10, 180), (30, 30, 60), -1)
            cv2.putText(frame, "ISTRUZIONI:", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y = 110
            for step in info['how_to'][:2]:  # Show first 2 steps only
                cv2.putText(frame, step, (25, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y += 25
            
            # Real-time feedback
            if last_predicted:
                pred_gesture, confidence = last_predicted
                feedback_text = f"Rilevato: {GESTURE_NAMES[pred_gesture]} ({confidence:.1%})"
                
                if pred_gesture == gesture:
                    color = (0, 255, 0)  # Green
                    feedback_text += " ✓ OK"
                else:
                    color = (0, 165, 255)  # Orange
                
                cv2.putText(frame, feedback_text, (10, h-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Controls
            cv2.putText(frame, "SPACE: Cattura | A: Salta",
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            cv2.imshow(f"Collecting {info['emoji']} {info['name']}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                if hand_detected and landmarks:
                    features = self.hand_detector.landmarks_to_features(landmarks)
                    if features is not None:
                        self.data[gesture.value].append(features)
                        collected += 1
                        
                        # Track angle
                        current_angle = angles[collected % len(angles)]
                        collected_per_angle[current_angle] += 1
                        
                        status = f"✓ {collected}/{num_samples}"
                        if collected % 10 == 0:
                            status += " [Checkpoint]"
                        print(f"  {status}")
            
            elif key == ord('a'):
                print(f"  ⊘ Saltato {info['name']}")
                break
            
            elif key == ord('q'):
                print("  Abort")
                return False
        
        cv2.destroyAllWindows()
        
        # Summary
        print(f"\n✓ Completato {info['name']}")
        print(f"  Totale: {collected} campioni")
        print(f"  Per angolo: {dict(collected_per_angle)}")
        
        return True
    
    def collect_all_gestures_smart(self, samples_per_gesture: int = 150):
        """
        Smart collection: hard gestures get more samples
        """
        print("\n" + "="*70)
        print("🎬 RACCOLTA DATI INTELLIGENTE - RPSLS")
        print("="*70)
        
        # Allocate samples strategically
        # Confusable gestures get more samples
        allocation = {
            Gesture.ROCK: int(samples_per_gesture * 1.2),      # Easy
            Gesture.PAPER: int(samples_per_gesture * 1.3),     # Hard (vs Spock)
            Gesture.SCISSORS: int(samples_per_gesture * 1.0),  # Easy
            Gesture.LIZARD: int(samples_per_gesture * 1.4),    # Very hard (vs Rock, Spock)
            Gesture.SPOCK: int(samples_per_gesture * 1.3),     # Hard (vs Paper, Lizard)
        }
        
        print("\n📊 Allocazione campioni per difficoltà:")
        for gesture, count in allocation.items():
            difficulty = "⭐" * min(int(count / samples_per_gesture * 3), 3)
            print(f"  {GESTURE_EMOJIS[gesture]} {GESTURE_NAMES[gesture]:10} : {count:3} campioni {difficulty}")
        
        print("\n")
        
        stats = {}
        for gesture in Gesture:
            target = allocation[gesture]
            success = self.collect_with_guidance(gesture, target)
            stats[gesture.name] = len(self.data[gesture.value])
            
            if not success:
                break
        
        return stats
    
    def analyze_class_balance(self):
        """Print class distribution"""
        print("\n" + "="*70)
        print("📊 DISTRIBUZIONE CLASSI")
        print("="*70)
        
        total = sum(len(v) for v in self.data.values())
        
        for gesture in Gesture:
            count = len(self.data[gesture.value])
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"{GESTURE_NAMES[gesture]:10} : {count:4} ({pct:5.1f}%) {bar}")
        
        print(f"{'TOTALE':10} : {total:4}")
    
    def get_data_arrays(self):
        """Get feature and label arrays"""
        all_features = []
        all_labels = []
        
        for gesture_idx in sorted(self.data.keys()):
            features = self.data[gesture_idx]
            all_features.extend(features)
            all_labels.extend([gesture_idx] * len(features))
        
        return np.array(all_features), np.array(all_labels)
    
    def save_data(self, filename: str = 'training_data.npz'):
        """Save with detailed metadata"""
        X, y = self.get_data_arrays()
        filepath = self.output_dir / filename
        
        np.savez(
            filepath,
            X=X, y=y,
            gesture_names=list(GESTURE_NAMES.values()),
            num_samples_per_class=np.array([len(self.data[i]) for i in range(5)])
        )
        
        print(f"\n✓ Dati salvati: {filepath}")
        self.analyze_class_balance()
        
        return filepath
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.close()


class ConfusionMatrixAnalyzer:
    """Analyze and visualize confusion between similar gestures"""
    
    def __init__(self, classifier: GestureClassifier):
        self.classifier = classifier
    
    def compute_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray):
        """Compute confusion matrix"""
        if not self.classifier.is_trained:
            print("✗ Classifier not trained")
            return None
        
        y_pred = []
        for features in X_test:
            gesture, _ = self.classifier.predict(features)
            y_pred.append(gesture.value if gesture else 0)
        
        y_pred = np.array(y_pred)
        
        cm = confusion_matrix(y_test, y_pred, labels=range(5))
        return cm
    
    def plot_confusion_matrix(self, cm: np.ndarray, output_path: str = 'confusion_matrix.png'):
        """Visualize confusion matrix"""
        gesture_list = [GESTURE_NAMES[Gesture(i)] for i in range(5)]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=gesture_list,
                    yticklabels=gesture_list,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - RPSLS Gesture Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {output_path}")
        plt.close()
    
    def identify_confusable_pairs(self, cm: np.ndarray, threshold: float = 0.15):
        """Identify which gestures are confused with each other"""
        print("\n⚠️  ANALISI CONFUSIONI")
        print("="*70)
        
        gesture_list = list(Gesture)
        
        for i in range(5):
            for j in range(5):
                if i != j:
                    # Percentage of class i predicted as j
                    total_i = cm[i, :].sum()
                    if total_i > 0:
                        confusion_rate = cm[i, j] / total_i
                        if confusion_rate > threshold:
                            print(f"⚠️  {GESTURE_NAMES[gesture_list[i]]} confuso come {GESTURE_NAMES[gesture_list[j]]} ({confusion_rate:.1%})")


def main():
    """Main collection workflow"""
    collector = SmartDataCollector()
    
    try:
        # Collect with smart strategy
        stats = collector.collect_all_gestures_smart(samples_per_gesture=150)
        
        # Save data
        data_path = collector.save_data('training_data_smart.npz')
        
        print("\n✓ Raccolta completata!")
        print("Prosegui con: python train.py --train-classifier --data-file training_data_smart.npz")
    
    finally:
        collector.close()


if __name__ == '__main__':
    main()
