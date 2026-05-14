"""
Post-Training Analysis and Improvement Strategy
Identifies weak points and recommends data collection improvements
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from config import Gesture, GESTURE_NAMES, GESTURE_EMOJIS
from gesture_classifier import GestureClassifier


class ModelDiagnostics:
    """Diagnose model weaknesses and recommend improvements"""
    
    def __init__(self, classifier: GestureClassifier):
        self.classifier = classifier
    
    def evaluate_on_dataset(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Comprehensive evaluation
        
        Args:
            X: Feature array (n_samples, 63)
            y: Label array (n_samples,)
            
        Returns:
            Dict with metrics
        """
        if not self.classifier.is_trained:
            raise ValueError("Classifier must be trained first")
        
        # Get predictions
        y_pred = []
        confidences = []
        
        for features in X:
            gesture, conf = self.classifier.predict(features)
            y_pred.append(gesture.value if gesture else 0)
            confidences.append(conf if conf else 0)
        
        y_pred = np.array(y_pred)
        confidences = np.array(confidences)
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=range(5))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'confidences': confidences,
            'classification_report': classification_report(y, y_pred, 
                                                           target_names=[g.name for g in Gesture],
                                                           output_dict=True)
        }
    
    def identify_weak_gestures(self, metrics: dict) -> dict:
        """
        Identify which gestures have low accuracy
        
        Returns:
            Dict mapping gesture -> problem_type -> suggestions
        """
        report = metrics['classification_report']
        cm = metrics['confusion_matrix']
        
        weak_gestures = {}
        
        for gesture_idx, gesture in enumerate(Gesture):
            precision = report[str(gesture_idx)]['precision']
            recall = report[str(gesture_idx)]['recall']
            
            gesture_name = GESTURE_NAMES[gesture]
            
            # Identify problem type
            problems = []
            suggestions = []
            
            if precision < 0.85:
                problems.append(f"Bassa precisione ({precision:.1%})")
                suggestions.append(f"Il classificatore confonde {gesture_name} con altri gesti")
                suggestions.append("→ Ricontrolla i campioni: contengono posizioni errate?")
            
            if recall < 0.85:
                problems.append(f"Basso recall ({recall:.1%})")
                suggestions.append(f"Non tutti i {gesture_name} sono riconosciuti correttamente")
                suggestions.append("→ Aggiungi più campioni variati")
            
            if problems:
                weak_gestures[gesture] = {
                    'problems': problems,
                    'suggestions': suggestions,
                    'precision': precision,
                    'recall': recall,
                    'support': report[str(gesture_idx)]['support']
                }
        
        return weak_gestures
    
    def find_confusion_pairs(self, metrics: dict, threshold: float = 0.10) -> list:
        """
        Find gesture pairs that are confused with each other
        
        Args:
            metrics: Evaluation metrics dict
            threshold: Min confusion rate to report (10%)
            
        Returns:
            List of tuples (gesture1, gesture2, confusion_rate, suggestion)
        """
        cm = metrics['confusion_matrix']
        pairs = []
        
        for i in range(5):
            for j in range(i+1, 5):
                # Rate of class i misclassified as j
                total_i = cm[i, :].sum()
                if total_i > 0:
                    confusion_ij = cm[i, j] / total_i
                    
                    # Rate of class j misclassified as i
                    total_j = cm[j, :].sum()
                    confusion_ji = cm[j, i] / total_j if total_j > 0 else 0
                    
                    max_confusion = max(confusion_ij, confusion_ji)
                    
                    if max_confusion > threshold:
                        gesture_i = Gesture(i)
                        gesture_j = Gesture(j)
                        
                        # Generate suggestion
                        suggestion = self._suggest_for_confusion(gesture_i, gesture_j)
                        
                        pairs.append({
                            'gesture1': gesture_i,
                            'gesture2': gesture_j,
                            'confusion_rate': max_confusion,
                            'direction': 'i→j' if confusion_ij > confusion_ji else 'j→i',
                            'suggestion': suggestion
                        })
        
        # Sort by confusion rate descending
        pairs.sort(key=lambda x: x['confusion_rate'], reverse=True)
        return pairs
    
    @staticmethod
    def _suggest_for_confusion(g1: Gesture, g2: Gesture) -> str:
        """
        Generate specific suggestions for confused gesture pairs
        """
        suggestions = {
            (Gesture.PAPER, Gesture.SPOCK): 
                "PAPER: unisci dita | SPOCK: separa tutte le dita (V doppia)",
            (Gesture.SPOCK, Gesture.PAPER):
                "SPOCK: tutte dita aperte separate | PAPER: dita unite",
            (Gesture.ROCK, Gesture.LIZARD):
                "ROCK: pugno chiuso, pollice DENTRO | LIZARD: pollice fuori e visibile",
            (Gesture.LIZARD, Gesture.ROCK):
                "LIZARD: pollice separato, indice-medio aperti | ROCK: pugno completamente chiuso",
            (Gesture.SCISSORS, Gesture.LIZARD):
                "SCISSORS: solo indice-medio aperti a V | LIZARD: indice-medio aperti + pollice fuori",
            (Gesture.LIZARD, Gesture.SPOCK):
                "LIZARD: 3 dita chiuse (ring+pinky) | SPOCK: tutte 5 dita separate",
        }
        
        key = (g1, g2)
        if key in suggestions:
            return suggestions[key]
        return "Aumenta campioni e varia angoli/distanze"
    
    def generate_report(self, X: np.ndarray, y: np.ndarray, 
                       output_file: str = 'diagnostics_report.txt'):
        """
        Generate comprehensive diagnostic report
        """
        print("\n" + "="*70)
        print("🔍 ANALISI DIAGNOSTICA CLASSIFIER")
        print("="*70)
        
        # Evaluate
        metrics = self.evaluate_on_dataset(X, y)
        
        print(f"\n📊 ACCURATEZZA GLOBALE: {metrics['accuracy']:.1%}")
        
        # Weak gestures
        weak = self.identify_weak_gestures(metrics)
        
        if weak:
            print("\n⚠️  GESTI DEBOLI (accuratezza < 85%):")
            for gesture, info in weak.items():
                print(f"\n  {GESTURE_EMOJIS[gesture]} {GESTURE_NAMES[gesture]}")
                print(f"     Precisione: {info['precision']:.1%}")
                print(f"     Recall:     {info['recall']:.1%}")
                print(f"     Campioni:   {info['support']}")
                for suggestion in info['suggestions']:
                    print(f"     {suggestion}")
        else:
            print("\n✅ Tutti i gesti hanno buona accuratezza (>85%)")
        
        # Confusion pairs
        pairs = self.find_confusion_pairs(metrics)
        
        if pairs:
            print("\n⚡ COPPIE CONFUSE (confusion rate > 10%):")
            for pair in pairs:
                print(f"\n  {GESTURE_EMOJIS[pair['gesture1']]} {GESTURE_NAMES[pair['gesture1']]} " +
                      f"↔ {GESTURE_EMOJIS[pair['gesture2']]} {GESTURE_NAMES[pair['gesture2']]}")
                print(f"     Confusion: {pair['confusion_rate']:.1%} ({pair['direction']})")
                print(f"     → {pair['suggestion']}")
        else:
            print("\n✅ Nessuna confusione significativa tra gesti")
        
        # Feature importance
        importances = self.classifier.get_feature_importance(top_n=15)
        
        if importances:
            print("\n🎯 FEATURE IMPORTANTI (top 15):")
            landmark_names = [
                "Wrist", "Thumb0", "Thumb1", "Thumb2", "Index0", "Index1", "Index2", "Index3",
                "Middle0", "Middle1", "Middle2", "Middle3", "Ring0", "Ring1", "Ring2", "Ring3",
                "Pinky0", "Pinky1", "Pinky2", "Pinky3"
            ]
            
            for idx, importance in importances[:10]:
                landmark_idx = idx // 3
                coord = ['X', 'Y', 'Z'][idx % 3]
                landmark = landmark_names[landmark_idx] if landmark_idx < len(landmark_names) else f"L{landmark_idx}"
                print(f"     {landmark}-{coord:1} : {importance:.4f}")
        
        # Recommendations
        print("\n📋 RACCOMANDAZIONI:")
        recommendations = self._generate_recommendations(weak, pairs, metrics['accuracy'])
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return {
            'metrics': metrics,
            'weak_gestures': weak,
            'confusion_pairs': pairs
        }
    
    @staticmethod
    def _generate_recommendations(weak: dict, pairs: list, accuracy: float) -> list:
        """Generate actionable recommendations"""
        recs = []
        
        if accuracy < 0.85:
            recs.append("❗ Accuratezza complessiva bassa. Considerare:")
            recs.append("   - Aumentare campioni per tutte le classi a 200+")
            recs.append("   - Migliorare illuminazione durante la raccolta")
            recs.append("   - Variare angoli e distanze della mano")
        
        if weak:
            weakest = min(weak.items(), key=lambda x: min(x[1]['precision'], x[1]['recall']))
            gesture, info = weakest
            recs.append(f"🔴 {GESTURE_NAMES[gesture]} è il gesto più debole:")
            recs.append(f"   - Attualmente: {len(info['support'])} campioni")
            recs.append(f"   - Target: almeno 300-400 campioni")
            recs.append(f"   - Varia: angolo frontale, laterale, dall'alto")
        
        for pair in pairs[:2]:  # Top 2 confusions
            g1_name = GESTURE_NAMES[pair['gesture1']]
            g2_name = GESTURE_NAMES[pair['gesture2']]
            recs.append(f"🟡 Confusione {g1_name}↔{g2_name}:")
            recs.append(f"   - {pair['suggestion']}")
        
        if not recs:
            recs.append("✅ Classifier funziona bene! Pronto per il deployment.")
        
        return recs
    
    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray,
                             output_path: str = 'confusion_matrix.png'):
        """Plot and save confusion matrix"""
        metrics = self.evaluate_on_dataset(X, y)
        cm = metrics['confusion_matrix']
        
        gesture_names = [GESTURE_NAMES[Gesture(i)] for i in range(5)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                    xticklabels=gesture_names,
                    yticklabels=gesture_names,
                    cbar_kws={'label': 'Frequency'},
                    ax=ax)
        
        ax.set_title('Confusion Matrix - RPSLS Gesture Classifier', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved: {output_path}")
        plt.close()
    
    def plot_per_gesture_metrics(self, X: np.ndarray, y: np.ndarray,
                                output_path: str = 'per_gesture_metrics.png'):
        """Plot precision/recall per gesture"""
        metrics = self.evaluate_on_dataset(X, y)
        report = metrics['classification_report']
        
        gesture_names = [GESTURE_NAMES[Gesture(i)] for i in range(5)]
        precisions = [report[str(i)]['precision'] for i in range(5)]
        recalls = [report[str(i)]['recall'] for i in range(5)]
        f1_scores = [report[str(i)]['f1-score'] for i in range(5)]
        
        x = np.arange(len(gesture_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precisions, width, label='Precision', color='skyblue')
        ax.bar(x, recalls, width, label='Recall', color='lightcoral')
        ax.bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
        
        ax.axhline(y=0.85, color='r', linestyle='--', linewidth=2, label='Target (85%)')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Gesture Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(gesture_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Per-gesture metrics saved: {output_path}")
        plt.close()


def main():
    """Run diagnostics on trained model"""
    # Load trained model
    classifier = GestureClassifier()
    if not classifier.load():
        print("✗ No trained model found. Train first with: python train.py --train-classifier")
        return
    
    # Load test data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/training_data.npz',
                       help='Path to test data')
    args = parser.parse_args()
    
    try:
        data = np.load(args.data)
        X, y = data['X'], data['y']
    except FileNotFoundError:
        print(f"✗ Data file not found: {args.data}")
        return
    
    # Run diagnostics
    diagnostics = ModelDiagnostics(classifier)
    results = diagnostics.generate_report(X, y)
    
    # Plot visualizations
    diagnostics.plot_confusion_matrix(X, y)
    diagnostics.plot_per_gesture_metrics(X, y)
    
    print("\n✓ Diagnostics complete!")
    print("  - confusion_matrix.png")
    print("  - per_gesture_metrics.png")


if __name__ == '__main__':
    main()
