"""
LSTM-based sequence predictor for AI opponent
Predicts the player's next move based on sequence history
"""

import numpy as np
from collections import deque
from typing import Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping

from config import (
    Gesture, LSTM_SEQUENCE_LENGTH, LSTM_BATCH_SIZE,
    LSTM_EPOCHS, LSTM_VALIDATION_SPLIT, LSTM_LAYERS,
    LSTM_PREDICTOR_PATH
)


class SequencePredictor:
    """LSTM model for predicting player's next move"""
    
    def __init__(self, sequence_length: int = LSTM_SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.n_gestures = len(Gesture)
        self.model = None
        self.is_trained = False
        self.move_history = deque(maxlen=sequence_length)
        
    def build_model(self):
        """Construct LSTM architecture"""
        self.model = Sequential([
            layers.LSTM(LSTM_LAYERS['lstm_units'], 
                       input_shape=(self.sequence_length, self.n_gestures),
                       return_sequences=False),
            layers.Dropout(LSTM_LAYERS['dropout_rate']),
            layers.Dense(LSTM_LAYERS['dense_units'], activation='relu'),
            layers.Dropout(LSTM_LAYERS['dropout_rate']),
            layers.Dense(self.n_gestures, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
    
    def _prepare_sequences(self, gesture_sequence: List[int]) -> tuple:
        """
        Convert gesture sequence into training data
        
        Args:
            gesture_sequence: List of gesture indices [0, 1, 2, ...]
            
        Returns:
            Tuple of (X, y) where:
            X: shape (n_samples, sequence_length, n_gestures)
            y: shape (n_samples, n_gestures) one-hot encoded
        """
        X, y = [], []
        
        for i in range(len(gesture_sequence) - self.sequence_length):
            # Get sequence of moves
            seq = gesture_sequence[i:i + self.sequence_length]
            next_move = gesture_sequence[i + self.sequence_length]
            
            # One-hot encode sequence
            seq_onehot = np.eye(self.n_gestures)[seq]
            next_onehot = np.eye(self.n_gestures)[next_move]
            
            X.append(seq_onehot)
            y.append(next_onehot)
        
        return np.array(X), np.array(y)
    
    def train(self, gesture_sequences: List[List[int]], verbose: bool = True) -> dict:
        """
        Train LSTM on player move sequences
        
        Args:
            gesture_sequences: List of sequences, each being a list of gesture indices
            verbose: Print training progress
            
        Returns:
            Training history dict
        """
        self.build_model()
        
        # Prepare training data
        all_X, all_y = [], []
        for seq in gesture_sequences:
            if len(seq) > self.sequence_length:
                X, y = self._prepare_sequences(seq)
                all_X.append(X)
                all_y.append(y)
        
        X_train = np.vstack(all_X)
        y_train = np.vstack(all_y)
        
        if len(X_train) < 2:
            if verbose:
                print("✗ Insufficient training data for LSTM. Need longer sequences.")
            return {}
        
        if verbose:
            print(f"Training LSTM on {len(X_train)} sequences...")
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=LSTM_BATCH_SIZE,
            epochs=LSTM_EPOCHS,
            validation_split=LSTM_VALIDATION_SPLIT,
            callbacks=[early_stop],
            verbose=1 if verbose else 0
        )
        
        self.is_trained = True
        
        if verbose:
            print(f"✓ LSTM training complete")
        
        return history.history
    
    def predict_next_move(self, recent_moves: Optional[List[int]] = None) -> tuple:
        """
        Predict probability distribution for next player move
        
        Args:
            recent_moves: List of recent gesture indices (will use history if None)
            
        Returns:
            Tuple of (probabilities array, most_likely_gesture)
        """
        if not self.is_trained or self.model is None:
            # Return uniform distribution if not trained
            uniform_proba = np.ones(self.n_gestures) / self.n_gestures
            return uniform_proba, Gesture(np.argmax(uniform_proba))
        
        # Use internal history if not provided
        if recent_moves is None:
            recent_moves = list(self.move_history)
        
        # Pad with neutral if not enough history
        if len(recent_moves) < self.sequence_length:
            recent_moves = [Gesture.ROCK.value] * (self.sequence_length - len(recent_moves)) + recent_moves
        else:
            recent_moves = recent_moves[-self.sequence_length:]
        
        # One-hot encode
        X = np.eye(self.n_gestures)[recent_moves]
        X = X.reshape(1, self.sequence_length, self.n_gestures)
        
        # Predict
        proba = self.model.predict(X, verbose=0)[0]
        proba = np.clip(proba, 0, 1)
        proba = proba / np.sum(proba)  # Ensure valid probabilities
        
        predicted_gesture = Gesture(np.argmax(proba))
        
        return proba, predicted_gesture
    
    def record_move(self, gesture: Gesture):
        """Record a player move in history"""
        self.move_history.append(gesture.value)
    
    def get_history(self) -> List[Gesture]:
        """Get recorded move history"""
        return [Gesture(idx) for idx in self.move_history]
    
    def clear_history(self):
        """Clear move history"""
        self.move_history.clear()
    
    def save(self, path: str = LSTM_PREDICTOR_PATH):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        self.model.save(path)
        print(f"✓ LSTM model saved to {path}")
    
    def load(self, path: str = LSTM_PREDICTOR_PATH) -> bool:
        """Load model from disk"""
        try:
            self.model = keras.models.load_model(path)
            self.is_trained = True
            print(f"✓ LSTM model loaded from {path}")
            return True
        except FileNotFoundError:
            print(f"✗ LSTM model not found at {path}")
            return False


class AIOpponent:
    """AI opponent that uses predictors to choose winning moves"""
    
    def __init__(self, predictor: SequencePredictor):
        self.predictor = predictor
        self.strategy = 'counter'  # 'counter', 'random', 'balanced'
    
    def choose_move(self, use_prediction: bool = True) -> Gesture:
        """
        Choose AI move based on strategy
        
        Args:
            use_prediction: Whether to use LSTM prediction or random
            
        Returns:
            Gesture to play
        """
        if use_prediction and self.predictor.is_trained:
            # Predict player's next move
            pred_proba, predicted_player_move = self.predictor.predict_next_move()
            
            if self.strategy == 'counter':
                # Choose move that beats predicted player move
                return self._get_winning_move(predicted_player_move)
            elif self.strategy == 'balanced':
                # Mix: 70% counter, 30% random
                if np.random.rand() < 0.7:
                    return self._get_winning_move(predicted_player_move)
                else:
                    return Gesture(np.random.randint(0, len(Gesture)))
        
        # Random fallback
        return Gesture(np.random.randint(0, len(Gesture)))
    
    @staticmethod
    def _get_winning_move(player_gesture: Gesture) -> Gesture:
        """
        Return a gesture that beats the given player gesture
        If multiple winning moves exist, choose randomly
        """
        from config import WINS
        winning_moves = WINS[player_gesture]
        return winning_moves[np.random.randint(0, len(winning_moves))]
    
    def set_strategy(self, strategy: str):
        """Set AI strategy: 'counter', 'random', 'balanced'"""
        if strategy in ['counter', 'random', 'balanced']:
            self.strategy = strategy
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class SyntheticDataGenerator:
    """Generate synthetic gesture sequences for LSTM training"""
    
    @staticmethod
    def generate_patterns(num_sequences: int = 10,
                         seq_length: int = 100) -> List[List[int]]:
        """
        Generate synthetic game sequences with patterns
        Mix of: random, cyclical, and biased patterns
        
        Args:
            num_sequences: Number of sequences to generate
            seq_length: Length of each sequence
            
        Returns:
            List of gesture sequences
        """
        sequences = []
        
        for _ in range(num_sequences):
            pattern = np.random.choice(['random', 'cycle', 'bias'], p=[0.3, 0.3, 0.4])
            seq = []
            
            if pattern == 'random':
                seq = np.random.randint(0, 5, seq_length).tolist()
            
            elif pattern == 'cycle':
                cycle = [0, 1, 2, 3, 4]
                seq = (cycle * (seq_length // 5 + 1))[:seq_length]
            
            elif pattern == 'bias':
                # Biased towards certain gestures
                bias_gesture = np.random.randint(0, 5)
                seq = []
                for _ in range(seq_length):
                    if np.random.rand() < 0.6:
                        seq.append(bias_gesture)
                    else:
                        seq.append(np.random.randint(0, 5))
            
            sequences.append(seq)
        
        return sequences
