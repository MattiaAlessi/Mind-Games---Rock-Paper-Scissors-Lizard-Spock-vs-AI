#!/usr/bin/env python3
"""
RPSLS Game - Advanced Adaptive AI opponent
- Markov chain of order 2
- Sliding window, Laplace smoothing, ε‑greedy with decay
- Persistent memory with async saving
- UI: pure OpenCV with streak counter and prediction histogram
"""

import cv2
import numpy as np
import time
import os
import threading
from collections import deque, defaultdict

from config import Gesture, GESTURE_NAMES, GESTURE_EMOJIS, WINS, STREAK_ICONS, SHOW_STREAK
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier
from logger import app_logger


# ----------------------------------------------------------------------
# Advanced Adaptive AI (Markov order 2 + window + smoothing + epsilon decay + async save)
# ----------------------------------------------------------------------
class AdaptiveAI:
    def __init__(self, n_gestures=5, memory_file="data/ai_memory.npz",
                 order=2, window_size=200, alpha=0.5, epsilon=0.1,
                 epsilon_min=0.02, epsilon_decay=0.9999):
        self.n = n_gestures
        self.order = order
        self.window_size = window_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_file = memory_file

        self.transitions = defaultdict(lambda: np.zeros(n_gestures, dtype=int))
        self.round_buffer = []
        self.history = deque(maxlen=order)
        self.total_rounds = 0
        self._save_lock = threading.Lock()
        self.load_memory()

    def _get_state_key(self):
        if len(self.history) < self.order:
            pad = [None] * (self.order - len(self.history))
            return tuple(pad + list(self.history))
        return tuple(self.history)

    def _rebuild_from_buffer(self):
        new_trans = defaultdict(lambda: np.zeros(self.n, dtype=int))
        buf = self.round_buffer
        if len(buf) < self.order + 1:
            self.transitions = new_trans
            return
        for i in range(len(buf) - self.order):
            state = tuple(buf[i:i+self.order])
            nxt = buf[i+self.order]
            new_trans[state][nxt] += 1
        self.transitions = new_trans
        if len(buf) >= self.order:
            self.history = deque(buf[-self.order:], maxlen=self.order)

    def update(self, player_move_idx):
        if len(self.round_buffer) >= self.window_size:
            self.round_buffer.pop(0)
        self.round_buffer.append(player_move_idx)
        self._rebuild_from_buffer()
        self.history.append(player_move_idx)
        self.total_rounds += 1

        # Auto-save every 20 rounds (async)
        if self.total_rounds % 20 == 0:
            self.save_memory_async()

    def predict_next(self):
        if len(self.history) < self.order:
            return None
        state = self._get_state_key()
        counts = self.transitions[state]
        if np.sum(counts) == 0:
            return None
        probs = (counts + self.alpha) / (np.sum(counts) + self.alpha * self.n)
        return int(np.argmax(probs))
    
    def get_prediction_probabilities(self):
        """Return probability distribution over next player moves."""
        if len(self.history) < self.order:
            return None
        state = self._get_state_key()
        counts = self.transitions[state].copy()
        if np.sum(counts) == 0:
            return np.ones(self.n) / self.n
        probs = (counts + self.alpha) / (np.sum(counts) + self.alpha * self.n)
        return probs

    def choose_move(self):
        # epsilon-greedy with decay
        if np.random.rand() < self.epsilon:
            move = np.random.randint(self.n)
        else:
            pred = self.predict_next()
            if pred is None:
                move = np.random.randint(self.n)
            else:
                player_gesture = Gesture(pred)
                winning_moves = [g.value for g in WINS[player_gesture]]
                if not winning_moves:
                    move = np.random.randint(self.n)
                else:
                    move = np.random.choice(winning_moves)
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return move

    def save_memory(self):
        """Synchronous save (used internally by async thread)."""
        with self._save_lock:
            trans_dict = {k: v for k, v in self.transitions.items()}
            data = {
                'order': self.order,
                'window_size': self.window_size,
                'alpha': self.alpha,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'transitions_keys': list(trans_dict.keys()),
                'transitions_vals': [trans_dict[k] for k in trans_dict.keys()],
                'round_buffer': np.array(self.round_buffer, dtype=int),
                'total_rounds': self.total_rounds
            }
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            np.savez(self.memory_file, **data)
            app_logger.info(f"AI memory saved ({len(self.round_buffer)} rounds)")

    def save_memory_async(self):
        """Save memory in a background thread."""
        thread = threading.Thread(target=self.save_memory, daemon=True)
        thread.start()

    def load_memory(self):
        if not os.path.exists(self.memory_file):
            app_logger.info("No previous AI memory found. Starting fresh.")
            return
        try:
            data = np.load(self.memory_file, allow_pickle=True)
            self.order = int(data['order'])
            self.window_size = int(data['window_size'])
            self.alpha = float(data['alpha'])
            self.epsilon = float(data['epsilon'])
            self.epsilon_min = float(data.get('epsilon_min', 0.02))
            self.epsilon_decay = float(data.get('epsilon_decay', 0.9999))
            keys = data['transitions_keys']
            vals = data['transitions_vals']
            self.transitions = defaultdict(lambda: np.zeros(self.n, dtype=int))
            for k, v in zip(keys, vals):
                self.transitions[tuple(k)] = v
            self.round_buffer = list(data['round_buffer'])
            self.total_rounds = int(data['total_rounds'])
            if len(self.round_buffer) >= self.order:
                self.history = deque(self.round_buffer[-self.order:], maxlen=self.order)
            app_logger.info(f"AI memory loaded (window: {len(self.round_buffer)} rounds, total: {self.total_rounds})")
        except Exception as e:
            app_logger.error(f"Could not load AI memory: {e}")


# ----------------------------------------------------------------------
# Main Game Class
# ----------------------------------------------------------------------
class Game:
    def __init__(self):
        self.classifier = GestureClassifier()
        if not self.classifier.load():
            raise RuntimeError("Gesture classifier not found. Train first.")
        self.hand_detector = HandDetector()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency

        self.ai = AdaptiveAI()   # with epsilon decay
        self.player_score = 0
        self.ai_score = 0
        self.draws = 0
        self.win_streak = 0          # consecutive wins (4.1)
        self.round_history = deque(maxlen=8)
        self.current_round_active = False
        self.countdown_start = 0
        self.player_gesture = None
        self.ai_gesture = None
        self.round_result = None
        self.result_display_end = 0
        self.frozen_gesture = None
        self.frozen_confidence = 0.0

    def determine_winner(self, player, ai):
        if player == ai:
            return "draw"
        if ai in WINS[player]:
            return "win"
        return "lose"

    def update_scores(self, result):
        if result == "win":
            self.player_score += 1
            self.win_streak += 1
        elif result == "lose":
            self.ai_score += 1
            self.win_streak = 0
        else:
            self.draws += 1
            self.win_streak = 0

    def get_streak_icon(self):
        if not SHOW_STREAK:
            return ""
        for threshold, icon in sorted(STREAK_ICONS.items(), reverse=True):
            if self.win_streak >= threshold:
                return icon
        return ""

    def draw_prediction_histogram(self, frame, probs, x, y, width=200, height=100):
        """Draw bar chart of AI's predicted next player moves."""
        if probs is None:
            return
        bar_width = width // len(probs)
        max_prob = np.max(probs)
        for i, prob in enumerate(probs):
            bar_height = int((prob / max_prob) * height) if max_prob > 0 else 0
            color = (0, int(255 * prob), 255)
            cv2.rectangle(frame,
                         (x + i * bar_width, y + height - bar_height),
                         (x + (i+1) * bar_width, y + height),
                         color, -1)
            # gesture emoji
            gesture = Gesture(i)
            cv2.putText(frame, GESTURE_EMOJIS[gesture],
                       (x + i * bar_width + 5, y + height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    def run(self):
        app_logger.info("Starting Mind Games RPSLS")
        print("\n🎮 RPSLS – Mind Games with Adaptive AI")
        print("   Press SPACE to start a round (after showing a clear gesture)")
        print("   Press ESC to quit\n")
        print("   AI learns from your last moves, decays exploration over time!\n")

        
        cv2.namedWindow("Mind Games - RPSLS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mind Games - RPSLS", 960, 720) 
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            _, landmarks, hand_detected = self.hand_detector.detect(frame)
            smoothed_landmarks = self.hand_detector.get_smoothed_landmarks()
            landmarks_to_use = smoothed_landmarks if smoothed_landmarks is not None else landmarks

            gesture = None
            confidence = 0.0
            if hand_detected and landmarks_to_use:
                features = self.hand_detector.landmarks_to_features(landmarks_to_use)
                if features is not None:
                    gesture, confidence = self.classifier.predict(features)
                frame = self.hand_detector.draw_skeleton(frame, landmarks_to_use, (0, 255, 0), 2)

            # ---------- UI overlays ----------
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"YOU: {self.player_score}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"AI: {self.ai_score}", (200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"DRAWS: {self.draws}", (380, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Streak display (4.1)
            streak_icon = self.get_streak_icon()
            if self.win_streak > 1:
                cv2.putText(frame, f"STREAK: {self.win_streak} {streak_icon}", (550, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.putText(frame, "SPACE: play | ESC: quit", (w-300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # current gesture feedback
            if not self.current_round_active and gesture and confidence > 0.6:
                emoji = GESTURE_EMOJIS[gesture]
                name = GESTURE_NAMES[gesture]
                cv2.putText(frame, f"{emoji} {name} ({confidence:.0%})", (20, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            elif not self.current_round_active:
                cv2.putText(frame, "No gesture", (20, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 1)

            # AI memory indicator
            if self.ai.total_rounds > 0:
                cv2.putText(frame, f"AI memory: {self.ai.total_rounds} rounds", (w-250, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # Show epsilon (optional)
                cv2.putText(frame, f"ε={self.ai.epsilon:.3f}", (w-250, h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            # recent rounds history
            x_hist = w - 210
            y_start = 100
            cv2.putText(frame, "Recent rounds:", (x_hist, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, (p, a, r) in enumerate(self.round_history):
                color = (0, 255, 0) if r == 'win' else (0, 0, 255) if r == 'lose' else (255, 255, 0)
                cv2.putText(frame, f"{p} vs {a}", (x_hist, y_start + 25 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Prediction histogram (2.3) - during countdown only to avoid clutter
            if self.current_round_active:
                probs = self.ai.get_prediction_probabilities()
                if probs is not None:
                    self.draw_prediction_histogram(frame, probs, w-220, 250, 180, 80)

            # ---------- Game round logic ----------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if key == ord(' ') and not self.current_round_active and time.time() > self.result_display_end:
                if gesture is None or confidence < 0.6:
                    cv2.rectangle(frame, (w//2-250, h//2-30), (w//2+250, h//2+30), (0, 0, 0), -1)
                    cv2.putText(frame, "Show a clear gesture first!", (w//2-230, h//2+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    
                    cv2.imshow("Mind Games - RPSLS", frame)
                    cv2.waitKey(500)
                    continue

                self.frozen_gesture = gesture
                self.frozen_confidence = confidence
                self.current_round_active = True
                self.countdown_start = time.time()
                self.player_gesture = self.frozen_gesture
                self.ai_gesture = Gesture(self.ai.choose_move())

            if self.current_round_active:
                elapsed = time.time() - self.countdown_start
                if elapsed < 2.0:
                    remaining = int(3 - elapsed * 1.5)
                    remaining = max(1, min(3, remaining))
                    angle = int(360 * (elapsed / 2.0))
                    center = (w//2, h//2)
                    radius = 80
                    cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, (0, 255, 255), 10)
                    cv2.putText(frame, str(remaining), (center[0]-30, center[1]+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
                    cv2.putText(frame, f"{GESTURE_EMOJIS[self.frozen_gesture]} {GESTURE_NAMES[self.frozen_gesture]}",
                                (center[0]-100, center[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    result = self.determine_winner(self.player_gesture, self.ai_gesture)
                    self.update_scores(result)
                    self.ai.update(self.player_gesture.value)

                    p_short = GESTURE_NAMES[self.player_gesture][:3]
                    a_short = GESTURE_NAMES[self.ai_gesture][:3]
                    self.round_history.appendleft((p_short, a_short, result))

                    self.round_result = result
                    self.result_display_end = time.time() + 2.0
                    self.current_round_active = False

            # show result overlay
            if time.time() < self.result_display_end and self.round_result:
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2-220, h//2-110), (w//2+220, h//2+110), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

                if self.round_result == "win":
                    msg = "YOU WIN! 🎉"
                    color = (0, 255, 0)
                elif self.round_result == "lose":
                    msg = "YOU LOSE! 😢"
                    color = (0, 0, 255)
                else:
                    msg = "DRAW 🤝"
                    color = (255, 255, 0)

                cv2.putText(frame, msg, (w//2-120, h//2-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(frame, f"You: {GESTURE_EMOJIS[self.player_gesture]} {GESTURE_NAMES[self.player_gesture]}",
                            (w//2-180, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(frame, f"AI : {GESTURE_EMOJIS[self.ai_gesture]} {GESTURE_NAMES[self.ai_gesture]}",
                            (w//2-180, h//2+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if not self.current_round_active and time.time() >= self.result_display_end:
                cv2.putText(frame, "Show a gesture and press SPACE", (w//2-250, h-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Mind Games - RPSLS", frame)

        self.ai.save_memory()
        self.cap.release()
        cv2.destroyAllWindows()
        app_logger.info(f"Game ended. Final score: You {self.player_score} - {self.ai_score} AI")
        print(f"\n👋 Game over. Final score: You {self.player_score} - {self.ai_score} AI")
        print("AI memory saved.")


if __name__ == "__main__":
    game = Game()
    game.run()