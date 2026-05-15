#!/usr/bin/env python3
"""
RPSLS Game - Advanced Adaptive AI opponent
- Markov chain of order 2 (looks at last two player moves)
- Sliding window to forget old habits
- Laplace smoothing to avoid zero probabilities
- ε‑greedy exploration
- Persistent memory across sessions
UI: pure OpenCV, no Pygame
"""

import cv2
import numpy as np
import time
import os
from collections import deque, defaultdict

from config import Gesture, GESTURE_NAMES, GESTURE_EMOJIS, WINS
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier


# ----------------------------------------------------------------------
# Advanced Adaptive AI (Markov order 2 + window + smoothing + epsilon)
# ----------------------------------------------------------------------
class AdaptiveAI:
    def __init__(self, n_gestures=5, memory_file="data/ai_memory.npz",
                 order=2, window_size=200, alpha=0.5, epsilon=0.1):
        self.n = n_gestures
        self.order = order                # number of previous moves to consider
        self.window_size = window_size    # how many past rounds to keep
        self.alpha = alpha                # Laplace smoothing factor
        self.epsilon = epsilon            # exploration rate
        self.memory_file = memory_file

        # transitions: key = tuple of last `order` moves -> array of counts (size n)
        self.transitions = defaultdict(lambda: np.zeros(n_gestures, dtype=int))
        self.round_buffer = []            # all moves in the sliding window
        self.history = deque(maxlen=order)  # last `order` moves (for current prediction)
        self.total_rounds = 0
        self.load_memory()

    def _get_state_key(self):
        """Return a tuple of the last `order` moves (padded with None if needed)."""
        if len(self.history) < self.order:
            pad = [None] * (self.order - len(self.history))
            return tuple(pad + list(self.history))
        return tuple(self.history)

    def _rebuild_from_buffer(self):
        """Rebuild transition counts from the current sliding window."""
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
        # restore history from the end of the buffer
        if len(buf) >= self.order:
            self.history = deque(buf[-self.order:], maxlen=self.order)

    def update(self, player_move_idx):
        """Call after each round: add the player's move to memory."""
        # sliding window management
        if len(self.round_buffer) >= self.window_size:
            self.round_buffer.pop(0)
        self.round_buffer.append(player_move_idx)

        # rebuild transitions from the window (simpler, safe)
        # For efficiency we could do incremental, but with window <= 200 it's fine
        self._rebuild_from_buffer()

        # update history (used for prediction)
        self.history.append(player_move_idx)
        self.total_rounds += 1

        # auto-save every 20 rounds
        if self.total_rounds % 20 == 0:
            self.save_memory()

    def predict_next(self):
        """Return most likely next player move based on current state."""
        if len(self.history) < self.order:
            return None
        state = self._get_state_key()
        counts = self.transitions[state]
        if np.sum(counts) == 0:
            return None
        # Laplace smoothing
        probs = (counts + self.alpha) / (np.sum(counts) + self.alpha * self.n)
        return int(np.argmax(probs))

    def choose_move(self):
        """AI chooses a move that beats the predicted player move (with exploration)."""
        # epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n)

        pred = self.predict_next()
        if pred is None:
            return np.random.randint(self.n)

        player_gesture = Gesture(pred)
        winning_moves = [g.value for g in WINS[player_gesture]]
        if not winning_moves:
            return np.random.randint(self.n)
        return np.random.choice(winning_moves)

    def save_memory(self):
        """Save transition dictionary and window to disk."""
        # convert defaultdict to plain dict for saving
        trans_dict = {k: v for k, v in self.transitions.items()}
        data = {
            'order': self.order,
            'window_size': self.window_size,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'transitions_keys': list(trans_dict.keys()),
            'transitions_vals': [trans_dict[k] for k in trans_dict.keys()],
            'round_buffer': np.array(self.round_buffer, dtype=int),
            'total_rounds': self.total_rounds
        }
        # create directory if needed
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        np.savez(self.memory_file, **data)
        print(f"AI memory saved ({len(self.round_buffer)} rounds in window)")

    def load_memory(self):
        """Load previously saved memory if it exists."""
        if not os.path.exists(self.memory_file):
            print("No previous AI memory found. Starting fresh.")
            return
        try:
            data = np.load(self.memory_file, allow_pickle=True)
            self.order = int(data['order'])
            self.window_size = int(data['window_size'])
            self.alpha = float(data['alpha'])
            self.epsilon = float(data['epsilon'])
            keys = data['transitions_keys']
            vals = data['transitions_vals']
            self.transitions = defaultdict(lambda: np.zeros(self.n, dtype=int))
            for k, v in zip(keys, vals):
                self.transitions[tuple(k)] = v
            self.round_buffer = list(data['round_buffer'])
            self.total_rounds = int(data['total_rounds'])
            # rebuild history from the end of the buffer
            if len(self.round_buffer) >= self.order:
                self.history = deque(self.round_buffer[-self.order:], maxlen=self.order)
            print(f"AI memory loaded (window: {len(self.round_buffer)} rounds, total: {self.total_rounds})")
        except Exception as e:
            print(f"Could not load AI memory: {e}")


# ----------------------------------------------------------------------
# Main Game Class
# ----------------------------------------------------------------------
class Game:
    def __init__(self):
        # Load gesture classifier
        self.classifier = GestureClassifier()
        if not self.classifier.load():
            raise RuntimeError("Gesture classifier not found. Train first.")
        self.hand_detector = HandDetector()

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Game state
        self.ai = AdaptiveAI()   # loads memory automatically
        self.player_score = 0
        self.ai_score = 0
        self.draws = 0
        self.round_history = deque(maxlen=8)   # recent results for UI
        self.current_round_active = False
        self.countdown_start = 0
        self.player_gesture = None
        self.ai_gesture = None
        self.round_result = None
        self.result_display_end = 0

        # Frozen gesture during countdown (improvement #9)
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
        elif result == "lose":
            self.ai_score += 1
        else:
            self.draws += 1

    def run(self):
        print("\n🎮 RPSLS – Mind Games with Adaptive AI")
        print("   Press SPACE to start a round (after showing a clear gesture)")
        print("   Press ESC to quit\n")
        print("   AI learns from your last moves and remembers across sessions!\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Hand detection and skeleton
            _, landmarks, hand_detected = self.hand_detector.detect(frame)
            # Use smoothed landmarks if available (improvement #10)
            smoothed_landmarks = self.hand_detector.get_smoothed_landmarks()
            landmarks_to_use = smoothed_landmarks if smoothed_landmarks is not None else landmarks

            gesture = None
            confidence = 0.0
            if hand_detected and landmarks_to_use:
                features = self.hand_detector.landmarks_to_features(landmarks_to_use)
                if features is not None:
                    gesture, confidence = self.classifier.predict(features)
                # draw skeleton using smoothed landmarks
                frame = self.hand_detector.draw_skeleton(frame, landmarks_to_use, (0, 255, 0), 2)

            # ---------- UI overlays ----------
            # top bar
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"YOU: {self.player_score}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"AI: {self.ai_score}", (200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"DRAWS: {self.draws}", (380, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, "SPACE: play | ESC: quit", (w-300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # current gesture feedback (only if round not active, otherwise show frozen)
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

            # recent rounds history (right side)
            x_hist = w - 210
            y_start = 100
            cv2.putText(frame, "Recent rounds:", (x_hist, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, (p, a, r) in enumerate(self.round_history):
                color = (0, 255, 0) if r == 'win' else (0, 0, 255) if r == 'lose' else (255, 255, 0)
                cv2.putText(frame, f"{p} vs {a}", (x_hist, y_start + 25 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # ---------- Game round logic ----------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                break

            if key == ord(' ') and not self.current_round_active and time.time() > self.result_display_end:
                if gesture is None or confidence < 0.6:
                    # show warning briefly
                    cv2.rectangle(frame, (w//2-250, h//2-30), (w//2+250, h//2+30), (0, 0, 0), -1)
                    cv2.putText(frame, "Show a clear gesture first!", (w//2-230, h//2+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Mind Games - RPSLS", frame)
                    cv2.waitKey(500)
                    continue

                # Freeze the gesture shown at this moment (improvement #9)
                self.frozen_gesture = gesture
                self.frozen_confidence = confidence

                # start a new round
                self.current_round_active = True
                self.countdown_start = time.time()
                self.player_gesture = self.frozen_gesture   # use frozen gesture
                self.ai_gesture = Gesture(self.ai.choose_move())

            if self.current_round_active:
                elapsed = time.time() - self.countdown_start
                if elapsed < 2.0:   # countdown for 2 seconds
                    # Calculate remaining number (3,2,1)
                    remaining = int(3 - elapsed * 1.5)
                    remaining = max(1, min(3, remaining))

                    # Animated circular arc
                    angle = int(360 * (elapsed / 2.0))
                    center = (w//2, h//2)
                    radius = 80
                    cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, (0, 255, 255), 10)

                    # Countdown number in center
                    cv2.putText(frame, str(remaining), (center[0]-30, center[1]+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)

                    # Show frozen gesture above the countdown (optional)
                    cv2.putText(frame, f"{GESTURE_EMOJIS[self.frozen_gesture]} {GESTURE_NAMES[self.frozen_gesture]}",
                                (center[0]-100, center[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    # round finished
                    result = self.determine_winner(self.player_gesture, self.ai_gesture)
                    self.update_scores(result)
                    self.ai.update(self.player_gesture.value)   # AI learns from this move

                    # store for UI history
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
                alpha = 0.8
                frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

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

            # idle instruction
            if not self.current_round_active and time.time() >= self.result_display_end:
                cv2.putText(frame, "Show a gesture and press SPACE", (w//2-250, h-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Mind Games - RPSLS", frame)

        # save AI memory before exiting
        self.ai.save_memory()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n👋 Game over. Final score: You {self.player_score} - {self.ai_score} AI")
        print("AI memory saved. Next time it will be even smarter!")


if __name__ == "__main__":
    game = Game()
    game.run()