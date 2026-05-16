#!/usr/bin/env python3
"""
RPSLS Game - Advanced Adaptive AI opponent
UI minimalista: nessun messaggio di disturbo, nessuna cronologia laterale.
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
# Adaptive AI (invariata)
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
        if len(self.history) < self.order:
            return None
        state = self._get_state_key()
        counts = self.transitions[state].copy()
        if np.sum(counts) == 0:
            return np.ones(self.n) / self.n
        probs = (counts + self.alpha) / (np.sum(counts) + self.alpha * self.n)
        return probs

    def choose_move(self):
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
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return move

    def save_memory(self):
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
# Helper: testo con sfondo nero
# ----------------------------------------------------------------------
def draw_text_with_bg(frame, text, x, y, font_scale, thickness, text_color, bg_color=(0,0,0), padding=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


# ----------------------------------------------------------------------
# Main Game Class (UI ultra-semplice)
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ai = AdaptiveAI()
        self.player_score = 0
        self.ai_score = 0
        self.draws = 0
        self.win_streak = 0
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

    def run(self):
        app_logger.info("Starting Mind Games RPSLS")
        print("\n🎮 RPSLS – Mind Games with Adaptive AI")
        print("   Press SPACE to start a round (after showing a clear gesture)")
        print("   Press ESC to quit\n")

        cv2.namedWindow("Mind Games - RPSLS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mind Games - RPSLS", 960, 720)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ---------- Hand detection ----------
            _, landmarks, hand_detected = self.hand_detector.detect(frame)
            smoothed = self.hand_detector.get_smoothed_landmarks()
            landmarks_to_use = smoothed if smoothed is not None else landmarks

            gesture = None
            confidence = 0.0
            if hand_detected and landmarks_to_use:
                features = self.hand_detector.landmarks_to_features(landmarks_to_use)
                if features is not None:
                    gesture, confidence = self.classifier.predict(features)
                frame = self.hand_detector.draw_skeleton(frame, landmarks_to_use, (0, 255, 0), 2)

            # ---------- Barra superiore (punteggi + streak) ----------
            top_bar_h = 70
            cv2.rectangle(frame, (0, 0), (w, top_bar_h), (0, 0, 0), -1)

            draw_text_with_bg(frame, f"YOU: {self.player_score}", 15, 45, 1, 2, (0, 255, 0))
            draw_text_with_bg(frame, f"AI: {self.ai_score}", 150, 45, 1, 2, (0, 0, 255))
            draw_text_with_bg(frame, f"DRAWS: {self.draws}", 280, 45, 1, 2, (255, 255, 0))

            if self.win_streak > 1:
                streak_text = f"STREAK: {self.win_streak} {self.get_streak_icon()}"
                (tw, _), _ = cv2.getTextSize(streak_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                x_streak = w - tw - 15
                draw_text_with_bg(frame, streak_text, x_streak, 45, 0.9, 2, (0, 165, 255))

            # ---------- Barra inferiore (comandi + stato AI) ----------
            bottom_bar_h = 60
            cv2.rectangle(frame, (0, h - bottom_bar_h), (w, h), (0, 0, 0), -1)
            draw_text_with_bg(frame, "SPACE: play | ESC: quit", 15, h - 25, 0.6, 1, (200, 200, 200))
            if self.ai.total_rounds > 0:
                mem_text = f"Memory: {self.ai.total_rounds} | epsilon={self.ai.epsilon:.3f}"
                (tw, _), _ = cv2.getTextSize(mem_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                x_mem = w - tw - 15
                draw_text_with_bg(frame, mem_text, x_mem, h - 25, 0.55, 1, (0, 255, 255))

            # ---------- Area centrale: solo gesto riconosciuto (se presente) ----------
            if not self.current_round_active and gesture and confidence > 0.6:
                gesture_text = f"{GESTURE_EMOJIS[gesture]} {GESTURE_NAMES[gesture]} ({confidence:.0%})"
                (tw, th), _ = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                x_center = (w - tw) // 2
                y_center = (h - top_bar_h - bottom_bar_h) // 2 + top_bar_h
                draw_text_with_bg(frame, gesture_text, x_center, y_center, 0.9, 2, (255, 255, 0))

            # ---------- Messaggio di attesa (solo quando nessun gesto e non in partita) ----------
            # Lo mettiamo in basso, poco invadente, solo se serve davvero
            if not self.current_round_active and time.time() > self.result_display_end:
                if gesture is None or confidence < 0.6:
                    wait_text = "Show a clear gesture, then press SPACE"
                else:
                    wait_text = "Press SPACE to start the round!"
                (tw, th), _ = cv2.getTextSize(wait_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                x_center = (w - tw) // 2
                y_center = h - bottom_bar_h - 15
                draw_text_with_bg(frame, wait_text, x_center, y_center, 0.6, 2, (255, 255, 255))

            # ---------- Game logic ----------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if key == ord(' ') and not self.current_round_active and time.time() > self.result_display_end:
                if gesture is None or confidence < 0.6:
                    warn_msg = "Show a clear gesture first!"
                    (tw, th), _ = cv2.getTextSize(warn_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    x_center = (w - tw) // 2
                    y_center = h // 2
                    draw_text_with_bg(frame, warn_msg, x_center, y_center, 0.8, 2, (0, 0, 255))
                    cv2.imshow("Mind Games - RPSLS", frame)
                    cv2.waitKey(500)
                    continue

                self.frozen_gesture = gesture
                self.frozen_confidence = confidence
                self.current_round_active = True
                self.countdown_start = time.time()
                self.player_gesture = self.frozen_gesture
                self.ai_gesture = Gesture(self.ai.choose_move())

            # Countdown
            if self.current_round_active:
                elapsed = time.time() - self.countdown_start
                if elapsed < 2.0:
                    remaining = int(3 - elapsed * 1.5)
                    remaining = max(1, min(3, remaining))
                    angle = int(360 * (elapsed / 2.0))
                    center = (w // 2, h // 2)
                    radius = 70
                    overlay = frame.copy()
                    cv2.circle(overlay, center, radius + 20, (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                    cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, (0, 255, 255), 8)
                    count_text = str(remaining)
                    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)
                    draw_text_with_bg(frame, count_text,
                                      center[0] - tw//2, center[1] + th//2,
                                      3, 4, (0, 255, 255))
                    gesture_text = f"{GESTURE_EMOJIS[self.frozen_gesture]} {GESTURE_NAMES[self.frozen_gesture]}"
                    (gw, gh), _ = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    draw_text_with_bg(frame, gesture_text,
                                      center[0] - gw//2, center[1] - 50,
                                      0.8, 2, (255, 255, 255))
                else:
                    result = self.determine_winner(self.player_gesture, self.ai_gesture)
                    self.update_scores(result)
                    self.ai.update(self.player_gesture.value)
                    self.round_result = result
                    self.result_display_end = time.time() + 2.0
                    self.current_round_active = False

            # Risultato
            if time.time() < self.result_display_end and self.round_result:
                overlay = frame.copy()
                cv2.rectangle(overlay, (w//2 - 200, h//2 - 100), (w//2 + 200, h//2 + 100), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
                if self.round_result == "win":
                    msg = "YOU WIN! 🎉"
                    color = (0, 255, 0)
                elif self.round_result == "lose":
                    msg = "YOU LOSE! 😢"
                    color = (0, 0, 255)
                else:
                    msg = "DRAW 🤝"
                    color = (255, 255, 0)
                (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
                draw_text_with_bg(frame, msg, w//2 - mw//2, h//2 - 30, 1.4, 3, color)
                p_text = f"You: {GESTURE_EMOJIS[self.player_gesture]} {GESTURE_NAMES[self.player_gesture]}"
                a_text = f"AI : {GESTURE_EMOJIS[self.ai_gesture]} {GESTURE_NAMES[self.ai_gesture]}"
                (pw, ph), _ = cv2.getTextSize(p_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                draw_text_with_bg(frame, p_text, w//2 - pw//2, h//2 + 35, 0.7, 2, (200, 200, 200))
                (aw, ah), _ = cv2.getTextSize(a_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                draw_text_with_bg(frame, a_text, w//2 - aw//2, h//2 + 75, 0.7, 2, (200, 200, 200))

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