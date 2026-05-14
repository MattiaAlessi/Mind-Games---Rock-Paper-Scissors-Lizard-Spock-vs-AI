#!/usr/bin/env python3
"""
RPSLS Game - Adaptive AI opponent using real-time gesture recognition.
UI based purely on OpenCV (no Pygame).
AI learns from your moves using a transition matrix.
"""

import cv2
import numpy as np
import time
import os
import pickle
from collections import defaultdict, deque

from config import Gesture, GESTURE_NAMES, GESTURE_EMOJIS, WINS
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier

# -------------------------------
# Adaptive AI using transition matrix
# -------------------------------
class AdaptiveAI:
    """
    Learns player's move patterns online.
    Maintains a transition matrix P(next_move | current_move) and chooses
    the move that most likely beats the next predicted move.
    """
    def __init__(self, n_gestures=5):
        self.n = n_gestures
        # transition counts: from -> to
        self.transitions = np.zeros((n_gestures, n_gestures), dtype=int)
        self.last_move = None
        self.total_rounds = 0

    def update(self, player_move_idx):
        """Call after each round with the player's move."""
        if self.last_move is not None:
            self.transitions[self.last_move, player_move_idx] += 1
        self.last_move = player_move_idx
        self.total_rounds += 1

    def predict_next(self):
        """Predict player's next move based on last move (maximum likelihood)."""
        if self.last_move is None or self.total_rounds < 2:
            return None  # not enough data
        
        probs = self.transitions[self.last_move, :]
        if np.sum(probs) == 0:
            return None
        next_move = np.argmax(probs)
        return next_move

    def choose_move(self):
        """AI chooses a move that beats the predicted player move."""
        pred = self.predict_next()
        if pred is None:
            # random move until we learn
            return np.random.randint(self.n)
        
        # pred is the player's most likely next gesture
        # we need a gesture that beats pred
        player_gesture = Gesture(pred)
        # find a gesture that beats player_gesture
        winning_moves = [g.value for g in WINS[player_gesture]]
        if not winning_moves:
            return np.random.randint(self.n)
        return np.random.choice(winning_moves)

# -------------------------------
# UI and game loop (OpenCV)
# -------------------------------
class Game:
    def __init__(self):
        # Load models
        self.classifier = GestureClassifier()
        if not self.classifier.load():
            raise RuntimeError("Gesture classifier not found. Train first.")
        self.hand_detector = HandDetector()
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Game state
        self.ai = AdaptiveAI()
        self.player_score = 0
        self.ai_score = 0
        self.draws = 0
        self.round_history = deque(maxlen=10)  # for display
        self.current_round_active = False
        self.countdown_start = 0
        self.player_gesture = None
        self.ai_gesture = None
        self.round_result = None  # "win", "lose", "draw"
        self.result_display_end = 0
        
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
        print("🎮 Game started. Press SPACE to start a round. Press ESC to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Hand detection and skeleton (every frame)
            _, landmarks, hand_detected = self.hand_detector.detect(frame)
            gesture = None
            confidence = 0.0
            if hand_detected and landmarks:
                features = self.hand_detector.landmarks_to_features(landmarks)
                if features is not None:
                    gesture, confidence = self.classifier.predict(features)
                # Draw skeleton
                frame = self.hand_detector.draw_skeleton(frame, landmarks, (0, 255, 0), 2)
            
            # ---------- UI overlays ----------
            # Top bar: scores and instructions
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"YOU: {self.player_score}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"AI: {self.ai_score}", (200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"DRAWS: {self.draws}", (380, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, "SPACE: play | ESC: quit", (w-300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Current detected gesture
            if gesture and confidence > 0.6:
                emoji = GESTURE_EMOJIS[gesture]
                name = GESTURE_NAMES[gesture]
                cv2.putText(frame, f"{emoji} {name} ({confidence:.0%})", (20, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No gesture", (20, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 1)
            
            # Recent moves history
            y_hist = 100
            cv2.putText(frame, "Recent rounds:", (w-200, y_hist),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, (p, a, r) in enumerate(self.round_history):
                color = (0,255,0) if r=='win' else (0,0,255) if r=='lose' else (255,255,0)
                cv2.putText(frame, f"{p} vs {a}", (w-200, y_hist+25+i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Round logic
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            if key == ord(' ') and not self.current_round_active and time.time() > self.result_display_end:
                # Start countdown only if not already in round and result displayed
                self.current_round_active = True
                self.countdown_start = time.time()
                self.player_gesture = gesture  # store the current gesture
                if self.player_gesture is None:
                    self.current_round_active = False
                    continue
                # AI chooses move based on learned pattern
                ai_choice_idx = self.ai.choose_move()
                self.ai_gesture = Gesture(ai_choice_idx)
                # Countdown duration 2 seconds
                
            if self.current_round_active:
                elapsed = time.time() - self.countdown_start
                if elapsed < 2:
                    countdown = int(3 - elapsed)
                    if countdown > 0:
                        cv2.rectangle(frame, (w//2-60, h//2-60), (w//2+60, h//2+60), (0,0,0), -1)
                        cv2.putText(frame, str(countdown), (w//2-30, h//2+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 4)
                else:
                    # Round result
                    result_str = self.determine_winner(self.player_gesture, self.ai_gesture)
                    self.update_scores(result_str)
                    # Record for AI learning
                    self.ai.update(self.player_gesture.value)
                    # Store history
                    p_name = GESTURE_NAMES[self.player_gesture][:3]
                    a_name = GESTURE_NAMES[self.ai_gesture][:3]
                    self.round_history.appendleft((p_name, a_name, result_str))
                    # Display result for 2 seconds
                    self.round_result = result_str
                    self.result_display_end = time.time() + 2
                    self.current_round_active = False
            
            # Show result overlay if active
            if time.time() < self.result_display_end and self.round_result:
                cv2.rectangle(frame, (w//2-200, h//2-100), (w//2+200, h//2+100), (0,0,0), -1)
                if self.round_result == "win":
                    msg = "YOU WIN! 🎉"
                    color = (0,255,0)
                elif self.round_result == "lose":
                    msg = "YOU LOSE! 😢"
                    color = (0,0,255)
                else:
                    msg = "DRAW 🤝"
                    color = (255,255,0)
                cv2.putText(frame, msg, (w//2-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                # Show gestures
                cv2.putText(frame, f"You: {GESTURE_EMOJIS[self.player_gesture]} {GESTURE_NAMES[self.player_gesture]}", 
                            (w//2-180, h//2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                cv2.putText(frame, f"AI : {GESTURE_EMOJIS[self.ai_gesture]} {GESTURE_NAMES[self.ai_gesture]}", 
                            (w//2-180, h//2+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            
            # Show instruction if no round active and not showing result
            if not self.current_round_active and time.time() >= self.result_display_end:
                cv2.putText(frame, "Show a gesture and press SPACE", (w//2-250, h-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            cv2.imshow("Mind Games - RPSLS", frame)
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Game over. Final score: You {} - {} AI".format(self.player_score, self.ai_score))

if __name__ == "__main__":
    game = Game()
    game.run()