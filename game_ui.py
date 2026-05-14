"""
Main game interface using Pygame
Integrates webcam feed, gesture recognition, game logic, and UI
"""

import cv2
import pygame
import numpy as np
from typing import Optional
from datetime import datetime
import time

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_FPS, BACKGROUND_COLOR,
    WEBCAM_AREA_WIDTH, STATS_AREA_WIDTH, COUNTDOWN_DURATION,
    GESTURE_NAMES, GESTURE_EMOJIS, Gesture,
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING, COLOR_TEXT,
    FONT_LARGE, FONT_MEDIUM, FONT_SMALL
)

from hand_detector import HandDetector
from gesture_classifier import GestureClassifier
from sequence_predictor import SequencePredictor, AIOpponent
from game_engine import GameEngine, GameResult, RoundResultAnalyzer


class GameUI:
    """Main UI controller for the game"""
    
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🧠 Mind Games - RPSLS vs AI")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, FONT_LARGE)
        self.font_medium = pygame.font.Font(None, FONT_MEDIUM)
        self.font_small = pygame.font.Font(None, FONT_SMALL)
        
        # Initialize components
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
        self.sequence_predictor = SequencePredictor()
        self.ai_opponent = AIOpponent(self.sequence_predictor)
        self.game_engine = GameEngine()
        
        # Game state
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.running = False
        self.game_state = 'waiting'  # 'waiting', 'countdown', 'showing_result'
        self.current_player_gesture: Optional[Gesture] = None
        self.current_ai_gesture: Optional[Gesture] = None
        self.countdown_time = 0
        self.result_display_time = 0
        self.last_frame_rgb = None
        
        # Try to load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """Attempt to load pre-trained models"""
        if self.gesture_classifier.load():
            print("✓ Gesture classifier loaded")
        else:
            print("⚠ Gesture classifier not found. Please train first.")
        
        if self.sequence_predictor.load():
            print("✓ Sequence predictor loaded")
        else:
            print("⚠ Sequence predictor not found. AI will play randomly.")
    
    def _convert_cv_frame_to_pygame(self, cv_frame: np.ndarray) -> pygame.Surface:
        """Convert OpenCV BGR frame to Pygame surface"""
        # Flip for mirror effect and resize
        cv_frame = cv2.flip(cv_frame, 1)
        cv_frame = cv2.resize(cv_frame, (WEBCAM_AREA_WIDTH, SCREEN_HEIGHT))
        
        # BGR to RGB
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to pygame surface
        pygame._frame_obj = rgb_frame
        surface = pygame.image.fromstring(
            rgb_frame.tobytes(),
            rgb_frame.shape[1::-1],
            'RGB'
        )
        
        return surface
    
    def _draw_webcam_area(self, frame: pygame.Surface):
        """Draw webcam feed area"""
        self.screen.blit(frame, (0, 0))
        
        # Draw border
        pygame.draw.rect(self.screen, COLOR_PRIMARY, 
                        (0, 0, WEBCAM_AREA_WIDTH, SCREEN_HEIGHT), 3)
    
    def _draw_stats_area(self):
        """Draw right sidebar with stats"""
        # Background
        pygame.draw.rect(self.screen, (20, 20, 40),
                        (WEBCAM_AREA_WIDTH, 0, STATS_AREA_WIDTH, SCREEN_HEIGHT))
        
        x_base = WEBCAM_AREA_WIDTH + 20
        y_pos = 30
        line_height = 50
        
        # Title
        title = self.font_large.render("STATISTICHE", True, COLOR_PRIMARY)
        self.screen.blit(title, (x_base, y_pos))
        y_pos += 70
        
        # Scores
        stats = self.game_engine.get_stats()
        
        # Player score
        player_score_text = self.font_medium.render(
            f"Tu: {stats.player_wins}", True, COLOR_SUCCESS
        )
        self.screen.blit(player_score_text, (x_base, y_pos))
        y_pos += line_height
        
        # AI score
        ai_score_text = self.font_medium.render(
            f"IA: {stats.ai_wins}", True, COLOR_DANGER
        )
        self.screen.blit(ai_score_text, (x_base, y_pos))
        y_pos += line_height
        
        # Draws
        draws_text = self.font_medium.render(
            f"Pareggi: {stats.draws}", True, COLOR_WARNING
        )
        self.screen.blit(draws_text, (x_base, y_pos))
        y_pos += line_height * 1.5
        
        # Total rounds
        if stats.total_rounds > 0:
            total_text = self.font_small.render(
                f"Totale: {stats.total_rounds} round", True, COLOR_TEXT
            )
            self.screen.blit(total_text, (x_base, y_pos))
            y_pos += 40
            
            # Win rate
            wr_text = self.font_small.render(
                f"Tuo Win Rate: {stats.win_rate:.1f}%", True, COLOR_TEXT
            )
            self.screen.blit(wr_text, (x_base, y_pos))
            y_pos += 60
        
        # Recent rounds
        recent_text = self.font_medium.render("ULTIMI ROUND", True, COLOR_PRIMARY)
        self.screen.blit(recent_text, (x_base, y_pos))
        y_pos += 50
        
        for round_obj in self.game_engine.get_recent_rounds(3):
            emoji_summary = RoundResultAnalyzer.get_round_emoji_summary(round_obj)
            round_text = self.font_small.render(emoji_summary, True, COLOR_TEXT)
            self.screen.blit(round_text, (x_base, y_pos))
            y_pos += 35
    
    def _draw_countdown(self):
        """Draw countdown timer"""
        remaining = COUNTDOWN_DURATION - (time.time() - self.countdown_time)
        
        if remaining > 0:
            countdown_num = int(remaining) + 1
            text = self.font_large.render(str(countdown_num), True, COLOR_WARNING)
            text_rect = text.get_rect(center=(WEBCAM_AREA_WIDTH // 2, SCREEN_HEIGHT // 2))
            
            # Semi-transparent background
            pygame.draw.circle(self.screen, (0, 0, 0), text_rect.center, 80)
            self.screen.blit(text, text_rect)
        else:
            self.game_state = 'showing_result'
            self.result_display_time = time.time()
    
    def _draw_result(self):
        """Draw round result"""
        if self.game_engine.current_round is None:
            return
        
        round_obj = self.game_engine.current_round
        result = round_obj.result
        
        # Get result message
        msg, color_name = RoundResultAnalyzer.get_result_message(result)
        
        # Color mapping
        colors = {
            'success': COLOR_SUCCESS,
            'danger': COLOR_DANGER,
            'warning': COLOR_WARNING
        }
        color = colors.get(color_name, COLOR_TEXT)
        
        # Gestures
        player_name = GESTURE_NAMES[round_obj.player_gesture]
        ai_name = GESTURE_NAMES[round_obj.ai_gesture]
        player_emoji = GESTURE_EMOJIS[round_obj.player_gesture]
        ai_emoji = GESTURE_EMOJIS[round_obj.ai_gesture]
        
        # Draw result message
        result_text = self.font_large.render(msg, True, color)
        result_rect = result_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
        self.screen.blit(result_text, result_rect)
        
        # Draw gestures
        player_text = self.font_medium.render(
            f"{player_emoji} {player_name}", True, COLOR_SUCCESS
        )
        ai_text = self.font_medium.render(
            f"{ai_emoji} {ai_name}", True, COLOR_DANGER
        )
        
        self.screen.blit(player_text, (100, 250))
        self.screen.blit(ai_text, (SCREEN_WIDTH - 400, 250))
        
        # Draw explanation
        explanation = RoundResultAnalyzer.get_gesture_explanation(
            round_obj.player_gesture, round_obj.ai_gesture
        )
        expl_text = self.font_small.render(explanation, True, COLOR_TEXT)
        expl_rect = expl_text.get_rect(center=(SCREEN_WIDTH // 2, 400))
        self.screen.blit(expl_text, expl_rect)
        
        # Check if display time is over
        if time.time() - self.result_display_time > 2:
            self.game_state = 'waiting'
    
    def run(self):
        """Main game loop"""
        self.running = True
        frame_count = 0
        
        print("\n🎮 Starting Mind Games...")
        print("Press 'R' to reset, 'Q' to quit\n")
        
        while self.running:
            # Limit FPS
            self.clock.tick(SCREEN_FPS)
            
            # Read and process frame
            ret, cv_frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process gesture every N frames for performance
            gesture = None
            confidence = 0
            
            if frame_count % 2 == 0:
                # Detect hand
                _, landmarks, hand_detected = self.hand_detector.detect(cv_frame)
                
                if hand_detected and landmarks:
                    # Extract features and classify
                    features = self.hand_detector.landmarks_to_features(landmarks)
                    if features is not None:
                        gesture, confidence = self.gesture_classifier.predict(features)
                        
                        # Record for LSTM
                        if gesture:
                            self.sequence_predictor.record_move(gesture)
                    
                    # Draw skeleton
                    landmarks_smooth = self.hand_detector.get_smoothed_landmarks()
                    if landmarks_smooth:
                        cv_frame = self.hand_detector.draw_skeleton(
                            cv_frame, landmarks_smooth,
                            color=(0, 255, 0), thickness=2
                        )
            
            # Convert frame for display
            pygame_frame = self._convert_cv_frame_to_pygame(cv_frame)
            
            # Clear screen
            self.screen.fill(BACKGROUND_COLOR)
            
            # Draw main areas
            self._draw_webcam_area(pygame_frame)
            self._draw_stats_area()
            
            # Draw gesture info (if detected)
            if gesture and confidence > 0.6:
                gesture_name = GESTURE_NAMES[gesture]
                gesture_emoji = GESTURE_EMOJIS[gesture]
                gesture_text = self.font_medium.render(
                    f"{gesture_emoji} {gesture_name} ({confidence:.1%})",
                    True, COLOR_PRIMARY
                )
                self.screen.blit(gesture_text, (20, SCREEN_HEIGHT - 60))
            
            # Handle game states
            if self.game_state == 'countdown':
                self._draw_countdown()
            elif self.game_state == 'showing_result':
                self._draw_result()
            elif self.game_state == 'waiting' and gesture and confidence > 0.7:
                # Auto-start countdown when gesture detected
                self.current_player_gesture = gesture
                self.current_ai_gesture = self.ai_opponent.choose_move()
                self.game_state = 'countdown'
                self.countdown_time = time.time()
                
                # Play game round
                self.game_engine.play_round(self.current_player_gesture, self.current_ai_gesture)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.game_engine.reset()
                        self.sequence_predictor.clear_history()
                        print("✓ Game reset")
            
            # Update display
            pygame.display.flip()
        
        self.quit()
    
    def quit(self):
        """Cleanup"""
        print("\n👋 Closing Mind Games...")
        self.running = False
        self.cap.release()
        self.hand_detector.close()
        pygame.quit()


def main():
    """Entry point"""
    try:
        game = GameUI()
        game.run()
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == '__main__':
    main()
