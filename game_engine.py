"""
Game logic for Rock-Paper-Scissors-Lizard-Spock
Handles game state, scoring, and win/lose/draw determination
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

from config import Gesture, WINS, GESTURE_NAMES, GESTURE_EMOJIS


class GameResult(Enum):
    WIN = 1
    LOSE = -1
    DRAW = 0


@dataclass
class GameRound:
    """Represents a single game round"""
    timestamp: datetime
    player_gesture: Gesture
    ai_gesture: Gesture
    result: GameResult
    
    def __str__(self) -> str:
        emoji_player = GESTURE_EMOJIS[self.player_gesture]
        emoji_ai = GESTURE_EMOJIS[self.ai_gesture]
        name_player = GESTURE_NAMES[self.player_gesture]
        name_ai = GESTURE_NAMES[self.ai_gesture]
        
        if self.result == GameResult.WIN:
            result_str = "✓ VINTO"
        elif self.result == GameResult.LOSE:
            result_str = "✗ PERSO"
        else:
            result_str = "= PAREGGIO"
        
        return f"{emoji_player} {name_player:8} vs {emoji_ai} {name_ai:8} | {result_str}"


@dataclass
class GameStats:
    """Track cumulative game statistics"""
    total_rounds: int = 0
    player_wins: int = 0
    ai_wins: int = 0
    draws: int = 0
    
    @property
    def win_rate(self) -> float:
        """Player win rate percentage"""
        if self.total_rounds == 0:
            return 0.0
        return (self.player_wins / self.total_rounds) * 100
    
    @property
    def ai_win_rate(self) -> float:
        """AI win rate percentage"""
        if self.total_rounds == 0:
            return 0.0
        return (self.ai_wins / self.total_rounds) * 100
    
    def __str__(self) -> str:
        return (f"Rounds: {self.total_rounds} | "
                f"Player: {self.player_wins} ({self.win_rate:.1f}%) | "
                f"AI: {self.ai_wins} ({self.ai_win_rate:.1f}%) | "
                f"Draws: {self.draws}")


class GameEngine:
    """Core game logic"""
    
    def __init__(self):
        self.history = []
        self.stats = GameStats()
        self.current_round = None
    
    @staticmethod
    def determine_winner(player_gesture: Gesture,
                        ai_gesture: Gesture) -> GameResult:
        """
        Determine round winner
        
        Args:
            player_gesture: Player's choice
            ai_gesture: AI's choice
            
        Returns:
            GameResult enum
        """
        if player_gesture == ai_gesture:
            return GameResult.DRAW
        
        # Check if player gesture beats AI gesture
        if ai_gesture in WINS[player_gesture]:
            return GameResult.WIN
        else:
            return GameResult.LOSE
    
    def play_round(self, player_gesture: Gesture, ai_gesture: Gesture) -> GameRound:
        """
        Execute a single game round
        
        Args:
            player_gesture: Player's move
            ai_gesture: AI's move
            
        Returns:
            GameRound object with result
        """
        result = self.determine_winner(player_gesture, ai_gesture)
        
        round_obj = GameRound(
            timestamp=datetime.now(),
            player_gesture=player_gesture,
            ai_gesture=ai_gesture,
            result=result
        )
        
        # Update history and stats
        self.history.append(round_obj)
        self.stats.total_rounds += 1
        
        if result == GameResult.WIN:
            self.stats.player_wins += 1
        elif result == GameResult.LOSE:
            self.stats.ai_wins += 1
        else:
            self.stats.draws += 1
        
        self.current_round = round_obj
        return round_obj
    
    def get_recent_rounds(self, n: int = 5) -> list:
        """Get last N rounds"""
        return self.history[-n:]
    
    def get_player_gesture_sequence(self) -> list:
        """Get sequence of player gestures"""
        return [round.player_gesture for round in self.history]
    
    def reset(self):
        """Reset game state"""
        self.history.clear()
        self.stats = GameStats()
        self.current_round = None
    
    def get_stats(self) -> GameStats:
        """Get game statistics"""
        return self.stats


class RoundResultAnalyzer:
    """Analyze and format round results for display"""
    
    @staticmethod
    def get_result_message(result: GameResult) -> Tuple[str, str]:
        """
        Get display message and color for result
        
        Args:
            result: GameResult enum
            
        Returns:
            Tuple of (message, color_name)
        """
        if result == GameResult.WIN:
            return "HAI VINTO! 🎉", "success"
        elif result == GameResult.LOSE:
            return "Hai perso... 😢", "danger"
        else:
            return "Pareggio", "warning"
    
    @staticmethod
    def get_gesture_explanation(player: Gesture, ai: Gesture) -> str:
        """
        Explain why one gesture beats another
        
        Args:
            player: Player gesture
            ai: AI gesture
            
        Returns:
            Explanation string
        """
        explanations = {
            (Gesture.ROCK, Gesture.SCISSORS): "Sasso schiaccia forbice",
            (Gesture.ROCK, Gesture.LIZARD): "Sasso schiaccia lucertola",
            (Gesture.PAPER, Gesture.ROCK): "Carta avvolge sasso",
            (Gesture.PAPER, Gesture.SPOCK): "Carta smentisce Spock",
            (Gesture.SCISSORS, Gesture.PAPER): "Forbice taglia carta",
            (Gesture.SCISSORS, Gesture.LIZARD): "Forbice decapita lucertola",
            (Gesture.LIZARD, Gesture.PAPER): "Lucertola mangia carta",
            (Gesture.LIZARD, Gesture.SPOCK): "Lucertola avvelena Spock",
            (Gesture.SPOCK, Gesture.SCISSORS): "Spock spezza forbice",
            (Gesture.SPOCK, Gesture.ROCK): "Spock vaporizza sasso",
        }
        
        key = (player, ai)
        if key in explanations:
            return explanations[key]
        
        # Draw or other case
        return f"{GESTURE_NAMES[player]} vs {GESTURE_NAMES[ai]}"
    
    @staticmethod
    def get_round_emoji_summary(round_obj: GameRound) -> str:
        """Get emoji summary of round"""
        emoji_p = GESTURE_EMOJIS[round_obj.player_gesture]
        emoji_ai = GESTURE_EMOJIS[round_obj.ai_gesture]
        
        if round_obj.result == GameResult.WIN:
            result_emoji = "✓"
        elif round_obj.result == GameResult.LOSE:
            result_emoji = "✗"
        else:
            result_emoji = "="
        
        return f"{emoji_p} {emoji_ai} {result_emoji}"
