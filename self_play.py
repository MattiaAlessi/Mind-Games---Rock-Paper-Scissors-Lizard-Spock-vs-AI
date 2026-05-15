#!/usr/bin/env python3
"""
Self-play: due AI si sfidano a vicenda e imparano l'una dall'altra.
Con epsilon decay e logging.
"""

import time
import signal
import sys
import os
from game import AdaptiveAI
from logger import app_logger

ROUNDS_BEFORE_SAVE = 1000
SLEEP_MS = 0

class SelfPlay:
    def __init__(self):
        self.ai_a = AdaptiveAI(memory_file="data/ai_memory.npz",
                               epsilon=0.2, epsilon_decay=0.99995)
        self.ai_b = AdaptiveAI(memory_file="data/opponent_memory.npz",
                               epsilon=0.2, epsilon_decay=0.99995)
        self.running = True
        self.round_count = 0
        self.last_save = 0

    def run(self):
        print("🎮 Self-play avviato. Due AI si sfidano all'infinito.")
        print("   Ogni AI impara dalle mosse dell'avversaria.")
        print("   Premi CTRL+C per fermare e salvare le memorie.\n")
        signal.signal(signal.SIGINT, self.signal_handler)
        try:
            while self.running:
                move_a = self.ai_a.choose_move()
                move_b = self.ai_b.choose_move()
                self.ai_a.update(move_b)
                self.ai_b.update(move_a)
                self.round_count += 1
                if self.round_count % 500 == 0:
                    app_logger.info(f"Self-play round {self.round_count} | ε_A={self.ai_a.epsilon:.4f} ε_B={self.ai_b.epsilon:.4f}")
                    print(f"Round {self.round_count} | Memoria A: {self.ai_a.total_rounds} | Memoria B: {self.ai_b.total_rounds}")
                if self.round_count - self.last_save >= ROUNDS_BEFORE_SAVE:
                    self.ai_a.save_memory_async()
                    self.ai_b.save_memory_async()
                    self.last_save = self.round_count
                    print(f"💾 Salvataggio intermedio al round {self.round_count}")
                if SLEEP_MS > 0:
                    time.sleep(SLEEP_MS / 1000.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.save_and_exit()

    def signal_handler(self, sig, frame):
        print("\n🛑 Interruzione ricevuta, salvataggio in corso...")
        self.running = False

    def save_and_exit(self):
        self.ai_a.save_memory()
        self.ai_b.save_memory()
        print(f"\n✅ Self-play terminato dopo {self.round_count} round.")
        print(f"   Memoria A (giocatore): {self.ai_a.total_rounds} mosse apprese")
        print(f"   Memoria B (opponente): {self.ai_b.total_rounds} mosse apprese")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    sp = SelfPlay()
    sp.run()