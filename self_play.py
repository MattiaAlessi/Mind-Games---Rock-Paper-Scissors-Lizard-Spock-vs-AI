#!/usr/bin/env python3
"""
Self-play: due AI si sfidano a vicenda e imparano l'una dall'altra.
- AI_A salva su data/ai_memory.npz
- AI_B salva su data/opponent_memory.npz
Premi CTRL+C per interrompere e salvare entrambe le memorie.
"""

import time
import signal
import sys
import os
from game import AdaptiveAI

# Configurazione
ROUNDS_BEFORE_SAVE = 1000   # salvataggio intermedio ogni N round
SLEEP_MS = 0                # ritardo tra round (0 = massima velocità)

class SelfPlay:
    def __init__(self):
        self.ai_a = AdaptiveAI(memory_file="data/ai_memory.npz")
        self.ai_b = AdaptiveAI(memory_file="data/opponent_memory.npz")
        self.running = True
        self.round_count = 0
        self.last_save = 0

    def run(self):
        print("🎮 Self-play avviato. Due AI si sfidano all'infinito.")
        print("   Ogni AI impara dalle mosse dell'avversaria.")
        print("   Premi CTRL+C per fermare e salvare le memorie.\n")
        
        # Gestisce l'interruzione da tastiera
        signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            while self.running:
                # AI A sceglie una mossa (basata sulla storia delle mosse di B)
                move_a = self.ai_a.choose_move()
                # AI B sceglie una mossa (basata sulla storia delle mosse di A)
                move_b = self.ai_b.choose_move()
                
                # Ogni AI aggiorna la propria memoria con la mossa dell'avversaria
                # (per A, l'avversario è B; per B, l'avversario è A)
                self.ai_a.update(move_b)
                self.ai_b.update(move_a)
                
                self.round_count += 1
                
                # Mostra progresso ogni 500 round
                if self.round_count % 500 == 0:
                    total_a = self.ai_a.total_rounds
                    total_b = self.ai_b.total_rounds
                    print(f"Round {self.round_count} | Memoria A: {total_a} | Memoria B: {total_b}")
                
                # Salvataggio periodico
                if self.round_count - self.last_save >= ROUNDS_BEFORE_SAVE:
                    self.ai_a.save_memory()
                    self.ai_b.save_memory()
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
        print("\nOra puoi lanciare 'python game.py' e l'AI userà la memoria addestrata.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    sp = SelfPlay()
    sp.run()