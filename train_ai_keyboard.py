#!/usr/bin/env python3
"""
Addestra rapidamente l'AI con input da tastiera (1-5).
Usa la stessa classe AdaptiveAI del gioco.
Dopo l'addestramento, l'AI sarà più intelligente in game.py
"""

import sys
from game import AdaptiveAI   

GESTI = {
    1: "Sasso",
    2: "Carta",
    3: "Forbice",
    4: "Lizard",
    5: "Spock"
}

def main():
    print("\n=== Addestramento rapido AI (Markov ordine 2) ===")
    print("Inserisci il numero della tua mossa (1-5).")
    print("Dopo ogni mossa l'AI aggiorna la sua memoria.")
    print("Comandi: 'q' per uscire e salvare, 'r' per resettare la memoria.\n")

    ai = AdaptiveAI()   # carica memoria esistente se presente

    while True:
        cmd = input("👉 La tua mossa (1-5, q=exit, r=reset): ").strip()
        if cmd == 'q':
            break
        if cmd == 'r':
            # reset memoria
            ai.transitions.clear()
            ai.round_buffer.clear()
            ai.history.clear()
            ai.total_rounds = 0
            print("🧹 Memoria resettata.\n")
            continue
        try:
            move = int(cmd)
            if move < 1 or move > 5:
                print("❌ Numero non valido. Usa 1-5.\n")
                continue
        except ValueError:
            print("❌ Inserisci un numero tra 1 e 5.\n")
            continue

        # aggiorna l'AI con la mossa del giocatore
        ai.update(move - 1)   # la classe vuole 0-based
        print(f"✓ Registrato: {GESTI[move]}")
        # mostra ultime mosse registrate
        if len(ai.round_buffer) > 0:
            last_moves = [GESTI[m+1] for m in ai.round_buffer[-5:]]
            print(f"   Ultime mosse: {' → '.join(last_moves)}")
        print()

    ai.save_memory()
    print("\n✅ Addestramento completato. Ora puoi lanciare 'python game.py'")
    print(f"   L'AI ha memorizzato {ai.total_rounds} mosse.\n")

if __name__ == "__main__":
    main()