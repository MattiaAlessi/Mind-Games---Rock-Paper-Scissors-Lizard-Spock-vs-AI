#!/usr/bin/env python3
"""
QUICK START GUIDE - Mind Games Dataset Creation
Segui questo script passo per passo!
"""

import os
import sys

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def print_step(step_num, title):
    print(f"\n📍 STEP {step_num}: {title}")
    print("-" * 70)

def main():
    print_header("🎮 MIND GAMES - QUICK START GUIDE")
    
    print("""
Questo script guida te attraverso la creazione di un dataset robusto
per il riconoscimento di gesti RPSLS.

⏱️  Tempo stimato: 2-3 giorni
""")
    
    # =======================
    # STEP 0: PREREQUISITI
    # =======================
    print_step(0, "PREREQUISITI")
    
    print("""
Assicurati di avere:
✓ Python 3.9+
✓ Webcam funzionante
✓ Buona illuminazione
✓ Spazio libero (30cm davanti alla webcam)

Installa dipendenze:
    pip install -r requirements.txt

Verifica che tutto funziona:
    python -c "import cv2, mediapipe, pygame; print('✓ OK')"
""")
    
    input("Premi INVIO quando pronto...")
    
    # =======================
    # STEP 1: COMPRENSIONE GESTI
    # =======================
    print_step(1, "COMPRENSIONE DEI GESTI")
    
    print("""
Leggi le ISTRUZIONI PRECISE per ogni gesto:
    python DATASET_STRATEGY.py

Questo script mostra:
1. Come fare ESATTAMENTE ogni gesto
2. Quali sono gli ERRORI COMUNI
3. Come differenziare gesti simili

⚠️  CRITICO: SPOCK e CARTA sono molto simili!
   - CARTA: dita UNITE (distanza 0)
   - SPOCK: dita SEPARATE (distanza massima)

Prenditi tempo per capire le differenze!
""")
    
    input("Premi INVIO dopo aver letto le istruzioni...")
    
    # =======================
    # STEP 2: RACCOLTA DATI
    # =======================
    print_step(2, "RACCOLTA DATI INTELLIGENTE")
    
    print("""
Usa lo smart collector che fornisce:
✓ Istruzioni in tempo reale
✓ Feedback visuale (gesto riconosciuto)
✓ Progress tracking

COMANDO:
    python smart_collector.py

WORKFLOW DURANTE LA RACCOLTA:
1. Leggi le istruzioni per il gesto
2. Mostra il gesto davanti alla webcam
3. Quando il gesto è CORRETTO, premi SPACE
4. Il sistema dirà se è stato riconosciuto bene

TARGET CAMPIONI PER GESTO:
- ✊ SASSO:    150-200 campioni
- ✋ CARTA:    150-180 campioni  
- ✌️ FORBICE:  100-120 campioni
- 🦎 LIZARD:   150-200 campioni
- 🖖 SPOCK:    150-180 campioni

VARIAZIONI IMPORTANTI:
Per ogni gesto, varia:
✓ Angolo: frontale, destra, sinistra, alto, basso
✓ Distanza: vicina (30cm), media (60cm), lontana (100cm)
✓ Illuminazione: naturale, artificiale, contro-luce

⏱️  Tempo previsto: 3-4 ore

CONSIGLI:
- Non affrettarti
- Leggi le istruzioni per OGNI gesto
- Se il feedback dice ARANCIONE, significa che il gesto
  non è completamente corretto. Osserva il suggerimento!
- Fai pause ogni 30 minuti per evitare stanchezza
""")
    
    input("Premi INVIO per iniziare la raccolta...")
    
    print("""
AVVIA ORA:
    python smart_collector.py
    
Quando finisci, torna qui.
""")
    
    input("Premi INVIO quando hai finito la raccolta...")
    
    # =======================
    # STEP 3: TRAINING
    # =======================
    print_step(3, "TRAINING DEL CLASSIFICATORE")
    
    print("""
Allena il modello LightGBM sui dati raccolti:
    
    python train.py --train-classifier --data-file training_data_smart.npz

Questo farà:
✓ Normalizzazione feature
✓ Data augmentation (aumenta variabilità)
✓ Training LightGBM con cross-validation
✓ Salvataggio del modello

ASPETTATIVE:
- Tempo: 5-10 minuti
- Accuratezza target: > 90%

QUANDO FINISCE, vedrai:
- Mean CV Accuracy (cross-validation)
- File salvati in models/
""")
    
    input("Premi INVIO per iniziare il training...")
    
    print("""
AVVIA ORA:
    python train.py --train-classifier --data-file training_data_smart.npz
    
Quando finisci, torna qui.
""")
    
    input("Premi INVIO quando il training è finito...")
    
    # =======================
    # STEP 4: DIAGNOSTICA
    # =======================
    print_step(4, "ANALISI E DIAGNOSTICA")
    
    print("""
Analizza il modello per identificare problemi:
    
    python diagnostics.py --data data/training_data_smart.npz

Questo mostra:
✓ Accuratezza globale
✓ Gesti "deboli" (accuracy < 85%)
✓ Coppie di gesti confusi
✓ Feature importanti
✓ Raccomandazioni specifiche

OUTPUT:
- confusion_matrix.png (quale gesto è confuso con quale)
- per_gesture_metrics.png (precision/recall per gesto)

ANALIZZA I RISULTATI:
Se accuracy > 90% e no confusioni significative:
    → Vai a STEP 6 (LSTM)

Se accuracy < 90% o confusioni > 10%:
    → Vai a STEP 5 (Miglioramento)
""")
    
    input("Premi INVIO per avviare la diagnostica...")
    
    print("""
AVVIA ORA:
    python diagnostics.py --data data/training_data_smart.npz
    
Guarda i file PNG generati e leggi il report.
Quando finisci, torna qui.
""")
    
    input("Premi INVIO quando hai analizzato i risultati...")
    
    # =======================
    # STEP 5: IMPROVEMENT (OPZIONALE)
    # =======================
    print_step(5, "MIGLIORAMENTO MIRATO (OPZIONALE)")
    
    print("""
Se la diagnostica ha evidenziato problemi:

PROBLEMI COMUNI E SOLUZIONI:
    
1️⃣  CARTA vs SPOCK confusi
   └─ SOLUZIONE: Unisci le dita in CARTA, separa in SPOCK
   └─ Raccogli 100 campioni extra per ognuno
   
2️⃣  SASSO vs LIZARD confusi
   └─ SOLUZIONE: Pollice DENTRO in SASSO, FUORI in LIZARD
   └─ Raccogli 100 campioni extra con luce migliore
   
3️⃣  LIZARD vs SPOCK confusi
   └─ SOLUZIONE: LIZARD ha 2 dita chiuse, SPOCK tutte aperte
   └─ Raccogli 100 campioni extra enfatizzando la differenza

WORKFLOW:
1. Torna al STEP 2 (smart_collector.py)
2. Raccogli campioni AGGIUNTIVI mirati
3. Riallena (STEP 3)
4. Rivaluta (STEP 4)
5. Ripeti finché accuracy > 90%

⚠️  NON saltare questo step se accuracy < 85%!
""")
    
    response = input("Hai problemi che richiedono miglioramento? (s/n): ").strip().lower()
    
    if response == 's':
        print("""
Torna al STEP 2:
    python smart_collector.py
    
Dopo aver raccolto nuovi campioni:
    python train.py --train-classifier --data-file training_data_smart.npz
    python diagnostics.py --data data/training_data_smart.npz
    
Quando accuracy > 90%, procedi a STEP 6.
""")
        input("Premi INVIO quando accuracy è buona...")
    
    # =======================
    # STEP 6: LSTM
    # =======================
    print_step(6, "TRAINING LSTM (AI OPPONENT)")
    
    print("""
Allena il modello LSTM che fa imparare all'IA
le tue abitudini di gioco:
    
    python train.py --train-lstm

Questo farà:
✓ Generazione dati sintetici
✓ Training LSTM per predizione sequenze
✓ Salvataggio modello LSTM

TEMPO: 10-15 minuti
OUTPUT: models/sequence_predictor.h5
""")
    
    input("Premi INVIO per avviare il training LSTM...")
    
    print("""
AVVIA ORA:
    python train.py --train-lstm
    
Quando finisci, torna qui.
""")
    
    input("Premi INVIO quando LSTM training è completo...")
    
    # =======================
    # STEP 7: TESTING
    # =======================
    print_step(7, "TESTING DEL GIOCO")
    
    print("""
Avvia il gioco e testa il sistema:
    
    python game_ui.py

DURANTE IL TEST:
✓ Mostra i 5 gesti davanti alla webcam
✓ Verifica che siano riconosciuti correttamente
✓ Gioca 50+ round contro l'IA
✓ Verifica che l'IA imparare dalle tue mosse

FEEDBACK DA CERCARE:
✓ Riconoscimento gesti: accurato?
✓ IA impara? (vince più spesso col tempo?)
✓ Performance: niente lag?
✓ UI: chiara e fruibile?

CONTROLI:
- R: Reset gioco
- Q: Esci

Se tutto funziona bene → ✅ FATTO!

Se ci sono problemi:
- Bassa accuratezza gesti: torna a STEP 5 (più dati)
- IA non impara: riallena LSTM
- Lag: riduci SCREEN_FPS in config.py
""")
    
    input("Premi INVIO per avviare il gioco...")
    
    print("""
AVVIA ORA:
    python game_ui.py
    
Gioca e testa! Quando finisci, torna qui.
""")
    
    input("Premi INVIO quando hai testato il gioco...")
    
    # =======================
    # STEP 8: FINALIZZAZIONE
    # =======================
    print_step(8, "FINALIZZAZIONE E DEPLOYMENT")
    
    print("""
✅ SETUP COMPLETO!

Hai creato un sistema completo di:
✓ Hand tracking in tempo reale (MediaPipe)
✓ Classificatore di gesti (LightGBM)
✓ Predittore di sequenze (LSTM)
✓ Interfaccia di gioco (Pygame)

STRUTTURA FINALE:
mind_games/
├── config.py                 # Configurazione
├── hand_detector.py          # Hand tracking
├── gesture_classifier.py     # Classifier
├── sequence_predictor.py     # LSTM
├── game_engine.py            # Game logic
├── game_ui.py                # UI
├── models/
│   ├── gesture_classifier.pkl
│   ├── gesture_scaler.pkl
│   └── sequence_predictor.h5
└── data/
    └── training_data_smart.npz

PER CONTINUARE:
- Salva i modelli in version control
- Documenta il performance (accuracy, confusion matrix)
- Raccogli più dati per miglioramenti futuri
- Prova diverse strategie di IA

PER DEPLOYMENT FUTURO:
- Esporta modelli per TensorFlow Lite (mobile)
- Crea API REST
- Deploy in cloud

RACCOMANDAZIONI:
1. Fai un backup dei modelli
2. Documentazione chiara del dataset
3. Versioning dei modelli (v1.0, v1.1, ecc)
4. Monitora performance in produzione
""")
    
    print("\n" + "="*70)
    print("  ✅ GUIDA COMPLETATA!")
    print("="*70)
    
    print("""
🎯 NEXT STEPS:
1. Migliora il dataset se necessario
2. Esperimenta con hyperparameter
3. Aggiungi funzionalità (replay, ranking, ecc)
4. Testa con altri giocatori

📧 Buona fortuna con il progetto!
""")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Guida interrotta. Puoi riavviarla in qualsiasi momento con:")
        print("    python QUICK_START.py")
        sys.exit(0)
