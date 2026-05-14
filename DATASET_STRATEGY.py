"""
STRATEGIE COMPLETE PER DATASET ROBUSTO - RPSLS
==============================================

PROBLEMA IDENTIFICATO:
- CARTA e SPOCK sono facilissime da confondere (entrambe mano aperta)
- SASSO e LIZARD: pollice dentro vs fuori
- LIZARD e SPOCK: livello di separazione dita diverso

SOLUZIONE MULTI-LAYER:
"""

# ===========================================================================
# LAYER 1: ISTRUZIONI PRECISE E DIFFERENZIATE
# ===========================================================================

"""
✅ ROCCIA (✊)
- Pugno COMPLETAMENTE CHIUSO
- Pollice DENTRO il pugno (non fuori!)
- Dorso della mano verso la telecamera
- Palmo completamente invisibile

COSA NON FARE:
❌ Pollice fuori → sembra forbice
❌ Dita semi-aperte → non è roccia pura
❌ Postura rilassata → confuso con lizard


✅ CARTA (✋)  
- Mano COMPLETAMENTE aperta
- Dita UNITE (non divaricate!)
- Palmo piatto verso telecamera
- Niente gaps tra le dita

DIFFERENZA DA SPOCK:
CARTA:  dita UNITE ____ ____ ____
SPOCK:  dita SEPARATE __ __ __ __ __
        (incluso pollice)

COSA NON FARE:
❌ Separare il pollice → diventa Spock
❌ Divarica indice-medio → non è carta
❌ Piegare le dita → sembra forbice


✅ FORBICE (✌️)
- SOLO indice e medio aperti a V
- Ring e pinky CHIUSI dentro pugno
- Pollice DENTRO (non fuori!)
- V ben marcata e visibile

DIFFERENZA DA SPOCK:
FORBICE: 2 dita aperte (__  __)  + 2 chiuse
SPOCK:   4 dita aperte (__  __)(__  __) + pollice

COSA NON FARE:
❌ Pollice fuori → sembra Spock
❌ Aprire altre dita → non è forbice


✅ LIZARD (🦎)
- Pollice FUORI e SEPARATO dal resto
- Indice e medio APERTI a V (NON chiusi!)
- Ring e pinky CHIUSI dentro pugno
- Forma di "rana" con bocca aperta

DIFFERENZA DA SASSO:
SASSO:  pugno chiuso, pollice DENTRO
LIZARD: pugno parziale, pollice FUORI, indice-medio V

DIFFERENZA DA SPOCK:
SPOCK:  TUTTE 5 dita separate (forma due V)
LIZARD: SOLO 3 dita aperte (pollice + indice-medio)

COSA NON FARE:
❌ Chiudere indice-medio → diventa sasso
❌ Separare ring-pinky → diventa Spock
❌ Tenere pollice dentro → diventa forbice


✅ SPOCK (🖖)
- Mano COMPLETAMENTE aperta
- TUTTE 5 dita separate (visibili, non toccate)
- Indice-medio in V + ring-pinky in V (due V)
- Pollice separato lateralmente

DIFFERENZA DA CARTA:
CARTA:  dita UNITE (distanza 0)
SPOCK:  dita SEPARATE (distanza massima)

COSA NON FARE:
❌ Unire le dita → diventa carta
❌ Chiudere ring-pinky → diventa lizard
❌ Tenere pollice dentro → diventa carta

"""

# ===========================================================================
# LAYER 2: STRATEGIE DI RACCOLTA DATI
# ===========================================================================

RACCOLTA_STRATEGICA = """

1️⃣  CAMPIONI BASEATI SU DIFFICOLTÀ
===============================

Allocazione campioni:
- FORBICE:    100-120 campioni   ⭐       (molto distinta)
- SASSO:      140-160 campioni   ⭐⭐     (confusa vs Lizard)
- CARTA:      160-180 campioni   ⭐⭐⭐   (confusa vs Spock)
- LIZARD:     180-200 campioni   ⭐⭐⭐   (molto confusa: vs Sasso, vs Spock)
- SPOCK:      160-180 campioni   ⭐⭐⭐   (confusa vs Carta, vs Lizard)

TOTALE: 740-840 campioni


2️⃣  VARIAZIONI ANGOLARI (per ogni gesto)
==========================================

Raccogliere da ALMENO 5 angoli:
├── FRONTALE (0°)           → 25% campioni
├── DESTRA (45°)            → 20% campioni
├── SINISTRA (-45°)         → 20% campioni
├── DALL'ALTO (70°)         → 20% campioni
└── DAL BASSO (-70°)        → 15% campioni

Perché? Il classificatore vede il gesto da angoli diversi
e i landmark cambiano significativamente!


3️⃣  VARIAZIONI DI DISTANZA
============================

Raccogliere con mano a distanze diverse:
├── VICINA (30cm)     → 30% campioni  (dettagli chiari)
├── MEDIA (60cm)      → 40% campioni  (distanza normale)
└── LONTANA (100cm)   → 30% campioni  (robustezza)


4️⃣  VARIAZIONI DI ILLUMINAZIONE
==================================

Per ogni gesto, raccogliere in:
├── Luce naturale fronte        ✓
├── Luce naturale da lato       ✓
├── Contro-luce                 ✓
├── Luce artificiale warm       ✓
└── Luce artificiale fredda     ✓


5️⃣  POSE E POSTURA
====================

Variare anche:
├── Mano davanti (normale)
├── Mano inclinata 15-30°
├── Mano girata (pronazione/supinazione)
├── Braccia a diverse altezze
└── Movimento lento vs statico

"""

# ===========================================================================
# LAYER 3: CONTROLLO QUALITÀ DURANTE LA RACCOLTA
# ===========================================================================

QUALITY_CONTROL = """

✓ DURANTE LA RACCOLTA (smart_collector.py fornisce):

1. REAL-TIME FEEDBACK
   - Quando premi SPACE, il sistema classifica
   - Se il gesto riconosciuto ≠ target: segnala con ARANCIONE
   - Se il gesto riconosciuto = target: conferma con VERDE
   
2. PROGRESS INDICATOR
   - Barra di progresso visuale
   - Conteggio campioni attuali
   - Allocazione per angolo

3. CONFUSION DETECTION
   - Se rileva confusione frequente tra due gesti:
   - Suggerisce angoli alternativi
   - Chiede di enfatizzare le differenze


✓ POST-RACCOLTA (diagnostics.py fornisce):

1. CONFUSION MATRIX VISUALIZATION
   - Matrice che mostra chi è confuso con chi
   - Percentuali di confusione
   - Heatmap colorata (rosso = confusione)

2. PER-GESTURE METRICS
   - Precision, Recall, F1-score per ogni gesto
   - Identifica gesti deboli
   - Suggerisce quali ricampionare

3. FEATURE IMPORTANCE
   - Quali coordinate (x,y,z) sono più importanti?
   - Es: per SPOCK è importante la "separazione" dita
   - Per SASSO è importante la "chiusura" pugno

"""

# ===========================================================================
# LAYER 4: DECISIONI DI TRAINING
# ===========================================================================

TRAINING_DECISIONS = """

1️⃣  DATA AUGMENTATION AGGRESSIVA
================================

Per gesti confusabili, applicare:
- Noise gaussiano: std = 0.03 (invece di 0.02)
- Rotazioni: ±15° (invece di ±10°)
- Scaling: ±15% (invece di ±10%)
- Brightness/contrast augmentation sui frame

Generare 3-5 augmentation per campione (non 2)


2️⃣  HYPERPARAMETER TUNING
==========================

Per LightGBM:
- Aumentare max_depth a 10-12 (dai 8 default)
- Aumentare n_estimators a 500 (dai 300)
- Ridurre learning_rate a 0.03 (dai 0.05)

Perché? Permettere al modello di catturare
le sottili differenze tra gesti simili


3️⃣  CLASS WEIGHTS (OPZIONALE)
==============================

Se LIZARD e SPOCK rimangono confusi:
- Aumentare peso per classe meno rappresentata
- LightGBM supporta scale_pos_weight

Esempio:
lgb.LGBMClassifier(
    class_weight='balanced',  # Bilancia automaticamente
    ...
)


4️⃣  ENSEMBLE APPROACH
======================

Se singolo modello non funziona bene:
- Trainare N modelli con bootstrap diversi
- Votazione maggioritaria per predizione
- Aumenta robustezza

"""

# ===========================================================================
# LAYER 5: WORKFLOW PASSO PER PASSO
# ===========================================================================

COMPLETE_WORKFLOW = """

GIORNO 1: RACCOLTA DATI
=======================
1. Leggi le istruzioni per OGNI gesto (in smart_collector.py)
2. Raccogli 200 campioni per SASSO:
   $ python smart_collector.py
   → Seleziona "Rock"
   → Osserva feedback real-time
   → Varia angoli/distanze come suggerito

3. Ripeti per tutti 5 i gesti
   - FORBICE: 120 campioni
   - SASSO: 200 campioni
   - CARTA: 180 campioni
   - LIZARD: 200 campioni
   - SPOCK: 180 campioni
   
4. Salva: training_data_balanced.npz


GIORNO 2: TRAINING E DIAGNOSTICA
==================================
1. Allena il classificatore:
   $ python train.py --train-classifier --data-file training_data_balanced.npz

2. Analizza i risultati:
   $ python diagnostics.py --data data/training_data_balanced.npz

3. Leggi il report. Domande da porsi:
   - Ci sono gesti con accuracy < 85%?
   - Quali coppie di gesti si confondono?
   - Quali coordinate (x,y,z) sono importanti?

4. Se risultati BUONI (accuracy > 90%):
   → Salta al giorno 4

5. Se risultati SCARSI:
   → Vai al giorno 3


GIORNO 3: MIGLIORAMENTI MIRATI
===============================
1. Identifica i gesti/coppie problematiche dal report

2. Raccogli campioni AGGIUNTIVI FOCALIZZATI:
   $ python smart_collector.py
   → Raccogli 100 campioni extra per SPOCK
   → Enfatizza: dita ben separate, tutte visibili

3. Aggiungi ai dati precedenti:
   ```python
   import numpy as np
   old = np.load('training_data_balanced.npz')
   new = np.load('spock_extra.npz')
   X = np.vstack([old['X'], new['X']])
   y = np.concatenate([old['y'], new['y']])
   np.savez('training_data_improved.npz', X=X, y=y)
   ```

4. Riallena:
   $ python train.py --train-classifier --data-file training_data_improved.npz

5. Rivaluta:
   $ python diagnostics.py --data data/training_data_improved.npz


GIORNO 4: LSTM E TESTING
========================
1. Allena LSTM su sequenze:
   $ python train.py --train-lstm

2. Prova il gioco:
   $ python game_ui.py

3. Gioca 50-100 round e verifica:
   - L'IA impara le tue abitudini?
   - Il riconoscimento dei gesti è accurato?
   - Ci sono lag o problemi di performance?

"""

# ===========================================================================
# LAYER 6: TROUBLESHOOTING SPECIFICO
# ===========================================================================

SPECIFIC_FIXES = """

PROBLEMA: CARTA vs SPOCK confusi
==================================
CAUSA: Le dita sembrano separate in entrambi
SOLUZIONE:
1. In CARTA: unisci FORZA mente le dita
2. In SPOCK: apri la mano il più possibile, separa CHIARAMENTE
3. Raccogli 100 campioni extra di CARTA con dita ESPLICITAMENTE unite
4. Raccogli 100 campioni extra di SPOCK con massima separazione
5. Riallena con augmentation più aggressiva


PROBLEMA: SASSO vs LIZARD confusi
===================================
CAUSA: Difficile distinguere pollice dentro/fuori
SOLUZIONE:
1. In SASSO: chiudi il pugno COMPLETAMENTE, pollice COPERTO dalle altre dita
2. In LIZARD: mostra il pollice CHIARAMENTE separato
3. Raccogli in luce migliore (oppure sasso sembra dark, lizard light)
4. Aumenta max_depth di LightGBM a 12


PROBLEMA: LIZARD vs SPOCK confusi
===================================
CAUSA: Entrambi hanno dita aperte
SOLUZIONE:
1. In LIZARD: tieni 2 dita chiuse (ring + pinky)
2. In SPOCK: apri TUTTE le dita, nessuna chiusa
3. Augmentation: focus su "numero dita aperte"
4. Add feature engineering: "conteggio dita aperte"


PROBLEMA: Bassa accuracy complessiva
=====================================
1. Aumenta campioni a 300+ per gesto
2. Migliora illuminazione (fonte di luce principale)
3. Varia di più angoli/distanze
4. Usa data augmentation più aggressiva
5. Aumenta LightGBM estimators a 500-700

"""

# ===========================================================================
# METRICHE TARGET
# ===========================================================================

TARGET_METRICS = """

✅ OBIETTIVO FINALE:

Per ogni gesto:
- Accuratezza: > 92%
- Precision: > 90%
- Recall: > 90%
- F1-score: > 90%

Coppie confusabili:
- CARTA ↔ SPOCK: < 5% confusione
- SASSO ↔ LIZARD: < 5% confusione
- LIZARD ↔ SPOCK: < 8% confusione

Accuratezza globale: > 90%

Se raggiungi questi target → pronto per deployment!

"""

print(__doc__)
