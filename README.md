# 🧠 Mind Games - Rock Paper Scissors Lizard Spock vs AI

Un sistema interattivo che combina **Computer Vision**, **Machine Learning** e **Game Theory** per un duello uomo-macchina intelligente.

## 📋 Requisiti

- **Python 3.9+**
- **Webcam** (per acquisizione video real-time)
- **CUDA** (opzionale, per accelerare TensorFlow/LightGBM)

## 🚀 Setup

### 1. Clona il repository
```bash
git clone <repo-url>
cd mind_games
```

### 2. Crea un ambiente virtuale
```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate
```

### 3. Installa dipendenze
```bash
pip install -r requirements.txt
```

## 📚 Architettura

```
mind_games/
├── config.py                 # Configurazione centralizzata
├── hand_detector.py          # MediaPipe hand tracking
├── gesture_classifier.py     # LightGBM classifier
├── sequence_predictor.py     # LSTM sequence prediction
├── game_engine.py            # Game logic (RPSLS rules)
├── game_ui.py                # Pygame interface
├── train.py                  # Data collection & training
└── models/                   # Pre-trained models (creato automaticamente)
    ├── gesture_classifier.pkl
    ├── gesture_scaler.pkl
    └── sequence_predictor.h5
```

## 🎮 Workflow

### Fase 1: Raccolta Dati e Training

```bash
python train.py --train-all --samples-per-gesture 100
```

Questo comando:
1. **Apre la webcam** e raccoglie 100 campioni per ogni gesto
   - Mostra il nome del gesto in tempo reale
   - Premi SPACE per catturare, 'N' per saltare
   
2. **Aumenta i dati** con noise, rotazioni e scaling

3. **Allena LightGBM** con cross-validation
   - Accuratezza tipica: >95%
   
4. **Allena LSTM** su sequenze sintetiche
   - Predice il prossimo gesto del giocatore

**Comandi alternativi:**
```bash
python train.py --collect --samples-per-gesture 150     # Solo raccolta
python train.py --train-classifier                        # Solo classifier
python train.py --train-lstm                              # Solo LSTM
```

### Fase 2: Giocare

```bash
python game_ui.py
```

**Interfaccia di gioco:**
- **Sinistra (60%)**: Feed webcam con scheletro della mano
- **Destra (40%)**: Statistiche, punteggio, cronologia mosse

**Controlli:**
- Mostra un gesto davanti alla webcam
- L'IA automaticamente rileva e risponde
- Countdown 3-2-1... SHOOT!
- **R**: Reset gioco
- **Q**: Esci

## 🔧 Configurazione

Modifica `config.py` per personalizzare:

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `MP_HANDS_MIN_CONFIDENCE` | Confidenza detection mano | 0.7 |
| `LGBM_N_ESTIMATORS` | Alberi LightGBM | 300 |
| `LSTM_SEQUENCE_LENGTH` | Finestra temporale LSTM | 10 |
| `COUNTDOWN_DURATION` | Secondi countdown | 3 |
| `GESTURE_MODEL_PATH` | Path classifier | `models/gesture_classifier.pkl` |

## 📊 Componenti Principali

### HandDetector (MediaPipe)
- 21 landmark 3D per mano
- Estrae features normalizzate (63 valori)
- Smooth temporale per ridurre tremori

```python
detector = HandDetector()
_, landmarks, detected = detector.detect(frame)
features = detector.landmarks_to_features(landmarks)
```

### GestureClassifier (LightGBM)
- Classifica gesti statici: Rock, Paper, Scissors, Lizard, Spock
- Data augmentation integrata
- Accuratezza validazione tramite cross-fold

```python
classifier = GestureClassifier()
classifier.train(X, y)  # Train
gesture, confidence = classifier.predict(features)
classifier.save()
```

### SequencePredictor (LSTM)
- Analizza sequenza ultime N mosse
- Predice probabilità prossima mossa
- Alimenta strategia IA "contromossa"

```python
predictor = SequencePredictor()
predictor.train(gesture_sequences)
proba, predicted_gesture = predictor.predict_next_move()
predictor.record_move(gesture)
```

### AIOpponent
- Sceglie mossa che batte previsione LSTM
- Strategie: 'counter', 'random', 'balanced'

```python
ai = AIOpponent(predictor)
ai.set_strategy('counter')
ai_move = ai.choose_move(use_prediction=True)
```

### GameEngine (RPSLS Logic)
- Implementa regole Rock-Paper-Scissors-Lizard-Spock
- Traccia cronologia e statistiche
- Spiega chi vince e perché

```python
engine = GameEngine()
result = engine.play_round(player_gesture, ai_gesture)
stats = engine.get_stats()
```

## 🎯 Regole RPSLS

| vs | Sasso | Carta | Forbice | Lizard | Spock |
|----|-------|-------|---------|--------|-------|
| **Sasso** | = | L | W | W | L |
| **Carta** | W | = | L | L | W |
| **Forbice** | L | W | = | W | L |
| **Lizard** | L | W | L | = | W |
| **Spock** | W | L | W | L | = |

## 📈 Metriche di Performance

Dopo training, controlla:

```python
from gesture_classifier import GestureClassifier

clf = GestureClassifier()
clf.load()

# Feature importance
importances = clf.get_feature_importance(top_n=15)
for idx, imp in importances:
    print(f"Feature {idx}: {imp:.4f}")
```

## 🐛 Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| **Webcam non viene riconosciuta** | Controlla permessi, prova `cv2.VideoCapture(1)` |
| **Bassa accuratezza gesture** | Raccogli più campioni (300+), migliora illuminazione |
| **LSTM non predice bene** | Aumenta `LSTM_SEQUENCE_LENGTH`, genera più sequenze sintetiche |
| **Scheletro mano sfarfalla** | Aumenta `SMOOTHING_WINDOW` in config |
| **Gioco rallentato** | Riduci `SCREEN_FPS` o processa ogni N frame |

## 📝 Struttura Training Data

Il file `training_data.npz` contiene:
- **X**: Array (n_samples, 63) - feature normalize [0,1]
- **y**: Array (n_samples,) - gesture label [0-4]

Formato compresso NumPy per efficienza.

## 🔐 Modelli Salvi

**gesture_classifier.pkl** (LightGBM)
- Dimensione: ~1-2 MB
- Caricabile con joblib

**gesture_scaler.pkl** (StandardScaler)
- Normalizzazione feature
- Essenziale per inference

**sequence_predictor.h5** (Keras/TensorFlow)
- Dimensione: ~2-5 MB
- Formato HDF5

## 🎨 Personalizzazioni

### Cambiare colori
In `config.py`:
```python
COLOR_PRIMARY = (0, 200, 255)    # Cyan
COLOR_SUCCESS = (0, 255, 100)    # Green
```

### Cambiare dimensioni UI
```python
FONT_LARGE = 48
FONT_MEDIUM = 32
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
```

### Regolare difficoltà IA
```python
ai.set_strategy('random')      # Più semplice
ai.set_strategy('balanced')    # Intermedio
ai.set_strategy('counter')     # Hardest
```

## 📚 Documentazione Codice

Ogni modulo ha docstring dettagliati:
```bash
python -c "import hand_detector; help(hand_detector.HandDetector.detect)"
```

## 🚀 Deployment Future

- [ ] Salva replay delle partite in video
- [ ] Dashboard live con grafico accuratezza IA
- [ ] Multiplayer online
- [ ] Modelli più avanzati (Transformer, Vision Transformer)
- [ ] Android/iOS app con TensorFlow Lite

## 📄 Licenza

MIT License - Libero per uso educativo e commerciale.

## 👨‍💻 Autore

Progetto per esame universitario - Computer Vision & Machine Learning.

---

**Buona fortuna a impressionare il professore! 🎓**
