# 🧠 Mind Games - Rock Paper Scissors Lizard Spock vs AI

Un sistema interattivo che combina **Computer Vision**, **Machine Learning** e **Game Theory** per un duello uomo-macchina intelligente.

## 📋 Requisiti

- **Versione di Python utilizzata per i test:  3.11.x**
- **Webcam** (per acquisizione video real-time)

## 🚀 Setup

### 1. Clona il repository
```bash
git clone https://github.com/MattiaAlessi/Mind-Games---Rock-Paper-Scissors-Lizard-Spock-vs-AI
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
├── config.py # Configurazione centralizzata
├── hand_detector.py # MediaPipe hand tracking
├── gesture_classifier.py # LightGBM classifier
├── game.py # Game logic + UI OpenCV
├── train.py # Data collection & training
├── self_play.py # Addestramento AI contro se stessa
└── models/ # Modelli salvati
├── gesture_classifier.pkl
└── gesture_scaler.pkl
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


**Comandi alternativi:**
```bash
python train.py --collect --samples-per-gesture 150     # Solo raccolta
python train.py --train-classifier                        # Solo classifier
python train.py --train-lstm                              # Solo LSTM
```

### Fase 2: Giocare

```bash
python game.py
```

**Controlli:**
- Mostra un gesto davanti alla webcam
- **SPAZIO**: inizia il gioco
- Countdown 3-2-1... !
- **ESC**: Esci


## 📊 Componenti Principali

### HandDetector (MediaPipe)
- 21 landmark 3D per mano
- Estrae features normalizzate (63 valori)
- Smooth temporale per ridurre tremori

### GestureClassifier (LightGBM)
- Classifica gesti statici: Rock, Paper, Scissors, Lizard, Spock
- Data augmentation integrata
- Accuratezza validazione tramite cross-fold

### SequencePredictor (LSTM)
- Analizza sequenza ultime N mosse
- Predice probabilità prossima mossa
- Alimenta strategia IA "contromossa"

### AdaptiveAI (Markov chain ordine 2)
- Memorizza sequenze delle tue ultime mosse
- Sliding window (dimentica le abitudini vecchie)
- Laplace smoothing + esplorazione ε‑greedy
- Memoria persistente su disco (data/ai_memory.npz)

## 🎯 Regole RPSLS
L = Loss
W = Win
'=' = pareggio

| vs | Sasso | Carta | Forbice | Lizard | Spock |
|----|-------|-------|---------|--------|-------|
| **Sasso** | = | L | W | W | L |
| **Carta** | W | = | L | L | W |
| **Forbice** | L | W | = | W | L |
| **Lizard** | L | W | L | = | W |
| **Spock** | W | L | W | L | = |

```

## 🐛 Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| **Webcam non viene riconosciuta** | Controlla permessi, prova `cv2.VideoCapture(1)` |
| **Bassa accuratezza gesture** | Raccogli più campioni (300+), migliora illuminazione |
| **Scheletro mano sfarfalla** | Aumenta `SMOOTHING_WINDOW` in config |
| **Gioco rallentato** | Riduci risoluzione webcam |

## 📝 Struttura Training Data

Il file `training_data.npz` contiene:
- **X**: Array (n_samples, 63) - feature normalize [0,1]
- **y**: Array (n_samples,) - gesture label [0-4]

Formato compresso NumPy per efficienza.

## 🔐 Modelli Salvi

**gesture_classifier.pkl** (LightGBM)
- Caricabile con joblib

**gesture_scaler.pkl** (StandardScaler)
- Normalizzazione feature
- Essenziale per inference

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


MIT License - Libero per uso educativo e commerciale.

## 👨‍💻 Autore

Alessi Mattia - Velli Vinicio
