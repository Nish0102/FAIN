# FAIN
> A real-time AI pipeline that sees who you are, feels what you're feeling, and hears what you're saying — simultaneously, with no wake word.
 
Inspired by the **TWEC stack** (Identity Engine + Emotion Engine + Voice Pipeline layers).
 
---
 
## What it does
 
| Capability | How |
|------------|-----|
| Recognizes your face | ResNet18 embeddings + cosine similarity |
| Detects your emotion | Fine-tuned EmotionCNN (7 emotions) |
| Listens without wake word | Voice Activity Detection (VAD) via RMS energy |
| Verifies your voice | Random Forest on 222 audio features |
| Understands what you said | Whisper ASR + intent classification |
| Responds in real time | Claude API or keyword fallback |
 
---
 
## Project Structure
 
```
PRAC/
├── data/
│   └── raw/
│       ├── myvoice/          # your voice samples (.wav)
│       └── others/           # other speakers (.wav)
├── faces/
│   ├── dark.npy              # registered face embeddings
│   └── nishanth.npy
├── fer2013/
│   ├── train/                # 7 emotion classes
│   └── test/
├── models/
│   └── speaker_model.pkl     # trained voice identity model
├── src/
│   ├── audio/
│   │   └── recorder.py       # mic recording
│   ├── features/
│   │   └── extract_features.py  # MFCC, chroma, mel, ZCR, RMS
│   ├── intent/
│   │   └── detect_intent.py  # Claude API + keyword fallback
│   ├── speaker/
│   │   ├── predict_speaker.py
│   │   └── train_speaker.py
│   └── speech/
│       └── speech_to_text.py # Whisper transcription
├── emotion_cnn.pth            # trained emotion model weights
├── emotions.py                # training + real-time emotion pipeline
├── register.py                # face registration script
└── integrated_pipeline.py    # full pipeline (face + voice together)
```
 
---
 
## Setup
 
### 1. Clone and create virtual environment
 
```bash
git clone https://github.com/yourusername/home-ai-pipeline.git
cd home-ai-pipeline
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```
 
### 2. Install dependencies
 
```bash
pip install torch torchvision
pip install mediapipe opencv-python
pip install openai-whisper sounddevice scipy
pip install librosa joblib scikit-learn
pip install Pillow numpy matplotlib
```
 
### 3. Download FER2013 dataset
 
Download from [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013) and place it as:
 
```
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── (same structure)
```
 
---
 
## Usage
 
### Step 1 — Train the emotion model
 
```bash
python emotions.py
```
 
Trains for 30 epochs, saves `emotion_cnn.pth`. Expected accuracy: ~55-60%.
 
### Step 2 — Register your face
 
```bash
python register.py
```
 
Enter your name when prompted. Move your head slightly in different angles. Saves `faces/yourname.npy`.
 
### Step 3 — Record your voice samples
 
```python
# run 15 times, changing filename each time
python -c "from src.audio.recorder import record_audio; record_audio('data/raw/myvoice/sample1.wav', duration=3)"
```
 
Say things like:
- *"I'm feeling really stressed today"*
- *"Play some music please"*
- *"I'm exhausted and need rest"*
- *"I feel okay today"*
Record 10 samples for `data/raw/others/` with a different voice.
 
### Step 4 — Train the speaker model
 
```bash
python src/speaker/train_speaker.py
```
 
Expected validation accuracy: ~85%+ with 15+ samples.
 
### Step 5 — Run the full pipeline
 
```bash
python integrated_pipeline.py
```
 
---
 
## How it works
 
```
Camera Thread                    Voice Thread (background)
─────────────────                ────────────────────────
MediaPipe detects face           VAD listens continuously
ResNet18 extracts embedding      Speech detected → save wav
Cosine similarity → who are you  Random Forest → is it you?
EmotionCNN → what emotion?       Whisper → what did you say?
Draw on screen                   Claude API → what do you mean?
         ↘                                ↙
           Shared state (thread-safe lock)
                      ↓
           Overlay response on camera feed
```
 
---
 
## Supported Intents
 
| Intent | Example phrase |
|--------|---------------|
| `need_support` | "I'm feeling stressed and overwhelmed" |
| `need_relax` | "I feel okay, want to chill" |
| `need_rest` | "I'm exhausted and sleepy" |
| `play_music` | "Play some music please" |
| `unknown` | anything else |
 
---
 
## Model Architecture
 
### EmotionCNN
 
```
ResNet18 (pretrained ImageNet)
    ├── Freeze: layer1, layer2        # already know edges/textures
    ├── Unfreeze: layer3, layer4, fc  # fine-tune for faces
    └── Custom heads:
        ├── head_emotion   → 7 classes (angry/disgust/fear/happy/neutral/sad/surprise)
        ├── head_intensity → 1 value  (0.0 to 1.0)
        └── head_valence   → 2 classes (positive/negative)
```
 
### Speaker Model
 
```
Audio (3s wav)
    └── Feature extraction (222 features)
        ├── MFCC mean + std    (80)
        ├── Chroma mean        (12)
        ├── Mel spectrogram    (128)
        ├── Zero Crossing Rate (1)
        └── RMS Energy         (1)
            └── Random Forest Classifier → you / not you
```
 
## Known Limitations
 
- FER2013 `disgust` class only has 547 images — model underperforms on this emotion
- Speaker model needs 15+ voice samples for reliable verification
- PyTorch used instead of TF Lite/ONNX — not optimized for edge deployment
- Fixed confidence threshold (0.75) may need tuning per environment
---
 
## Tech Stack
 
| Category | Technology |
|----------|-----------|
| Deep Learning | PyTorch, ResNet18 |
| Computer Vision | OpenCV, MediaPipe |
| Speech Recognition | OpenAI Whisper |
| Audio Processing | librosa, sounddevice |
| Speaker Verification | scikit-learn Random Forest |
| Intent Detection | Anthropic Claude API |
| Dataset | FER2013 (35,000 images, 7 emotions) |
 
---
 
