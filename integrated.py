# ─────────────────────────────────────────────────────────
# integrated_pipeline.py
# Face Identity + Emotion + Voice + Telegram Bot
# ─────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
import queue
import time
import os
import sys
import sounddevice as sd
from scipy.io.wavfile import write
from torchvision import transforms, models
from PIL import Image
import asyncio
import telegram

# voice modules
sys.path.insert(0, "./src")
from src.speaker.predict_speaker import is_user
from src.speech.speech_to_text import transcribe_audio
from src.intent.detect_intent import detect_intent

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# telegram config — replace with your actual values
TELEGRAM_TOKEN   = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
COOLDOWN_SECONDS = 600  # 10 minutes between messages

# voice config
SAMPLE_RATE      = 16000
CHUNK_DURATION   = 0.03
CHUNK_SIZE       = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_LIMIT    = 1.5
MIN_SPEECH_SECS  = 0.5
ENERGY_THRESHOLD = 0.01
TEMP_AUDIO_PATH  = "data/live_input.wav"

# emotions that trigger a telegram alert
NEGATIVE_EMOTIONS = ["angry", "disgust", "fear", "sad"]

# intents that trigger a telegram alert
NEGATIVE_INTENTS  = ["need_support", "need_rest", "need_relax"]

RESPONSES = {
    "need_support": "That sounds tough. Want to talk or listen to something calming?",
    "need_relax":   "Let's unwind. Playing something calm...",
    "need_rest":    "You should take some rest. Want me to set a timer?",
    "play_music":   "Playing music...",
    "unknown":      "I didn't quite catch that. Could you say it again?",
}

TELEGRAM_MESSAGES = {
    "need_support": "Hey Nishanth, you seem stressed 😔 Want to talk or listen to something calming?",
    "need_relax":   "Hey Nishanth, looks like you need to unwind 😌 Take a breather!",
    "need_rest":    "Hey Nishanth, you look exhausted 😴 Maybe time for a break?",
    "angry":        "Hey Nishanth, you seem angry 😤 Take a deep breath, it'll be okay.",
    "sad":          "Hey Nishanth, you seem sad 😢 Hope you feel better soon.",
    "fear":         "Hey Nishanth, you seem anxious 😰 You've got this.",
    "disgust":      "Hey Nishanth, something bothering you? 😣 Take a moment.",
}

# ─────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────

shared = {
    "name":           "Unknown",
    "emotion":        "neutral",
    "intent":         "",
    "response":       "",
    "last_alert_time": 0,       # tracks cooldown
    "lock":           threading.Lock()
}

voice_queue = queue.Queue()

# ─────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────

def send_telegram(message: str):
    """sends a telegram message in a separate thread so it doesn't block"""
    async def _send():
        try:
            bot = telegram.Bot(token=TELEGRAM_TOKEN)
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            print(f"[Telegram] Sent: {message}")
        except Exception as e:
            print(f"[Telegram] Failed to send: {e}")

    # run async in a new event loop inside this thread
    asyncio.run(_send())


def maybe_send_alert(trigger: str):
    """
    sends telegram alert only if:
    1. trigger is negative (emotion or intent)
    2. cooldown has passed
    """
    now = time.time()

    with shared["lock"]:
        last = shared["last_alert_time"]
        if now - last < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - (now - last))
            print(f"[Telegram] Cooldown active — {remaining}s remaining")
            return
        shared["last_alert_time"] = now

    message = TELEGRAM_MESSAGES.get(trigger)
    if message:
        t = threading.Thread(target=send_telegram, args=(message,), daemon=True)
        t.start()


# ─────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base = models.resnet18(pretrained=False)
        for name, param in self.base.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.head_emotion   = nn.Linear(512, num_classes)
        self.head_intensity = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.head_valence   = nn.Linear(512, 2)

    def forward(self, x):
        x = self.base(x)
        return self.head_emotion(x), self.head_intensity(x), self.head_valence(x)


def load_models():
    emotion_model = EmotionCNN(num_classes=7).to(DEVICE)
    emotion_model.load_state_dict(torch.load("emotion_cnn.pth", map_location=DEVICE))
    emotion_model.eval()

    embedder    = models.resnet18(pretrained=True)
    embedder.fc = torch.nn.Identity()
    embedder    = embedder.to(DEVICE)
    embedder.eval()

    return emotion_model, embedder


# ─────────────────────────────────────────────────────────
# FACE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

embed_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

inference_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def load_all_faces(folder="faces"):
    known_faces = {}
    if not os.path.exists(folder):
        print("No faces folder — run register.py first")
        return known_faces
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            name       = file.replace(".npy", "")
            embeddings = np.load(f"{folder}/{file}")
            known_faces[name] = torch.tensor(
                embeddings, dtype=torch.float32
            ).to(DEVICE)
            print(f"Loaded face: {name} ({len(embeddings)} embeddings)")
    return known_faces


def identify_face(live_emb, known_faces, threshold=0.75):
    best_name  = "Unknown"
    best_score = 0.0
    for person_name, stored_embs in known_faces.items():
        sims     = F.cosine_similarity(
            live_emb.unsqueeze(0).expand(stored_embs.shape[0], -1),
            stored_embs
        )
        top_sims = sims.topk(min(5, len(sims))).values
        score    = top_sims.mean().item()
        if score > best_score:
            best_score = score
            best_name  = person_name if score > threshold else "Unknown"
    return best_name, round(best_score, 2)


def get_face_embedding(face_img, embedder):
    pil    = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = embed_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = embedder(tensor)
    return emb.squeeze()


# ─────────────────────────────────────────────────────────
# VOICE THREAD
# ─────────────────────────────────────────────────────────

def voice_pipeline():
    print("[Voice] Listening continuously — no wake word needed")

    while True:
        speech_frames  = []
        silence_frames = 0
        speaking       = False

        def callback(indata, frames, time_info, status):
            nonlocal speaking, silence_frames
            rms = np.sqrt(np.mean(indata**2))
            if rms > ENERGY_THRESHOLD:
                speaking = True
                silence_frames = 0
                speech_frames.append(indata.copy())
            elif speaking:
                silence_frames += 1
                speech_frames.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype="float32",
                            blocksize=CHUNK_SIZE,
                            callback=callback):
            while not speaking:
                time.sleep(0.01)

            max_silence = int(SILENCE_LIMIT / CHUNK_DURATION)
            while silence_frames < max_silence:
                time.sleep(0.01)

        if not speech_frames:
            continue

        total_secs = len(speech_frames) * CHUNK_DURATION
        if total_secs < MIN_SPEECH_SECS:
            print(f"[Voice] Too short ({total_secs:.1f}s) — ignoring")
            continue

        # save wav
        audio_np   = np.concatenate(speech_frames, axis=0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        os.makedirs(os.path.dirname(TEMP_AUDIO_PATH), exist_ok=True)
        write(TEMP_AUDIO_PATH, SAMPLE_RATE, audio_int16)

        # verify speaker
        print("[Voice] Verifying speaker...")
        authorized = is_user(TEMP_AUDIO_PATH)
        if not authorized:
            print("[Voice] Unknown speaker — ignoring")
            continue

        print("[Voice] Authorized speaker detected")

        # transcribe
        text = transcribe_audio(TEMP_AUDIO_PATH)
        if not text:
            print("[Voice] Could not transcribe — ignoring")
            continue

        print(f"[Voice] You said: {text}")

        # detect intent
        intent   = detect_intent(text)
        response = RESPONSES.get(intent, RESPONSES["unknown"])
        print(f"[Voice] Intent: {intent} | Response: {response}")

        # update shared state
        with shared["lock"]:
            shared["intent"]   = intent
            shared["response"] = response

        voice_queue.put({
            "text":     text,
            "intent":   intent,
            "response": response
        })

        # ── telegram alert for negative intent ────────────
        if intent in NEGATIVE_INTENTS:
            maybe_send_alert(intent)


# ─────────────────────────────────────────────────────────
# FACE THREAD
# ─────────────────────────────────────────────────────────

def face_pipeline():
    emotion_model, embedder = load_models()
    known_faces = load_all_faces()

    mp_face  = mp.solutions.face_detection
    detector = mp_face.FaceDetection(min_detection_confidence=0.7)
    cap      = cv2.VideoCapture(0)

    latest_voice    = {"text": "", "intent": "", "response": ""}
    prev_emotion    = "neutral"
    emotion_counter = 0
    # need same emotion for 30 consecutive frames before alerting
    # prevents flickering false alerts
    EMOTION_STABILITY = 30

    print("[Face] Webcam running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            latest_voice = voice_queue.get_nowait()
        except queue.Empty:
            pass

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x  = max(0, int(bbox.xmin * w))
                y  = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                face_crop = frame[y:y+bh, x:x+bw]
                if face_crop.size == 0:
                    continue

                # ── face identity ─────────────────────────
                live_emb         = get_face_embedding(face_crop, embedder)
                name, similarity = identify_face(live_emb, known_faces)
                color            = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                # ── emotion ───────────────────────────────
                pil_img = Image.fromarray(
                    cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                )
                tensor  = inference_transform(pil_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    emotion_out, _, _ = emotion_model(tensor)
                    class_idx         = emotion_out.argmax(dim=1).item()
                    confidence        = torch.softmax(emotion_out, dim=1).max().item()

                emotion = EMOTIONS[class_idx]

                # ── emotion stability check ───────────────
                # only alert if same negative emotion
                # held for EMOTION_STABILITY frames
                if emotion == prev_emotion:
                    emotion_counter += 1
                else:
                    emotion_counter = 0
                    prev_emotion    = emotion

                if (emotion_counter >= EMOTION_STABILITY
                        and emotion in NEGATIVE_EMOTIONS
                        and name != "Unknown"):
                    maybe_send_alert(emotion)
                    emotion_counter = 0  # reset after alert

                # update shared state
                with shared["lock"]:
                    shared["name"]    = name
                    shared["emotion"] = emotion

                # ── draw on screen ────────────────────────
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
                cv2.putText(frame, f"{name} ({similarity})",
                           (x, y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"{emotion} {confidence*100:.0f}%",
                           (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # ── voice overlay at bottom ───────────────────────
        if latest_voice["text"]:
            overlay_y = frame.shape[0] - 80
            cv2.rectangle(frame,
                         (0, overlay_y - 10),
                         (frame.shape[1], frame.shape[0]),
                         (0, 0, 0), -1)
            cv2.putText(frame,
                       f"You: {latest_voice['text'][:60]}",
                       (10, overlay_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame,
                       f"AI: {latest_voice['response'][:60]}",
                       (10, overlay_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1)

        cv2.imshow("Home AI — Face + Emotion + Voice", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    voice_thread = threading.Thread(target=voice_pipeline, daemon=True)
    voice_thread.start()
    face_pipeline()
