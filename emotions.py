# ─────────────────────────────────────────────────────────
# LIBRARIES
# ─────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch import optim
from PIL import Image
from collections import Counter

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

DATA_DIR   = "./fer2013"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS   = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print(f"Training on: {DEVICE}")


# ─────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING + TRANSFORMS
# ─────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(DATA_DIR + "/train", transform=train_transforms)
test_data  = datasets.ImageFolder(DATA_DIR + "/test",  transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {train_data.classes}")
print(f"Train: {len(train_data)} | Test: {len(test_data)}")


# ─────────────────────────────────────────────────────────
# STEP 2 — MODEL DEFINITION
# ─────────────────────────────────────────────────────────

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.base = models.resnet18(pretrained=True)

        # freeze only early layers
        # unfreeze layer3, layer4, fc so they adapt to FER2013
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
        self.head_intensity = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.head_valence   = nn.Linear(512, 2)

    def forward(self, x):
        x = self.base(x)
        return self.head_emotion(x), \
               self.head_intensity(x), \
               self.head_valence(x)


model = EmotionCNN(num_classes=len(train_data.classes)).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────────────────────
# STEP 3 — TRAINING
# ─────────────────────────────────────────────────────────

# class weights — penalizes model more for getting
# rare emotions wrong (fixes happy dominance)
label_counts = Counter(train_data.targets)
total        = len(train_data)
weights      = [total / label_counts[i] for i in range(len(EMOTIONS))]
weights      = torch.tensor(weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

train_losses, train_accs, test_accs = [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        emotion_out, intensity_out, valence_out = model(images)
        loss = criterion(emotion_out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = emotion_out.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total * 100
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            emotion_out, _, _ = model(images)
            preds    = emotion_out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    test_acc = correct / total * 100
    test_accs.append(test_acc)
    scheduler.step()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Train: {train_acc:.1f}% | "
          f"Test: {test_acc:.1f}%")

torch.save(model.state_dict(), "emotion_cnn.pth")
print("Model saved → emotion_cnn.pth")


# ─────────────────────────────────────────────────────────
# STEP 4 — PLOT TRAINING RESULTS
# ─────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses)
ax1.set_title("Loss over epochs")
ax1.set_xlabel("Epoch")
ax2.plot(train_accs, label="Train")
ax2.plot(test_accs,  label="Test")
ax2.set_title("Accuracy over epochs")
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.savefig("training_results.png")
plt.show()


# ─────────────────────────────────────────────────────────
# STEP 5 — REAL TIME: FACE ID + EMOTION
# ─────────────────────────────────────────────────────────

# ── load emotion model ────────────────────────────────────
model.load_state_dict(torch.load("emotion_cnn.pth"))
model.eval()

# ── load face embedder ────────────────────────────────────
embedder    = models.resnet18(pretrained=True)
embedder.fc = torch.nn.Identity()
embedder    = embedder.to(DEVICE)
embedder.eval()

# ── transforms ────────────────────────────────────────────
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

# ── function definitions ──────────────────────────────────
def load_all_faces(folder="faces"):
    known_faces = {}
    if not os.path.exists(folder):
        print("No faces folder found — run register.py first")
        return known_faces
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            person_name = file.replace(".npy", "")
            embeddings  = np.load(f"{folder}/{file}")
            known_faces[person_name] = torch.tensor(
                embeddings, dtype=torch.float32
            ).to(DEVICE)
            print(f"Loaded: {person_name} ({len(embeddings)} embeddings)")
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


def get_embedding(face_img):
    pil    = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = embed_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = embedder(tensor)
    return emb.squeeze()


# ── load all registered faces ─────────────────────────────
known_faces = load_all_faces()

# ── webcam loop ───────────────────────────────────────────
mp_face  = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.7)
cap      = cv2.VideoCapture(0)

print("Webcam running — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

            # ── who is this? ──────────────────────────────
            live_emb         = get_embedding(face_crop)
            name, similarity = identify_face(live_emb, known_faces)
            color            = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # ── what emotion? ─────────────────────────────
            pil_img = Image.fromarray(
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            )
            tensor  = inference_transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emotion_out, _, _ = model(tensor)
                class_idx         = emotion_out.argmax(dim=1).item()
                probs             = torch.softmax(emotion_out, dim=1)
                confidence        = probs.max().item()
                # intensity derived from confidence
                # high confidence = strong emotion
                intensity         = round(confidence, 2)

            emotion = EMOTIONS[class_idx]

            # ── draw on screen ────────────────────────────
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(frame, f"{name} ({similarity})",
                       (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"{emotion} {confidence*100:.0f}% | intensity: {intensity}",
                       (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            print({
                "name":       name,
                "similarity": similarity,
                "emotion":    emotion,
                "confidence": round(confidence * 100, 1),
                "intensity":  intensity
            })

    cv2.imshow("Home AI — Identity + Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
