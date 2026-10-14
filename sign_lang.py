"""
Intelligent Sign Language Recognition System (prototype)
Pipeline: Webcam -> MediaPipe Hands (landmarks) -> LSTM (sequence model) -> Text + Speech
 
Usage:
    python sign_lang.py collect   # record training sequences from your webcam
    python sign_lang.py train     # train the LSTM, saves model + plots
    python sign_lang.py live      # real-time demo (text + speech)
 
Setup (Python 3.9-3.11):
    pip install mediapipe==0.10.14 tensorflow==2.15.0 "numpy<2" opencv-python scikit-learn matplotlib pyttsx3
"""
import os
import sys
import threading
from collections import deque
 
import cv2
import numpy as np
import mediapipe as mp
 
# ---------------- CONFIG ----------------
ACTIONS = np.array(["hello", "thanks", "yes", "no", "iloveyou"])  # edit/add gestures
DATA_DIR = "data"
NUM_SEQUENCES = 30      # videos per gesture
SEQ_LEN = 30            # frames per video
THRESHOLD = 0.85        # min confidence to accept a prediction
MODEL_PATH = "sign_model.h5"
SPEAK = True
 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
 
 
# ---------------- FEATURES ----------------
def extract_features(results):
    """132 features: for each hand (Left, Right) -> 21 landmarks relative to wrist (63) + wrist position (3)."""
    feats = {"Left": np.zeros(66), "Right": np.zeros(66)}
    if results.multi_hand_landmarks:
        for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hd.classification[0].label
            pts = np.array([[p.x, p.y, p.z] for p in hl.landmark])
            wrist = pts[0].copy()
            rel = (pts - wrist).flatten()
            feats[label] = np.concatenate([rel, wrist])
    return np.concatenate([feats["Left"], feats["Right"]])
 
 
def make_hands():
    return mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
 
 
def read_frame(cap, hands):
    ok, frame = cap.read()
    if not ok:
        return None, None
    frame = cv2.flip(frame, 1)  # selfie view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
    return frame, results
 
 
# ---------------- 1. COLLECT ----------------
def collect():
    cap = cv2.VideoCapture(0)
    hands = make_hands()
    for action in ACTIONS:
        os.makedirs(os.path.join(DATA_DIR, action), exist_ok=True)
        existing = len(os.listdir(os.path.join(DATA_DIR, action)))
        for seq in range(existing, NUM_SEQUENCES):
            # countdown / reset pause
            for t in range(30):
                frame, _ = read_frame(cap, hands)
                if frame is None:
                    return
                cv2.putText(frame, f"GET READY: {action}  ({seq + 1}/{NUM_SEQUENCES})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                cv2.imshow("Collect", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return
            window = []
            for _ in range(SEQ_LEN):
                frame, results = read_frame(cap, hands)
                if frame is None:
                    return
                window.append(extract_features(results))
                cv2.putText(frame, f"RECORDING: {action}  ({seq + 1}/{NUM_SEQUENCES})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Collect", frame)
                cv2.waitKey(1)
            np.save(os.path.join(DATA_DIR, action, f"{seq}.npy"), np.array(window))
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection done.")
 
 
# ---------------- 2. TRAIN ----------------
def train():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay  # type: ignore[import-not-found]
    from tensorflow.keras.models import Sequential  # type: ignore[import-not-found]
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore[import-not-found]
    from tensorflow.keras.utils import to_categorical  # type: ignore[import-not-found]
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[import-not-found]
 
    X, y = [], []
    for idx, action in enumerate(ACTIONS):
        folder = os.path.join(DATA_DIR, action)
        for f in sorted(os.listdir(folder)):
            X.append(np.load(os.path.join(folder, f)))
            y.append(idx)
    X, y = np.array(X), np.array(y)
    print("Dataset:", X.shape, y.shape)
 
    X_tr, X_te, y_tr, y_te = train_test_split(X, to_categorical(y), test_size=0.2, stratify=y, random_state=42)
 
    model = Sequential([
        LSTM(64, return_sequences=True, activation="tanh", input_shape=(SEQ_LEN, X.shape[2])),
        Dropout(0.3),
        LSTM(128, return_sequences=False, activation="tanh"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(len(ACTIONS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.summary()
    hist = model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=150, batch_size=16,
                     callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)])
    model.save(MODEL_PATH)
 
    # --- evaluation + plots for the PPT ---
    pred = np.argmax(model.predict(X_te), axis=1)
    true = np.argmax(y_te, axis=1)
    print(classification_report(true, pred, target_names=ACTIONS))
 
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["categorical_accuracy"], label="train")
    plt.plot(hist.history["val_categorical_accuracy"], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Training accuracy")
    plt.tight_layout(); plt.savefig("accuracy_curve.png", dpi=150); plt.close()
 
    cm = confusion_matrix(true, pred)
    ConfusionMatrixDisplay(cm, display_labels=ACTIONS).plot(cmap="Blues")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()
    print("Saved model + accuracy_curve.png + confusion_matrix.png")
 
 
# ---------------- 3. LIVE DEMO ----------------
def speak(text):
    if not SPEAK:
        return
    def _run():
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_run, daemon=True).start()
 
 
def live():
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    hands = make_hands()
    window = deque(maxlen=SEQ_LEN)
    recent = deque(maxlen=8)
    sentence = []
    while True:
        frame, results = read_frame(cap, hands)
        if frame is None:
            break
        window.append(extract_features(results))
        label, conf = "...", 0.0
        if len(window) == SEQ_LEN:
            probs = model.predict(np.expand_dims(np.array(window), 0), verbose=0)[0]
            k = int(np.argmax(probs))
            conf = float(probs[k])
            if conf > THRESHOLD:
                label = ACTIONS[k]
                recent.append(k)
                # accept only if stable across recent frames
                if len(recent) == recent.maxlen and len(set(recent)) == 1:
                    if not sentence or sentence[-1] != label:
                        sentence.append(label)
                        speak(label)
                        recent.clear()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (30, 30, 30), -1)
        cv2.putText(frame, f"Detected: {label} ({conf:.0%})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Text: " + " ".join(sentence[-6:]), (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence.clear()
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"collect": collect, "train": train, "live": live}.get(mode, lambda: print(__doc__))()
 

