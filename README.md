# Posture Alarm

A "vibe coded" computer vision tool designed to detect head posture (looking up/down) even when the user's face is covered by surgical bandages.

**Status:** *100% Vibecoded (with failures)*

## The Problem (Why this exists)
I recently underwent surgery and needed a way to strictly monitor my head posture during recovery while using laptop.
* **The Constraint:** My face was covered in bandages.
* **The Failure:** Standard AI models (MediaPipe Pose) completely failed. They rely on seeing a clear nose/mouth, so they either couldn't find my face or hallucinated landmarks on my chin/forehead.
* **The Goal:** A system that ignores the messy, bandaged center of the face and tracks **only the eyes** to enforce safe posture.

## 🛠️ The Solution
We abandoned the standard body trackers and switched to the **MediaPipe Face Mesh (478 points)** model with custom logic.
* **Robust Tracking:** Uses high-density mesh to find eyes even when the rest of the face is obscured.
* **Audio Feedback:** Emits a beep to signal "Look Up" or "Look Down" so I can rectify my posture immediately.
* **1s Delay:** Tuned to a 1-second interval to prevent audio spam while adjusting.

---

## 📉 The "Vibe Coding" Journey (A History of Failures)
This project was built through a series of failures with vibecoding .

 **The "Ghost Nose" Bug:**
    * *Issue:* The Pose model couldn't handle the medical dressing. It kept losing the face entirely.
    * *Fix:* Switched to **Face Mesh** (478 landmarks) which is far more aggressive at finding facial features.

---

## ⚡ Installation & Usage

### 1. Requirements
```bash
pip install opencv-python mediapipe sounddevice numpy
```
*Linux users:* `sudo apt-get install libportaudio2` (for sound).

### 2. Download Model
```bash
wget -O face_landmarker.task -q [https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
```

### 3. Run It
```bash
python3 posture_app.py
```
* **Sit Straight:** Look at the screen comfortably.
* **Press 'C':** This calibrates the "Good" state.
* **Audio Signal:** If you deviate, it waits **1000ms** before beeping, giving you a second to self-correct.

## 🧠 Configuration
The delay is set to **1 second** to avoid annoyance:

```python
# --- CONFIGURATION ---
BEEP_INTERVAL = 1.0  # 1000 ms delay between beeps
```
