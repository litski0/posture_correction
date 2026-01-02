import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import sounddevice as sd
import time

# --- CONFIGURATION ---
MODEL_PATH = 'face_landmarker.task' 
BEEP_INTERVAL = 1  # 1 s delay between beeps

# --- SOUND SETUP ---
# We generate a 100ms sine wave tone (1000 Hz)
sample_rate = 44100
duration = 1 # seconds
frequency = 1000.0  # Hz
t = np.linspace(0, duration, int(sample_rate * duration), False)
# Generate audio data (numpy array)
beep_sound = 0.5 * np.sin(2 * np.pi * frequency * t)

def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        min_tracking_confidence=0.2
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    
    # STATE VARIABLES
    calibrated = False
    ref_nose_y = 0
    ref_eye_y = 0
    last_sound_time = 0 # Timer for the sound delay
    
    print("--- POSTURE ALARM ENABLED ---")
    print("1. Look straight and press 'c' to calibrate.")
    print("2. If you look UP or DOWN, it will beep every 100ms.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = detector.detect(mp_image)
        output_img = frame.copy()
        h, w, _ = output_img.shape

        if results.face_landmarks:
            lm = results.face_landmarks[0]
            
            # --- KEY LANDMARKS ---
            nose = np.array([lm[1].x * w, lm[1].y * h])
            l_iris = np.array([lm[468].x * w, lm[468].y * h])
            r_iris = np.array([lm[473].x * w, lm[473].y * h])
            eye_y_avg = (l_iris[1] + r_iris[1]) / 2
            
            # --- VISUALIZATION ---
            cv2.circle(output_img, tuple(nose.astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(output_img, tuple(l_iris.astype(int)), 3, (0, 255, 255), -1)
            cv2.circle(output_img, tuple(r_iris.astype(int)), 3, (0, 255, 255), -1)

            # --- LOGIC ---
            if not calibrated:
                cv2.putText(output_img, "Press 'C' to Calibrate", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                current_dist = abs(eye_y_avg - nose[1])
                target_dist = abs(ref_eye_y - ref_nose_y)
                diff = target_dist - current_dist
                
                # Logic (Swapped as per previous request)
                if diff > (target_dist * 0.50):
                    status = "LOOKING UP"
                    color = (0, 0, 255)
                    trigger_sound = True
                elif diff < -(target_dist * 0.60):
                    status = "LOOKING DOWN"
                    color = (0, 0, 255)
                    trigger_sound = True
                else:
                    status = "GOOD"
                    color = (0, 255, 0)
                    trigger_sound = False
                
                cv2.putText(output_img, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                # --- SOUND TRIGGER (With 100ms Delay/Interval) ---
                current_time = time.time()
                if trigger_sound and (current_time - last_sound_time > BEEP_INTERVAL):
                    # Play sound asynchronously (doesn't pause the video)
                    sd.play(beep_sound, sample_rate)
                    last_sound_time = current_time

        cv2.imshow('Posture Alarm', output_img)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'): break
        elif key == ord('c') and results.face_landmarks:
            lm = results.face_landmarks[0]
            ref_nose_y = lm[1].y * h
            l_iris_y = lm[468].y * h
            r_iris_y = lm[473].y * h
            ref_eye_y = (l_iris_y + r_iris_y) / 2
            calibrated = True
            print("Calibrated.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()