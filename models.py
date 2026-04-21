"""
models.py - Lightweight version for Render free tier
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("[Models] Lightweight mode active")


class ResumeMatcher:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.vectorizer = TfidfVectorizer(
                stop_words='english', ngram_range=(1, 2), max_features=5000
            )
        return cls._instance

    def match(self, resume_text: str, job_description: str) -> float:
        if not resume_text.strip() or not job_description.strip():
            return 50.0
        try:
            tfidf = self.vectorizer.fit_transform([resume_text, job_description])
            sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
            return round(float(min(96.0, max(30.0, sim * 100 * 1.8))), 1)
        except Exception as e:
            print(f"[ResumeMatcher] Error: {e}")
            return 55.0


class FaceAnalyzer:
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        import cv2
        self._cv2 = cv2
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_face(self, frame):
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def detect_emotion(self, frame) -> str:
        face = self.detect_face(frame)
        if face is None:
            return "neutral"
        x, y, w, h = face
        crop = frame[y:y+h, x:x+w]
        gray = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        if brightness > 145 and contrast > 40:
            return "happy"
        elif brightness < 80:
            return "sad"
        elif contrast > 70:
            return "surprise"
        else:
            return "neutral"

    def gaze_deviation(self, frame, threshold_ratio=0.28) -> bool:
        face = self.detect_face(frame)
        if face is None:
            return False
        x, y, w, h = face
        face_cx = x + w // 2
        frame_cx = frame.shape[1] // 2
        return abs(face_cx - frame_cx) > frame.shape[1] * threshold_ratio


class SpeechAnalyzer:
    def analyze(self, audio_path: str) -> str:
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, duration=60, mono=True)
            energy = float(np.mean(librosa.feature.rms(y=y)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(np.atleast_1d(tempo_arr)[0])
            if energy > 0.05 and tempo > 90:
                return "positive"
            elif energy < 0.01 or tempo < 50:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            print(f"[SpeechAnalyzer] Error: {e}")
            return "neutral"


_EMOTION_WEIGHT = {
    "happy": 1.0, "surprise": 0.8, "neutral": 0.65,
    "sad": 0.4, "fear": 0.3, "angry": 0.2, "disgust": 0.1
}
_SPEECH_WEIGHT = {"positive": 1.0, "neutral": 0.55, "negative": 0.15}


def fuse_scores(resume_score, emotions, speech_sentiment, fraud_flag, fraud_msg=""):
    if fraud_flag:
        return 35.0, "Not Suitable", (
            f"Fraud indicators detected — {fraud_msg}. Candidate flagged for manual review."
        ), {"resume": resume_score, "emotion": 0, "speech": 0}

    raw_emotion = np.mean([_EMOTION_WEIGHT.get(e, 0.5) for e in emotions]) if emotions else 0.5
    emotion_score = float(np.clip(raw_emotion * 100, 20, 100))
    speech_score = _SPEECH_WEIGHT.get(speech_sentiment, 0.55) * 100

    final = 0.50 * resume_score + 0.25 * emotion_score + 0.25 * speech_score
    final = round(float(np.clip(final, 30, 97)), 1)

    breakdown = {
        "resume": round(resume_score, 1),
        "emotion": round(emotion_score, 1),
        "speech": round(speech_score, 1),
    }

    if final >= 82:
        suitability = "Highly Suitable"
        comment = ("Exceptional candidate — strong technical alignment, confident communication, "
                   "and consistently positive engagement throughout the interview.")
    elif final >= 68:
        suitability = "Suitable"
        comment = ("Strong candidate — good technical match with solid communication skills. "
                   "Minor areas for improvement in non-verbal confidence.")
    elif final >= 52:
        suitability = "Borderline"
        comment = ("Average candidate — technical skills are present but communication confidence "
                   "and engagement need improvement.")
    else:
        suitability = "Not Suitable"
        comment = ("Below-average candidate — significant gaps in technical alignment "
                   "or communication effectiveness.")

    return final, suitability, comment, breakdown
