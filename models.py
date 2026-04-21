"""
models.py - Lightweight version for Render free tier (512MB RAM)
Uses scikit-learn + lightweight NLP instead of heavy PyTorch models
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("[Models] Lightweight mode — optimised for free tier")

# ── 1. Resume Matcher (TF-IDF instead of BERT) ──────────────────────────
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
        corpus = [resume_text, job_description]
        try:
            tfidf = self.vectorizer.fit_transform(corpus)
            sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
            score = float(min(96.0, max(30.0, sim * 100 * 1.8)))
            return round(score, 1)
        except Exception as e:
            print(f"[ResumeMatcher] Error: {e}")
            return 55.0


# ── 2. Face Analyzer (OpenCV only, no PyTorch) ──────────────────────────
class FaceAnalyzer:
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        import cv2
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._cv2 = cv2

    def detect_face(self, frame):
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def detect_emotion(self, frame) -> str:
        """Rule-based brightness/contrast heuristic — no model needed."""
        face = self.detect_face(frame)
        if face is None:
            return "neutral"
        x, y, w, h = face
        crop = frame[y:y+h, x:x+w]
        gray = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast   = float(np.std(gray))
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
        x, y,
