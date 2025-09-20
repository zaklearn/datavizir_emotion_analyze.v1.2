import streamlit as st
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

class EmotionClassifier:
    _models = {}  # Cache statique des modèles par langue
    
    def __init__(self, lang="en"):
        """
        Initialise le modèle d'analyse des émotions avec support multilingue.
        """
        self.lang = lang
        self._load_model()
        
    @st.cache_resource
    def _load_model(self):
        """
        Charge les modèles d'analyse des émotions et les met en cache.
        """
        # Modèle pour l'anglais
        if "en" not in EmotionClassifier._models:
            try:
                EmotionClassifier._models["en"] = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None
                )
                st.success("✅ Modèle d'analyse des émotions (EN) chargé avec succès.")
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du chargement du modèle EN: {e}")
                EmotionClassifier._models["en"] = None
                
        # Modèle pour le français
        if "fr" not in EmotionClassifier._models:
            try:
                EmotionClassifier._models["fr"] = pipeline(
                    "text-classification",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    top_k=1
                )
                st.success("✅ Modèle d'analyse des émotions (FR) chargé avec succès.")
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du chargement du modèle FR: {e}")
                EmotionClassifier._models["fr"] = None
                
        # VADER comme fallback
        if "vader" not in EmotionClassifier._models:
            try:
                nltk.download("vader_lexicon", quiet=True)
                EmotionClassifier._models["vader"] = SentimentIntensityAnalyzer()
            except Exception:
                EmotionClassifier._models["vader"] = None

    def classify(self, text):
        """
        Retourne un dictionnaire avec les émotions détectées et leurs scores.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {"message": "Texte vide ou invalide", "scores": {}}
            
        model = EmotionClassifier._models.get(self.lang)
        
        # Fallback si le modèle principal n'est pas disponible
        if model is None:
            if self.lang == "en" and EmotionClassifier._models.get("vader"):
                # Fallback pour l'anglais avec VADER
                scores = EmotionClassifier._models["vader"].polarity_scores(text)
                
                emotion_map = {
                    "joy": max(0, scores["pos"] * 2 - 0.5),
                    "sadness": max(0, scores["neg"] - 0.1),
                    "anger": max(0, scores["neg"] - 0.2),
                    "fear": max(0, scores["neg"] - 0.3),
                    "surprise": max(0, scores["compound"] if scores["compound"] > 0.5 else 0),
                    "neutral": max(0, 1 - abs(scores["compound"]))
                }
                
                return {
                    "message": "Analyse par VADER (fallback)",
                    "scores": emotion_map
                }
            else:
                return {"message": "Modèle non disponible", "scores": {}}
                
        try:
            if self.lang == "fr":
                result = model(text)
                sentiment_score = int(result[0]['label'].split()[0]) / 5.0
                
                scores = {
                    "satisfaction": sentiment_score if sentiment_score > 0.5 else 0,
                    "déception": max(0, 1 - sentiment_score - 0.2) if sentiment_score < 0.5 else 0,
                    "neutre": max(0, 1 - abs((sentiment_score - 0.5) * 2)),
                    "joie": max(0, sentiment_score - 0.7) * 2 if sentiment_score > 0.7 else 0,
                    "colère": max(0, 0.3 - sentiment_score) * 2 if sentiment_score < 0.3 else 0
                }
                
                return {"message": "Analyse de sentiment (FR)", "scores": scores}
            else:
                result = model(text)
                scores = {r["label"]: r["score"] for r in result}
                return {"message": "Analyse d'émotions (EN)", "scores": scores}
                
        except Exception as e:
            return {"message": f"Erreur lors de l'analyse: {str(e)}", "scores": {}}
