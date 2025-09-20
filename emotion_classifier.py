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
        
        Parameters:
            lang (str): Langue pour l'analyse ('fr' ou 'en')
        """
        self.lang = lang
        self._load_model()
        
    @st.cache_resource
    def _load_model(_self):
        """
        Charge les modèles d'analyse des émotions et les met en cache.
        Utilise des modèles adaptés selon la langue.
        """
        # Si le modèle n'est pas déjà en cache
        if "en" not in EmotionClassifier._models:
            try:
                # Modèle pour l'anglais (plus précis)
                EmotionClassifier._models["en"] = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                st.success("✅ Modèle d'analyse des émotions (EN) chargé avec succès.")
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du chargement du modèle: {e}")
                EmotionClassifier._models["en"] = None
                
        # Modèle pour le français
        if "fr" not in EmotionClassifier._models:
            try:
                # Pour le français, on utilise un modèle multilingue
                EmotionClassifier._models["fr"] = pipeline(
                    "text-classification",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    return_all_scores=False
                )
                st.success("✅ Modèle d'analyse des émotions (FR) chargé avec succès.")
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du chargement du modèle: {e}")
                EmotionClassifier._models["fr"] = None
                
        # VADER comme fallback pour l'anglais
        if "vader" not in EmotionClassifier._models:
            try:
                nltk.download("vader_lexicon", quiet=True)
                EmotionClassifier._models["vader"] = SentimentIntensityAnalyzer()
            except Exception:
                EmotionClassifier._models["vader"] = None

    @st.cache_data
    def classify(_self, text):
        """
        Retourne un dictionnaire avec les émotions détectées et leurs scores.
        
        Parameters:
            text (str): Texte à analyser
            
        Returns:
            dict: Dictionnaire avec les émotions et scores
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {"message": "Texte vide ou invalide", "scores": {}}
            
        model = EmotionClassifier._models.get(_self.lang)
        
        # Si le modèle principal n'est pas disponible, utiliser un fallback
        if model is None:
            if _self.lang == "en" and EmotionClassifier._models.get("vader"):
                # Fallback pour l'anglais avec VADER
                scores = EmotionClassifier._models["vader"].polarity_scores(text)
                
                # Mappings simples (approximatifs) des scores VADER vers des émotions
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
                # Aucun modèle disponible
                return {"message": "Modèle non disponible", "scores": {}}
                
        try:
            # Traitement différent selon le type de modèle
            if _self.lang == "fr":
                # Le modèle français retourne une seule note de 1 à 5
                result = model(text)
                # Conversion en émotions approximatives
                sentiment_score = int(result[0]['label'].split()[0]) / 5.0  # Normaliser entre 0 et 1
                
                # Mapper le score à des émotions approximatives
                scores = {
                    "satisfaction": sentiment_score if sentiment_score > 0.5 else 0,
                    "déception": max(0, 1 - sentiment_score - 0.2) if sentiment_score < 0.5 else 0,
                    "neutre": max(0, 1 - abs((sentiment_score - 0.5) * 2)),
                    "joie": max(0, sentiment_score - 0.7) * 2 if sentiment_score > 0.7 else 0,
                    "colère": max(0, 0.3 - sentiment_score) * 2 if sentiment_score < 0.3 else 0
                }
                
                return {"message": "Analyse de sentiment (FR)", "scores": scores}
            else:
                # Le modèle anglais retourne directement des émotions
                result = model(text)
                scores = {r["label"]: r["score"] for r in result[0]}
                return {"message": "Analyse d'émotions (EN)", "scores": scores}
                
        except Exception as e:
            return {"message": f"Erreur lors de l'analyse: {str(e)}", "scores": {}}
