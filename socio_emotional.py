import streamlit as st
from transformers import pipeline
import numpy as np

class SocioEmotionalAnalyzer:
    _model = None  # Cache statique du modèle
    
    def __init__(self, lang="fr"):
        """
        Initialise l'analyseur socio-émotionnel.
        """
        self.lang = lang
        
        # Définition des catégories socio-émotionnelles selon la langue
        if lang == "fr":
            self.categories = [
                "Autonomie", "Compétence", "Affiliation",  # Self-Determination Theory
                "Émotion positive", "Engagement", "Relations sociales", "Sens", "Accomplissement",  # PERMA Model
                "Satisfaction", "Confiance", "Reconnaissance", "Respect"  # Catégories supplémentaires
            ]
        else:  # anglais
            self.categories = [
                "Autonomy", "Competence", "Relatedness",  # Self-Determination Theory
                "Positive emotion", "Engagement", "Relationships", "Meaning", "Accomplishment",  # PERMA Model
                "Satisfaction", "Trust", "Recognition", "Respect"  # Catégories supplémentaires
            ]
            
        # Charger le modèle
        self._ensure_model_loaded()
        
    @staticmethod
    @st.cache_resource
    def _load_model():
        """Charge le modèle zero-shot et le met en cache."""
        try:
            # Utiliser un modèle multilingue
            model = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",
                device=-1  # CPU par défaut
            )
            st.success("✅ Modèle socio-émotionnel chargé avec succès.")
            return model
        except Exception as e:
            st.warning(f"⚠️ Erreur lors du chargement du modèle socio-émotionnel: {e}")
            return None
    
    def _ensure_model_loaded(self):
        """
        S'assure que le modèle est chargé dans le cache de classe.
        """
        if SocioEmotionalAnalyzer._model is None:
            SocioEmotionalAnalyzer._model = self._load_model()
    
    def analyze(self, text):
        """
        Retourne les scores pour chaque catégorie socio-émotionnelle.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {"message": "Texte vide ou invalide", "scores": {}}
            
        if SocioEmotionalAnalyzer._model is None:
            return {"message": "Modèle non disponible", "scores": {}}
            
        try:
            # Limiter la longueur du texte si nécessaire
            if len(text) > 1000:
                text = text[:1000] + "..."
                
            result = SocioEmotionalAnalyzer._model(text, self.categories, multi_label=True)
            scores = dict(zip(result["labels"], result["scores"]))
            
            # Analyse des dimensions principales
            main_dimensions = {
                "Bien-être psychologique": np.mean([
                    scores.get("Autonomie" if self.lang == "fr" else "Autonomy", 0),
                    scores.get("Compétence" if self.lang == "fr" else "Competence", 0),
                    scores.get("Accomplissement" if self.lang == "fr" else "Accomplishment", 0)
                ]),
                "Bien-être social": np.mean([
                    scores.get("Affiliation" if self.lang == "fr" else "Relatedness", 0),
                    scores.get("Relations sociales" if self.lang == "fr" else "Relationships", 0)
                ]),
                "Bien-être émotionnel": np.mean([
                    scores.get("Émotion positive" if self.lang == "fr" else "Positive emotion", 0),
                    scores.get("Satisfaction" if self.lang == "fr" else "Satisfaction", 0)
                ])
            }
            
            return {
                "message": "Succès",
                "scores": scores,
                "dimensions": main_dimensions
            }
            
        except Exception as e:
            return {"message": f"Erreur lors de l'analyse: {str(e)}", "scores": {}}
