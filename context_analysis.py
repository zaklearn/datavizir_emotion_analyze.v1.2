import spacy
import streamlit as st

class ContextAnalyzer:
    def __init__(self, lang="fr"):
        """
        Initialise l'analyseur de contexte avec fallback pour Streamlit Cloud
        """
        try:
            from sentiment_analyzer import load_spacy_model
            self.nlp = load_spacy_model(lang)
            self.lang = lang
        except ImportError:
            self.nlp = None
            self.lang = lang
            st.warning("⚠️ Analyseur de contexte en mode limité - SpaCy non disponible")

    def analyze_context(self, text):
        """
        Retourne les entités et relations dans le texte avec fallback
        """
        if not text or not isinstance(text, str):
            return {"entities": [], "dependencies": [], "syntactic_analysis": {}}
        
        # Fallback si SpaCy n'est pas disponible
        if self.nlp is None:
            return {
                "entities": [],
                "dependencies": [],
                "syntactic_analysis": {
                    "sentences": len(text.split('.')),
                    "verbs": [],
                    "subjects": [],
                    "objects": [],
                    "voice": "unknown",
                    "message": "Analyse de contexte basique - SpaCy non disponible"
                }
            }
            
        doc = self.nlp(text)
        
        # Extraction des entités nommées
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extraction des dépendances syntaxiques
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        
        # Analyse syntaxique avancée
        syntactic_analysis = {
            "sentences": len(list(doc.sents)),
            "verbs": [token.lemma_ for token in doc if token.pos_ == "VERB"],
            "subjects": [token.text for token in doc if "subj" in token.dep_],
            "objects": [token.text for token in doc if "obj" in token.dep_]
        }
        
        # Détection de la voix (active/passive)
        passive_constructs = [token.text for token in doc if token.dep_ == "auxpass"]
        syntactic_analysis["voice"] = "passive" if passive_constructs else "active"
        
        return {
            "entities": entities,
            "dependencies": dependencies,
            "syntactic_analysis": syntactic_analysis
        }
