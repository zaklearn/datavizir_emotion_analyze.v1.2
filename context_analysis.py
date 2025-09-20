import spacy
import streamlit as st

class ContextAnalyzer:
    def __init__(_self, lang="fr"):
        """
        Initialise l'analyseur de contexte avec le modèle approprié.
        Réutilise les modèles SpaCy déjà chargés dans sentiment_analyzer.py.
        
        Parameters:
            lang (str): Langue de l'analyse ('fr' ou 'en')
        """
        try:
            # Réutiliser les modèles pré-chargés via sentiment_analyzer
            from sentiment_analyzer import load_spacy_model
            _self.nlp = load_spacy_model(lang)
            _self.lang = lang
        except ImportError:
            # Fallback si le modèle n'est pas disponible via sentiment_analyzer
            if lang == "fr":
                try:
                    _self.nlp = spacy.load("fr_core_news_sm")
                except OSError:
                    import os
                    os.system("python -m spacy download fr_core_news_sm")
                    _self.nlp = spacy.load("fr_core_news_sm")
            else:  # Anglais par défaut
                try:
                    _self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    import os
                    os.system("python -m spacy download en_core_web_sm")
                    _self.nlp = spacy.load("en_core_web_sm")

    def analyze_context(_self, text):
        """
        Retourne les entités et relations dans le texte.
        Mise en cache avec Streamlit pour améliorer les performances.
        
        Parameters:
            text (str): Texte à analyser
            
        Returns:
            dict: Dictionnaire contenant les entités, relations et analyse syntaxique
        """
        if not text or not isinstance(text, str):
            return {"entities": [], "dependencies": [], "syntactic_analysis": {}}
            
        doc = _self.nlp(text)
        
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
