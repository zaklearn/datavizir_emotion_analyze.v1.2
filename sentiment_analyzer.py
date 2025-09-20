import nltk
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os

@st.cache_resource
def load_spacy_models():
    """
    Charge et met en cache les modèles SpaCy avec fallback pour Streamlit Cloud.
    """
    try:
        import en_core_web_sm
        import fr_core_news_sm
        nlp_en = en_core_web_sm.load()
        nlp_fr = fr_core_news_sm.load()
        st.success("✅ Modèles SpaCy chargés avec succès")
        return nlp_en, nlp_fr
    except ImportError:
        st.warning("⚠️ Modèles SpaCy non disponibles - Fonctionnalités limitées à TextBlob et VADER")
        return None, None

# Chargement des modèles au démarrage
nlp_en, nlp_fr = load_spacy_models()

@st.cache_resource
def load_vader():
    """
    Charge et met en cache le modèle VADER.
    """
    try:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.warning(f"⚠️ Erreur lors du chargement de VADER: {e}")
        class SimpleSentimentAnalyzer:
            def polarity_scores(self, text):
                return {"compound": 0}
        return SimpleSentimentAnalyzer()

# Initialisation de VADER
sia = load_vader()

def load_spacy_model(lang):
    """
    Retourne le modèle spaCy préchargé correspondant à la langue.
    """
    if lang == "fr":
        return nlp_fr
    else:
        return nlp_en

def analyze_sentiment(text, lang):
    """
    Analyse de sentiment de base
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return "Neutre", 0.0
        
    if lang == "fr":
        blob = TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        score = blob.sentiment[0]
    elif lang == "en":
        score = sia.polarity_scores(text)["compound"]
    else:
        return "Inconnu", 0
    
    if score > 0.1:
        sentiment = "Positif"
    elif score < -0.1:
        sentiment = "Négatif"
    else:
        sentiment = "Neutre"
    return sentiment, score

def analyze_text_advanced(text, lang):
    """
    Réalise une analyse avancée du texte avec fallback si SpaCy n'est pas disponible
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return {
            "exclamative": False,
            "negations": False,
            "entities": [],
            "keywords": [],
            "adjectives": []
        }
        
    nlp = load_spacy_model(lang)
    
    # Vérifier si SpaCy est disponible
    if nlp is None:
        return {
            "exclamative": "!" in text,
            "negations": any(word in text.lower() for word in ["not", "no", "never", "pas", "non", "jamais"]),
            "entities": [],
            "keywords": text.split()[:10] if text else [],  # Mots simples comme fallback
            "adjectives": [],
            "message": "Analyse basique - SpaCy non disponible"
        }
    
    doc = nlp(text)
    
    # Détection de phrases exclamatives
    exclamative = any(sent.text.strip().endswith('!') for sent in doc.sents)
    
    # Vérification des négations
    negations = any(token.dep_ == "neg" for token in doc)
    
    # Extraction des entités nommées (texte et type)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extraction de mots-clés
    keywords = [token.lemma_.lower() for token in doc 
                if token.pos_ in ["NOUN", "ADJ"] and not token.is_stop and token.is_alpha]
    
    # Extraction des adjectifs
    adjectives = [token.lemma_.lower() for token in doc if token.pos_ == "ADJ"]
    
    return {
        "exclamative": exclamative,
        "negations": negations,
        "entities": entities,
        "keywords": keywords,
        "adjectives": adjectives
    }

@st.cache_data
def get_tfidf_keywords(corpus, lang, max_features=20):
    """
    Extrait les mots-clés du corpus d'avis en utilisant TF-IDF.
    """
    if not corpus or len(corpus) == 0:
        return []
        
    # Définir les stop words selon la langue
    if lang == "fr":
        try:
            import nltk
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            stop_words = list(stopwords.words('french'))
        except:
            stop_words = ["le", "la", "les", "un", "une", "des", "et", "est", "en", "que", "qui", 
                        "pour", "dans", "ce", "cette", "ces", "il", "elle", "ils", "elles", 
                        "nous", "vous", "je", "tu", "on", "son", "sa", "ses", "mon", "ma", "mes",
                        "ton", "ta", "tes", "leur", "leurs", "de", "du", "au", "aux", "par", "avec"]
    else:
        try:
            import nltk
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            stop_words = list(stopwords.words('english'))
        except:
            stop_words = "english"
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return list(feature_names)

def get_combined_analysis(text, lang, emotion_classifier=None, socio_analyzer=None):
    """
    Combine les analyses de sentiment, d'émotion et socio-émotionnelle.
    """
    # Analyse de base
    sentiment, score = analyze_sentiment(text, lang)
    
    # Analyse avancée
    advanced = analyze_text_advanced(text, lang)
    
    # Analyse des émotions
    emotions = {}
    if emotion_classifier:
        emotions = emotion_classifier.classify(text)
        
    # Analyse socio-émotionnelle
    socio_emotional = {}
    if socio_analyzer:
        socio_emotional = socio_analyzer.analyze(text)
        
    return {
        "sentiment": sentiment,
        "score": score,
        "advanced": advanced,
        "emotions": emotions,
        "socio_emotional": socio_emotional
    }
