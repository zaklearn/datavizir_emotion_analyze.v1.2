import streamlit as st
st.set_page_config(page_title="Analyse des Avis Patients", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data
from sentiment_analyzer import analyze_sentiment, analyze_text_advanced, get_tfidf_keywords
from visualization import visualize_sentiments, generate_wordcloud
from report_generator import export_report, generate_streamlit_report
from context_analysis import ContextAnalyzer
from emotion_classifier import EmotionClassifier
#from trend_analyzer import TrendAnalyzer
from socio_emotional import SocioEmotionalAnalyzer
import time
import seaborn as sns
# Configuration de la page (doit être le premier appel Streamlit)
#st.set_page_config(page_title="Analyse des Avis Patients", layout="wide")

# CSS personnalisé pour améliorer l'interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des données de cache de page
if 'df_cache' not in st.session_state:
    st.session_state.df_cache = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'emotions_analyzed' not in st.session_state:
    st.session_state.emotions_analyzed = False
if 'socio_analyzed' not in st.session_state:
    st.session_state.socio_analyzed = False
if 'context_analyzed' not in st.session_state:
    st.session_state.context_analyzed = False
@st.cache_data
# Fonction pour exécuter l'analyse de sentiment
def run_sentiment_analysis(df, lang):
    sentiments_scores = []
    progress_bar = st.progress(0)
    for i, review in enumerate(df["review"]):
        sentiments_scores.append(analyze_sentiment(review, lang))
        progress_bar.progress((i + 1) / len(df))
    df["Sentiment"], df["Score"] = zip(*sentiments_scores)
    progress_bar.empty()
    return df
@st.cache_data
# Fonction pour exécuter l'analyse avancée
def run_advanced_analysis(df, lang):
    progress_bar = st.progress(0)
    advanced_results = []
    for i, review in enumerate(df["review"]):
        advanced_results.append(analyze_text_advanced(review, lang))
        progress_bar.progress((i + 1) / len(df))
    df["Advanced"] = advanced_results
    progress_bar.empty()  # Effacer la barre de progression
    return df
@st.cache_data
# Fonction pour exécuter l'analyse de contexte
def run_context_analysis(df, lang):
    progress_bar = st.progress(0)
    context_analyzer = ContextAnalyzer(lang=lang)
    context_results = []
    for i, review in enumerate(df["review"]):
        context_results.append(context_analyzer.analyze_context(review))
        progress_bar.progress((i + 1) / len(df))
    df["Context"] = context_results
    progress_bar.empty()
    return df
@st.cache_data
# Fonction pour exécuter l'analyse des émotions
def run_emotion_analysis(df, lang):
    progress_bar = st.progress(0)
    emotion_classifier = EmotionClassifier(lang=lang)
    emotion_results = []
    emotion_columns = {}
    
    for i, review in enumerate(df["review"]):
        result = emotion_classifier.classify(review)
        emotion_results.append(result)
        
        # Créer des colonnes distinctes pour chaque émotion
        for emotion, score in result.get("scores", {}).items():
            if emotion not in emotion_columns:
                emotion_columns[emotion] = []
            while len(emotion_columns[emotion]) < i:
                emotion_columns[emotion].append(0)  # Remplir avec des zéros pour les lignes précédentes
            emotion_columns[emotion].append(score)
        
        # Assurer que toutes les colonnes ont la même longueur
        for emotion in emotion_columns:
            while len(emotion_columns[emotion]) <= i:
                emotion_columns[emotion].append(0)
                
        progress_bar.progress((i + 1) / len(df))
    
    # Ajouter les colonnes d'émotions au DataFrame
    for emotion, values in emotion_columns.items():
        # Normaliser le nom de la colonne
        col_name = f"Emotion_{emotion.replace(' ', '_')}"
        df[col_name] = values
    
    df["Emotions"] = emotion_results
    progress_bar.empty()
    return df
@st.cache_data
# Fonction pour exécuter l'analyse socio-émotionnelle
def run_socio_emotional_analysis(df, lang):
    progress_bar = st.progress(0)
    socio_analyzer = SocioEmotionalAnalyzer(lang=lang)
    socio_results = []
    socio_columns = {}
    dimension_columns = {}
    
    for i, review in enumerate(df["review"]):
        result = socio_analyzer.analyze(review)
        socio_results.append(result)
        
        # Créer des colonnes distinctes pour chaque dimension socio-émotionnelle
        for category, score in result.get("scores", {}).items():
            col_name = f"Socio_{category.replace(' ', '_')}"
            if col_name not in socio_columns:
                socio_columns[col_name] = []
            while len(socio_columns[col_name]) < i:
                socio_columns[col_name].append(0)
            socio_columns[col_name].append(score)
            
        # Dimensions principales
        for dimension, score in result.get("dimensions", {}).items():
            col_name = f"Dim_{dimension.replace(' ', '_')}"
            if col_name not in dimension_columns:
                dimension_columns[col_name] = []
            while len(dimension_columns[col_name]) < i:
                dimension_columns[col_name].append(0)
            dimension_columns[col_name].append(score)
        
        # Assurer que toutes les colonnes ont la même longueur
        for col_dict in [socio_columns, dimension_columns]:
            for col_name in col_dict:
                while len(col_dict[col_name]) <= i:
                    col_dict[col_name].append(0)
                    
        progress_bar.progress((i + 1) / len(df))
    
    # Ajouter les colonnes au DataFrame
    for col_dict, columns in [(socio_columns, socio_columns), (dimension_columns, dimension_columns)]:
        for col_name, values in columns.items():
            df[col_name] = values
    
    df["SocioEmotional"] = socio_results
    progress_bar.empty()
    return df

# Interface dans la sidebar
st.sidebar.markdown("<h1 class='main-header'>Analyse des avis patients</h1>", unsafe_allow_html=True)
lang = st.sidebar.radio("Choisissez la langue de l'analyse :", ["fr", "en"])
file = st.sidebar.file_uploader("Chargez un fichier d'avis", type=["csv", "xls", "xlsx", "txt"])

# Section principale
st.markdown("<h1 class='main-header'>Analyse des Avis Patients</h1>", unsafe_allow_html=True)

# Chargement des données
if file is not None:
    try:
        with st.spinner("Chargement des données..."):
            df = load_data(file)
            st.session_state.df_cache = df
            st.success(f"✅ {len(df)} avis chargés avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        st.session_state.df_cache = None

# Affichage et analyse des données
if st.session_state.df_cache is not None:
    df = st.session_state.df_cache
    
    # Afficher un aperçu des données
    with st.expander("Aperçu des données"):
        st.dataframe(df.head())
        
    # Options d'analyse
    st.markdown("<h2 class='sub-header'>Options d'analyse</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_basic = st.checkbox("Analyse de sentiment de base", value=True)
        run_advanced = st.checkbox("Analyse avancée", value=True)
    with col2:
        run_emotions = st.checkbox("Analyse des émotions")
        run_context = st.checkbox("Analyse du contexte")
    with col3:
        run_socio = st.checkbox("Analyse socio-émotionnelle")
        generate_report = st.checkbox("Générer un rapport")
    
    # Bouton pour lancer l'analyse
    if st.button("Lancer l'analyse"):
        with st.spinner("Analyse en cours..."):
            # Analyse de sentiment de base
            if run_basic:
                df = run_sentiment_analysis(df, lang)
                
            # Analyse avancée
            if run_advanced:
                df = run_advanced_analysis(df, lang)
                
            # Analyse du contexte
            if run_context:
                df = run_context_analysis(df, lang)
                st.session_state.context_analyzed = True
                
            # Analyse des émotions
            if run_emotions:
                df = run_emotion_analysis(df, lang)
                st.session_state.emotions_analyzed = True
                
            # Analyse socio-émotionnelle
            if run_socio:
                df = run_socio_emotional_analysis(df, lang)
                st.session_state.socio_analyzed = True
                
            # Mise à jour du cache
            st.session_state.df_cache = df
            st.session_state.analysis_complete = True
            
            st.success("✅ Analyse terminée avec succès !")
    
    # Affichage des résultats
    if st.session_state.analysis_complete:
        st.markdown("<h2 class='sub-header'>Résultats de l'analyse</h2>", unsafe_allow_html=True)
        
        # Extraire les mots-clés TF-IDF
        with st.spinner("Extraction des thèmes récurrents..."):
            keywords = get_tfidf_keywords(df["review"].tolist(), lang)
            
        tab1, tab2, tab3, tab4 = st.tabs(["Statistiques générales", "Sentiments", "Mots-clés", "Tendances"])
        
        with tab1:
            # Statistiques générales
            if "Sentiment" in df.columns:
                sentiment_counts = df["Sentiment"].value_counts()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avis positifs", f"{sentiment_counts.get('Positif', 0)} ({sentiment_counts.get('Positif', 0)/len(df)*100:.1f}%)")
                with col2:
                    st.metric("Avis neutres", f"{sentiment_counts.get('Neutre', 0)} ({sentiment_counts.get('Neutre', 0)/len(df)*100:.1f}%)")
                with col3:
                    st.metric("Avis négatifs", f"{sentiment_counts.get('Négatif', 0)} ({sentiment_counts.get('Négatif', 0)/len(df)*100:.1f}%)")
            
            # Émotions moyennes si disponibles
            if st.session_state.emotions_analyzed:
                st.subheader("Émotions dominantes")
                emotion_cols = [col for col in df.columns if col.startswith("Emotion_")]
                if emotion_cols:
                    emotion_means = df[emotion_cols].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    emotion_means.plot(kind="bar", ax=ax)
                    plt.title("Émotions moyennes dans les avis")
                    plt.ylabel("Score moyen")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Dimensions socio-émotionnelles si disponibles
            if st.session_state.socio_analyzed:
                st.subheader("Dimensions socio-émotionnelles")
                dimension_cols = [col for col in df.columns if col.startswith("Dim_")]
                if dimension_cols:
                    dim_means = df[dimension_cols].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    dim_means.plot(kind="bar", ax=ax)
                    plt.title("Dimensions socio-émotionnelles moyennes")
                    plt.ylabel("Score moyen")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with tab2:
            # Visualisation des sentiments
            if "Sentiment" in df.columns:
                visualize_sentiments(df)
                
                # Distribution des scores de sentiment
                st.write("### Distribution des scores de sentiment")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df["Score"], kde=True, ax=ax)
                plt.xlabel("Score de sentiment")
                plt.ylabel("Nombre d'avis")
                st.pyplot(fig)
        
        with tab3:
            # Mots-clés et nuage de mots
            st.write("### Thèmes récurrents (TF-IDF)")
            st.write(", ".join(keywords))
            
            # Nuage de mots
            st.write("### Nuage de mots")
            corpus = " ".join(df["review"].tolist())
            generate_wordcloud(corpus, lang)
            
            # Entités nommées les plus fréquentes
            if "Advanced" in df.columns:
                st.write("### Entités nommées les plus fréquentes")
                entities = []
                for adv in df["Advanced"]:
                    entities.extend([ent[0].lower() for ent in adv.get("entities", [])])
                
                from collections import Counter
                entity_counts = Counter(entities).most_common(20)
                
                if entity_counts:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    labels, values = zip(*entity_counts)
                    plt.barh(range(len(values)), values, tick_label=labels)
                    plt.xlabel("Fréquence")
                    plt.ylabel("Entité")
                    plt.tight_layout()
                    st.pyplot(fig)
        
        
        
        # Génération de rapport
        if generate_report:
            st.markdown("<h2 class='sub-header'>Génération de rapport</h2>", unsafe_allow_html=True)
            generate_streamlit_report(df, keywords=keywords, lang=lang)


else:
    st.info("👈 Veuillez charger un fichier d'avis pour commencer l'analyse.")
    