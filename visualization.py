import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def visualize_sentiments(df):
    """
    Affiche la distribution des sentiments sous forme d'histogramme.
    """
    st.write("### Distribution des sentiments")
    plt.figure(figsize=(8, 5))
    # Correction du warning seaborn
    ax = sns.countplot(x="Sentiment", data=df, hue="Sentiment", 
                      palette={"Positif": "green", "Neutre": "gray", "Négatif": "red"}, 
                      legend=False)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Nombre d'avis")
    st.pyplot(plt)

def generate_wordcloud(text, lang):
    """
    Génère et affiche un WordCloud à partir du texte fourni.
    """
    # Définir les stop words selon la langue
    if lang == "fr":
        stop_words = ["le", "la", "les", "un", "une", "des", "et", "est", "en", "que", "qui", 
                     "pour", "dans", "ce", "cette", "ces", "il", "elle", "ils", "elles"]
    else:
        stop_words = "english"
    
    wordcloud = WordCloud(width=800, height=400, 
                         background_color="white",
                         stopwords=stop_words,
                         max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
