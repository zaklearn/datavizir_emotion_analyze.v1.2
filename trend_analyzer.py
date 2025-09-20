import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

class TrendAnalyzer:
    def __init__(_self, df):
        """
        Initialise l'analyseur de tendances.
        
        Parameters:
            df (pandas.DataFrame): DataFrame contenant les avis et analyses
        """
        _self.df = df.copy()  # Copie pour éviter les modifications non intentionnelles
        
        # Tenter d'extraire une date si elle existe
        _self._extract_date_if_exists()
    
    def _extract_date_if_exists(_self):
        """Tente d'extraire une colonne de date pour l'analyse temporelle."""
        # Chercher une colonne de date existante
        date_columns = [col for col in _self.df.columns if any(date_term in col.lower() 
                                                             for date_term in ["date", "jour", "mois", "année", "day", "month", "year"])]
        
        if date_columns:
            # Utiliser la première colonne de date trouvée
            try:
                _self.df["date_parsed"] = pd.to_datetime(_self.df[date_columns[0]], errors="coerce")
                return True
            except:
                pass
        
        # Tenter d'extraire une date à partir du texte des avis
        if "review" in _self.df.columns:
            # Expression régulière pour détecter des dates courantes
            date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{1,2}\s(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|jan|fév|mar|avr|mai|juin|juil|août|sept|oct|nov|déc|january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s\d{2,4})"
            
            # Appliquer l'extraction
            _self.df["extracted_date"] = _self.df["review"].str.extract(date_pattern, expand=False)
            
            if _self.df["extracted_date"].notna().any():
                # Convertir en datetime
                _self.df["date_parsed"] = pd.to_datetime(_self.df["extracted_date"], errors="coerce")
                return True
                
        return False
    
    @st.cache_data
    def correlation_matrix(_self, target_columns=None):
        """
        Affiche une matrice de corrélation des émotions ou des colonnes spécifiées.
        
        Parameters:
            target_columns (list, optional): Liste des colonnes à inclure dans la matrice
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib pour affichage dans Streamlit
        """
        # Sélectionner les colonnes numériques si aucune colonne n'est spécifiée
        if not target_columns:
            target_columns = _self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclure certaines colonnes non pertinentes
            exclude = ["index", "id", "user_id", "rating"]
            target_columns = [col for col in target_columns if not any(ex in col.lower() for ex in exclude)]
        
        # Utiliser uniquement les colonnes qui existent
        valid_columns = [col for col in target_columns if col in _self.df.columns]
        
        if not valid_columns:
            return None
            
        # Calculer la matrice de corrélation
        corr = _self.df[valid_columns].corr()
        
        # Créer la visualisation
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, mask=mask, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.title("Matrice de corrélation")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        return fig
    
    @st.cache_data
    def sentiment_over_time(_self, time_unit="M"):
        """
        Analyse l'évolution des sentiments au fil du temps si une date est disponible.
        
        Parameters:
            time_unit (str): Unité de temps pour le regroupement ('D'=jour, 'W'=semaine, 'M'=mois)
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib pour affichage dans Streamlit
        """
        if "date_parsed" not in _self.df.columns or _self.df["date_parsed"].isna().all():
            return None
            
        if "Sentiment" not in _self.df.columns:
            return None
            
        # Regrouper par période et sentiment
        sentiment_counts = _self.df.groupby([pd.Grouper(key="date_parsed", freq=time_unit), "Sentiment"]).size().unstack(fill_value=0)
        
        # Calculer les pourcentages
        sentiment_pcts = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
        
        # Créer la visualisation
        fig, ax = plt.subplots(figsize=(12, 6))
        sentiment_pcts.plot(kind="area", stacked=True, alpha=0.7, ax=ax,
                         color={"Positif": "green", "Neutre": "gray", "Négatif": "red"})
        
        plt.title("Évolution des sentiments au fil du temps")
        plt.ylabel("Pourcentage")
        plt.xlabel("Date")
        plt.legend(title="Sentiment")
        plt.tight_layout()
        
        return fig
    
    @st.cache_data    
    def keyword_trends(_self, keywords, window=5):
        """
        Analyse la fréquence des mots-clés au fil du temps.
        
        Parameters:
            keywords (list): Liste des mots-clés à rechercher
            window (int): Fenêtre pour la moyenne mobile
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib pour affichage dans Streamlit
        """
        if "date_parsed" not in _self.df.columns or "review" not in _self.df.columns:
            return None
            
        # Créer un DataFrame pour stocker les tendances des mots-clés
        trends_df = pd.DataFrame(index=_self.df.index)
        
        # Pour chaque mot-clé, vérifier sa présence dans chaque avis
        for keyword in keywords:
            trends_df[keyword] = _self.df["review"].str.contains(keyword, case=False, regex=False).astype(int)
        
        # Regrouper par date et calculer la moyenne d'occurrence
        if trends_df.shape[1] > 0:
            # Regrouper par date
            grouped = trends_df.join(_self.df["date_parsed"]).groupby("date_parsed").mean()
            
            # Appliquer une moyenne mobile pour lisser les tendances
            smoothed = grouped.rolling(window=window, min_periods=1).mean()
            
            # Visualiser les tendances
            fig, ax = plt.subplots(figsize=(12, 6))
            smoothed.plot(ax=ax)
            plt.title("Tendances des mots-clés au fil du temps")
            plt.ylabel("Fréquence relative")
            plt.xlabel("Date")
            plt.legend(title="Mots-clés")
            plt.tight_layout()
            
            return fig
            
        return None
