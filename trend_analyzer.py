import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from collections import Counter

class TrendAnalyzer:
    def __init__(self, df):
        """
        Initialise l'analyseur de tendances.
        
        Parameters:
            df (pandas.DataFrame): DataFrame contenant les avis et analyses
        """
        self.df = df.copy()  # Copie pour éviter les modifications non intentionnelles
        
        # Tenter d'extraire une date si elle existe
        self.has_date = self._extract_date_if_exists()
    
    def _extract_date_if_exists(self):
        """Tente d'extraire une colonne de date pour l'analyse temporelle."""
        # Chercher une colonne de date existante
        date_columns = [col for col in self.df.columns if any(date_term in col.lower() 
                                                             for date_term in ["date", "jour", "mois", "année", "day", "month", "year", "time", "timestamp"])]
        
        if date_columns:
            # Utiliser la première colonne de date trouvée
            try:
                self.df["date_parsed"] = pd.to_datetime(self.df[date_columns[0]], errors="coerce")
                if self.df["date_parsed"].notna().sum() > 0:
                    return True
            except:
                pass
        
        # Tenter d'extraire une date à partir du texte des avis
        if "review" in self.df.columns:
            # Expression régulière pour détecter des dates courantes
            date_patterns = [
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",  # 01/01/2023, 1-1-23
                r"(\d{1,2}\s(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|jan|fév|mar|avr|mai|juin|juil|août|sept|oct|nov|déc)\s\d{2,4})",  # français
                r"(\d{1,2}\s(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s\d{2,4})",  # anglais
                r"(\d{4}-\d{1,2}-\d{1,2})"  # ISO format 2023-01-01
            ]
            
            extracted_dates = []
            for pattern in date_patterns:
                matches = self.df["review"].str.extract(pattern, expand=False)
                extracted_dates.extend(matches.dropna().tolist())
            
            if extracted_dates:
                # Prendre la première date trouvée par ligne
                self.df["extracted_date"] = None
                for i, review in enumerate(self.df["review"]):
                    for pattern in date_patterns:
                        match = re.search(pattern, str(review), re.IGNORECASE)
                        if match:
                            self.df.loc[i, "extracted_date"] = match.group(1)
                            break
                
                # Convertir en datetime
                try:
                    self.df["date_parsed"] = pd.to_datetime(self.df["extracted_date"], errors="coerce")
                    if self.df["date_parsed"].notna().sum() > 0:
                        return True
                except:
                    pass
                
        return False
    
    def correlation_matrix(self, target_columns=None):
        """
        Affiche une matrice de corrélation des émotions ou des colonnes spécifiées.
        
        Parameters:
            target_columns (list, optional): Liste des colonnes à inclure dans la matrice
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib pour affichage dans Streamlit
        """
        # Sélectionner les colonnes numériques si aucune colonne n'est spécifiée
        if not target_columns:
            target_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclure certaines colonnes non pertinentes
            exclude = ["index", "id", "user_id", "rating", "date_parsed"]
            target_columns = [col for col in target_columns if not any(ex in col.lower() for ex in exclude)]
        
        # Utiliser uniquement les colonnes qui existent
        valid_columns = [col for col in target_columns if col in self.df.columns]
        
        if len(valid_columns) < 2:
            st.warning("Pas assez de colonnes numériques pour créer une matrice de corrélation.")
            return None
            
        # Calculer la matrice de corrélation
        corr = self.df[valid_columns].corr()
        
        # Créer la visualisation
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Utiliser une palette de couleurs divergente
        sns.heatmap(corr, annot=True, mask=mask, cmap="RdBu_r", 
                   fmt=".2f", ax=ax, center=0,
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": .8})
        
        plt.title("Matrice de corrélation des variables", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def sentiment_over_time(self, time_unit="M"):
        """
        Analyse l'évolution des sentiments au fil du temps si une date est disponible.
        
        Parameters:
            time_unit (str): Unité de temps pour le regroupement ('D'=jour, 'W'=semaine, 'M'=mois)
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib pour affichage dans Streamlit
        """
        if not self.has_date or "date_parsed" not in self.df.columns:
            st.warning("Aucune information de date disponible pour l'analyse temporelle.")
            return None
            
        if "Sentiment" not in self.df.columns:
            st.warning("Colonne 'Sentiment' requise pour l'analyse temporelle.")
            return None
        
        # Filtrer les lignes avec des dates valides
        df_with_dates = self.df[self.df["date_parsed"].notna()].copy()
        
        if len(df_with_dates) == 0:
            st.warning("Aucune date valide trouvée dans les données.")
            return None
            
        # Regrouper par période et sentiment
        try:
            sentiment_counts = df_with_dates.groupby([
                pd.Grouper(key="date_parsed", freq=time_unit), 
                "Sentiment"
            ]).size().unstack(fill_value=0)
            
            # Calculer les pourcentages
            sentiment_pcts = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
            
            # Créer la visualisation
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Définir les couleurs
            colors = {"Positif": "#2E8B57", "Neutre": "#808080", "Négatif": "#DC143C"}
            available_colors = [colors.get(col, "#1f77b4") for col in sentiment_pcts.columns]
            
            sentiment_pcts.plot(kind="area", stacked=True, alpha=0.7, ax=ax, color=available_colors)
            
            plt.title("Évolution des sentiments au fil du temps", fontsize=16, pad=20)
            plt.ylabel("Pourcentage", fontsize=12)
            plt.xlabel("Date", fontsize=12)
            plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Erreur lors de la création du graphique temporel: {str(e)}")
            return None
    
    def keyword_trends(self, keywords, window=5):
        """
        Analyse la fréquence des mots-clés au fil du temps.
        
        Parameters:
            keywords (list): Liste des mots-clés à rechercher
            window (int): Fenêtre pour la moyenne mobile
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib pour affichage dans Streamlit
        """
        if not self.has_date or "date_parsed" not in self.df.columns:
            st.warning("Aucune information de date disponible pour l'analyse des tendances de mots-clés.")
            return None
            
        if "review" not in self.df.columns:
            st.warning("Colonne 'review' requise pour l'analyse des mots-clés.")
            return None
        
        if not keywords:
            st.warning("Aucun mot-clé fourni pour l'analyse.")
            return None
            
        # Filtrer les lignes avec des dates valides
        df_with_dates = self.df[self.df["date_parsed"].notna()].copy()
        
        if len(df_with_dates) == 0:
            st.warning("Aucune date valide trouvée dans les données.")
            return None
        
        # Créer un DataFrame pour stocker les tendances des mots-clés
        trends_data = []
        
        # Pour chaque mot-clé, vérifier sa présence dans chaque avis
        for keyword in keywords:
            keyword_presence = df_with_dates["review"].str.contains(
                keyword, case=False, regex=False, na=False
            ).astype(int)
            
            # Grouper par date et calculer la fréquence
            keyword_trend = df_with_dates.groupby("date_parsed")[keyword_presence.name].apply(
                lambda x: x.sum() / len(x) if len(x) > 0 else 0
            )
            
            trends_data.append(pd.Series(keyword_trend.values, 
                                       index=keyword_trend.index, 
                                       name=keyword))
        
        if not trends_data:
            st.warning("Aucune tendance de mots-clés à analyser.")
            return None
        
        # Combiner toutes les tendances
        trends_df = pd.concat(trends_data, axis=1)
        
        # Appliquer une moyenne mobile pour lisser les tendances
        smoothed = trends_df.rolling(window=window, min_periods=1).mean()
        
        # Visualiser les tendances
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, keyword in enumerate(keywords):
            if keyword in smoothed.columns:
                ax.plot(smoothed.index, smoothed[keyword], 
                       label=keyword, linewidth=2, marker='o', markersize=4)
        
        plt.title("Tendances des mots-clés au fil du temps", fontsize=16, pad=20)
        plt.ylabel("Fréquence relative", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.legend(title="Mots-clés", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def emotion_evolution(self, top_n=5):
        """
        Analyse l'évolution des émotions principales au fil du temps.
        
        Parameters:
            top_n (int): Nombre d'émotions principales à analyser
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib pour affichage dans Streamlit
        """
        if not self.has_date:
            st.warning("Aucune information de date disponible pour l'analyse de l'évolution des émotions.")
            return None
        
        # Identifier les colonnes d'émotions
        emotion_cols = [col for col in self.df.columns if col.startswith("Emotion_")]
        
        if not emotion_cols:
            st.warning("Aucune colonne d'émotion trouvée dans les données.")
            return None
        
        # Filtrer les lignes avec des dates valides
        df_with_dates = self.df[self.df["date_parsed"].notna()].copy()
        
        if len(df_with_dates) == 0:
            st.warning("Aucune date valide trouvée dans les données.")
            return None
        
        # Sélectionner les top N émotions par score moyen
        emotion_means = df_with_dates[emotion_cols].mean().sort_values(ascending=False)
        top_emotions = emotion_means.head(top_n).index.tolist()
        
        # Grouper par date et calculer les moyennes des émotions
        emotion_trends = df_with_dates.groupby("date_parsed")[top_emotions].mean()
        
        # Créer la visualisation
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for emotion in top_emotions:
            emotion_name = emotion.replace("Emotion_", "").replace("_", " ")
            ax.plot(emotion_trends.index, emotion_trends[emotion], 
                   label=emotion_name, linewidth=2, marker='o', markersize=4)
        
        plt.title("Évolution des émotions principales au fil du temps", fontsize=16, pad=20)
        plt.ylabel("Score moyen d'émotion", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.legend(title="Émotions", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def sentiment_score_distribution(self):
        """
        Analyse la distribution des scores de sentiment.
        
        Returns:
            matplotlib.figure.Figure: Figure matplotlib pour affichage dans Streamlit
        """
        if "Score" not in self.df.columns:
            st.warning("Colonne 'Score' requise pour l'analyse de la distribution.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogramme des scores
        ax1.hist(self.df["Score"].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title("Distribution des scores de sentiment", fontsize=14)
        ax1.set_xlabel("Score de sentiment", fontsize=12)
        ax1.set_ylabel("Fréquence", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Box plot par sentiment si disponible
        if "Sentiment" in self.df.columns:
            sentiment_data = []
            sentiment_labels = []
            
            for sentiment in self.df["Sentiment"].unique():
                if pd.notna(sentiment):
                    scores = self.df[self.df["Sentiment"] == sentiment]["Score"].dropna()
                    if len(scores) > 0:
                        sentiment_data.append(scores)
                        sentiment_labels.append(sentiment)
            
            if sentiment_data:
                ax2.boxplot(sentiment_data, labels=sentiment_labels)
                ax2.set_title("Distribution des scores par sentiment", fontsize=14)
                ax2.set_xlabel("Sentiment", fontsize=12)
                ax2.set_ylabel("Score", fontsize=12)
                ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Données de sentiment\nnon disponibles", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Distribution par sentiment", fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def get_summary_stats(self):
        """
        Retourne un résumé statistique des tendances.
        
        Returns:
            dict: Dictionnaire contenant les statistiques de résumé
        """
        stats = {
            "total_reviews": len(self.df),
            "has_temporal_data": self.has_date,
            "date_range": None,
            "sentiment_distribution": None,
            "top_emotions": None,
            "numeric_columns": len(self.df.select_dtypes(include=[np.number]).columns)
        }
        
        # Informations temporelles
        if self.has_date and "date_parsed" in self.df.columns:
            valid_dates = self.df["date_parsed"].dropna()
            if len(valid_dates) > 0:
                stats["date_range"] = {
                    "start": valid_dates.min(),
                    "end": valid_dates.max(),
                    "span_days": (valid_dates.max() - valid_dates.min()).days
                }
        
        # Distribution des sentiments
        if "Sentiment" in self.df.columns:
            stats["sentiment_distribution"] = self.df["Sentiment"].value_counts().to_dict()
        
        # Émotions principales
        emotion_cols = [col for col in self.df.columns if col.startswith("Emotion_")]
        if emotion_cols:
            emotion_means = self.df[emotion_cols].mean().sort_values(ascending=False)
            stats["top_emotions"] = emotion_means.head(5).to_dict()
        
        return stats
