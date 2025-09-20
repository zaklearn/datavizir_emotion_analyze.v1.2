# Patient Review Analysis System

## Overview
This application is a comprehensive text analysis platform designed specifically for patient reviews, providing deep insights through advanced natural language processing (NLP) techniques. The system analyzes patient feedback to extract sentiments, emotions, context, and socio-emotional aspects, with full support for both French and English languages.

## Key Features

### Multilingual Support
- Complete analysis capabilities in both French and English
- Language-specific models for optimal accuracy

### Sentiment Analysis
- Basic sentiment categorization (Positive, Neutral, Negative)
- Sentiment scoring with numerical values between -1 and 1
- Detection of exclamations and negations for nuanced analysis

### Emotion Analysis
- Identification of specific emotions in patient feedback
- Emotion intensity scoring
- Language-specific emotion detection models

### Socio-Emotional Analysis
- Advanced psychological dimension analysis based on:
  - Self-Determination Theory (Autonomy, Competence, Relatedness)
  - PERMA Model (Positive emotion, Engagement, Relationships, Meaning, Accomplishment)
  - Additional dimensions: Satisfaction, Trust, Recognition, Respect

### Contextual Analysis
- Named entity recognition
- Syntactic relationship extraction
- Advanced syntactic analysis (subjects, objects, verbs)
- Active/passive voice detection

### Trend Analysis
- Temporal analysis of sentiments and emotions
- Keyword frequency tracking over time
- Correlation analysis between different metrics

### Report Generation
- Comprehensive DOCX reports with all analysis results
- Data visualizations embedded in reports
- Customizable report sections

### Data Visualization
- Interactive sentiment distribution charts
- Word clouds for key term visualization
- Emotion distribution visualization
- Correlation matrices
- Temporal trend visualization

## Components

### Data Loading
- Support for multiple file formats (CSV, Excel, TXT)
- Automatic encoding detection
- Validation of input data structure

### Modular Architecture
- Separate analysis components for different aspects of text
- Caching system for improved performance
- Fallback mechanisms when primary models are unavailable

### User Interface
- Built with Streamlit for interactive web experience
- Simple file upload and analysis options
- Dynamic visualization of results
- Downloadable reports

## Technical Details

### Core Technologies
- **spaCy**: For contextual analysis and named entity recognition
- **Transformers**: For emotion classification and zero-shot learning
- **TextBlob**: For basic sentiment analysis
- **VADER**: For enhanced English sentiment analysis
- **Streamlit**: For the web interface
- **Pandas**: For data processing
- **Matplotlib/Seaborn**: For data visualization
- **WordCloud**: For generating word clouds
- **python-docx**: For report generation

### Performance Optimization
- Model caching to prevent reloading
- Results caching for faster repeated analysis
- Automatic resource management

## Requirements

### Python Libraries
- spacy
- transformers
- textblob
- textblob-fr
- vaderSentiment
- nltk
- scikit-learn
- streamlit
- pandas
- matplotlib
- seaborn
- wordcloud
- python-docx
- numpy
- torch (or tensorflow, depending on chosen backend for transformers)

### Language Models
- SpaCy:
  - fr_core_news_sm (French)
  - en_core_web_sm (English)
- NLTK resources:
  - vader_lexicon
  - stopwords
- Hugging Face Transformer models:
  - j-hartmann/emotion-english-distilroberta-base
  - nlptown/bert-base-multilingual-uncased-sentiment
  - facebook/bart-large-mnli

## Installation and Setup

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required language models:
   ```bash
   python -m spacy download fr_core_news_sm
   python -m spacy download en_core_web_sm
   python -m nltk.downloader vader_lexicon
   python -m nltk.downloader stopwords
   ```
4. Launch the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a CSV, Excel, or text file containing patient reviews (must include a 'review' column)
2. Select the language (French or English)
3. Choose which analysis components to enable
4. View the interactive analysis results
5. Generate and download a comprehensive report

## Note on Performance

The initial startup may be slow as the application downloads and loads the necessary language models. After the first run, subsequent analyses will be faster due to the caching system.

## System Requirements

- Python 3.7+
- 4GB+ RAM recommended for optimal performance
- Internet connection for initial model downloads