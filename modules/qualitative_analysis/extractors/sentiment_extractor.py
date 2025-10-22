"""
Extractor de Sentimientos para Análisis Cualitativo
Implementa VADER, TextBlob y análisis contextual para identificación de sentimientos
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Importaciones para análisis de sentimientos
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Importaciones para procesamiento de texto
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Importaciones para visualización
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Importaciones del sistema
from ..core.config import AnalysisConfig
from ..core.citation_manager import CitationManager

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Datos de un análisis de sentimiento"""
    text: str
    polarity: float  # -1.0 a 1.0
    subjectivity: float  # 0.0 a 1.0
    compound_score: float  # VADER compound score
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    confidence: float  # Confianza en la clasificación
    context: str  # Contexto del sentimiento
    citations: List[Dict[str, Any]]  # Referencias a fuentes

@dataclass
class SentimentAnalysisResult:
    """Resultado del análisis de sentimientos"""
    sentiments: List[SentimentData]
    algorithm_used: str
    total_analyzed: int
    sentiment_distribution: Dict[str, int]
    average_polarity: float
    average_subjectivity: float
    processing_time: float
    metadata: Dict[str, Any]
    citations: List[Dict[str, Any]]

class SentimentExtractor:
    """
    Extractor de sentimientos especializado para análisis cualitativo
    
    Características:
    - Múltiples algoritmos (VADER, TextBlob, Híbrido)
    - Análisis contextual avanzado
    - Sistema de citación integrado
    - Visualizaciones interactivas
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.citation_manager = CitationManager()
        self.stopwords = self._get_spanish_stopwords()
        
        # Inicializar VADER
        self._initialize_vader()
    
    def _initialize_vader(self):
        """Inicializar VADER para análisis de sentimientos"""
        try:
            # Descargar recursos de NLTK si es necesario
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Error inicializando VADER: {e}")
            self.vader_analyzer = None
    
    def _get_spanish_stopwords(self) -> List[str]:
        """Obtener stopwords en español"""
        try:
            spanish_stopwords = set(stopwords.words('spanish'))
            # Agregar stopwords personalizados para análisis de sentimientos
            custom_stopwords = {
                'muy', 'más', 'menos', 'mucho', 'poco', 'bastante',
                'realmente', 'verdaderamente', 'absolutamente', 'completamente'
            }
            return list(spanish_stopwords.union(custom_stopwords))
        except:
            return []
    
    def analyze_sentiments_vader(self, chunks: List[Dict[str, Any]]) -> SentimentAnalysisResult:
        """
        Analizar sentimientos usando VADER
        
        Args:
            chunks: Lista de chunks de texto para análisis
            
        Returns:
            Resultado del análisis de sentimientos
        """
        start_time = datetime.now()
        
        try:
            if not self.vader_analyzer:
                raise ValueError("VADER no está disponible. Verifica la instalación de NLTK.")
            
            sentiments = []
            
            for chunk in chunks:
                if isinstance(chunk, dict) and 'content' in chunk:
                    content = chunk['content']
                    
                    # Preprocesar texto
                    processed_text = self._preprocess_text(content)
                    
                    if len(processed_text.strip()) < 10:  # Saltar textos muy cortos
                        continue
                    
                    # Análisis VADER
                    vader_scores = self.vader_analyzer.polarity_scores(processed_text)
                    
                    # Análisis TextBlob para comparación
                    blob = TextBlob(processed_text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    # Determinar etiqueta de sentimiento
                    sentiment_label = self._classify_sentiment(
                        vader_scores['compound'], 
                        polarity
                    )
                    
                    # Calcular confianza
                    confidence = self._calculate_sentiment_confidence(
                        vader_scores, polarity, subjectivity
                    )
                    
                    # Generar contexto
                    context = self._generate_sentiment_context(
                        processed_text, sentiment_label, vader_scores
                    )
                    
                    # Crear citación
                    citation = self._create_sentiment_citation(
                        chunk, processed_text, sentiment_label, confidence
                    )
                    
                    sentiment_data = SentimentData(
                        text=processed_text,
                        polarity=polarity,
                        subjectivity=subjectivity,
                        compound_score=vader_scores['compound'],
                        sentiment_label=sentiment_label,
                        confidence=confidence,
                        context=context,
                        citations=[citation]
                    )
                    
                    sentiments.append(sentiment_data)
            
            # Calcular estadísticas
            sentiment_distribution = self._calculate_sentiment_distribution(sentiments)
            average_polarity = np.mean([s.polarity for s in sentiments])
            average_subjectivity = np.mean([s.subjectivity for s in sentiments])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SentimentAnalysisResult(
                sentiments=sentiments,
                algorithm_used="VADER + TextBlob",
                total_analyzed=len(sentiments),
                sentiment_distribution=sentiment_distribution,
                average_polarity=average_polarity,
                average_subjectivity=average_subjectivity,
                processing_time=processing_time,
                metadata={
                    'n_chunks': len(chunks),
                    'n_analyzed': len(sentiments),
                    'vader_available': self.vader_analyzer is not None
                },
                citations=self.citation_manager.export_citations()
            )
            
        except Exception as e:
            logger.error(f"Error en análisis VADER: {str(e)}")
            raise
    
    def analyze_sentiments_textblob(self, chunks: List[Dict[str, Any]]) -> SentimentAnalysisResult:
        """
        Analizar sentimientos usando TextBlob
        
        Args:
            chunks: Lista de chunks de texto para análisis
            
        Returns:
            Resultado del análisis de sentimientos
        """
        start_time = datetime.now()
        
        try:
            sentiments = []
            
            for chunk in chunks:
                if isinstance(chunk, dict) and 'content' in chunk:
                    content = chunk['content']
                    
                    # Preprocesar texto
                    processed_text = self._preprocess_text(content)
                    
                    if len(processed_text.strip()) < 10:
                        continue
                    
                    # Análisis TextBlob
                    blob = TextBlob(processed_text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    # Determinar etiqueta de sentimiento
                    sentiment_label = self._classify_sentiment_textblob(polarity, subjectivity)
                    
                    # Calcular confianza
                    confidence = self._calculate_textblob_confidence(polarity, subjectivity)
                    
                    # Generar contexto
                    context = self._generate_sentiment_context_textblob(
                        processed_text, sentiment_label, polarity, subjectivity
                    )
                    
                    # Crear citación
                    citation = self._create_sentiment_citation(
                        chunk, processed_text, sentiment_label, confidence
                    )
                    
                    sentiment_data = SentimentData(
                        text=processed_text,
                        polarity=polarity,
                        subjectivity=subjectivity,
                        compound_score=polarity,  # Usar polarity como compound
                        sentiment_label=sentiment_label,
                        confidence=confidence,
                        context=context,
                        citations=[citation]
                    )
                    
                    sentiments.append(sentiment_data)
            
            # Calcular estadísticas
            sentiment_distribution = self._calculate_sentiment_distribution(sentiments)
            average_polarity = np.mean([s.polarity for s in sentiments])
            average_subjectivity = np.mean([s.subjectivity for s in sentiments])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SentimentAnalysisResult(
                sentiments=sentiments,
                algorithm_used="TextBlob",
                total_analyzed=len(sentiments),
                sentiment_distribution=sentiment_distribution,
                average_polarity=average_polarity,
                average_subjectivity=average_subjectivity,
                processing_time=processing_time,
                metadata={
                    'n_chunks': len(chunks),
                    'n_analyzed': len(sentiments)
                },
                citations=self.citation_manager.export_citations()
            )
            
        except Exception as e:
            logger.error(f"Error en análisis TextBlob: {str(e)}")
            raise
    
    def analyze_sentiments_hybrid(self, chunks: List[Dict[str, Any]]) -> SentimentAnalysisResult:
        """
        Analizar sentimientos usando enfoque híbrido (VADER + TextBlob)
        
        Args:
            chunks: Lista de chunks de texto para análisis
            
        Returns:
            Resultado del análisis de sentimientos
        """
        start_time = datetime.now()
        
        try:
            sentiments = []
            
            for chunk in chunks:
                if isinstance(chunk, dict) and 'content' in chunk:
                    content = chunk['content']
                    
                    # Preprocesar texto
                    processed_text = self._preprocess_text(content)
                    
                    if len(processed_text.strip()) < 10:
                        continue
                    
                    # Análisis TextBlob
                    blob = TextBlob(processed_text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    # Análisis VADER si está disponible
                    vader_scores = None
                    if self.vader_analyzer:
                        vader_scores = self.vader_analyzer.polarity_scores(processed_text)
                    
                    # Combinar resultados
                    final_polarity, final_subjectivity, confidence = self._combine_sentiment_scores(
                        polarity, subjectivity, vader_scores
                    )
                    
                    # Determinar etiqueta de sentimiento
                    sentiment_label = self._classify_sentiment_hybrid(
                        final_polarity, final_subjectivity, vader_scores
                    )
                    
                    # Generar contexto
                    context = self._generate_hybrid_context(
                        processed_text, sentiment_label, final_polarity, vader_scores
                    )
                    
                    # Crear citación
                    citation = self._create_sentiment_citation(
                        chunk, processed_text, sentiment_label, confidence
                    )
                    
                    sentiment_data = SentimentData(
                        text=processed_text,
                        polarity=final_polarity,
                        subjectivity=final_subjectivity,
                        compound_score=vader_scores['compound'] if vader_scores else final_polarity,
                        sentiment_label=sentiment_label,
                        confidence=confidence,
                        context=context,
                        citations=[citation]
                    )
                    
                    sentiments.append(sentiment_data)
            
            # Calcular estadísticas
            sentiment_distribution = self._calculate_sentiment_distribution(sentiments)
            average_polarity = np.mean([s.polarity for s in sentiments])
            average_subjectivity = np.mean([s.subjectivity for s in sentiments])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SentimentAnalysisResult(
                sentiments=sentiments,
                algorithm_used="Híbrido (VADER + TextBlob)",
                total_analyzed=len(sentiments),
                sentiment_distribution=sentiment_distribution,
                average_polarity=average_polarity,
                average_subjectivity=average_subjectivity,
                processing_time=processing_time,
                metadata={
                    'n_chunks': len(chunks),
                    'n_analyzed': len(sentiments),
                    'vader_available': self.vader_analyzer is not None
                },
                citations=self.citation_manager.export_citations()
            )
            
        except Exception as e:
            logger.error(f"Error en análisis híbrido: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocesar texto para análisis de sentimientos"""
        # Limpiar texto
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        
        # Remover stopwords
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        
        return ' '.join(words)
    
    def _classify_sentiment(self, compound_score: float, polarity: float) -> str:
        """Clasificar sentimiento basado en scores"""
        # Usar compound score de VADER como principal
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_sentiment_textblob(self, polarity: float, subjectivity: float) -> str:
        """Clasificar sentimiento usando TextBlob"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_sentiment_hybrid(self, polarity: float, subjectivity: float, vader_scores: Optional[Dict]) -> str:
        """Clasificar sentimiento usando enfoque híbrido"""
        if vader_scores:
            compound = vader_scores['compound']
            if compound >= 0.05:
                return 'positive'
            elif compound <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        else:
            return self._classify_sentiment_textblob(polarity, subjectivity)
    
    def _calculate_sentiment_confidence(self, vader_scores: Dict, polarity: float, subjectivity: float) -> float:
        """Calcular confianza en la clasificación de sentimiento"""
        # Combinar confianza de VADER y TextBlob
        vader_confidence = abs(vader_scores['compound'])
        textblob_confidence = abs(polarity)
        
        # Promedio ponderado
        confidence = (vader_confidence * 0.7 + textblob_confidence * 0.3)
        return min(confidence, 1.0)
    
    def _calculate_textblob_confidence(self, polarity: float, subjectivity: float) -> float:
        """Calcular confianza usando solo TextBlob"""
        # Confianza basada en polaridad y subjetividad
        polarity_confidence = abs(polarity)
        subjectivity_confidence = subjectivity
        
        # Combinar ambos factores
        confidence = (polarity_confidence * 0.6 + subjectivity_confidence * 0.4)
        return min(confidence, 1.0)
    
    def _combine_sentiment_scores(self, polarity: float, subjectivity: float, vader_scores: Optional[Dict]) -> Tuple[float, float, float]:
        """Combinar scores de diferentes algoritmos"""
        if vader_scores:
            # Promedio ponderado entre VADER y TextBlob
            final_polarity = (vader_scores['compound'] * 0.6 + polarity * 0.4)
            final_subjectivity = subjectivity  # TextBlob es mejor para subjetividad
            confidence = self._calculate_sentiment_confidence(vader_scores, polarity, subjectivity)
        else:
            final_polarity = polarity
            final_subjectivity = subjectivity
            confidence = self._calculate_textblob_confidence(polarity, subjectivity)
        
        return final_polarity, final_subjectivity, confidence
    
    def _generate_sentiment_context(self, text: str, sentiment_label: str, vader_scores: Dict) -> str:
        """Generar contexto para el sentimiento"""
        context_parts = []
        
        # Contexto basado en el sentimiento
        if sentiment_label == 'positive':
            context_parts.append("Contenido con tono positivo")
        elif sentiment_label == 'negative':
            context_parts.append("Contenido con tono negativo")
        else:
            context_parts.append("Contenido con tono neutral")
        
        # Contexto basado en scores
        if vader_scores['pos'] > vader_scores['neg']:
            context_parts.append("predominan elementos positivos")
        elif vader_scores['neg'] > vader_scores['pos']:
            context_parts.append("predominan elementos negativos")
        
        return ". ".join(context_parts) + "."
    
    def _generate_sentiment_context_textblob(self, text: str, sentiment_label: str, polarity: float, subjectivity: float) -> str:
        """Generar contexto usando TextBlob"""
        context_parts = []
        
        if sentiment_label == 'positive':
            context_parts.append("Contenido con tono positivo")
        elif sentiment_label == 'negative':
            context_parts.append("Contenido con tono negativo")
        else:
            context_parts.append("Contenido con tono neutral")
        
        if subjectivity > 0.5:
            context_parts.append("alto contenido subjetivo")
        else:
            context_parts.append("contenido más objetivo")
        
        return ". ".join(context_parts) + "."
    
    def _generate_hybrid_context(self, text: str, sentiment_label: str, polarity: float, vader_scores: Optional[Dict]) -> str:
        """Generar contexto usando enfoque híbrido"""
        if vader_scores:
            return self._generate_sentiment_context(text, sentiment_label, vader_scores)
        else:
            return self._generate_sentiment_context_textblob(text, sentiment_label, polarity, 0.5)
    
    def _calculate_sentiment_distribution(self, sentiments: List[SentimentData]) -> Dict[str, int]:
        """Calcular distribución de sentimientos"""
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for sentiment in sentiments:
            distribution[sentiment.sentiment_label] += 1
        
        return distribution
    
    def _create_sentiment_citation(self, chunk: Dict[str, Any], text: str, sentiment_label: str, confidence: float) -> Dict[str, Any]:
        """Crear citación para análisis de sentimientos"""
        try:
            citation = self.citation_manager.add_citation(
                source_file=chunk.get('metadata', {}).get('source_file', 'unknown'),
                chunk_id=chunk.get('metadata', {}).get('chunk_index', 0),
                content=text,
                context_before="",
                context_after="",
                relevance_score=confidence
            )
            return citation.to_dict()
        except Exception as e:
            logger.error(f"Error creando citación: {e}")
            return {
                'source_file': chunk.get('metadata', {}).get('source_file', 'unknown'),
                'chunk_id': chunk.get('metadata', {}).get('chunk_index', 0),
                'content': text[:100] + "...",
                'sentiment_label': sentiment_label,
                'confidence': confidence
            }
    
    def generate_sentiment_visualization(self, result: SentimentAnalysisResult) -> Dict[str, Any]:
        """Generar visualizaciones para el análisis de sentimientos"""
        try:
            # Preparar datos para visualización
            sentiments_data = []
            for sentiment in result.sentiments:
                sentiments_data.append({
                    'text': sentiment.text[:50] + "...",
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity,
                    'sentiment_label': sentiment.sentiment_label,
                    'confidence': sentiment.confidence
                })
            
            # Gráfico de distribución de sentimientos
            df = pd.DataFrame(sentiments_data)
            
            # Gráfico de barras de distribución
            fig_distribution = px.bar(
                x=list(result.sentiment_distribution.keys()),
                y=list(result.sentiment_distribution.values()),
                title='Distribución de Sentimientos',
                labels={'x': 'Sentimiento', 'y': 'Cantidad'},
                color=list(result.sentiment_distribution.keys()),
                color_discrete_map={
                    'positive': '#2ecc71',
                    'negative': '#e74c3c',
                    'neutral': '#95a5a6'
                }
            )
            
            # Gráfico de dispersión polaridad vs subjetividad
            fig_scatter = px.scatter(
                df,
                x='polarity',
                y='subjectivity',
                color='sentiment_label',
                title='Polaridad vs Subjetividad',
                labels={'polarity': 'Polaridad', 'subjectivity': 'Subjetividad'},
                color_discrete_map={
                    'positive': '#2ecc71',
                    'negative': '#e74c3c',
                    'neutral': '#95a5a6'
                }
            )
            
            # Gráfico de confianza
            fig_confidence = px.histogram(
                df,
                x='confidence',
                title='Distribución de Confianza',
                labels={'confidence': 'Confianza', 'count': 'Frecuencia'},
                nbins=20
            )
            
            return {
                'distribution_chart': fig_distribution,
                'scatter_chart': fig_scatter,
                'confidence_chart': fig_confidence,
                'sentiments_data': sentiments_data,
                'algorithm': result.algorithm_used,
                'total_analyzed': result.total_analyzed,
                'avg_polarity': result.average_polarity,
                'avg_subjectivity': result.average_subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {str(e)}")
            return {}
    
    def export_sentiment_analysis(self, result: SentimentAnalysisResult, format: str = 'json') -> Dict[str, Any]:
        """Exportar análisis de sentimientos"""
        try:
            export_data = {
                'metadata': {
                    'algorithm': result.algorithm_used,
                    'total_analyzed': result.total_analyzed,
                    'avg_polarity': result.average_polarity,
                    'avg_subjectivity': result.average_subjectivity,
                    'processing_time': result.processing_time,
                    'export_date': datetime.now().isoformat()
                },
                'sentiment_distribution': result.sentiment_distribution,
                'sentiments': []
            }
            
            for sentiment in result.sentiments:
                sentiment_data = {
                    'text': sentiment.text,
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity,
                    'compound_score': sentiment.compound_score,
                    'sentiment_label': sentiment.sentiment_label,
                    'confidence': sentiment.confidence,
                    'context': sentiment.context,
                    'citations_count': len(sentiment.citations)
                }
                export_data['sentiments'].append(sentiment_data)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exportando análisis: {str(e)}")
            return {}
