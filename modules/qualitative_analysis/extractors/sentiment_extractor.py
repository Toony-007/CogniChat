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
from utils.ollama_client import ollama_client

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
                    
                    if len(processed_text.strip()) < 5:  # Saltar textos muy cortos (reducido de 10 a 5)
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
            
            print(f"📊 VADER: {len(sentiments)} sentimientos analizados de {len(chunks)} chunks")
            
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
                    
                    if len(processed_text.strip()) < 5:  # Saltar textos muy cortos (reducido de 10 a 5)
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
            
            print(f"📊 TextBlob: {len(sentiments)} sentimientos analizados de {len(chunks)} chunks")
            
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
                    
                    if len(processed_text.strip()) < 5:  # Saltar textos muy cortos (reducido de 10 a 5)
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
    
    def analyze_sentiments_hybrid(self, chunks: List[Dict[str, Any]]) -> SentimentAnalysisResult:
        """
        Análisis híbrido de sentimientos: algoritmo + IA
        
        Este método combina múltiples algoritmos para identificar sentimientos
        y luego usa IA para refinarlos y generar análisis emocionales profundos.
        
        Args:
            chunks: Lista de chunks de documentos
            
        Returns:
            Resultado del análisis híbrido de sentimientos
        """
        print("😊 Iniciando análisis híbrido de sentimientos...")
        
        # FASE 1: Extracción inicial con algoritmos
        print(f"🔍 Fase 1: Extracción inicial con {self.config.sentiment_algorithm.upper()}...")
        
        if self.config.sentiment_algorithm == "vader":
            raw_result = self.analyze_sentiments_vader(chunks)
        elif self.config.sentiment_algorithm == "textblob":
            raw_result = self.analyze_sentiments_textblob(chunks)
        elif self.config.sentiment_algorithm == "hybrid":
            # Combinar múltiples algoritmos
            result_vader = self.analyze_sentiments_vader(chunks)
            result_textblob = self.analyze_sentiments_textblob(chunks)
            
            # Combinar resultados
            combined_sentiments = result_vader.sentiments + result_textblob.sentiments
            print(f"📊 Híbrido: {len(combined_sentiments)} sentimientos combinados (VADER: {len(result_vader.sentiments)}, TextBlob: {len(result_textblob.sentiments)})")
            
            raw_result = SentimentAnalysisResult(
                sentiments=combined_sentiments,
                algorithm_used='hybrid',
                total_analyzed=len(combined_sentiments),
                sentiment_distribution=self._calculate_combined_distribution(result_vader, result_textblob),
                average_polarity=(result_vader.average_polarity + result_textblob.average_polarity) / 2,
                average_subjectivity=(result_vader.average_subjectivity + result_textblob.average_subjectivity) / 2,
                processing_time=result_vader.processing_time + result_textblob.processing_time,
                metadata={
                    'vader_count': len(result_vader.sentiments),
                    'textblob_count': len(result_textblob.sentiments),
                    'combined_analysis': True
                },
                citations=[]
            )
        else:
            raise ValueError(f"Algoritmo desconocido: {self.config.sentiment_algorithm}")
        
        # FASE 2: Refinamiento con IA (si está habilitado)
        if self.config.enable_sentiment_refinement:
            print(f"🤖 Fase 2: Refinamiento con {self.config.llm_model}...")
            refined_result = self._refine_sentiments_with_llm(raw_result, chunks)
        else:
            print("⚠️ Refinamiento con IA deshabilitado, usando resultados algorítmicos")
            refined_result = raw_result
        
        print(f"✅ Análisis híbrido completado: {len(refined_result.sentiments)} sentimientos analizados")
        return refined_result
    
    def _calculate_combined_distribution(self, result_vader: SentimentAnalysisResult, result_textblob: SentimentAnalysisResult) -> Dict[str, int]:
        """Calcular distribución combinada de sentimientos"""
        combined_dist = {}
        
        # Combinar distribuciones
        for sentiment, count in result_vader.sentiment_distribution.items():
            combined_dist[sentiment] = combined_dist.get(sentiment, 0) + count
        
        for sentiment, count in result_textblob.sentiment_distribution.items():
            combined_dist[sentiment] = combined_dist.get(sentiment, 0) + count
        
        return combined_dist
    
    def _refine_sentiments_with_llm(
        self, 
        raw_result: SentimentAnalysisResult, 
        chunks: List[Dict[str, Any]]
    ) -> SentimentAnalysisResult:
        """
        Refinar sentimientos usando modelo LLM
        
        Args:
            raw_result: Resultado del análisis algorítmico
            chunks: Chunks originales para contexto
            
        Returns:
            Resultado refinado con IA
        """
        try:
            # Preparar datos para el LLM
            sentiments_data = []
            for sentiment in raw_result.sentiments[:50]:  # Máximo 50 sentimientos para análisis profundo (aumentado de 20)
                sentiments_data.append({
                    'text': sentiment.text[:200],  # Limitar texto para el prompt
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity,
                    'sentiment_label': sentiment.sentiment_label,
                    'confidence': sentiment.confidence
                })
            
            if not sentiments_data:
                print("⚠️ No hay sentimientos para refinar")
                return raw_result
            
            # Crear contexto del documento
            document_context = " ".join([chunk.get('content', '') for chunk in chunks[:5]])
            context_preview = document_context[:2000] + "..." if len(document_context) > 2000 else document_context
            
            # Crear prompt para el LLM
            prompt = self._create_sentiment_refinement_prompt(sentiments_data, context_preview)
            
            # Llamar al LLM
            response = ollama_client.generate_response(
                model=self.config.llm_model,
                prompt=prompt
            )
            
            # Parsear respuesta del LLM
            refined_sentiments = self._parse_llm_sentiment_response(response, raw_result.sentiments)
            
            print(f"📊 Refinamiento LLM: {len(refined_sentiments)} sentimientos refinados de {len(raw_result.sentiments)} originales")
            
            # Crear resultado refinado
            refined_result = SentimentAnalysisResult(
                sentiments=refined_sentiments,
                algorithm_used=f"{raw_result.algorithm_used}_refined",
                total_analyzed=len(refined_sentiments),
                sentiment_distribution=self._calculate_refined_distribution(refined_sentiments),
                average_polarity=sum(s.polarity for s in refined_sentiments) / len(refined_sentiments) if refined_sentiments else 0.0,
                average_subjectivity=sum(s.subjectivity for s in refined_sentiments) / len(refined_sentiments) if refined_sentiments else 0.0,
                processing_time=raw_result.processing_time,
                metadata={
                    **raw_result.metadata,
                    'llm_refinement': True,
                    'original_sentiments': len(raw_result.sentiments),
                    'refined_sentiments': len(refined_sentiments)
                },
                citations=[]
            )
            
            return refined_result
            
        except Exception as e:
            print(f"Error en refinamiento LLM: {e}")
            return raw_result
    
    def _calculate_refined_distribution(self, sentiments: List[SentimentData]) -> Dict[str, int]:
        """Calcular distribución de sentimientos refinados"""
        distribution = {}
        for sentiment in sentiments:
            label = sentiment.sentiment_label
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def _create_sentiment_refinement_prompt(self, sentiments_data: List[Dict], context_preview: str) -> str:
        """
        Crear prompt para refinamiento de sentimientos con LLM
        
        Args:
            sentiments_data: Datos de sentimientos a refinar
            context_preview: Contexto del documento
            
        Returns:
            Prompt formateado para el LLM
        """
        sentiments_text = ""
        for i, sent in enumerate(sentiments_data, 1):
            sentiments_text += f"{i}. Texto: \"{sent['text']}\" | Polaridad: {sent['polarity']:.3f} | Etiqueta: {sent['sentiment_label']}\n"
        
        return f"""Eres un experto en análisis cualitativo de sentimientos y emociones. Tu tarea es refinar y mejorar el análisis de sentimientos, generando interpretaciones emocionales profundas y académicas.

IMPORTANTE: Responde ÚNICAMENTE en español. Todas las interpretaciones y análisis deben estar en español.

CONTEXTO DEL DOCUMENTO:
{context_preview}

SENTIMIENTOS IDENTIFICADOS:
{sentiments_text}

DEFINICIÓN DE ANÁLISIS EMOCIONAL PROFUNDO:
Un análisis emocional profundo en investigación cualitativa va más allá de la simple clasificación positivo/negativo para identificar:
- Emociones específicas (alegría, tristeza, ira, miedo, sorpresa, asco)
- Intensidad emocional y matices
- Contexto emocional y significado cultural
- Patrones emocionales emergentes
- Relación entre emociones y contenido temático

INSTRUCCIONES ESPECÍFICAS:
1. ANALIZA cada sentimiento identificado en su contexto específico
2. REFINA las clasificaciones eliminando interpretaciones superficiales
3. GENERA interpretaciones emocionales profundas que expliquen "qué emociones" y "por qué"
4. IDENTIFICA emociones específicas más allá de positivo/negativo/neutral
5. PRIORIZA análisis que revelen patrones emocionales significativos
6. CADA interpretación debe ser ÚNICA y específica al contexto emocional
7. NO uses plantillas genéricas - cada análisis debe ser diferente

FORMATO DE RESPUESTA (JSON estricto):
{{
    "sentimientos_refinados": [
        {{
            "texto_original": "texto_analizado",
            "polaridad_refinada": -1.0 a 1.0,
            "subjetividad_refinada": 0.0 a 1.0,
            "etiqueta_emocional": "positivo|negativo|neutral",
            "emocion_especifica": "alegria|tristeza|ira|miedo|sorpresa|asco|confianza|anticipacion",
            "intensidad_emocional": 0.0 a 1.0,
            "interpretacion_emocional": "Interpretación detallada y ÚNICA de las emociones presentes y su significado contextual.",
            "patron_emocional": "Descripción del patrón emocional que emerge de este análisis",
            "relevancia_investigacion": "Por qué este análisis emocional es importante para la investigación"
        }}
    ]
}}

CRITERIOS DE CALIDAD:
- Los análisis deben ser emocionalmente específicos y no genéricos
- Las interpretaciones deben ser únicas y contextuales
- Deben revelar patrones emocionales profundos
- Deben ser útiles para la comprensión del fenómeno estudiado
- NO deben ser análisis superficiales o obvios
- Cada interpretación debe aportar valor emocional específico

EJEMPLOS DE ANÁLISIS VÁLIDOS:
✅ "El texto expresa una alegría contenida mezclada con nostalgia" (emocionalmente específico)
✅ "Se detecta una ira justificada hacia la injusticia social" (contextual y específico)
✅ "La confianza se mezcla con incertidumbre sobre el futuro" (emocionalmente complejo)

EJEMPLOS DE ANÁLISIS INVÁLIDOS:
❌ "El texto es positivo" (genérico)
❌ "Se siente tristeza" (superficial)
❌ "Es neutral" (obvio)

REGLAS CRÍTICAS:
- Cada interpretación debe ser completamente diferente
- Las interpretaciones deben ser específicas al contexto emocional del documento
- NO uses frases genéricas o plantillas repetidas
- Si un sentimiento no es emocionalmente significativo, es mejor excluirlo
- Prioriza calidad emocional sobre cantidad

IMPORTANTE:
- Responde SOLO con JSON válido
- Máximo {self.config.max_sentiment_samples} sentimientos refinados
- Cada sentimiento debe tener una interpretación emocional única y específica
- TODO debe estar en español
- Si no encuentras sentimientos emocionalmente significativos, es mejor devolver menos análisis pero de mayor calidad"""

    def _parse_llm_sentiment_response(self, response: str, original_sentiments: List[SentimentData]) -> List[SentimentData]:
        """
        Parsear respuesta del LLM para sentimientos refinados
        
        Args:
            response: Respuesta del LLM
            original_sentiments: Sentimientos originales para preservar metadatos
            
        Returns:
            Lista de sentimientos refinados
        """
        import json
        import re
        
        try:
            # Limpiar caracteres de control inválidos
            cleaned_response = self._clean_json_response(response)
            
            # Buscar JSON en la respuesta
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No se encontró JSON válido en la respuesta")
            
            json_str = cleaned_response[json_start:json_end]
            
            # Verificar si el JSON está completo
            if not self._is_json_complete(json_str):
                print("⚠️ JSON parece estar truncado, intentando reparar...")
                json_str = self._repair_truncated_json(json_str)
            
            # Intentar parsear JSON con manejo de errores mejorado
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ Error JSON inicial: {e}")
                print(f"📄 JSON problemático: {json_str[:500]}...")
                
                # Si falla, intentar limpiar más agresivamente
                json_str = self._aggressive_json_clean(json_str)
                print(f"🔧 JSON después de limpieza: {json_str[:500]}...")
                
                try:
                    data = json.loads(json_str)
                    print("✅ JSON parseado exitosamente después de limpieza")
                except json.JSONDecodeError as e2:
                    print(f"❌ Error JSON persistente: {e2}")
                    print(f"📄 JSON final: {json_str}")
                    
                    # Último intento: reparación manual básica
                    json_str = self._manual_json_repair(json_str)
                    try:
                        data = json.loads(json_str)
                        print("✅ JSON reparado manualmente")
                    except json.JSONDecodeError as e3:
                        print(f"❌ Error JSON final: {e3}")
                        raise ValueError(f"No se pudo parsear JSON después de múltiples intentos: {e3}")
            
            refined_sentiments = []
            
            for item in data.get('sentimientos_refinados', []):
                # Buscar sentimiento original más similar para preservar metadatos
                original_sentiment = self._find_most_similar_original_sentiment(
                    item['texto_original'], original_sentiments
                )
                
                # Crear sentimiento refinado
                refined_sentiment = SentimentData(
                    text=item['texto_original'],
                    polarity=float(item['polaridad_refinada']),
                    subjectivity=float(item['subjetividad_refinada']),
                    compound_score=original_sentiment.compound_score if original_sentiment else 0.0,
                    sentiment_label=item['etiqueta_emocional'],
                    confidence=original_sentiment.confidence if original_sentiment else 0.5,
                    context=item.get('interpretacion_emocional', ''),
                    citations=original_sentiment.citations if original_sentiment else []
                )
                
                # Agregar metadatos adicionales
                if not hasattr(refined_sentiment, 'metadata'):
                    refined_sentiment.metadata = {}
                
                refined_sentiment.metadata.update({
                    'emocion_especifica': item.get('emocion_especifica', ''),
                    'intensidad_emocional': float(item.get('intensidad_emocional', 0.5)),
                    'patron_emocional': item.get('patron_emocional', ''),
                    'relevancia_investigacion': item.get('relevancia_investigacion', ''),
                    'llm_refined': True,
                    'original_polarity': original_sentiment.polarity if original_sentiment else 0.0
                })
                
                refined_sentiments.append(refined_sentiment)
            
            return refined_sentiments
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parseando respuesta LLM: {e}")
            return original_sentiments[:self.config.max_sentiment_samples]
    
    def _clean_json_response(self, response: str) -> str:
        """Limpiar caracteres de control inválidos de la respuesta del LLM"""
        import re
        
        # Remover caracteres de control excepto \n, \r, \t
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', response)
        
        # Normalizar saltos de línea
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remover espacios múltiples
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # Limpiar caracteres especiales problemáticos
        cleaned = cleaned.replace('"', '"').replace('"', '"')  # Comillas tipográficas
        cleaned = cleaned.replace(''', "'").replace(''', "'")  # Apostrofes tipográficos
        cleaned = cleaned.replace('…', '...')  # Puntos suspensivos
        cleaned = cleaned.replace('–', '-').replace('—', '-')  # Guiones tipográficos
        
        # Remover caracteres Unicode problemáticos que pueden causar problemas
        cleaned = re.sub(r'[^\x20-\x7E\n\t]', '', cleaned)
        
        return cleaned
    
    def _is_json_complete(self, json_str: str) -> bool:
        """Verificar si el JSON está completo"""
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        return open_braces == close_braces and json_str.strip().endswith('}')
    
    def _repair_truncated_json(self, json_str: str) -> str:
        """Reparar JSON truncado"""
        if not json_str.strip().endswith('}'):
            json_str = json_str.rstrip(',') + '\n}'
        return json_str
    
    def _aggressive_json_clean(self, json_str: str) -> str:
        """Limpieza agresiva de JSON"""
        import re
        
        print(f"🔧 Limpieza agresiva de JSON...")
        print(f"📄 JSON original (últimos 200 chars): ...{json_str[-200:]}")
        
        # Paso 1: Limpiar caracteres problemáticos específicos
        json_str = json_str.replace('"', '"').replace('"', '"')  # Comillas tipográficas
        json_str = json_str.replace(''', "'").replace(''', "'")  # Apostrofes tipográficos
        json_str = json_str.replace('…', '...')  # Puntos suspensivos
        json_str = json_str.replace('–', '-').replace('—', '-')  # Guiones tipográficos
        
        # Paso 2: Escapar caracteres problemáticos en strings
        def escape_string(match):
            string_content = match.group(1)
            # Escapar caracteres problemáticos
            string_content = string_content.replace('\\', '\\\\')
            string_content = string_content.replace('"', '\\"')
            string_content = string_content.replace('\n', '\\n')
            string_content = string_content.replace('\t', '\\t')
            string_content = string_content.replace('\r', '\\r')
            # Remover caracteres Unicode problemáticos
            string_content = re.sub(r'[^\x20-\x7E]', '', string_content)
            return f'"{string_content}"'
        
        # Aplicar escape a strings
        json_str = re.sub(r'"([^"]*)"', escape_string, json_str)
        
        # Paso 3: Corregir comas faltantes antes de llaves de cierre
        json_str = re.sub(r'"\s*}', '",\n}', json_str)
        json_str = re.sub(r'"\s*]', '",\n]', json_str)
        
        # Paso 4: Corregir comas faltantes entre elementos del array
        json_str = re.sub(r'}\s*{', '},\n{', json_str)
        
        # Paso 5: Limpiar comas duplicadas
        json_str = re.sub(r',\s*,', ',', json_str)
        
        # Paso 6: Remover comas antes de llaves de cierre
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Paso 7: Remover caracteres de control restantes
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', json_str)
        
        print(f"📄 JSON después de limpieza agresiva (últimos 200 chars): ...{json_str[-200:]}")
        return json_str
    
    def _is_json_complete(self, json_str: str) -> bool:
        """
        Verificar si el JSON está completo (no truncado)
        
        Args:
            json_str: String JSON a verificar
            
        Returns:
            True si el JSON parece completo, False si está truncado
        """
        # Contar llaves de apertura y cierre
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        
        # Contar corchetes de apertura y cierre
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Verificar que las llaves estén balanceadas
        braces_balanced = open_braces == close_braces
        brackets_balanced = open_brackets == close_brackets
        
        # Verificar que termine con } o ]
        ends_properly = json_str.strip().endswith('}') or json_str.strip().endswith(']')
        
        return braces_balanced and brackets_balanced and ends_properly
    
    def _repair_truncated_json(self, json_str: str) -> str:
        """
        Reparar JSON truncado agregando elementos faltantes
        
        Args:
            json_str: JSON truncado a reparar
            
        Returns:
            JSON reparado
        """
        import re
        
        print(f"🔧 Reparando JSON truncado...")
        print(f"📄 JSON original (últimos 200 chars): ...{json_str[-200:]}")
        
        # Si el JSON está incompleto, intentar cerrarlo apropiadamente
        if not json_str.strip().endswith('}'):
            # Buscar el último objeto completo
            # Encontrar el último } completo
            last_complete_brace = json_str.rfind('},')
            if last_complete_brace != -1:
                # Cortar hasta el último objeto completo y cerrar el array y objeto principal
                json_str = json_str[:last_complete_brace + 1] + '\n    ]\n}'
                print(f"✅ JSON reparado cortando en último objeto completo")
            else:
                # Si no hay objetos completos, cerrar lo que hay
                # Buscar el último objeto incompleto y cerrarlo
                last_open_brace = json_str.rfind('{')
                if last_open_brace != -1:
                    # Encontrar el último objeto incompleto
                    incomplete_part = json_str[last_open_brace:]
                    # Cerrar el objeto incompleto
                    if not incomplete_part.strip().endswith('}'):
                        # Agregar comas faltantes y cerrar
                        incomplete_part = incomplete_part.rstrip(',') + '}'
                        json_str = json_str[:last_open_brace] + incomplete_part + '\n    ]\n}'
                        print(f"✅ JSON reparado cerrando objeto incompleto")
                    else:
                        json_str = json_str.rstrip(',') + '\n    ]\n}'
                        print(f"✅ JSON reparado con cierre básico")
                else:
                    json_str = json_str.rstrip(',') + '\n    ]\n}'
                    print(f"✅ JSON reparado con cierre básico")
        
        # Limpiar comas duplicadas
        json_str = re.sub(r',\s*,', ',', json_str)
        
        # Remover comas antes de llaves de cierre
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        print(f"📄 JSON reparado (últimos 200 chars): ...{json_str[-200:]}")
        return json_str
    
    def _manual_json_repair(self, json_str: str) -> str:
        """
        Reparación manual básica de JSON para casos extremos
        
        Args:
            json_str: String JSON a reparar
            
        Returns:
            JSON reparado
        """
        import re
        
        print(f"🔧 Reparación manual de JSON...")
        print(f"📄 JSON problemático (últimos 300 chars): ...{json_str[-300:]}")
        
        # Paso 1: Limpiar caracteres problemáticos específicos
        json_str = json_str.replace('"', '"').replace('"', '"')  # Comillas tipográficas
        json_str = json_str.replace(''', "'").replace(''', "'")  # Apostrofes tipográficos
        json_str = json_str.replace('…', '...')  # Puntos suspensivos
        json_str = json_str.replace('–', '-').replace('—', '-')  # Guiones tipográficos
        
        # Paso 2: Reparaciones básicas comunes
        # 1. Agregar comas faltantes antes de llaves de cierre
        json_str = re.sub(r'"\s*}\s*', '",\n}', json_str)
        json_str = re.sub(r'"\s*]\s*', '",\n]', json_str)
        
        # 2. Corregir comas faltantes entre elementos del array
        json_str = re.sub(r'}\s*{', '},\n{', json_str)
        
        # 3. Limpiar comas duplicadas
        json_str = re.sub(r',\s*,', ',', json_str)
        
        # 4. Remover comas antes de llaves de cierre
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Paso 3: Reparación específica para casos problemáticos
        # Buscar patrones específicos que causan problemas
        json_str = re.sub(r'"([^"]*)"([^,}\]]*)([}\]])', r'"\1"\2,\n\3', json_str)
        
        # Paso 4: Asegurar que el JSON esté bien formado
        if not json_str.strip().startswith('{'):
            json_str = '{' + json_str
        if not json_str.strip().endswith('}'):
            json_str = json_str + '}'
        
        # Paso 5: Reparación específica para el caso del usuario
        # Buscar objetos incompletos y cerrarlos
        lines = json_str.split('\n')
        repaired_lines = []
        in_object = False
        brace_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Contar llaves
            brace_count += line.count('{') - line.count('}')
            
            if '{' in line and not in_object:
                in_object = True
                brace_count = 1
            
            if in_object:
                # Si la línea no termina con } o }, agregar coma si es necesario
                if line.endswith('"') and not line.endswith('",') and not line.endswith('"}') and not line.endswith('"]'):
                    line = line + ','
                
                repaired_lines.append(line)
                
                # Si cerramos el objeto
                if brace_count == 0:
                    in_object = False
        
        json_str = '\n'.join(repaired_lines)
        
        # Paso 6: Limpieza final
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        print(f"📄 JSON reparado manualmente (últimos 200 chars): ...{json_str[-200:]}")
        return json_str
    
    def _find_most_similar_original_sentiment(
        self, 
        text: str, 
        original_sentiments: List[SentimentData]
    ) -> Optional[SentimentData]:
        """Encontrar el sentimiento original más similar"""
        if not original_sentiments:
            return None
        
        text_lower = text.lower()
        
        for original in original_sentiments:
            if text_lower in original.text.lower() or original.text.lower() in text_lower:
                return original
        
        return original_sentiments[0] if original_sentiments else None
    
    def get_sentiment_summary(self, result: SentimentAnalysisResult) -> Dict[str, Any]:
        """
        Generar resumen estadístico de los sentimientos analizados
        
        Args:
            result: Resultado del análisis de sentimientos
            
        Returns:
            Diccionario con estadísticas
        """
        if not result.sentiments:
            return {
                'total_sentiments': 0,
                'avg_polarity': 0.0,
                'avg_subjectivity': 0.0,
                'sentiment_distribution': {},
                'unique_sources': 0,
                'total_citations': 0
            }
        
        avg_polarity = sum(s.polarity for s in result.sentiments) / len(result.sentiments)
        avg_subjectivity = sum(s.subjectivity for s in result.sentiments) / len(result.sentiments)
        
        all_sources = set()
        total_citations = 0
        
        for sentiment in result.sentiments:
            for citation in sentiment.citations:
                if isinstance(citation, dict):
                    all_sources.add(citation.get('source_file', 'unknown'))
                else:
                    all_sources.add(citation.source_file)
            total_citations += len(sentiment.citations)
        
        return {
            'total_sentiments': len(result.sentiments),
            'avg_polarity': avg_polarity,
            'avg_subjectivity': avg_subjectivity,
            'sentiment_distribution': result.sentiment_distribution,
            'unique_sources': len(all_sources),
            'total_citations': total_citations,
            'top_sentiment': result.sentiments[0] if result.sentiments else None,
            'algorithm_used': result.algorithm_used,
            'metadata': result.metadata
        }
    
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
