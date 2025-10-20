"""
Módulo de Análisis Cualitativo Avanzado
Análisis profundo de contenido RAG con técnicas de NLP y visualizaciones interactivas

ORGANIZACIÓN DEL CÓDIGO POR FUNCIONALIDADES:
================================================================================
1. IMPORTS Y CONFIGURACIÓN GLOBAL
2. ENUMS, DATACLASSES Y ESTRUCTURAS DE DATOS
3. CLASES BASE Y INTERFACES
4. GESTIÓN DE CACHE Y MEMORIA
5. PREPROCESAMIENTO DE TEXTO
6. EXTRACCIÓN DE CONCEPTOS
7. ANÁLISIS DE TEMAS
8. ANÁLISIS DE SENTIMIENTOS
9. CLUSTERING Y AGRUPACIÓN
10. MAPAS CONCEPTUALES INTERACTIVOS
11. MAPAS MENTALES
12. RESUMENES AUTOMÁTICOS
13. ANÁLISIS DE TRIANGULACIÓN
14. NUBES DE PALABRAS
15. VISUALIZACIONES Y GRÁFICOS
16. ANÁLISIS PARALELO Y OPTIMIZACIÓN
17. MÉTODOS DE CONFIGURACIÓN
18. FUNCIONES DE RENDERIZADO (STREAMLIT)
19. FUNCIÓN PRINCIPAL DE RENDERIZADO
================================================================================

Cada sección está claramente marcada y separada para fácil localización y modificación.

ÍNDICE DE NAVEGACIÓN RÁPIDA:
================================================================================
Para encontrar y modificar cualquier funcionalidad, busca estos separadores:

🔧 CONFIGURACIÓN Y ESTRUCTURAS:
   - Línea 145:  # 2. ENUMS, DATACLASSES Y ESTRUCTURAS DE DATOS
   - Línea 199:  # 3. CLASES BASE Y INTERFACES
   - Línea 223:  # 4. GESTIÓN DE CACHE Y MEMORIA
   - Línea 1711: # 17. MÉTODOS DE CONFIGURACIÓN

📝 PROCESAMIENTO DE TEXTO:
   - Línea 276:  # 5. PREPROCESAMIENTO DE TEXTO

🔍 ANÁLISIS DE CONTENIDO:
   - Línea 346:  # 6. EXTRACCIÓN DE CONCEPTOS
   - Línea 533:  # 7. ANÁLISIS DE TEMAS
   - Línea 749:  # 8. ANÁLISIS DE SENTIMIENTOS
   - Línea 3637: # 9. CLUSTERING Y AGRUPACIÓN
   - Línea 2320: # 13. ANÁLISIS DE TRIANGULACIÓN

🗺️ VISUALIZACIONES INTERACTIVAS:
   - Línea 1918: # 10. MAPAS CONCEPTUALES INTERACTIVOS
   - Línea 2214: # 11. MAPAS MENTALES
   - Línea 4363: # 15. VISUALIZACIONES Y GRÁFICOS

📄 RESUMENES Y REPORTES:
   - Línea 1118: # 12. RESUMENES AUTOMÁTICOS
   - Línea 3819: # 14. NUBES DE PALABRAS

⚡ OPTIMIZACIÓN Y RENDIMIENTO:
   - Línea 1589: # 16. ANÁLISIS PARALELO Y OPTIMIZACIÓN

🖥️ INTERFAZ DE USUARIO:
   - Línea 5890: # 19. FUNCIÓN PRINCIPAL DE RENDERIZADO

FUNCIONES CLAVE POR SECCIÓN:
================================================================================
🔍 EXTRACCIÓN DE CONCEPTOS (Línea 346):
   - extract_key_concepts()          → Extraer conceptos clave
   - _extract_concepts_legacy()      → Método legacy de respaldo
   - _extract_concepts_with_ngrams() → Extracción inteligente con n-gramas (NUEVO)
   - _identify_intelligent_main_theme() → Tema central inteligente (NUEVO)
   - ConceptExtractor.analyze()      → Análisis modular

🎯 ANÁLISIS DE TEMAS (Línea 533):
   - extract_advanced_themes()       → Análisis LDA avanzado
   - _extract_advanced_themes_detailed() → Método legacy detallado
   - ThemeAnalyzer.analyze()         → Análisis modular

😊 ANÁLISIS DE SENTIMIENTOS (Línea 749):
   - advanced_sentiment_analysis()   → Análisis con VADER/TextBlob
   - _advanced_sentiment_analysis_detailed() → Método legacy detallado
   - SentimentAnalyzer.analyze()     → Análisis modular

🗺️ MAPAS CONCEPTUALES (Línea 1918):
   - create_interactive_concept_map() → Mapa con PyVis (MEJORADO: n-gramas, mejor separación)
   - _analyze_concept_hierarchy()    → Jerarquía de conceptos (MEJORADO: n-gramas)
   - _analyze_concept_hierarchy_with_ai() → Análisis con IA (NUEVO)
   - _extract_concepts_with_ngrams() → Extracción inteligente (NUEVO)
   - _identify_intelligent_main_theme() → Tema central inteligente (NUEVO)
   - generate_advanced_concept_map() → Mapa avanzado

🧠 MAPAS MENTALES (Línea 2214):
   - create_interactive_mind_map()   → Mapa con streamlit-agraph (MEJORADO: texto legible, contenedor completo)
   - _analyze_intelligent_mind_map_structure() → Estructura inteligente (MEJORADO: n-gramas)
   - _extract_concepts_with_ngrams() → Extracción inteligente (COMPARTIDO)
   - _identify_intelligent_main_theme() → Tema central inteligente (COMPARTIDO)

📝 RESUMENES (Línea 1118):
   - generate_intelligent_summary()  → Resumen con LLM
   - generate_rag_summary()          → Resumen básico
   - generate_basic_summary()        → Resumen por frecuencia

🔺 TRIANGULACIÓN (Línea 2320):
   - perform_triangulation_analysis() → Validación multi-fuente (MEJORADO: soporte una fuente)
   - _perform_single_source_triangulation() → Triangulación interna (NUEVO)

☁️ NUBES DE PALABRAS (Línea 3819):
   - generate_word_cloud()           → Generación de nube

🔍 CLUSTERING (Línea 3637):
   - perform_clustering()            → Agrupación K-means/DBSCAN

⚡ OPTIMIZACIÓN (Línea 1589):
   - perform_parallel_analysis()     → Análisis paralelo
   - optimize_performance()          → Optimización automática
   - get_performance_metrics()       → Métricas del sistema

================================================================================

MEJORAS IMPLEMENTADAS (Última actualización):
================================================================================
✅ MAPAS CONCEPTUALES:
   - Extracción inteligente con n-gramas (frases completas en lugar de palabras sueltas)
   - Modo normal por defecto (3-5x más rápido que IA)
   - Mejor separación visual entre nodos
   - Análisis con IA como opción avanzada
   - Identificación más coherente del tema central

✅ MAPAS MENTALES:
   - Texto blanco/gris claro para legibilidad en fondo oscuro
   - Contenedor más grande (1400x700px) para cubrir pantalla completa
   - Espaciado mejorado entre nodos (450px por defecto, hasta 800px)
   - Física más suave para evitar nodos "pegados"
   - Extracción inteligente con n-gramas
   - Modo normal por defecto

✅ TRIANGULACIÓN:
   - Soporte para una sola fuente (triangulación interna)
   - Análisis por secciones del mismo documento
   - Validación cruzada mejorada

✅ ARQUITECTURA:
   - Clases especializadas (ConceptExtractor, ThemeAnalyzer, SentimentAnalyzer)
   - Sistema de cache optimizado
   - Procesamiento paralelo
   - Configuración dinámica

================================================================================
"""

# =============================================================================
# 1. IMPORTS Y CONFIGURACIÓN GLOBAL
# =============================================================================

import streamlit as st
import json
import os
import re
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import sys
import tempfile
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Análisis de texto avanzado
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import LatentDirichletAllocation, PCA, TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE, MDS
    from sklearn.preprocessing import StandardScaler
    import nltk
    from textblob import TextBlob
    from wordcloud import WordCloud
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False
    st.warning("⚠️ Algunas funcionalidades avanzadas no están disponibles. Instala las dependencias adicionales.")

# Mapas conceptuales interactivos
try:
    from pyvis.network import Network
    import streamlit.components.v1 as components
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from utils.rag_processor import RAGProcessor
from utils.logger import setup_logger
from config.settings import config

logger = setup_logger()

# =============================================================================
# 2. ENUMS, DATACLASSES Y ESTRUCTURAS DE DATOS
# =============================================================================
# En esta sección se definen los tipos de datos y estructuras fundamentales
# que usa todo el módulo. Modificar aquí si necesitas:
# - Agregar nuevos tipos de análisis
# - Cambiar la configuración por defecto
# - Modificar estructuras de datos

class AnalysisType(Enum):
    """Tipos de análisis disponibles"""
    CONCEPT_EXTRACTION = "concept_extraction"
    THEME_ANALYSIS = "theme_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CLUSTERING = "clustering"
    CONCEPT_MAP = "concept_map"
    MIND_MAP = "mind_map"
    SUMMARY = "summary"
    TRIANGULATION = "triangulation"

class VisualizationType(Enum):
    """Tipos de visualización disponibles"""
    INTERACTIVE_NETWORK = "interactive_network"
    STATIC_GRAPH = "static_graph"
    HIERARCHICAL_TREE = "hierarchical_tree"
    FORCE_DIRECTED = "force_directed"
    CIRCULAR = "circular"

@dataclass
class AnalysisConfig:
    """Configuración para análisis cualitativo"""
    min_frequency: int = 2
    max_concepts: int = 50
    similarity_threshold: float = 0.6
    chunk_size: int = 2000
    enable_cache: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
@dataclass
class ConceptData:
    """Datos de un concepto extraído"""
    concept: str
    score: float
    frequency: int
    context: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    sentiment: Optional[float] = None
    category: Optional[str] = None

@dataclass
class AnalysisResult:
    """Resultado de un análisis"""
    analysis_type: AnalysisType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0

# =============================================================================
# 3. CLASES BASE Y INTERFACES
# =============================================================================

class BaseAnalyzer(ABC):
    """Clase base para todos los analizadores"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.cache = {}
        self.logger = logger
    
    @abstractmethod
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """Realizar análisis específico"""
        pass
    
    def _validate_input(self, chunks: List[Dict]) -> bool:
        """Validar entrada de datos"""
        if not chunks or not isinstance(chunks, list):
            return False
        return any(isinstance(chunk, dict) and chunk.get('content') 
                  for chunk in chunks)

# =============================================================================
# 4. GESTIÓN DE CACHE Y MEMORIA
# =============================================================================

class CacheManager:
    """Gestor de cache optimizado para análisis cualitativo"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener elemento del cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = datetime.now()
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Guardar elemento en cache"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            self.cache[key] = value
            self.access_times[key] = datetime.now()
    
    def _evict_oldest(self) -> None:
        """Eliminar el elemento más antiguo"""
        if not self.access_times:
            return
        oldest_key = min(self.access_times.keys(), 
                        key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self) -> None:
        """Limpiar cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_ratio': getattr(self, '_hit_ratio', 0.0)
            }

# =============================================================================
# 5. PREPROCESAMIENTO DE TEXTO
# =============================================================================

class TextPreprocessor:
    """Preprocesador de texto especializado"""
    
    def __init__(self):
        self.stopwords_cache = None
        self.logger = logger
    
    def get_spanish_stopwords(self) -> List[str]:
        """Obtener stopwords en español con cache"""
        if self.stopwords_cache is None:
            try:
                if ADVANCED_ANALYSIS_AVAILABLE:
                    from nltk.corpus import stopwords
                    spanish_stopwords = set(stopwords.words('spanish'))
                    # Agregar stopwords adicionales específicas del dominio
                    additional_stopwords = {
                        'también', 'puede', 'ser', 'está', 'están', 'hacer', 'hace',
                        'tiene', 'tienen', 'dice', 'dice', 'muy', 'más', 'menos',
                        'bien', 'mal', 'bueno', 'malo', 'nuevo', 'viejo', 'grande',
                        'pequeño', 'mucho', 'poco', 'todo', 'nada', 'algo', 'alguno',
                        'algunos', 'cada', 'cualquier', 'mismo', 'misma', 'otros',
                        'otras', 'varios', 'varias', 'diferentes', 'importante',
                        'necesario', 'posible', 'imposible', 'general', 'específico'
                    }
                    spanish_stopwords.update(additional_stopwords)
                    self.stopwords_cache = list(spanish_stopwords)
                else:
                    # Fallback básico
                    self.stopwords_cache = [
                        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se',
                        'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con',
                        'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más',
                        'pero', 'sus', 'todo', 'esta', 'sobre', 'entre', 'cuando',
                        'muy', 'sin', 'hasta', 'desde', 'está', 'mi', 'porque',
                        'qué', 'sólo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos',
                        'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde',
                        'bien', 'tiempo', 'mismo', 'ahora', 'cada', 'e', 'vida',
                        'otro', 'después', 'te', 'otros', 'me', 'esas', 'le', 'suya',
                        'misma', 'yo', 'también', 'hasta', 'año', 'dos', 'bajo',
                        'arriba', 'encima', 'debajo', 'dentro', 'fuera', 'aquí',
                        'allí', 'donde', 'cuando', 'como', 'porque', 'aunque',
                        'mientras', 'antes', 'después', 'durante', 'hasta', 'desde'
                    ]
            except Exception as e:
                self.logger.warning(f"Error cargando stopwords: {e}")
                self.stopwords_cache = []
        
        return self.stopwords_cache
    
    def preprocess_text(self, text: str) -> str:
        """Preprocesar texto para análisis"""
        if not text:
            return ""
        
        # Limpiar texto
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remover stopwords
        words = text.split()
        stopwords = self.get_spanish_stopwords()
        filtered_words = [word for word in words 
                         if word not in stopwords and len(word) > 2]
        
        return ' '.join(filtered_words)

# =============================================================================
# 6. EXTRACCIÓN DE CONCEPTOS
# =============================================================================
# Esta sección maneja la extracción de conceptos clave del contenido.
# MODIFICAR AQUÍ PARA:
# - Cambiar algoritmos de extracción (TF-IDF, frecuencia, etc.)
# - Ajustar umbrales de relevancia
# - Mejorar la detección de conceptos relacionados
# 
# MÉTODOS PRINCIPALES:
# - extract_key_concepts() → Punto de entrada principal
# - ConceptExtractor.analyze() → Análisis modular
# - _extract_concepts_legacy() → Fallback compatible

class ConceptExtractor(BaseAnalyzer):
    """Extractor de conceptos especializado"""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.preprocessor = TextPreprocessor()
        self.cache_manager = CacheManager()
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """Extraer conceptos clave del contenido"""
        start_time = datetime.now()
        
        if not self._validate_input(chunks):
            return AnalysisResult(
                analysis_type=AnalysisType.CONCEPT_EXTRACTION,
                data={'concepts': []},
                metadata={'error': 'Invalid input data'}
            )
        
        # Generar clave de cache
        cache_key = self._generate_cache_key(chunks)
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result and self.config.enable_cache:
            return cached_result
        
        # Procesar chunks
        concepts = self._extract_concepts_advanced(chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = AnalysisResult(
            analysis_type=AnalysisType.CONCEPT_EXTRACTION,
            data={'concepts': concepts},
            metadata={
                'total_chunks': len(chunks),
                'concepts_found': len(concepts),
                'processing_method': 'advanced_tfidf' if ADVANCED_ANALYSIS_AVAILABLE else 'frequency_based'
            },
            processing_time=processing_time
        )
        
        if self.config.enable_cache:
            self.cache_manager.set(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, chunks: List[Dict]) -> str:
        """Generar clave única para cache"""
        content_hash = hash(str(sorted([chunk.get('content', '')[:100] for chunk in chunks])))
        return f"concepts_{content_hash}_{self.config.min_frequency}_{self.config.max_concepts}"
    
    def _extract_concepts_advanced(self, chunks: List[Dict]) -> List[ConceptData]:
        """Extraer conceptos usando métodos avanzados"""
        # Preparar textos
        chunk_texts = []
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            if content:
                processed = self.preprocessor.preprocess_text(content)
                if processed and len(processed.split()) > 5:
                    chunk_texts.append(processed)
        
        if not chunk_texts:
            return []
        
        concepts = []
        
        # Método avanzado con TF-IDF
        if ADVANCED_ANALYSIS_AVAILABLE and len(chunk_texts) >= 2:
            concepts = self._extract_with_tfidf(chunk_texts)
        
        # Fallback con análisis de frecuencia
        if not concepts:
            concepts = self._extract_with_frequency(chunk_texts)
        
        # Enriquecer conceptos con contexto
        concepts = self._enrich_concepts_with_context(concepts, chunks)
        
        return concepts[:self.config.max_concepts]
    
    def _extract_with_tfidf(self, texts: List[str]) -> List[ConceptData]:
        """Extraer conceptos usando TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words=self.preprocessor.get_spanish_stopwords(),
                ngram_range=(1, 3),
                min_df=max(1, len(texts) // 10),
                max_df=0.9
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calcular puntuaciones promedio
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            concepts = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    concept = ConceptData(
                        concept=feature_names[i],
                        score=float(score),
                        frequency=int(score * len(texts) * 10)  # Aproximación
                    )
                    concepts.append(concept)
            
            return sorted(concepts, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error en TF-IDF: {e}")
            return []
    
    def _extract_with_frequency(self, texts: List[str]) -> List[ConceptData]:
        """Extraer conceptos usando análisis de frecuencia"""
        all_text = " ".join(texts)
        words = all_text.split()
        word_freq = Counter(words)
        
        total_words = len(words)
        concepts = []
        
        for word, freq in word_freq.most_common(100):
            if (len(word) > 3 and 
                word not in self.preprocessor.get_spanish_stopwords() and 
                freq >= self.config.min_frequency and
                word.isalpha()):
                
                concept = ConceptData(
                    concept=word,
                    score=freq / total_words,
                    frequency=freq
                )
                concepts.append(concept)
        
        return concepts
    
    def _enrich_concepts_with_context(self, concepts: List[ConceptData], chunks: List[Dict]) -> List[ConceptData]:
        """Enriquecer conceptos con contexto y relaciones"""
        for concept in concepts:
            # Buscar contexto
            concept.context = self._find_concept_context(concept.concept, chunks)
            # Buscar conceptos relacionados
            concept.related_concepts = self._find_related_concepts(concept.concept, concepts)
        
        return concepts
    
    def _find_concept_context(self, concept: str, chunks: List[Dict], max_contexts: int = 3) -> List[str]:
        """Encontrar contexto para un concepto"""
        contexts = []
        for chunk in chunks:
            content = chunk.get('content', '')
            if concept.lower() in content.lower():
                # Extraer oración que contiene el concepto
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    if concept.lower() in sentence.lower() and len(sentence.strip()) > 20:
                        contexts.append(sentence.strip())
                        if len(contexts) >= max_contexts:
                            break
                if len(contexts) >= max_contexts:
                    break
        return contexts
    
    def _find_related_concepts(self, concept: str, all_concepts: List[ConceptData], max_related: int = 5) -> List[str]:
        """Encontrar conceptos relacionados"""
        related = []
        concept_words = set(concept.lower().split())
        
        for other_concept in all_concepts:
            if other_concept.concept == concept:
                continue
            
            other_words = set(other_concept.concept.lower().split())
            # Calcular similitud basada en palabras compartidas
            if concept_words & other_words:
                related.append(other_concept.concept)
                if len(related) >= max_related:
                    break
        
        return related

# =============================================================================
# 7. ANÁLISIS DE TEMAS
# =============================================================================

class ThemeAnalyzer(BaseAnalyzer):
    """Analizador de temas especializado"""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.preprocessor = TextPreprocessor()
        self.cache_manager = CacheManager()
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """Analizar temas del contenido"""
        start_time = datetime.now()
        
        if not self._validate_input(chunks):
            return AnalysisResult(
                analysis_type=AnalysisType.THEME_ANALYSIS,
                data={'themes': []},
                metadata={'error': 'Invalid input data'}
            )
        
        # Generar clave de cache
        cache_key = self._generate_cache_key(chunks)
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result and self.config.enable_cache:
            return cached_result
        
        # Analizar temas
        themes = self._extract_themes_advanced(chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = AnalysisResult(
            analysis_type=AnalysisType.THEME_ANALYSIS,
            data={'themes': themes},
            metadata={
                'total_chunks': len(chunks),
                'themes_found': len(themes),
                'processing_method': 'lda_analysis' if ADVANCED_ANALYSIS_AVAILABLE else 'keyword_clustering'
            },
            processing_time=processing_time
        )
        
        if self.config.enable_cache:
            self.cache_manager.set(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, chunks: List[Dict]) -> str:
        """Generar clave única para cache"""
        content_hash = hash(str(sorted([chunk.get('content', '')[:100] for chunk in chunks])))
        return f"themes_{content_hash}_{self.config.max_concepts}"
    
    def _extract_themes_advanced(self, chunks: List[Dict]) -> List[Dict]:
        """Extraer temas usando métodos avanzados"""
        # Preparar textos
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            if content:
                processed = self.preprocessor.preprocess_text(content)
                if processed and len(processed.split()) > 10:
                    texts.append(processed)
        
        if not texts:
            return []
        
        themes = []
        
        # Análisis LDA si está disponible
        if ADVANCED_ANALYSIS_AVAILABLE and len(texts) >= 3:
            themes = self._extract_themes_with_lda(texts)
        
        # Fallback con clustering de palabras clave
        if not themes:
            themes = self._extract_themes_with_clustering(texts)
        
        return themes
    
    def _extract_themes_with_lda(self, texts: List[str]) -> List[Dict]:
        """Extraer temas usando LDA"""
        try:
            # Configurar LDA
            n_topics = min(8, max(3, len(texts) // 3))
            
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=self.preprocessor.get_spanish_stopwords(),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            
            lda.fit(tfidf_matrix)
            
            # Extraer temas
            feature_names = vectorizer.get_feature_names_out()
            themes = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                theme = {
                    'id': topic_idx,
                    'name': f"Tema {topic_idx + 1}",
                    'keywords': top_words,
                    'weights': top_weights,
                    'coherence': self._calculate_topic_coherence(top_words, texts),
                    'description': self._generate_theme_description(top_words)
                }
                themes.append(theme)
            
            return sorted(themes, key=lambda x: x['coherence'], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error en LDA: {e}")
            return []
    
    def _extract_themes_with_clustering(self, texts: List[str]) -> List[Dict]:
        """Extraer temas usando clustering de palabras clave"""
        try:
            # Extraer palabras clave
            all_words = []
            for text in texts:
                words = text.split()
                all_words.extend([w for w in words if len(w) > 3])
            
            word_freq = Counter(all_words)
            stopwords = self.preprocessor.get_spanish_stopwords()
            
            # Filtrar palabras clave
            keywords = [word for word, freq in word_freq.most_common(50)
                       if word not in stopwords and freq >= 2]
            
            if len(keywords) < 5:
                return []
            
            # Crear vectores de palabras
            vectorizer = TfidfVectorizer(
                vocabulary=keywords,
                stop_words=self.preprocessor.get_spanish_stopwords()
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Clustering
            n_clusters = min(5, max(2, len(keywords) // 10))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Agrupar palabras por cluster
            themes = []
            for cluster_id in range(n_clusters):
                cluster_words = []
                for i, word in enumerate(keywords):
                    if cluster_labels[i] == cluster_id:
                        cluster_words.append(word)
                
                if cluster_words:
                    theme = {
                        'id': cluster_id,
                        'name': f"Tema {cluster_id + 1}",
                        'keywords': cluster_words[:10],
                        'weights': [1.0] * len(cluster_words[:10]),
                        'coherence': len(cluster_words) / len(keywords),
                        'description': self._generate_theme_description(cluster_words[:5])
                    }
                    themes.append(theme)
            
            return themes
            
        except Exception as e:
            self.logger.warning(f"Error en clustering: {e}")
            return []
    
    def _calculate_topic_coherence(self, words: List[str], texts: List[str]) -> float:
        """Calcular coherencia del tema"""
        if len(words) < 2:
            return 0.0
        
        # Coherencia simple basada en co-ocurrencia
        co_occurrences = 0
        total_pairs = 0
        
        for text in texts:
            text_words = set(text.split())
            for i, word1 in enumerate(words):
                for word2 in words[i+1:]:
                    total_pairs += 1
                    if word1 in text_words and word2 in text_words:
                        co_occurrences += 1
        
        return co_occurrences / total_pairs if total_pairs > 0 else 0.0
    
    def _generate_theme_description(self, keywords: List[str]) -> str:
        """Generar descripción del tema basada en palabras clave"""
        if not keywords:
            return "Tema sin descripción"
        
        # Crear descripción simple
        main_keywords = keywords[:3]
        return f"Tema relacionado con: {', '.join(main_keywords)}"

# =============================================================================
# 8. ANÁLISIS DE SENTIMIENTOS
# =============================================================================

class SentimentAnalyzer(BaseAnalyzer):
    """Analizador de sentimientos especializado"""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.preprocessor = TextPreprocessor()
        self.cache_manager = CacheManager()
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """Analizar sentimientos del contenido"""
        start_time = datetime.now()
        
        if not self._validate_input(chunks):
            return AnalysisResult(
                analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
                data={'sentiments': []},
                metadata={'error': 'Invalid input data'}
            )
        
        # Generar clave de cache
        cache_key = self._generate_cache_key(chunks)
        cached_result = self.cache_manager.get(cache_key)
        
        if cached_result and self.config.enable_cache:
            return cached_result
        
        # Analizar sentimientos
        sentiments = self._analyze_sentiments_advanced(chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = AnalysisResult(
            analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
            data={'sentiments': sentiments},
            metadata={
                'total_chunks': len(chunks),
                'processing_method': 'textblob_vader' if ADVANCED_ANALYSIS_AVAILABLE else 'basic_analysis'
            },
            processing_time=processing_time
        )
        
        if self.config.enable_cache:
            self.cache_manager.set(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, chunks: List[Dict]) -> str:
        """Generar clave única para cache"""
        content_hash = hash(str(sorted([chunk.get('content', '')[:100] for chunk in chunks])))
        return f"sentiments_{content_hash}"
    
    def _analyze_sentiments_advanced(self, chunks: List[Dict]) -> Dict:
        """Analizar sentimientos usando métodos avanzados"""
        sentiments = {
            'overall': {'positive': 0, 'negative': 0, 'neutral': 0},
            'by_chunk': [],
            'by_source': {},
            'trends': [],
            'emotions': {}
        }
        
        if ADVANCED_ANALYSIS_AVAILABLE:
            sentiments = self._analyze_with_textblob_vader(chunks)
        else:
            sentiments = self._analyze_basic_sentiment(chunks)
        
        return sentiments
    
    def _analyze_with_textblob_vader(self, chunks: List[Dict]) -> Dict:
        """Analizar sentimientos usando TextBlob y VADER"""
        try:
            from textblob import TextBlob
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            sia = SentimentIntensityAnalyzer()
            
            sentiments = {
                'overall': {'positive': 0, 'negative': 0, 'neutral': 0},
                'by_chunk': [],
                'by_source': {},
                'trends': [],
                'emotions': {}
            }
            
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                source = chunk.get('source', 'Unknown')
                
                if not content:
                    continue
                
                # Análisis con TextBlob
                blob = TextBlob(content)
                polarity = blob.sentiment.polarity
                
                # Análisis con VADER
                vader_scores = sia.polarity_scores(content)
                
                # Determinar sentimiento
                if polarity > 0.1:
                    sentiment = 'positive'
                elif polarity < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                chunk_sentiment = {
                    'content': content[:100] + '...' if len(content) > 100 else content,
                    'source': source,
                    'sentiment': sentiment,
                    'polarity': polarity,
                    'subjectivity': blob.sentiment.subjectivity,
                    'vader_scores': vader_scores,
                    'confidence': abs(polarity)
                }
                
                sentiments['by_chunk'].append(chunk_sentiment)
                sentiments['overall'][sentiment] += 1
                
                # Agrupar por fuente
                if source not in sentiments['by_source']:
                    sentiments['by_source'][source] = {'positive': 0, 'negative': 0, 'neutral': 0}
                sentiments['by_source'][source][sentiment] += 1
            
            # Calcular tendencias
            sentiments['trends'] = self._calculate_sentiment_trends(sentiments['by_chunk'])
            
            return sentiments
            
        except Exception as e:
            self.logger.warning(f"Error en análisis de sentimientos: {e}")
            return self._analyze_basic_sentiment(chunks)
    
    def _analyze_basic_sentiment(self, chunks: List[Dict]) -> Dict:
        """Análisis básico de sentimientos"""
        sentiments = {
            'overall': {'positive': 0, 'negative': 0, 'neutral': 0},
            'by_chunk': [],
            'by_source': {},
            'trends': [],
            'emotions': {}
        }
        
        # Palabras positivas y negativas básicas
        positive_words = ['bueno', 'excelente', 'mejor', 'positivo', 'favorable', 'éxito', 'logro']
        negative_words = ['malo', 'terrible', 'peor', 'negativo', 'problema', 'error', 'fallo']
        
        for chunk in chunks:
            content = chunk.get('content', '').strip().lower()
            source = chunk.get('source', 'Unknown')
            
            if not content:
                continue
            
            # Contar palabras positivas y negativas
            pos_count = sum(1 for word in positive_words if word in content)
            neg_count = sum(1 for word in negative_words if word in content)
            
            if pos_count > neg_count:
                sentiment = 'positive'
            elif neg_count > pos_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            chunk_sentiment = {
                'content': content[:100] + '...' if len(content) > 100 else content,
                'source': source,
                'sentiment': sentiment,
                'polarity': (pos_count - neg_count) / max(1, pos_count + neg_count),
                'subjectivity': 0.5,
                'confidence': abs(pos_count - neg_count) / max(1, pos_count + neg_count)
            }
            
            sentiments['by_chunk'].append(chunk_sentiment)
            sentiments['overall'][sentiment] += 1
            
            # Agrupar por fuente
            if source not in sentiments['by_source']:
                sentiments['by_source'][source] = {'positive': 0, 'negative': 0, 'neutral': 0}
            sentiments['by_source'][source][sentiment] += 1
        
        return sentiments
    
    def _calculate_sentiment_trends(self, chunk_sentiments: List[Dict]) -> List[Dict]:
        """Calcular tendencias de sentimiento"""
        if len(chunk_sentiments) < 3:
            return []
        
        # Agrupar por ventanas de tiempo (simulado por orden)
        window_size = max(1, len(chunk_sentiments) // 5)
        trends = []
        
        for i in range(0, len(chunk_sentiments), window_size):
            window = chunk_sentiments[i:i + window_size]
            
            positive_count = sum(1 for chunk in window if chunk['sentiment'] == 'positive')
            negative_count = sum(1 for chunk in window if chunk['sentiment'] == 'negative')
            neutral_count = sum(1 for chunk in window if chunk['sentiment'] == 'neutral')
            
            total = len(window)
            trend = {
                'window': i // window_size + 1,
                'positive_ratio': positive_count / total,
                'negative_ratio': negative_count / total,
                'neutral_ratio': neutral_count / total,
                'dominant_sentiment': 'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'neutral'
            }
            trends.append(trend)
        
        return trends

class AdvancedQualitativeAnalyzer:
    """Analizador cualitativo avanzado con arquitectura modular mejorada"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.rag_processor = RAGProcessor()
        
        # Usar la configuración global para las rutas de cache
        from config.settings import config as global_config
        self.cache_path = Path(global_config.CACHE_DIR) / "rag_cache.json"
        self.analysis_cache_path = Path(global_config.CACHE_DIR) / "qualitative_analysis_cache.json"
        
        # Inicializar componentes especializados
        self.preprocessor = TextPreprocessor()
        self.cache_manager = CacheManager()
        self.concept_extractor = ConceptExtractor(self.config)
        self.theme_analyzer = ThemeAnalyzer(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        
        # Cache legacy para compatibilidad
        self._spanish_stopwords_cache = None
        self._tfidf_vectorizers_cache = {}
        self._processed_text_cache = {}
        self._concept_analysis_cache = {}
        
        self._initialize_nltk()
        
    def _initialize_nltk(self):
        """Inicializar recursos de NLTK con manejo de errores mejorado"""
        if ADVANCED_ANALYSIS_AVAILABLE:
            try:
                resources = ['punkt', 'stopwords', 'vader_lexicon', 'wordnet']
                for resource in resources:
                    try:
                        nltk.download(resource, quiet=True)
                    except Exception as e:
                        logger.warning(f"No se pudo descargar {resource}: {e}")
            except Exception as e:
                logger.error(f"Error crítico inicializando NLTK: {e}")
    
    # =============================================================================
    # MÉTODOS DE VALIDACIÓN Y UTILIDADES
    # =============================================================================
    
    def _validate_chunks(self, chunks: List[Dict]) -> bool:
        """Validar que los chunks sean válidos para procesamiento"""
        if not chunks:
            logger.warning("Lista de chunks vacía")
            return False
        
        if not isinstance(chunks, list):
            logger.error("Los chunks deben ser una lista")
            return False
        
        valid_chunks = sum(1 for chunk in chunks 
                          if isinstance(chunk, dict) and 
                          chunk.get('content') and 
                          len(chunk.get('content', '').strip()) > 10)
        
        if valid_chunks == 0:
            logger.warning("No hay chunks válidos con contenido suficiente")
            return False
        
        logger.info(f"Validados {valid_chunks} chunks de {len(chunks)} totales")
        return True
    
    def clear_cache(self):
        """Limpiar todos los caches para liberar memoria"""
        try:
            # Limpiar cache nuevo
            self.cache_manager.clear()
            
            # Limpiar cache legacy para compatibilidad
            self._spanish_stopwords_cache = None
            self._tfidf_vectorizers_cache.clear()
            self._processed_text_cache.clear()
            self._concept_analysis_cache.clear()
            
            logger.info("Cache limpiado exitosamente")
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Obtener estadísticas del cache"""
        new_stats = self.cache_manager.get_stats()
        legacy_stats = {
            'stopwords_cached': self._spanish_stopwords_cache is not None,
            'tfidf_vectorizers': len(self._tfidf_vectorizers_cache),
            'processed_texts': len(self._processed_text_cache),
            'concept_analyses': len(self._concept_analysis_cache)
        }
        
        return {
            'new_cache': new_stats,
            'legacy_cache': legacy_stats,
            'total_cached_items': new_stats['size'] + sum(legacy_stats.values())
        }
    
    # =============================================================================
    # MÉTODOS DE COMPATIBILIDAD
    # =============================================================================
    
    def _get_spanish_stopwords(self) -> List[str]:
        """Método de compatibilidad para obtener stopwords"""
        return self.preprocessor.get_spanish_stopwords()
    
    def preprocess_text(self, text: str) -> str:
        """Método de compatibilidad para preprocesar texto"""
        return self.preprocessor.preprocess_text(text)
        
    def _get_term_color(self, term: str) -> str:
        """Generar color jerárquico consistente para un término basado en su importancia"""
        # Usar hash del término para generar color base consistente
        hash_value = hash(term) % 360
        
        # Definir colores jerárquicos basados en Material Design
        hierarchical_colors = [
            f'hsl({hash_value}, 80%, 45%)',  # Color base más saturado
            f'hsl({hash_value}, 70%, 55%)',  # Color medio
            f'hsl({hash_value}, 60%, 65%)',  # Color claro
            f'hsl({hash_value}, 50%, 75%)',  # Color muy claro
        ]
        
        # Determinar nivel jerárquico basado en características del término
        term_length = len(term)
        if term_length >= 12:  # Términos largos = conceptos complejos
            return hierarchical_colors[0]
        elif term_length >= 8:  # Términos medianos = conceptos principales
            return hierarchical_colors[1]
        elif term_length >= 5:  # Términos cortos = sub-conceptos
            return hierarchical_colors[2]
        else:  # Términos muy cortos = detalles
            return hierarchical_colors[3]
        
    def load_rag_data(self) -> List[Dict]:
        """Cargar datos del cache RAG"""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = []
                    
                    if 'chunks' in data and isinstance(data['chunks'], dict):
                        for filename, chunk_list in data['chunks'].items():
                            if isinstance(chunk_list, list):
                                chunks.extend(chunk_list)
                    
                    return chunks
            return []
        except Exception as e:
            logger.error(f"Error cargando datos RAG: {e}")
            return []
    
# =============================================================================
# 12. RESUMENES AUTOMÁTICOS
# =============================================================================
# Esta sección maneja la generación de resúmenes automáticos del contenido.
# MODIFICAR AQUÍ PARA:
# - Cambiar estrategias de resumen (LLM, extractivo, abstractivo)
# - Ajustar longitud y nivel de detalle
# - Modificar prompts para resúmenes con IA
# 
# MÉTODOS PRINCIPALES:
# - generate_intelligent_summary() → Resumen con LLM (IA)
# - generate_rag_summary() → Resumen básico (TextBlob)
# - generate_basic_summary() → Resumen por frecuencia
    
    def generate_rag_summary(self, chunks: List[Dict], max_length: int = 500) -> str:
        """Generar resumen automático básico (fallback)"""
        try:
            if not chunks:
                return "No hay contenido disponible para resumir."
            
            # Usar TODOS los chunks disponibles para un resumen más completo
            all_content = []
            sources = set()
            
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                source = chunk.get('source', 'Documento')
                
                if content and len(content) > 30:
                    all_content.append(content)
                    sources.add(source)
            
            if not all_content:
                return "No hay contenido suficiente para generar un resumen."
            
            combined_text = " ".join(all_content)
            
            # Análisis básico con TextBlob (fallback)
            blob = TextBlob(combined_text)
            sentences = list(blob.sentences)
            
            if len(sentences) <= 5:
                result = combined_text
                if len(result) > max_length:
                    result = result[:max_length] + "..."
                return f"**Resumen Básico** (basado en {len(sources)} fuente(s)):\n\n{result}"
            
            # Seleccionar oraciones más relevantes
            num_sentences = min(10, len(sentences) // 2)
            selected_sentences = sentences[:num_sentences]
            
            summary = " ".join([str(s) for s in selected_sentences])
            
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return f"**Resumen Básico** (basado en {len(sources)} fuente(s)):\n\n{summary}"
            
        except Exception as e:
            logger.error(f"Error generando resumen básico: {e}")
            return f"Error generando resumen: {str(e)}"

    def generate_intelligent_summary(self, chunks: List[Dict], summary_type: str = "comprehensive") -> Dict:
        """Generar resumen inteligente usando LLM con contexto RAG completo"""
        try:
            if not chunks:
                return {
                    'summary': "No hay contenido disponible para resumir.",
                    'type': 'error',
                    'metadata': {}
                }
            
            # Preparar contenido completo
            all_content = []
            sources = set()
            total_words = 0
            
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                source = chunk.get('source', 'Documento')
                
                if content and len(content) > 50:
                    all_content.append(content)
                    sources.add(source)
                    total_words += len(content.split())
            
            if not all_content:
                return {
                    'summary': "No hay contenido suficiente para generar un resumen.",
                    'type': 'error',
                    'metadata': {}
                }
            
            # Combinar contenido inteligentemente
            combined_content = "\n\n".join(all_content)
            
            # Limitar contenido para eficiencia del LLM (máximo 8000 palabras)
            if total_words > 8000:
                words = combined_content.split()
                combined_content = " ".join(words[:8000]) + "\n\n[Contenido truncado para procesamiento eficiente]"
            
            # Configurar prompt según tipo de resumen
            prompts = {
                "comprehensive": self._get_comprehensive_summary_prompt(),
                "executive": self._get_executive_summary_prompt(),
                "analytical": self._get_analytical_summary_prompt(),
                "thematic": self._get_thematic_summary_prompt()
            }
            
            base_prompt = prompts.get(summary_type, prompts["comprehensive"])
            
            full_prompt = f"""{base_prompt}

CONTENIDO A RESUMIR:
{combined_content}

INSTRUCCIONES ESPECÍFICAS:
- Analiza TODO el contenido proporcionado
- Identifica los temas y conceptos más importantes
- Mantén la coherencia y estructura lógica
- Incluye datos específicos y ejemplos relevantes
- Proporciona un resumen de alta calidad académica

FORMATO DE RESPUESTA:
Proporciona un resumen estructurado y completo que capture la esencia del contenido."""
            
            # Usar LLM para generar resumen inteligente
            try:
                from utils.ollama_client import OllamaClient
                from config.settings import config
                
                ollama_client = OllamaClient()
                
                # Configurar parámetros optimizados para resumen
                response = ollama_client.generate_response(
                    model=config.DEFAULT_LLM_MODEL,
                    prompt=full_prompt,
                    max_tokens=2000  # Permitir resúmenes más largos
                )
                
                # Procesar y estructurar respuesta
                processed_summary = self._process_llm_summary_response(response, summary_type)
                
                return {
                    'summary': processed_summary,
                    'type': summary_type,
                    'metadata': {
                        'sources_count': len(sources),
                        'chunks_processed': len(chunks),
                        'total_words_input': total_words,
                        'summary_words': len(processed_summary.split()),
                        'sources': list(sources),
                        'model_used': config.DEFAULT_LLM_MODEL,
                        'compression_ratio': round(len(processed_summary.split()) / total_words * 100, 2)
                    }
                }
                
            except Exception as llm_error:
                logger.warning(f"Error con LLM, usando resumen básico: {llm_error}")
                # Fallback a resumen básico
                basic_summary = self.generate_rag_summary(chunks, 2000)
                return {
                    'summary': basic_summary,
                    'type': 'basic_fallback',
                    'metadata': {
                        'sources_count': len(sources),
                        'chunks_processed': len(chunks),
                        'fallback_reason': str(llm_error)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generando resumen inteligente: {e}")
            return {
                'summary': f"Error generando resumen: {str(e)}",
                'type': 'error',
                'metadata': {}
            }

    def _get_comprehensive_summary_prompt(self) -> str:
        """Prompt para resumen comprehensivo"""
        return """Eres un experto analista de contenido. Tu tarea es crear un resumen comprehensivo y detallado que capture todos los aspectos importantes del contenido proporcionado.

OBJETIVOS:
1. Identificar y explicar los temas principales y secundarios
2. Destacar hallazgos, conclusiones y datos importantes
3. Mantener la estructura lógica del contenido original
4. Proporcionar contexto y relaciones entre conceptos
5. Incluir detalles específicos y ejemplos relevantes"""

    def _get_executive_summary_prompt(self) -> str:
        """Prompt para resumen ejecutivo"""
        return """Eres un consultor senior creando un resumen ejecutivo. Enfócate en los puntos más críticos para la toma de decisiones.

OBJETIVOS:
1. Destacar hallazgos y conclusiones clave
2. Identificar implicaciones y recomendaciones
3. Presentar datos y métricas importantes
4. Mantener un enfoque estratégico y de alto nivel
5. Ser conciso pero completo en información crítica"""

    def _get_analytical_summary_prompt(self) -> str:
        """Prompt para resumen analítico"""
        return """Eres un investigador académico creando un análisis profundo del contenido. Enfócate en patrones, relaciones y insights.

OBJETIVOS:
1. Analizar patrones y tendencias en el contenido
2. Identificar relaciones causales y correlaciones
3. Destacar metodologías y enfoques utilizados
4. Proporcionar interpretación crítica de hallazgos
5. Incluir análisis de fortalezas y limitaciones"""

    def _get_thematic_summary_prompt(self) -> str:
        """Prompt para resumen temático"""
        return """Eres un especialista en análisis temático. Organiza el contenido por temas principales y sus interrelaciones.

OBJETIVOS:
1. Identificar y organizar temas principales
2. Explicar subtemas y conceptos relacionados
3. Mostrar conexiones entre diferentes temas
4. Proporcionar ejemplos específicos por tema
5. Crear una estructura temática clara y lógica"""

    def _process_llm_summary_response(self, response: str, summary_type: str) -> str:
        """Procesar y mejorar la respuesta del LLM"""
        try:
            # Limpiar respuesta
            processed = response.strip()
            
            # Agregar encabezado según tipo
            type_headers = {
                "comprehensive": "📋 **Resumen Comprehensivo**",
                "executive": "🎯 **Resumen Ejecutivo**", 
                "analytical": "🔍 **Análisis Detallado**",
                "thematic": "🏷️ **Análisis Temático**"
            }
            
            header = type_headers.get(summary_type, "📄 **Resumen Inteligente**")
            
            # Estructurar respuesta final
            final_response = f"{header}\n\n{processed}"
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error procesando respuesta LLM: {e}")
            return response
    
    # =============================================================================
    # MÉTODOS DE ANÁLISIS PRINCIPALES (REFACTORIZADOS)
    # =============================================================================
    
    def extract_key_concepts(self, chunks: List[Dict], min_freq: int = 2) -> List[Dict]:
        """Extraer conceptos clave usando la nueva arquitectura modular"""
        try:
            # Actualizar configuración temporal
            temp_config = AnalysisConfig(
                min_frequency=min_freq,
                max_concepts=self.config.max_concepts,
                similarity_threshold=self.config.similarity_threshold,
                enable_cache=self.config.enable_cache
            )
            
            # Usar el extractor especializado
            concept_extractor = ConceptExtractor(temp_config)
            result = concept_extractor.analyze(chunks)
            
            # Convertir ConceptData a formato legacy para compatibilidad
            concepts = []
            for concept_data in result.data.get('concepts', []):
                concepts.append({
                    'concept': concept_data.concept,
                    'score': concept_data.score,
                    'frequency': concept_data.frequency,
                    'context': concept_data.context,
                    'related_concepts': concept_data.related_concepts,
                    'type': 'advanced_extraction'
                })
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error extrayendo conceptos: {e}")
            # Fallback al método legacy
            return self._extract_concepts_legacy(chunks, min_freq)
    
    def extract_advanced_themes(self, chunks: List[Dict], n_topics: int = 10) -> Dict:
        """Extraer temas avanzados usando el analizador especializado"""
        try:
            # Intentar usar el método detallado legacy que retorna el formato esperado
            # por las funciones de renderizado existentes
            result = self._extract_advanced_themes_detailed(chunks, n_topics)
            
            # Si hay error o no hay temas, intentar con el analizador moderno
            if 'error' in result or not result.get('topics'):
                temp_config = AnalysisConfig(
                    min_frequency=self.config.min_frequency,
                    max_concepts=n_topics,
                    similarity_threshold=self.config.similarity_threshold,
                    enable_cache=self.config.enable_cache
                )
                
                theme_analyzer = ThemeAnalyzer(temp_config)
                analyzer_result = theme_analyzer.analyze(chunks)
                themes_data = analyzer_result.data.get('themes', [])
                
                return {
                    'topics': themes_data,
                    'metadata': analyzer_result.metadata,
                    'processing_time': analyzer_result.processing_time,
                    'method': 'advanced_lda_analysis'
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis de temas avanzado: {e}")
            # Fallback al método legacy
            return self._extract_themes_legacy(chunks, n_topics)
    
    def advanced_sentiment_analysis(self, chunks: List[Dict]) -> Dict:
        """Análisis de sentimientos avanzado usando el analizador especializado"""
        try:
            # Intentar usar el método detallado legacy que retorna el formato esperado
            # por las funciones de renderizado existentes
            result = self._advanced_sentiment_analysis_detailed(chunks)
            
            # Si hay error, intentar con el analizador moderno
            if 'error' in result:
                analyzer_result = self.sentiment_analyzer.analyze(chunks)
                sentiments_data = analyzer_result.data.get('sentiments', {})
                
                return {
                    'by_source': sentiments_data.get('by_source', {}),
                    'overall_stats': {
                        'distribution': sentiments_data.get('overall', {}),
                        'mean_score': 0.0
                    },
                    'metadata': analyzer_result.metadata,
                    'processing_time': analyzer_result.processing_time,
                    'method': 'advanced_sentiment_analysis'
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis de sentimientos avanzado: {e}")
            # Fallback al método legacy simple
            return self._sentiment_analysis_legacy(chunks)
    
    def _extract_themes_legacy(self, chunks: List[Dict], n_topics: int = 10) -> Dict:
        """Método legacy para extraer temas (fallback)"""
        try:
            if not self._validate_chunks(chunks):
                return {'themes': [], 'metadata': {'error': 'Invalid input'}}
            
            # Implementación básica de extracción de temas
            all_text = []
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                if content:
                    processed = self.preprocess_text(content)
                    if processed:
                        all_text.append(processed)
            
            if not all_text:
                return {'themes': [], 'metadata': {'error': 'No valid content'}}
            
            # Análisis básico de palabras clave
            all_words = []
            for text in all_text:
                words = text.split()
                all_words.extend([w for w in words if len(w) > 3])
            
            word_freq = Counter(all_words)
            stopwords = self._get_spanish_stopwords()
            
            # Filtrar palabras clave
            keywords = [word for word, freq in word_freq.most_common(50)
                       if word not in stopwords and freq >= 2]
            
            # Crear temas básicos
            themes = []
            chunk_size = max(1, len(keywords) // n_topics)
            
            for i in range(0, len(keywords), chunk_size):
                theme_keywords = keywords[i:i + chunk_size]
                if theme_keywords:
                    theme = {
                        'id': len(themes),
                        'name': f"Tema {len(themes) + 1}",
                        'keywords': theme_keywords,
                        'weights': [1.0] * len(theme_keywords),
                        'coherence': len(theme_keywords) / len(keywords),
                        'description': f"Tema relacionado con: {', '.join(theme_keywords[:3])}"
                    }
                    themes.append(theme)
            
            return {
                'themes': themes,
                'metadata': {
                    'total_chunks': len(chunks),
                    'themes_found': len(themes),
                    'processing_method': 'basic_keyword_clustering'
                },
                'processing_time': 0.0,
                'method': 'legacy_theme_extraction'
            }
            
        except Exception as e:
            logger.error(f"Error en extracción de temas legacy: {e}")
            return {'themes': [], 'metadata': {'error': str(e)}}
    
    def _sentiment_analysis_legacy(self, chunks: List[Dict]) -> Dict:
        """Método legacy para análisis de sentimientos (fallback)"""
        try:
            if not self._validate_chunks(chunks):
                return {'sentiments': {}, 'metadata': {'error': 'Invalid input'}}
            
            # Análisis básico de sentimientos
            sentiments = {
                'overall': {'positive': 0, 'negative': 0, 'neutral': 0},
                'by_chunk': [],
                'by_source': {},
                'trends': [],
                'emotions': {}
            }
            
            # Palabras positivas y negativas básicas
            positive_words = ['bueno', 'excelente', 'mejor', 'positivo', 'favorable', 'éxito', 'logro']
            negative_words = ['malo', 'terrible', 'peor', 'negativo', 'problema', 'error', 'fallo']
            
            for chunk in chunks:
                content = chunk.get('content', '').strip().lower()
                source = chunk.get('source', 'Unknown')
                
                if not content:
                    continue
                
                # Contar palabras positivas y negativas
                pos_count = sum(1 for word in positive_words if word in content)
                neg_count = sum(1 for word in negative_words if word in content)
                
                if pos_count > neg_count:
                    sentiment = 'positive'
                elif neg_count > pos_count:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                chunk_sentiment = {
                    'content': content[:100] + '...' if len(content) > 100 else content,
                    'source': source,
                    'sentiment': sentiment,
                    'polarity': (pos_count - neg_count) / max(1, pos_count + neg_count),
                    'subjectivity': 0.5,
                    'confidence': abs(pos_count - neg_count) / max(1, pos_count + neg_count)
                }
                
                sentiments['by_chunk'].append(chunk_sentiment)
                sentiments['overall'][sentiment] += 1
                
                # Agrupar por fuente
                if source not in sentiments['by_source']:
                    sentiments['by_source'][source] = {'positive': 0, 'negative': 0, 'neutral': 0}
                sentiments['by_source'][source][sentiment] += 1
            
            return {
                'sentiments': sentiments,
                'metadata': {
                    'total_chunks': len(chunks),
                    'processing_method': 'basic_sentiment_analysis'
                },
                'processing_time': 0.0,
                'method': 'legacy_sentiment_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de sentimientos legacy: {e}")
            return {'sentiments': {}, 'metadata': {'error': str(e)}}
    
# =============================================================================
# 16. ANÁLISIS PARALELO Y OPTIMIZACIÓN
# =============================================================================

    def perform_parallel_analysis(self, chunks: List[Dict], analysis_types: List[AnalysisType]) -> Dict[str, AnalysisResult]:
        """Realizar múltiples análisis en paralelo para optimizar rendimiento"""
        try:
            if not self.config.parallel_processing:
                return self._perform_sequential_analysis(chunks, analysis_types)
            
            results = {}
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Crear tareas para cada tipo de análisis
                future_to_analysis = {}
                
                for analysis_type in analysis_types:
                    if analysis_type == AnalysisType.CONCEPT_EXTRACTION:
                        future = executor.submit(self.concept_extractor.analyze, chunks)
                        future_to_analysis[future] = analysis_type
                    elif analysis_type == AnalysisType.THEME_ANALYSIS:
                        future = executor.submit(self.theme_analyzer.analyze, chunks)
                        future_to_analysis[future] = analysis_type
                    elif analysis_type == AnalysisType.SENTIMENT_ANALYSIS:
                        future = executor.submit(self.sentiment_analyzer.analyze, chunks)
                        future_to_analysis[future] = analysis_type
                
                # Recopilar resultados
                for future in as_completed(future_to_analysis):
                    analysis_type = future_to_analysis[future]
                    try:
                        result = future.result()
                        results[analysis_type.value] = result
                    except Exception as e:
                        logger.error(f"Error en análisis paralelo {analysis_type}: {e}")
                        results[analysis_type.value] = AnalysisResult(
                            analysis_type=analysis_type,
                            data={},
                            metadata={'error': str(e)}
                        )
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis paralelo: {e}")
            return self._perform_sequential_analysis(chunks, analysis_types)
    
    def _perform_sequential_analysis(self, chunks: List[Dict], analysis_types: List[AnalysisType]) -> Dict[str, AnalysisResult]:
        """Realizar análisis secuencial como fallback"""
        results = {}
        
        for analysis_type in analysis_types:
            try:
                if analysis_type == AnalysisType.CONCEPT_EXTRACTION:
                    result = self.concept_extractor.analyze(chunks)
                elif analysis_type == AnalysisType.THEME_ANALYSIS:
                    result = self.theme_analyzer.analyze(chunks)
                elif analysis_type == AnalysisType.SENTIMENT_ANALYSIS:
                    result = self.sentiment_analyzer.analyze(chunks)
                else:
                    continue
                
                results[analysis_type.value] = result
                
            except Exception as e:
                logger.error(f"Error en análisis secuencial {analysis_type}: {e}")
                results[analysis_type.value] = AnalysisResult(
                    analysis_type=analysis_type,
                    data={},
                    metadata={'error': str(e)}
                )
        
        return results
    
    def get_analysis_summary(self, chunks: List[Dict]) -> Dict:
        """Obtener resumen completo de análisis cualitativo"""
        try:
            start_time = datetime.now()
            
            # Realizar análisis paralelo
            analysis_types = [
                AnalysisType.CONCEPT_EXTRACTION,
                AnalysisType.THEME_ANALYSIS,
                AnalysisType.SENTIMENT_ANALYSIS
            ]
            
            results = self.perform_parallel_analysis(chunks, analysis_types)
            
            # Procesar resultados
            summary = {
                'overview': {
                    'total_chunks': len(chunks),
                    'analysis_types': [t.value for t in analysis_types],
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                },
                'concepts': results.get('concept_extraction', {}).data.get('concepts', []),
                'themes': results.get('theme_analysis', {}).data.get('themes', []),
                'sentiments': results.get('sentiment_analysis', {}).data.get('sentiments', {}),
                'metadata': {
                    'concept_metadata': results.get('concept_extraction', {}).metadata,
                    'theme_metadata': results.get('theme_analysis', {}).metadata,
                    'sentiment_metadata': results.get('sentiment_analysis', {}).metadata
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generando resumen de análisis: {e}")
            return {
                'overview': {
                    'total_chunks': len(chunks),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                },
                'concepts': [],
                'themes': [],
                'sentiments': {},
                'metadata': {}
            }
    
# =============================================================================
# 17. MÉTODOS DE CONFIGURACIÓN
# =============================================================================

    def update_config(self, **kwargs) -> None:
        """Actualizar configuración dinámicamente"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Configuración actualizada: {key} = {value}")
                else:
                    logger.warning(f"Parámetro de configuración no válido: {key}")
        except Exception as e:
            logger.error(f"Error actualizando configuración: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento del sistema"""
        try:
            cache_stats = self.get_cache_stats()
            
            metrics = {
                'cache_performance': cache_stats,
                'memory_usage': {
                    'concept_extractor_cache': len(self.concept_extractor.cache_manager.cache),
                    'theme_analyzer_cache': len(self.theme_analyzer.cache_manager.cache),
                    'sentiment_analyzer_cache': len(self.sentiment_analyzer.cache_manager.cache),
                    'total_cached_items': sum([
                        len(self.concept_extractor.cache_manager.cache),
                        len(self.theme_analyzer.cache_manager.cache),
                        len(self.sentiment_analyzer.cache_manager.cache)
                    ])
                },
                'configuration': {
                    'parallel_processing': self.config.parallel_processing,
                    'max_workers': self.config.max_workers,
                    'enable_cache': self.config.enable_cache,
                    'max_concepts': self.config.max_concepts,
                    'similarity_threshold': self.config.similarity_threshold
                },
                'system_info': {
                    'advanced_analysis_available': ADVANCED_ANALYSIS_AVAILABLE,
                    'pyvis_available': PYVIS_AVAILABLE,
                    'agraph_available': AGRAPH_AVAILABLE,
                    'graphviz_available': GRAPHVIZ_AVAILABLE
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas de rendimiento: {e}")
            return {'error': str(e)}
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimizar rendimiento del sistema"""
        try:
            optimizations = []
            
            # Limpiar caches si están muy llenos
            cache_stats = self.get_cache_stats()
            total_items = cache_stats.get('total_cached_items', 0)
            
            if total_items > 500:
                self.clear_cache()
                optimizations.append("Cache limpiado - demasiados elementos")
            
            # Ajustar configuración basada en recursos disponibles
            try:
                import psutil
                cpu_count = psutil.cpu_count()
                memory_gb = psutil.virtual_memory().total / (1024**3)
                
                if cpu_count >= 4 and memory_gb >= 8:
                    if not self.config.parallel_processing:
                        self.config.parallel_processing = True
                        self.config.max_workers = min(4, cpu_count)
                        optimizations.append("Procesamiento paralelo habilitado")
                else:
                    if self.config.parallel_processing:
                        self.config.parallel_processing = False
                        optimizations.append("Procesamiento paralelo deshabilitado - recursos limitados")
                
                system_resources = {
                    'cpu_count': cpu_count,
                    'memory_gb': round(memory_gb, 2)
                }
            except ImportError:
                system_resources = {'error': 'psutil no disponible'}
            
            return {
                'optimizations_applied': optimizations,
                'new_config': {
                    'parallel_processing': self.config.parallel_processing,
                    'max_workers': self.config.max_workers,
                    'enable_cache': self.config.enable_cache
                },
                'system_resources': system_resources
            }
            
        except Exception as e:
            logger.error(f"Error optimizando rendimiento: {e}")
            return {'error': str(e), 'optimizations_applied': []}
    
    def _extract_concepts_legacy(self, chunks: List[Dict], min_freq: int = 2) -> List[Dict]:
        """Método legacy para extraer conceptos (fallback)"""
        try:
            if not self._validate_chunks(chunks):
                return []
            
            # Crear clave de cache basada en el contenido
            content_hash = hash(str(sorted([chunk.get('content', '')[:100] for chunk in chunks])))
            cache_key = f"concepts_{content_hash}_{min_freq}"
            
            # Verificar cache
            if cache_key in self._concept_analysis_cache:
                logger.info("Usando conceptos desde cache legacy")
                return self._concept_analysis_cache[cache_key]
            
            # Procesar cada chunk por separado para TF-IDF
            chunk_texts = []
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                if content:
                    processed = self.preprocess_text(content)
                    if processed and len(processed.split()) > 5:  # Mínimo 5 palabras
                        chunk_texts.append(processed)
            
            if not chunk_texts:
                return []
            
            concepts = []
            
            # Extraer conceptos usando TF-IDF si está disponible y hay suficientes documentos
            if ADVANCED_ANALYSIS_AVAILABLE and len(chunk_texts) >= 2:
                try:
                    # Configurar TF-IDF para múltiples documentos
                    num_docs = len(chunk_texts)
                    adjusted_min_df = max(1, min(min_freq, num_docs // 4))
                    adjusted_max_df = min(0.95, max(0.5, (num_docs - 1) / num_docs))
                    
                    # Usar cache para vectorizador si es posible
                    vectorizer_key = f"tfidf_{num_docs}_{adjusted_min_df}_{adjusted_max_df}"
                    
                    if vectorizer_key not in self._tfidf_vectorizers_cache:
                        vectorizer = TfidfVectorizer(
                            max_features=100,
                            stop_words=self._get_spanish_stopwords(),
                            ngram_range=(1, 2),
                            min_df=adjusted_min_df,
                            max_df=adjusted_max_df
                        )
                        self._tfidf_vectorizers_cache[vectorizer_key] = vectorizer
                    else:
                        vectorizer = self._tfidf_vectorizers_cache[vectorizer_key]
                    
                    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Calcular puntuaciones promedio para todos los documentos
                    mean_scores = tfidf_matrix.mean(axis=0).A1
                    
                    for i, score in enumerate(mean_scores):
                        if score > 0:
                            concepts.append({
                                'concept': feature_names[i],
                                'score': float(score),
                                'type': 'tfidf'
                            })
                    
                    if concepts:
                        result = sorted(concepts, key=lambda x: x['score'], reverse=True)[:30]
                        # Guardar en cache
                        self._concept_analysis_cache[cache_key] = result
                        return result
                    
                except Exception as e:
                    logger.warning(f"Error en TF-IDF: {e}")
            
            # Fallback: análisis básico de frecuencia con todo el texto
            all_text = " ".join(chunk_texts)
            words = all_text.split()
            word_freq = Counter(words)
            stop_words = self._get_spanish_stopwords()
            
            total_words = len(words)
            
            for word, freq in word_freq.most_common(50):
                if (len(word) > 3 and 
                    word not in stop_words and 
                    freq >= min_freq and
                    word.isalpha()):  # Solo palabras alfabéticas
                    concepts.append({
                        'concept': word,
                        'score': freq / total_words,
                        'type': 'frequency'
                    })
            
            result = concepts[:30]
            # Guardar en cache
            self._concept_analysis_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error en extracción legacy: {e}")
            return []
    
# =============================================================================
# 10. MAPAS CONCEPTUALES INTERACTIVOS
# =============================================================================
# Esta sección maneja la generación de mapas conceptuales interactivos.
# MODIFICAR AQUÍ PARA:
# - Cambiar la visualización (PyVis, layouts, colores)
# - Ajustar la jerarquía de conceptos
# - Modificar relaciones entre nodos
# 
# MÉTODOS PRINCIPALES:
# - create_interactive_concept_map() → Genera mapa con PyVis
# - _analyze_concept_hierarchy() → Analiza jerarquía de conceptos
# - generate_advanced_concept_map() → Mapa avanzado con TF-IDF
    
    def create_interactive_concept_map(self, chunks: List[Dict], layout_type: str = "spring") -> Optional[str]:
        """Crear mapa conceptual interactivo estructurado usando PyVis con paleta de colores profesional mejorada"""
        if not PYVIS_AVAILABLE:
            return None
        
        try:
            # Análisis inteligente del contenido para crear estructura jerárquica
            concept_structure = self._analyze_concept_hierarchy(chunks)
            if not concept_structure:
                return None
            
            # Crear red con configuración mejorada y tamaño amplio
            net = Network(
                height="700px",  # Altura aumentada para mejor visualización
                width="100%",
                bgcolor="#f8f9fa",  # Fondo gris muy claro para mejor contraste
                font_color="#212529",  # Color de fuente oscuro para máxima legibilidad
                directed=True  # Dirigido para mostrar relaciones jerárquicas
            )
            
            # Configurar física para mejor organización jerárquica con MEJOR SEPARACIÓN
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 400},
                "hierarchicalRepulsion": {
                  "centralGravity": 0.2,
                  "springLength": 200,
                  "springConstant": 0.015,
                  "nodeDistance": 250,
                  "damping": 0.15
                }
              },
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "levelSeparation": 250,
                  "nodeSpacing": 180,
                  "treeSpacing": 350,
                  "blockShifting": true,
                  "edgeMinimization": true,
                  "parentCentralization": true,
                  "direction": "UD",
                  "sortMethod": "directed"
                }
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 150,
                "hideEdgesOnDrag": false,
                "dragNodes": true,
                "dragView": true,
                "zoomView": true,
                "selectConnectedEdges": true
              },
              "edges": {
                "arrows": {
                  "to": {"enabled": true, "scaleFactor": 1.2, "type": "arrow"}
                },
                "smooth": {
                  "enabled": true,
                  "type": "continuous",
                  "roundness": 0.5
                }
              },
              "nodes": {
                "borderWidth": 3,
                "borderWidthSelected": 5,
                "margin": 15,
                "widthConstraint": {
                  "minimum": 120,
                  "maximum": 250
                },
                "font": {
                  "size": 16,
                  "face": "Arial",
                  "strokeWidth": 2,
                  "strokeColor": "#ffffff",
                  "vadjust": 0
                }
              }
            }
            """)
            
            # Nueva paleta de colores profesional con excelente contraste
            professional_palette = {
                'main_theme': {
                    'background': '#2c3e50',    # Azul marino oscuro - tema principal
                    'border': '#1a252f',        # Borde más oscuro
                    'text': '#ffffff',          # Texto blanco para máximo contraste
                    'highlight': '#34495e'      # Resaltado
                },
                'branch_colors': [
                    # Colores distintos y profesionales para cada rama
                    {
                        'background': '#e74c3c',    # Rojo profesional
                        'border': '#c0392b',        # Borde más oscuro
                        'text': '#ffffff',          # Texto blanco
                        'highlight': '#ec7063'      # Resaltado más claro
                    },
                    {
                        'background': '#27ae60',    # Verde profesional
                        'border': '#229954',        # Borde más oscuro
                        'text': '#ffffff',          # Texto blanco
                        'highlight': '#58d68d'      # Resaltado más claro
                    },
                    {
                        'background': '#f39c12',    # Naranja profesional
                        'border': '#e67e22',        # Borde más oscuro
                        'text': '#ffffff',          # Texto blanco
                        'highlight': '#f8c471'      # Resaltado más claro
                    },
                    {
                        'background': '#9b59b6',    # Púrpura profesional
                        'border': '#8e44ad',        # Borde más oscuro
                        'text': '#ffffff',          # Texto blanco
                        'highlight': '#bb8fce'      # Resaltado más claro
                    },
                    {
                        'background': '#3498db',    # Azul cielo profesional
                        'border': '#2980b9',        # Borde más oscuro
                        'text': '#ffffff',          # Texto blanco
                        'highlight': '#85c1e9'      # Resaltado más claro
                    },
                    {
                        'background': '#e67e22',    # Naranja oscuro profesional
                        'border': '#d35400',        # Borde más oscuro
                        'text': '#ffffff',          # Texto blanco
                        'highlight': '#f0b27a'      # Resaltado más claro
                    },
                    {
                        'background': '#16a085',    # Verde azulado profesional
                        'border': '#138d75',        # Borde más oscuro
                        'text': '#ffffff',          # Texto blanco
                        'highlight': '#76d7c4'      # Resaltado más claro
                    },
                    {
                        'background': '#8e44ad',    # Violeta profesional
                        'border': '#7d3c98',        # Borde más oscuro
                        'text': '#ffffff',          # Texto blanco
                        'highlight': '#af7ac5'      # Resaltado más claro
                    }
                ],
                'sub_concept_base': {
                    'text': '#ffffff',          # Texto blanco para sub-conceptos
                    'border_factor': 0.8,      # Factor para oscurecer bordes
                    'background_factor': 0.7   # Factor para aclarar fondos
                },
                'relations': {
                    'primary': '#34495e',       # Relaciones principales - gris oscuro
                    'secondary': '#7f8c8d',     # Relaciones secundarias - gris medio
                    'cross': '#e74c3c'          # Relaciones cruzadas - rojo
                }
            }
            
            # Agregar nodo principal (tema central) con nueva paleta profesional
            main_theme = concept_structure['main_theme']
            main_colors = professional_palette['main_theme']
            net.add_node(
                "main",
                label=main_theme['name'],
                title=f"TEMA PRINCIPAL: {main_theme['name']}\n\nDescripción: {main_theme['description']}",
                size=80,  # Tamaño más grande para el tema principal
                color={
                    'background': main_colors['background'],
                    'border': main_colors['border'],
                    'highlight': {'background': main_colors['highlight'], 'border': main_colors['border']}
                },
                font={'size': 22, 'color': main_colors['text'], 'face': 'Arial Black', 'strokeWidth': 2, 'strokeColor': '#000000'},
                level=0,
                shape="box",
                borderWidth=4,
                borderWidthSelected=6,
                shadow={'enabled': True, 'color': 'rgba(0,0,0,0.3)', 'size': 10}
            )
            
            # Agregar conceptos principales (nivel 1) con colores distintos por rama
            for i, concept in enumerate(concept_structure['main_concepts']):
                node_id = f"concept_{i}"
                # Asignar color de rama basado en el índice
                branch_colors = professional_palette['branch_colors']
                branch_color = branch_colors[i % len(branch_colors)]
                
                net.add_node(
                    node_id,
                    label=concept['name'],
                    title=f"CONCEPTO PRINCIPAL: {concept['name']}\n\nRelevancia: {concept['relevance']:.2f}\n\nContexto: {concept['context']}",
                    size=max(45, min(65, concept['relevance'] * 130)),  # Tamaños jerárquicos
                    color={
                        'background': branch_color['background'],
                        'border': branch_color['border'],
                        'highlight': {'background': branch_color['highlight'], 'border': branch_color['border']}
                    },
                    font={'size': 18, 'color': branch_color['text'], 'face': 'Arial Bold', 'strokeWidth': 1, 'strokeColor': '#000000'},
                    level=1,
                    shape="ellipse",
                    borderWidth=3,
                    borderWidthSelected=5,
                    shadow={'enabled': True, 'color': 'rgba(0,0,0,0.2)', 'size': 8}
                )
                
                # Conectar con el tema principal con estilo jerárquico mejorado
                net.add_edge(
                    "main", 
                    node_id,
                    width=5,
                    color={'color': professional_palette['relations']['primary'], 'opacity': 0.8},
                    title="define",
                    smooth={'type': 'continuous'},
                    arrows={'to': {'enabled': True, 'scaleFactor': 1.2}}
                )
                
                # Agregar sub-conceptos (nivel 2) con versión más clara del color de rama
                for j, sub_concept in enumerate(concept['sub_concepts'][:4]):  # Máximo 4 sub-conceptos
                    sub_id = f"sub_{i}_{j}"
                    
                    # Crear versión más clara del color de rama para sub-conceptos
                    import colorsys
                    
                    # Convertir hex a RGB
                    hex_color = branch_color['background'].lstrip('#')
                    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    
                    # Convertir a HSV y aclarar
                    hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
                    lighter_hsv = (hsv[0], hsv[1] * 0.6, min(1.0, hsv[2] + 0.3))
                    lighter_rgb = colorsys.hsv_to_rgb(*lighter_hsv)
                    
                    # Convertir de vuelta a hex
                    lighter_hex = '#{:02x}{:02x}{:02x}'.format(
                        int(lighter_rgb[0] * 255),
                        int(lighter_rgb[1] * 255),
                        int(lighter_rgb[2] * 255)
                    )
                    
                    # Crear borde más oscuro
                    darker_hsv = (hsv[0], hsv[1], hsv[2] * 0.8)
                    darker_rgb = colorsys.hsv_to_rgb(*darker_hsv)
                    darker_hex = '#{:02x}{:02x}{:02x}'.format(
                        int(darker_rgb[0] * 255),
                        int(darker_rgb[1] * 255),
                        int(darker_rgb[2] * 255)
                    )
                    
                    net.add_node(
                        sub_id,
                        label=sub_concept['name'],
                        title=f"SUB-CONCEPTO: {sub_concept['name']}\n\nRelación: {sub_concept['relation']}\n\nContexto: {sub_concept['context']}",
                        size=max(30, sub_concept['relevance'] * 100),  # Tamaños jerárquicos menores
                        color={
                            'background': lighter_hex,
                            'border': darker_hex,
                            'highlight': {'background': branch_color['highlight'], 'border': darker_hex}
                        },
                        font={'size': 15, 'color': '#ffffff', 'face': 'Arial', 'strokeWidth': 1, 'strokeColor': '#000000'},
                        level=2,
                        shape="dot",
                        borderWidth=2,
                        borderWidthSelected=4,
                        shadow={'enabled': True, 'color': 'rgba(0,0,0,0.15)', 'size': 6}
                    )
                    
                    # Conectar sub-concepto con concepto principal con estilo jerárquico
                    net.add_edge(
                        node_id,
                        sub_id,
                        width=4,
                        color={'color': professional_palette['relations']['secondary'], 'opacity': 0.7},
                        title=sub_concept['relation'],
                        smooth={'type': 'continuous'},
                        arrows={'to': {'enabled': True, 'scaleFactor': 1.0}}
                    )
            
            # Agregar relaciones cruzadas entre conceptos del mismo nivel con estilo jerárquico
            for relation in concept_structure['cross_relations']:
                net.add_edge(
                    relation['from'],
                    relation['to'],
                    width=3,
                    color={'color': professional_palette['relations']['cross'], 'opacity': 0.6},
                    title=relation.get('relation', 'relacionado'),
                    dashes=True,
                    smooth={'type': 'continuous'},
                    arrows={'to': {'enabled': True, 'scaleFactor': 0.8}}
                )
            
            # Generar HTML
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8')
            net.save_graph(temp_file.name)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error creando mapa conceptual: {e}")
            return None
    
# =============================================================================
# 11. MAPAS MENTALES
# =============================================================================
    
    def create_interactive_mind_map(self, chunks: List[Dict], node_spacing: int = 250, return_data: bool = False) -> Optional[Dict]:
        """Crear mapa mental interactivo estructurado usando streamlit-agraph con análisis inteligente"""
        if not AGRAPH_AVAILABLE:
            return None
        
        try:
            # Análisis inteligente mejorado para estructura de mapa mental
            mind_structure = self._analyze_intelligent_mind_map_structure(chunks)
            if not mind_structure:
                return None
            
            nodes = []
            edges = []
            
            # Nodo central (tema principal) con mejor tamaño y separación
            central_theme = mind_structure['central_theme']
            nodes.append({
                'id': "central",
                'label': central_theme['name'],
                'title': f"Tema Central: {central_theme['name']}\n\n{central_theme['description']}",
                'size': 70,  # AUMENTADO: de 60 a 70 para mejor visibilidad
                'color': "#1a237e",  # Azul profundo para el tema central
                'font_size': 22,  # AUMENTADO: de 20 a 22
                'level': 0  # Nivel jerárquico
            })
            
            # Paleta de colores jerárquica mejorada para ramas principales
            hierarchical_branch_colors = [
                "#2196f3",  # Azul brillante
                "#4caf50",  # Verde
                "#ff9800",  # Naranja
                "#9c27b0",  # Púrpura
                "#f44336",  # Rojo
                "#00bcd4",  # Cian
                "#795548",  # Marrón
                "#607d8b"   # Azul gris
            ]
            
            for i, branch in enumerate(mind_structure['main_branches']):
                branch_id = f"branch_{i}"
                branch_color = hierarchical_branch_colors[i % len(hierarchical_branch_colors)]
                
                # Nodo de rama principal con mejor tamaño
                nodes.append({
                    'id': branch_id,
                    'label': branch['name'],
                    'title': f"Concepto Principal: {branch['name']}\nImportancia: {branch['importance']:.2f}\n\n{branch['description']}",
                    'size': max(45, int(branch['importance'] * 55)),  # AUMENTADO: de 35/45 a 45/55
                    'color': branch_color,
                    'font_size': 18,  # AUMENTADO: de 17 a 18
                    'level': 1  # Nivel jerárquico
                })
                
                # Conectar con el centro usando colores jerárquicos
                edges.append({
                    'from': "central",
                    'to': branch_id,
                    'width': max(4, int(branch['importance'] * 6)),  # Conexiones más gruesas
                    'color': branch_color
                })
                
                # Sub-ramas (conceptos de segundo nivel) con colores jerárquicos
                for j, sub_branch in enumerate(branch['sub_branches'][:4]):  # Máximo 4 sub-ramas
                    sub_id = f"sub_{i}_{j}"
                    
                    # Color más claro para sub-ramas (jerarquía visual)
                    sub_color = self._lighten_color_hex(branch_color, 0.3)
                    
                    nodes.append({
                        'id': sub_id,
                        'label': sub_branch['name'],
                        'title': f"Sub-concepto: {sub_branch['name']}\nRelación: {sub_branch['relation']}\n\n{sub_branch['context'][:150]}...",
                        'size': max(25, int(sub_branch['relevance'] * 35)),  # AUMENTADO: de 20/30 a 25/35
                        'color': sub_color,
                        'font_size': 15,  # AUMENTADO: de 14 a 15
                        'level': 2  # Nivel jerárquico
                    })
                    
                    # Conectar con la rama principal usando colores jerárquicos
                    edges.append({
                        'from': branch_id,
                        'to': sub_id,
                        'width': max(2, int(sub_branch['relevance'] * 4)),
                        'color': sub_color
                    })
            
            # Estadísticas
            stats = {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'main_branches': len(mind_structure['main_branches']),
                'detailed_concepts': sum(len(branch['sub_branches']) for branch in mind_structure['main_branches'])
            }
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': stats,
                'structure_info': mind_structure.get('info', {})
            }
            
        except Exception as e:
            logger.error(f"Error creando mapa mental: {e}")
            return None
    
# =============================================================================
# 13. ANÁLISIS DE TRIANGULACIÓN
# =============================================================================
    
    def perform_triangulation_analysis(self, chunks: List[Dict]) -> Dict:
        """
        Realizar triangulación de información para validar conceptos
        
        MEJORA: Ahora funciona con una sola fuente también
        - Con 1 fuente: Analiza frecuencia y distribución de conceptos por secciones
        - Con 2+ fuentes: Triangulación clásica entre fuentes
        """
        try:
            if not chunks:
                return {'error': 'No hay datos disponibles'}
            
            # Agrupar chunks por fuente
            sources = defaultdict(list)
            for chunk in chunks:
                source = chunk.get('metadata', {}).get('source_file', 'Desconocido')
                sources[source].append(chunk)
            
            # NUEVO: Manejar caso de una sola fuente
            if len(sources) == 1:
                return self._perform_single_source_triangulation(chunks, list(sources.keys())[0])
            
            # Caso normal: 2 o más fuentes (triangulación clásica)
            # Extraer conceptos por fuente
            concepts_by_source = {}
            for source, source_chunks in sources.items():
                concepts = self.extract_key_concepts(source_chunks)
                concepts_by_source[source] = {c['concept']: c['score'] for c in concepts}
            
            # Encontrar conceptos comunes (triangulación)
            all_concepts = set()
            for concepts in concepts_by_source.values():
                all_concepts.update(concepts.keys())
            
            triangulated_concepts = []
            for concept in all_concepts:
                sources_with_concept = []
                total_score = 0
                
                for source, concepts in concepts_by_source.items():
                    if concept in concepts:
                        sources_with_concept.append(source)
                        total_score += concepts[concept]
                
                # MODIFICADO: Con 2 fuentes, acepta conceptos en al menos 1
                min_sources = 2 if len(sources) > 2 else 1
                
                if len(sources_with_concept) >= min_sources:
                    triangulated_concepts.append({
                        'concept': concept,
                        'sources': sources_with_concept,
                        'source_count': len(sources_with_concept),
                        'avg_score': total_score / len(sources_with_concept),
                        'reliability': len(sources_with_concept) / len(sources),
                        'validation_type': 'multi-source' if len(sources_with_concept) > 1 else 'single-source'
                    })
            
            # Ordenar por confiabilidad
            triangulated_concepts.sort(key=lambda x: (x['reliability'], x['avg_score']), reverse=True)
            
            return {
                'triangulated_concepts': triangulated_concepts[:20],
                'total_sources': len(sources),
                'total_concepts': len(all_concepts),
                'validated_concepts': len(triangulated_concepts),
                'sources': list(sources.keys()),
                'analysis_mode': 'multi-source'
            }
            
        except Exception as e:
            logger.error(f"Error en triangulación: {e}")
            return {'error': str(e)}
    
    def _perform_single_source_triangulation(self, chunks: List[Dict], source_name: str) -> Dict:
        """
        Realizar análisis de triangulación interna para una sola fuente
        
        En lugar de triangular entre fuentes, triangula entre secciones del mismo documento:
        - Divide el documento en secciones
        - Analiza conceptos por sección
        - Identifica conceptos que aparecen en múltiples secciones (triangulación interna)
        """
        try:
            # Dividir chunks en secciones (grupos de 3-5 chunks)
            section_size = max(3, len(chunks) // 5)  # Al menos 5 secciones si es posible
            sections = []
            
            for i in range(0, len(chunks), section_size):
                section_chunks = chunks[i:i + section_size]
                if section_chunks:
                    sections.append({
                        'id': f"Sección {len(sections) + 1}",
                        'chunks': section_chunks,
                        'start_chunk': i,
                        'end_chunk': min(i + section_size, len(chunks))
                    })
            
            # Extraer conceptos por sección
            concepts_by_section = {}
            for section in sections:
                section_concepts = self.extract_key_concepts(section['chunks'])
                concepts_by_section[section['id']] = {c['concept']: c['score'] for c in section_concepts}
            
            # Encontrar conceptos que aparecen en múltiples secciones
            all_concepts = set()
            for concepts in concepts_by_section.values():
                all_concepts.update(concepts.keys())
            
            triangulated_concepts = []
            for concept in all_concepts:
                sections_with_concept = []
                total_score = 0
                
                for section_id, concepts in concepts_by_section.items():
                    if concept in concepts:
                        sections_with_concept.append(section_id)
                        total_score += concepts[concept]
                
                # Concepto debe aparecer en al menos 2 secciones para ser considerado "triangulado"
                if len(sections_with_concept) >= 2:
                    triangulated_concepts.append({
                        'concept': concept,
                        'sources': sections_with_concept,  # Ahora son secciones
                        'source_count': len(sections_with_concept),
                        'avg_score': total_score / len(sections_with_concept),
                        'reliability': len(sections_with_concept) / len(sections),
                        'validation_type': 'internal-triangulation',
                        'distribution': f"Aparece en {len(sections_with_concept)}/{len(sections)} secciones"
                    })
                elif len(sections_with_concept) == 1:
                    # Conceptos únicos de una sola sección (baja confiabilidad)
                    triangulated_concepts.append({
                        'concept': concept,
                        'sources': sections_with_concept,
                        'source_count': len(sections_with_concept),
                        'avg_score': total_score,
                        'reliability': 1 / len(sections),  # Baja confiabilidad
                        'validation_type': 'single-section',
                        'distribution': f"Solo en {sections_with_concept[0]}"
                    })
            
            # Ordenar por confiabilidad
            triangulated_concepts.sort(key=lambda x: (x['reliability'], x['avg_score']), reverse=True)
            
            return {
                'triangulated_concepts': triangulated_concepts[:30],  # Más conceptos para fuente única
                'total_sources': 1,
                'total_sections': len(sections),
                'total_concepts': len(all_concepts),
                'validated_concepts': len([c for c in triangulated_concepts if c['source_count'] >= 2]),
                'unique_concepts': len([c for c in triangulated_concepts if c['source_count'] == 1]),
                'sources': [source_name],
                'analysis_mode': 'single-source-internal',
                'info': f"Análisis interno del documento dividido en {len(sections)} secciones"
            }
            
        except Exception as e:
            logger.error(f"Error en triangulación de fuente única: {e}")
            return {'error': str(e)}
    
    def _get_concept_context(self, concept: str, chunks: List[Dict], max_length: int = 300) -> str:
        """Obtener contexto de un concepto específico con caché"""
        try:
            # Crear clave de caché
            content_hash = hashlib.md5(
                f"{concept}_{str([chunk.get('content', '')[:50] for chunk in chunks])}".encode()
            ).hexdigest()
            cache_key = f"context_{content_hash}_{max_length}"
            
            # Verificar caché
            if cache_key in self._concept_analysis_cache:
                return self._concept_analysis_cache[cache_key]
            
            contexts = []
            concept_lower = concept.lower()
            
            # Búsqueda optimizada
            for chunk in chunks:
                content = chunk.get('content', '')
                if not content:
                    continue
                    
                content_lower = content.lower()
                if concept_lower in content_lower:
                    # Encontrar la posición del concepto de forma eficiente
                    pos = content_lower.find(concept_lower)
                    start = max(0, pos - 100)
                    end = min(len(content), pos + len(concept) + 100)
                    context = content[start:end].strip()
                    if context and context not in contexts:  # Evitar duplicados
                        contexts.append(context)
                        
                    if len(contexts) >= 3:  # Limitar para eficiencia
                        break
            
            result = ""
            if contexts:
                combined = " ... ".join(contexts)
                result = combined[:max_length]
            else:
                result = f"Contexto no encontrado para: {concept}"
            
            # Guardar en caché
            self._concept_analysis_cache[cache_key] = result
            return result
            
        except Exception as e:
            return f"Error obteniendo contexto: {str(e)}"
    
    def _find_related_concepts(self, main_concept: str, chunks: List[Dict]) -> List[Dict]:
        """Encontrar conceptos relacionados a un concepto principal con caché"""
        try:
            # Crear clave de caché
            content_hash = hashlib.md5(
                f"{main_concept}_{str([chunk.get('content', '')[:50] for chunk in chunks])}".encode()
            ).hexdigest()
            cache_key = f"related_{content_hash}"
            
            # Verificar caché
            if cache_key in self._concept_analysis_cache:
                return self._concept_analysis_cache[cache_key]
            
            related = []
            main_concept_lower = main_concept.lower()
            
            # Buscar en chunks que contengan el concepto principal de forma eficiente
            relevant_chunks = []
            for chunk in chunks:
                content = chunk.get('content', '')
                if content and main_concept_lower in content.lower():
                    relevant_chunks.append(chunk)
                    
                if len(relevant_chunks) >= 10:  # Limitar para eficiencia
                    break
            
            if relevant_chunks:
                # Extraer conceptos de chunks relevantes (ya optimizado con caché)
                related_concepts = self.extract_key_concepts(relevant_chunks)
                
                # Filtrar conceptos relacionados (excluyendo el principal)
                for concept in related_concepts:
                    if (concept['concept'].lower() != main_concept_lower and 
                        len(concept['concept']) > 3 and
                        concept['score'] > 0.01):
                        related.append({
                            'concept': concept['concept'],
                            'score': concept['score'],
                            'relation': 'relacionado con'
                        })
                        
                        if len(related) >= 5:  # Máximo 5 conceptos relacionados
                            break
            
            # Guardar en caché
            self._concept_analysis_cache[cache_key] = related
            return related
            
        except Exception as e:
            logger.warning(f"Error encontrando conceptos relacionados: {e}")
            return []
    
    def _extract_concepts_with_ngrams(self, chunks: List[Dict], max_concepts: int = 30) -> List[Dict]:
        """
        Extraer conceptos usando N-GRAMAS para detectar frases completas
        
        Mejora significativa sobre extracción simple:
        - Detecta frases de 1-3 palabras ("inteligencia artificial", "aprendizaje profundo")
        - Filtra conceptos más relevantes y coherentes
        - Prioriza conceptos compuestos sobre palabras sueltas
        """
        try:
            if not ADVANCED_ANALYSIS_AVAILABLE:
                # Fallback al método simple
                return self.extract_key_concepts(chunks)
            
            # Preparar textos
            all_texts = []
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                if content:
                    # Preprocesar manteniendo más contexto
                    processed = re.sub(r'[^\w\s]', ' ', content.lower())
                    processed = re.sub(r'\s+', ' ', processed).strip()
                    all_texts.append(processed)
            
            if not all_texts:
                return self.extract_key_concepts(chunks)
            
            # Usar TF-IDF con n-gramas para detectar frases importantes
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=self._get_spanish_stopwords(),
                ngram_range=(1, 3),  # Detecta frases de hasta 3 palabras
                min_df=max(1, len(all_texts) // 10),
                max_df=0.85,
                token_pattern=r'\b[a-záéíóúñü]+\b'  # Solo palabras válidas en español
            )
            
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            # Crear lista de conceptos con scores
            concepts = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    concept_text = feature_names[i]
                    # Priorizar conceptos compuestos (frases)
                    word_count = len(concept_text.split())
                    adjusted_score = score * (1 + (word_count - 1) * 0.3)  # Bonus para n-gramas
                    
                    concepts.append({
                        'concept': concept_text.title(),  # Capitalizar para mejor presentación
                        'score': float(adjusted_score),
                        'type': 'ngram' if word_count > 1 else 'unigram',
                        'word_count': word_count
                    })
            
            # Ordenar por score ajustado y retornar top conceptos
            concepts_sorted = sorted(concepts, key=lambda x: x['score'], reverse=True)
            
            return concepts_sorted[:max_concepts]
            
        except Exception as e:
            logger.warning(f"Error extrayendo conceptos con n-gramas: {e}")
            # Fallback al método simple
            return self.extract_key_concepts(chunks)
    
    def _identify_intelligent_main_theme(self, chunks: List[Dict], concepts: List[Dict]) -> Dict:
        """
        Identificar tema principal de forma INTELIGENTE
        
        Mejoras:
        - Considera los conceptos más frecuentes Y relevantes
        - Analiza el contexto completo del documento
        - Genera nombre descriptivo y coherente
        """
        try:
            # Tomar los top 3 conceptos más relevantes
            top_concepts = concepts[:3] if concepts else []
            
            if not top_concepts:
                return {
                    'name': 'Análisis de Contenido',
                    'description': 'Tema principal del documento'
                }
            
            # Crear nombre del tema combinando conceptos principales
            if len(top_concepts) >= 2:
                # Combinar los 2-3 conceptos más importantes
                theme_parts = [c['concept'] for c in top_concepts[:2]]
                theme_name = " y ".join(theme_parts)
            else:
                theme_name = top_concepts[0]['concept']
            
            # Generar descripción basada en contexto
            description_parts = []
            for concept in top_concepts[:3]:
                context = self._get_concept_context(concept['concept'], chunks, max_length=150)
                if context and "no encontrado" not in context.lower():
                    description_parts.append(context[:100])
            
            description = " | ".join(description_parts) if description_parts else f"Análisis centrado en {theme_name}"
            
            return {
                'name': theme_name,
                'description': description,
                'key_concepts': [c['concept'] for c in top_concepts],
                'confidence': sum(c['score'] for c in top_concepts) / len(top_concepts)
            }
            
        except Exception as e:
            logger.error(f"Error identificando tema principal: {e}")
            return {
                'name': 'Tema Principal',
                'description': 'Análisis de contenido del documento'
            }
    
    def _analyze_concept_hierarchy_with_ai(self, chunks: List[Dict]) -> Optional[Dict]:
        """
        Analizar jerarquía de conceptos usando IA (LLM) para análisis semántico profundo
        
        Este método usa el modelo de lenguaje para identificar:
        - Conceptos principales de forma semántica (no solo frecuencia)
        - Relaciones jerárquicas reales entre conceptos
        - Agrupaciones temáticas coherentes
        """
        try:
            from utils.ollama_client import OllamaClient
            from config.settings import config as global_config
            
            # Combinar contenido de chunks (limitado para eficiencia)
            combined_content = "\n\n".join([
                chunk.get('content', '')[:800] for chunk in chunks[:15]  # Máximo 15 chunks
            ])
            
            # Prompt para análisis con IA
            analysis_prompt = f"""Analiza el siguiente texto y extrae una jerarquía de conceptos coherente y bien estructurada.

TEXTO A ANALIZAR:
{combined_content}

INSTRUCCIONES:
1. Identifica el TEMA CENTRAL del texto (una frase descriptiva)
2. Identifica 5-6 CONCEPTOS PRINCIPALES más importantes
3. Para cada concepto principal, identifica 2-3 SUB-CONCEPTOS relacionados
4. Proporciona contexto breve para cada concepto

Responde SOLO en formato JSON válido:
{{
  "tema_central": "nombre del tema central",
  "descripcion": "descripción breve del tema",
  "conceptos_principales": [
    {{
      "nombre": "concepto 1",
      "relevancia": 0.9,
      "contexto": "contexto breve",
      "sub_conceptos": [
        {{"nombre": "sub-concepto 1", "relacion": "es parte de", "relevancia": 0.7}}
      ]
    }}
  ]
}}"""
            
            # Usar LLM para análisis
            ollama_client = OllamaClient()
            response = ollama_client.generate_response(
                model=global_config.DEFAULT_LLM_MODEL,
                prompt=analysis_prompt,
                max_tokens=1500
            )
            
            # Intentar parsear respuesta JSON
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    ai_analysis = json.loads(json_match.group())
                    
                    # Convertir a formato de jerarquía
                    hierarchy = {
                        'main_theme': {
                            'name': ai_analysis.get('tema_central', 'Tema Principal'),
                            'description': ai_analysis.get('descripcion', 'Análisis del documento')
                        },
                        'main_concepts': [],
                        'cross_relations': [],
                        'info': {
                            'total_concepts': len(ai_analysis.get('conceptos_principales', [])),
                            'analysis_method': 'ai_semantic_analysis',
                            'documents_analyzed': len(chunks),
                            'model_used': global_config.DEFAULT_LLM_MODEL
                        }
                    }
                    
                    # Convertir conceptos principales
                    for concepto in ai_analysis.get('conceptos_principales', [])[:6]:
                        concept_data = {
                            'name': concepto.get('nombre', 'Concepto'),
                            'relevance': concepto.get('relevancia', 0.5),
                            'context': concepto.get('contexto', 'Sin contexto')[:200],
                            'sub_concepts': []
                        }
                        
                        # Agregar sub-conceptos
                        for sub in concepto.get('sub_conceptos', [])[:4]:
                            sub_concept = {
                                'name': sub.get('nombre', 'Sub-concepto'),
                                'relation': sub.get('relacion', 'relacionado con'),
                                'relevance': sub.get('relevancia', 0.5),
                                'context': sub.get('contexto', f"Sub-concepto de {concept_data['name']}")[:150]
                            }
                            concept_data['sub_concepts'].append(sub_concept)
                        
                        hierarchy['main_concepts'].append(concept_data)
                    
                    # Identificar relaciones cruzadas
                    hierarchy['cross_relations'] = self._identify_cross_relations(hierarchy['main_concepts'])
                    
                    logger.info(f"Análisis con IA completado: {len(hierarchy['main_concepts'])} conceptos")
                    return hierarchy
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parseando JSON de IA: {e}")
                    # Fallback a análisis normal
                    logger.info("Fallback a análisis normal después de error JSON")
                    return None  # Retornar None para forzar uso del método normal
            else:
                # Si no hay JSON, usar análisis normal
                logger.warning("No se encontró JSON en respuesta de IA, usando análisis normal")
                return None  # Retornar None para forzar uso del método normal
            
        except Exception as e:
            logger.error(f"Error en análisis con IA: {e}")
            # Fallback a análisis normal
            return None  # Retornar None para forzar uso del método normal
    
    def _analyze_concept_hierarchy(self, chunks: List[Dict]) -> Optional[Dict]:
        """
        Analizar jerarquía de conceptos de forma INTELIGENTE Y COHERENTE
        
        MEJORAS:
        - Usa n-gramas para detectar conceptos compuestos (ej: "inteligencia artificial")
        - Agrupa conceptos relacionados semánticamente
        - Identifica relaciones jerárquicas reales (no solo frecuencia)
        - Detecta tema central considerando contexto completo
        """
        try:
            # Crear clave de caché basada en el contenido
            content_hash = hashlib.md5(
                str([chunk.get('content', '')[:100] for chunk in chunks]).encode()
            ).hexdigest()
            cache_key = f"concept_hierarchy_{content_hash}"
            
            # Verificar caché
            if cache_key in self._concept_analysis_cache:
                return self._concept_analysis_cache[cache_key]
            
            # Validar entrada
            if not self._validate_chunks(chunks):
                return None
            
            # MEJORA 1: Extraer conceptos con N-GRAMAS (detecta frases completas)
            concepts = self._extract_concepts_with_ngrams(chunks)
            if not concepts:
                return None
            
            # MEJORA 2: Identificar tema central de forma más inteligente
            main_theme = self._identify_intelligent_main_theme(chunks, concepts)
            
            # Clasificar conceptos por niveles jerárquicos
            hierarchy = {
                'main_theme': main_theme,
                'main_concepts': [],
                'sub_concepts': [],
                'cross_relations': [],
                'info': {
                    'total_concepts': len(concepts),
                    'analysis_method': 'hierarchical_semantic',
                    'documents_analyzed': len(chunks)
                }
            }
            
            # Conceptos principales (nivel 1) - procesamiento optimizado
            main_concepts = concepts[:6]  # Top 6 conceptos principales
            for concept in main_concepts:
                concept_data = {
                    'name': concept['concept'],
                    'relevance': concept['score'],
                    'context': self._get_concept_context(concept['concept'], chunks)[:200],
                    'sub_concepts': []
                }
                
                # Buscar sub-conceptos relacionados de forma eficiente
                related_concepts = self._find_related_concepts(concept['concept'], chunks)
                for related in related_concepts[:4]:  # Máximo 4 sub-conceptos
                    sub_concept = {
                        'name': related['concept'],
                        'relation': related.get('relation', 'relacionado con'),
                        'relevance': related['score'],
                        'context': self._get_concept_context(related['concept'], chunks)[:150]
                    }
                    concept_data['sub_concepts'].append(sub_concept)
                
                hierarchy['main_concepts'].append(concept_data)
            
            # Identificar relaciones cruzadas de forma eficiente
            hierarchy['cross_relations'] = self._identify_cross_relations(hierarchy['main_concepts'])
            
            # Guardar en caché
            self._concept_analysis_cache[cache_key] = hierarchy
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error analizando jerarquía de conceptos: {e}")
            return None
    
    def _analyze_intelligent_mind_map_structure(self, chunks: List[Dict]) -> Optional[Dict]:
        """
        Análisis MEJORADO para estructura de mapa mental
        
        MEJORAS:
        - Usa n-gramas para conceptos más coherentes
        - Identifica tema central inteligente
        - Agrupa conceptos relacionados semánticamente
        """
        try:
            logger.info("Iniciando análisis inteligente de estructura de mapa mental...")
            
            # MEJORA 1: Usar extracción con n-gramas para conceptos más coherentes
            concepts = self._extract_concepts_with_ngrams(chunks)
            if not concepts:
                logger.warning("No se encontraron conceptos, usando estructura básica")
                return self._fallback_mind_map_structure(chunks)
            
            # MEJORA 2: Identificar tema central de forma inteligente
            central_theme_data = self._identify_intelligent_main_theme(chunks, concepts)
            central_theme = {
                'name': central_theme_data.get('name', 'Análisis de Contenido'),
                'description': central_theme_data.get('description', 'Tema central del documento')
            }
            
            # 3. Convertir conceptos a ramas principales (método directo)
            main_branches = []
            main_concepts = concepts[:6]  # Top 6 conceptos principales
            
            for i, concept in enumerate(main_concepts):
                branch = {
                    'name': concept['concept'],
                    'importance': concept['score'],
                    'description': self._get_concept_context(concept['concept'], chunks)[:200],
                    'sub_branches': []
                }
                
                # Buscar sub-ramas relacionadas (método rápido)
                related_concepts = self._find_related_concepts(concept['concept'], chunks)
                for related in related_concepts[:3]:  # Máximo 3 sub-ramas
                    sub_branch = {
                        'name': related['concept'],
                        'relation': related.get('relation', 'relacionado con'),
                        'relevance': related['score'],
                        'context': self._get_concept_context(related['concept'], chunks)[:150]
                    }
                    branch['sub_branches'].append(sub_branch)
                
                main_branches.append(branch)
            
            # 4. Identificar conexiones cruzadas (método rápido)
            cross_connections = self._identify_branch_connections(main_branches)
            
            # 5. Crear estructura final
            structure = {
                'central_theme': central_theme,
                'main_branches': main_branches,
                'cross_connections': cross_connections,
                'info': {
                    'total_branches': len(main_branches),
                    'analysis_method': 'fast_traditional_analysis',
                    'semantic_depth': len(concepts)
                }
            }
            
            logger.info(f"Estructura rápida creada con {len(main_branches)} ramas en método tradicional")
            return structure
            
        except Exception as e:
            logger.error(f"Error en análisis rápido: {e}")
            logger.info("Usando estructura de fallback...")
            return self._fallback_mind_map_structure(chunks)
    
    def _llm_semantic_analysis(self, chunks: List[Dict]) -> Dict:
        """Análisis semántico profundo usando modelo de lenguaje"""
        try:
            from utils.ollama_client import OllamaClient
            
            # Combinar contenido de chunks
            combined_content = "\n\n".join([
                chunk.get('content', '')[:500] for chunk in chunks[:10]  # Limitar para eficiencia
            ])
            
            # Prompt para análisis semántico
            analysis_prompt = f"""
            Analiza el siguiente texto y extrae información estructurada para crear un mapa mental inteligente:

            TEXTO:
            {combined_content}

            INSTRUCCIONES:
            1. Identifica los 5-8 conceptos más importantes y relevantes
            2. Para cada concepto, identifica 2-4 sub-conceptos relacionados
            3. Determina el tema central que conecta todos los conceptos
            4. Identifica relaciones semánticas entre conceptos
            5. Proporciona contexto breve para cada concepto

            FORMATO DE RESPUESTA (JSON):
            {{
                "tema_central": "tema principal del contenido",
                "conceptos_principales": [
                    {{
                        "nombre": "nombre del concepto",
                        "importancia": 0.9,
                        "descripcion": "descripción breve",
                        "sub_conceptos": [
                            {{
                                "nombre": "sub-concepto",
                                "relacion": "tipo de relación",
                                "relevancia": 0.8
                            }}
                        ]
                    }}
                ],
                "relaciones": [
                    {{
                        "concepto_1": "nombre concepto 1",
                        "concepto_2": "nombre concepto 2",
                        "tipo_relacion": "causa-efecto/similitud/oposición/etc",
                        "fuerza": 0.7
                    }}
                ]
            }}
            """
            
            # Usar cliente Ollama para análisis
            ollama_client = OllamaClient()
            response = ollama_client.generate_response(
                model=config.DEFAULT_LLM_MODEL,
                prompt=analysis_prompt
            )
            
            # Intentar parsear respuesta JSON
            import json
            import re
            
            # Extraer JSON de la respuesta
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    semantic_data = json.loads(json_match.group())
                    return semantic_data
                except json.JSONDecodeError:
                    pass
            
            # Fallback: análisis básico si falla el LLM
            return self._fallback_semantic_analysis(chunks)
            
        except Exception as e:
            logger.error(f"Error en análisis semántico LLM: {e}")
            return self._fallback_semantic_analysis(chunks)
    
    def _llm_identify_central_theme(self, chunks: List[Dict], semantic_analysis: Dict) -> Dict:
        """Identificar tema central usando análisis LLM"""
        try:
            tema_central = semantic_analysis.get('tema_central', 'Análisis Cualitativo')
            
            return {
                'name': tema_central,
                'description': f"Tema central identificado mediante análisis semántico: {tema_central}"
            }
            
        except Exception as e:
            logger.error(f"Error identificando tema central LLM: {e}")
            return {'name': 'Tema Principal', 'description': 'Análisis de contenido'}
    
    def _llm_extract_semantic_branches(self, chunks: List[Dict], semantic_analysis: Dict) -> List[Dict]:
        """Extraer ramas principales usando análisis LLM"""
        try:
            branches = []
            conceptos_principales = semantic_analysis.get('conceptos_principales', [])
            
            for i, concepto in enumerate(conceptos_principales[:8]):  # Máximo 8 ramas
                branch = {
                    'name': concepto.get('nombre', f'Concepto {i+1}'),
                    'importance': concepto.get('importancia', 0.5),
                    'description': concepto.get('descripcion', 'Concepto identificado automáticamente'),
                    'sub_branches': []
                }
                
                # Agregar sub-conceptos
                sub_conceptos = concepto.get('sub_conceptos', [])
                for sub_concepto in sub_conceptos[:4]:  # Máximo 4 sub-ramas
                    sub_branch = {
                        'name': sub_concepto.get('nombre', 'Sub-concepto'),
                        'relation': sub_concepto.get('relacion', 'relacionado con'),
                        'relevance': sub_concepto.get('relevancia', 0.5),
                        'context': f"Sub-concepto de {branch['name']}"
                    }
                    branch['sub_branches'].append(sub_branch)
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.error(f"Error extrayendo ramas semánticas LLM: {e}")
            return self._fallback_extract_branches(chunks)
    
    def _llm_identify_semantic_connections(self, branches: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Identificar conexiones semánticas usando análisis LLM"""
        try:
            connections = []
            
            # Crear conexiones basadas en las relaciones identificadas por el LLM
            for i, branch1 in enumerate(branches):
                for j, branch2 in enumerate(branches[i+1:], i+1):
                    # Simular conexión basada en similitud semántica
                    if self._calculate_semantic_similarity(branch1['name'], branch2['name']) > 0.3:
                        connections.append({
                            'from': f"branch_{i}",
                            'to': f"branch_{j}",
                            'relation': 'relacionado semánticamente',
                            'strength': self._calculate_semantic_similarity(branch1['name'], branch2['name'])
                        })
            
            return connections[:5]  # Máximo 5 conexiones
            
        except Exception as e:
            logger.error(f"Error identificando conexiones semánticas LLM: {e}")
            return []
    
    def _calculate_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Calcular similitud semántica básica entre dos conceptos"""
        try:
            # Similitud básica basada en palabras comunes
            words1 = set(concept1.lower().split())
            words2 = set(concept2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def _fallback_semantic_analysis(self, chunks: List[Dict]) -> Dict:
        """Análisis semántico de respaldo si falla el LLM"""
        try:
            # Análisis básico usando técnicas tradicionales
            concept_network = self._build_concept_network(chunks)
            
            # Convertir a formato esperado
            conceptos_principales = []
            for concept, data in list(concept_network.items())[:6]:
                concepto = {
                    'nombre': concept.title(),
                    'importancia': min(1.0, data['frequency'] / 10),
                    'descripcion': f"Concepto extraído automáticamente: {concept}",
                    'sub_conceptos': []
                }
                
                # Agregar sub-conceptos basados en co-ocurrencias
                for related_concept, count in list(data['cooccurrences'].items())[:3]:
                    sub_concepto = {
                        'nombre': related_concept.title(),
                        'relacion': 'co-ocurre con',
                        'relevancia': min(1.0, count / 5)
                    }
                    concepto['sub_conceptos'].append(sub_concepto)
                
                conceptos_principales.append(concepto)
            
            return {
                'tema_central': 'Análisis de Contenido',
                'conceptos_principales': conceptos_principales,
                'relaciones': []
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de respaldo: {e}")
            return {
                'tema_central': 'Análisis Cualitativo',
                'conceptos_principales': [],
                'relaciones': []
            }
    
    def _fallback_mind_map_structure(self, chunks: List[Dict]) -> Dict:
        """Estructura de mapa mental de respaldo usando análisis tradicional"""
        try:
            logger.info("Generando estructura de mapa mental usando métodos tradicionales...")
            
            # Usar análisis de respaldo
            semantic_analysis = self._fallback_semantic_analysis(chunks)
            
            # Crear estructura básica
            central_theme = {
                'name': semantic_analysis.get('tema_central', 'Análisis de Contenido'),
                'description': 'Tema central identificado mediante análisis tradicional'
            }
            
            # Convertir conceptos principales a ramas
            main_branches = []
            conceptos = semantic_analysis.get('conceptos_principales', [])
            
            for i, concepto in enumerate(conceptos[:6]):  # Máximo 6 ramas
                branch = {
                    'name': concepto.get('nombre', f'Concepto {i+1}'),
                    'importance': concepto.get('importancia', 0.5),
                    'description': concepto.get('descripcion', 'Concepto identificado automáticamente'),
                    'sub_branches': []
                }
                
                # Agregar sub-ramas
                sub_conceptos = concepto.get('sub_conceptos', [])
                for sub_concepto in sub_conceptos[:3]:  # Máximo 3 sub-ramas
                    sub_branch = {
                        'name': sub_concepto.get('nombre', 'Sub-concepto'),
                        'relation': sub_concepto.get('relacion', 'relacionado con'),
                        'relevance': sub_concepto.get('relevancia', 0.5),
                        'context': f"Sub-concepto de {branch['name']}"
                    }
                    branch['sub_branches'].append(sub_branch)
                
                main_branches.append(branch)
            
            # Si no hay ramas, crear algunas básicas
            if not main_branches:
                logger.warning("No se encontraron conceptos, creando ramas básicas...")
                main_branches = [
                    {
                        'name': 'Contenido Principal',
                        'importance': 0.8,
                        'description': 'Contenido principal del documento',
                        'sub_branches': [
                            {
                                'name': 'Información Clave',
                                'relation': 'contiene',
                                'relevance': 0.7,
                                'context': 'Información relevante del documento'
                            }
                        ]
                    },
                    {
                        'name': 'Temas Secundarios',
                        'importance': 0.6,
                        'description': 'Temas de apoyo y contexto',
                        'sub_branches': [
                            {
                                'name': 'Detalles Adicionales',
                                'relation': 'complementa',
                                'relevance': 0.5,
                                'context': 'Información complementaria'
                            }
                        ]
                    }
                ]
            
            structure = {
                'central_theme': central_theme,
                'main_branches': main_branches,
                'cross_connections': [],  # Conexiones básicas
                'info': {
                    'total_branches': len(main_branches),
                    'analysis_method': 'traditional_fallback_analysis',
                    'semantic_depth': len(conceptos)
                }
            }
            
            logger.info(f"Estructura de fallback creada con {len(main_branches)} ramas")
            return structure
            
        except Exception as e:
            logger.error(f"Error en estructura de fallback: {e}")
            # Estructura mínima de emergencia
            return {
                'central_theme': {
                    'name': 'Análisis de Documento',
                    'description': 'Análisis básico del contenido'
                },
                'main_branches': [
                    {
                        'name': 'Contenido',
                        'importance': 0.7,
                        'description': 'Contenido del documento',
                        'sub_branches': [
                            {
                                'name': 'Información',
                                'relation': 'contiene',
                                'relevance': 0.6,
                                'context': 'Información del documento'
                            }
                        ]
                    }
                ],
                'cross_connections': [],
                'info': {
                    'total_branches': 1,
                    'analysis_method': 'emergency_fallback',
                    'semantic_depth': 0
                }
            }

    def _fallback_extract_branches(self, chunks: List[Dict]) -> List[Dict]:
        """Extracción de ramas de respaldo"""
        try:
            # Usar método tradicional como respaldo
            concept_network = self._build_concept_network(chunks)
            branches = []
            
            for i, (concept, data) in enumerate(list(concept_network.items())[:6]):
                branch = {
                    'name': concept.title(),
                    'importance': min(1.0, data['frequency'] / 10),
                    'description': f"Concepto identificado: {concept}",
                    'sub_branches': []
                }
                
                # Agregar sub-ramas
                for related_concept, count in list(data['cooccurrences'].items())[:3]:
                    sub_branch = {
                        'name': related_concept.title(),
                        'relation': 'relacionado con',
                        'relevance': min(1.0, count / 5),
                        'context': f"Relacionado con {concept}"
                    }
                    branch['sub_branches'].append(sub_branch)
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.error(f"Error en extracción de ramas de respaldo: {e}")
            return []
    
    def _build_concept_network(self, chunks: List[Dict]) -> Dict:
        """Construir red de conceptos basada en co-ocurrencia y frecuencia"""
        try:
            concept_network = defaultdict(lambda: {'frequency': 0, 'cooccurrences': defaultdict(int), 'contexts': []})
            
            # Extraer conceptos de cada chunk
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                
                # Extraer frases importantes (2-4 palabras)
                important_phrases = self._extract_important_phrases(content)
                
                # Registrar frecuencias y co-ocurrencias
                for phrase in important_phrases:
                    concept_network[phrase]['frequency'] += 1
                    concept_network[phrase]['contexts'].append(content[:200])
                    
                    # Co-ocurrencias con otras frases en el mismo chunk
                    for other_phrase in important_phrases:
                        if phrase != other_phrase:
                            concept_network[phrase]['cooccurrences'][other_phrase] += 1
            
            return dict(concept_network)
            
        except Exception as e:
            logger.error(f"Error construyendo red de conceptos: {e}")
            return {}
    
    def _extract_important_phrases(self, text: str) -> List[str]:
        """Extraer frases importantes usando técnicas de NLP"""
        try:
            # Limpiar texto
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            
            # Filtrar palabras vacías
            stop_words = set(self._get_spanish_stopwords())
            filtered_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
            
            phrases = []
            
            # Extraer bigramas y trigramas significativos
            for i in range(len(filtered_words) - 1):
                bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
                phrases.append(bigram.lower())
                
                if i < len(filtered_words) - 2:
                    trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
                    phrases.append(trigram.lower())
            
            # Filtrar frases por relevancia (frecuencia mínima)
            phrase_counts = Counter(phrases)
            relevant_phrases = [phrase for phrase, count in phrase_counts.items() if count >= 2]
            
            return relevant_phrases[:20]  # Top 20 frases más relevantes
            
        except Exception as e:
            logger.error(f"Error extrayendo frases importantes: {e}")
            return []
    
    def _identify_central_theme_advanced(self, chunks: List[Dict]) -> Dict:
        """Identificar tema central usando análisis semántico avanzado"""
        try:
            # Combinar todo el contenido
            all_content = " ".join([chunk.get('content', '') for chunk in chunks])
            
            # Extraer conceptos más frecuentes y significativos
            important_phrases = self._extract_important_phrases(all_content)
            phrase_counts = Counter(important_phrases)
            
            # Seleccionar el tema central basado en frecuencia y centralidad
            if phrase_counts:
                central_concept = phrase_counts.most_common(1)[0][0]
                
                # Generar descripción del tema central
                description = self._generate_theme_description(central_concept, chunks)
                
                return {
                    'name': central_concept.title(),
                    'description': description
                }
            
            return {
                'name': 'Análisis Cualitativo',
                'description': 'Tema central identificado automáticamente'
            }
            
        except Exception as e:
            logger.error(f"Error identificando tema central avanzado: {e}")
            return {'name': 'Tema Principal', 'description': 'Análisis de contenido'}
    
    def _extract_semantic_branches(self, chunks: List[Dict], concept_network: Dict) -> List[Dict]:
        """Extraer ramas principales usando clustering semántico"""
        try:
            branches = []
            
            # Ordenar conceptos por importancia (frecuencia + co-ocurrencias)
            concept_scores = []
            for concept, data in concept_network.items():
                importance_score = data['frequency'] + sum(data['cooccurrences'].values()) * 0.1
                concept_scores.append((concept, importance_score, data))
            
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Crear ramas principales (top 6 conceptos)
            for i, (concept, score, data) in enumerate(concept_scores[:6]):
                branch = {
                    'name': concept.title(),
                    'importance': min(1.0, score / max(concept_scores[0][1], 1)),  # Normalizar
                    'description': self._generate_concept_description(concept, chunks),
                    'sub_branches': []
                }
                
                # Encontrar sub-ramas relacionadas
                related_concepts = sorted(
                    data['cooccurrences'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:4]
                
                for related_concept, cooccurrence_count in related_concepts:
                    if related_concept in concept_network:
                        sub_branch = {
                            'name': related_concept.title(),
                            'relation': 'relacionado con',
                            'relevance': min(1.0, cooccurrence_count / max(data['cooccurrences'].values(), 1)),
                            'context': self._get_concept_context(related_concept, chunks)[:150]
                        }
                        branch['sub_branches'].append(sub_branch)
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.error(f"Error extrayendo ramas semánticas: {e}")
            return []
    
    def _generate_theme_description(self, theme: str, chunks: List[Dict]) -> str:
        """Generar descripción del tema basada en el contexto"""
        try:
            relevant_contexts = []
            theme_lower = theme.lower()
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if theme_lower in content:
                    # Extraer oración que contiene el tema
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if theme_lower in sentence and len(sentence.strip()) > 20:
                            relevant_contexts.append(sentence.strip())
                            break
                
                if len(relevant_contexts) >= 3:
                    break
            
            if relevant_contexts:
                return ". ".join(relevant_contexts[:2])[:200] + "..."
            
            return f"Tema central relacionado con {theme}"
            
        except Exception as e:
            logger.error(f"Error generando descripción del tema: {e}")
            return f"Análisis de {theme}"
    
    def _generate_concept_description(self, concept: str, chunks: List[Dict]) -> str:
        """Generar descripción del concepto basada en el contexto"""
        try:
            concept_lower = concept.lower()
            contexts = []
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if concept_lower in content:
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if concept_lower in sentence and len(sentence.strip()) > 15:
                            contexts.append(sentence.strip())
                            break
                
                if len(contexts) >= 2:
                    break
            
            if contexts:
                return ". ".join(contexts)[:200] + "..."
            
            return f"Concepto relacionado con {concept}"
            
        except Exception as e:
            logger.error(f"Error generando descripción del concepto: {e}")
            return f"Análisis de {concept}"
    
    def _identify_semantic_connections(self, branches: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Identificar conexiones semánticas entre ramas"""
        try:
            connections = []
            
            # Buscar conexiones entre ramas principales
            for i, branch1 in enumerate(branches):
                for j, branch2 in enumerate(branches[i+1:], i+1):
                    # Verificar si los conceptos aparecen juntos en el contenido
                    concept1 = branch1['name'].lower()
                    concept2 = branch2['name'].lower()
                    
                    connection_strength = 0
                    for chunk in chunks:
                        content = chunk.get('content', '').lower()
                        if concept1 in content and concept2 in content:
                            connection_strength += 1
                    
                    if connection_strength >= 2:  # Aparecen juntos en al menos 2 chunks
                        connections.append({
                            'from': f"branch_{i}",
                            'to': f"branch_{j}",
                            'relation': 'co-ocurre con',
                            'strength': connection_strength
                        })
            
            return connections[:5]  # Máximo 5 conexiones
            
        except Exception as e:
            logger.error(f"Error identificando conexiones semánticas: {e}")
            return []
    
    def _lighten_color_hex(self, hex_color: str, factor: float = 0.3) -> str:
        """Aclarar un color hexadecimal"""
        try:
            # Remover el # si está presente
            hex_color = hex_color.lstrip('#')
            
            # Convertir a RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Aclarar cada componente
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            
            # Convertir de vuelta a hex
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception:
            return "#cccccc"  # Color por defecto si hay error
    
    
    def _identify_main_theme(self, chunks: List[Dict]) -> Dict:
        """Identificar el tema principal del contenido con caché"""
        try:
            # Crear clave de caché basada en el contenido
            content_hash = hashlib.md5(
                str([chunk.get('content', '')[:100] for chunk in chunks]).encode()
            ).hexdigest()
            cache_key = f"main_theme_{content_hash}"
            
            # Verificar caché
            if cache_key in self._concept_analysis_cache:
                return self._concept_analysis_cache[cache_key]
            
            # Validar entrada
            if not chunks:
                return {'name': 'Sin Tema', 'description': 'No hay contenido disponible'}
            
            # Generar resumen principal de forma eficiente
            main_summary = self.generate_rag_summary(chunks, 150)
            
            # Extraer conceptos más frecuentes usando texto preprocesado
            all_text = " ".join([chunk.get('content', '') for chunk in chunks if chunk.get('content')])
            if not all_text.strip():
                return {'name': 'Sin Tema', 'description': 'Contenido vacío'}
            
            # Usar función de preprocesamiento optimizada
            processed_text = self.preprocess_text(all_text)
            words = processed_text.split()
            
            if not words:
                return {'name': 'Sin Tema', 'description': main_summary}
            
            word_freq = Counter(words)
            
            # Filtrar palabras usando stopwords cacheadas
            stop_words = set(self._get_spanish_stopwords())
            relevant_words = [
                word for word, count in word_freq.most_common(20) 
                if len(word) > 3 and word not in stop_words and word.isalpha()
            ]
            
            # Crear nombre del tema principal más descriptivo
            if relevant_words:
                # Tomar las 2-3 palabras más relevantes
                theme_words = relevant_words[:3]
                theme_name = " ".join(theme_words).title()
            else:
                theme_name = "Tema Principal"
            
            result = {
                'name': theme_name,
                'description': main_summary,
                'key_terms': relevant_words[:10],  # Términos clave adicionales
                'documents_analyzed': len(chunks)
            }
            
            # Guardar en caché
            self._concept_analysis_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error identificando tema principal: {e}")
            return {'name': 'Error en Análisis', 'description': 'No se pudo identificar el tema'}
    
    def _group_concepts_into_branches(self, concepts: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Agrupar conceptos en ramas temáticas para mapa mental"""
        try:
            branches = []
            
            # Tomar los conceptos principales como ramas
            main_concepts = concepts[:6]  # Máximo 6 ramas principales
            
            for concept in main_concepts:
                branch = {
                    'name': concept['concept'],
                    'importance': concept['score'],
                    'description': self._get_concept_context(concept['concept'], chunks)[:200],
                    'sub_branches': [],
                    'details': []
                }
                
                # Buscar sub-ramas relacionadas
                related_concepts = self._find_related_concepts(concept['concept'], chunks)
                for related in related_concepts[:4]:  # Máximo 4 sub-ramas
                    sub_branch = {
                        'name': related['concept'],
                        'relation': related.get('relation', 'relacionado'),
                        'relevance': related['score'],
                        'context': self._get_concept_context(related['concept'], chunks)[:150],
                        'details': []
                    }
                    
                    # Agregar detalles específicos
                    details = self._extract_concept_details(related['concept'], chunks)
                    sub_branch['details'] = details[:3]  # Máximo 3 detalles
                    
                    branch['sub_branches'].append(sub_branch)
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.error(f"Error agrupando conceptos en ramas: {e}")
            return []
    
    def _extract_concept_details(self, concept: str, chunks: List[Dict]) -> List[Dict]:
        """Extraer detalles específicos de un concepto"""
        try:
            details = []
            concept_lower = concept.lower()
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if concept_lower in content:
                    # Extraer oraciones que contienen el concepto
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if concept_lower in sentence and len(sentence.strip()) > 20:
                            # Extraer palabras clave de la oración
                            words = re.findall(r'\b\w+\b', sentence)
                            key_words = [w for w in words if len(w) > 4 and w != concept_lower]
                            
                            if key_words:
                                detail_name = key_words[0].title()
                                details.append({
                                    'name': detail_name,
                                    'description': sentence.strip()[:100]
                                })
                            
                            if len(details) >= 5:  # Máximo 5 detalles por concepto
                                break
                
                if len(details) >= 5:
                    break
            
            return details
            
        except Exception as e:
            logger.error(f"Error extrayendo detalles del concepto: {e}")
            return []
    
    def _identify_cross_relations(self, main_concepts: List[Dict]) -> List[Dict]:
        """Identificar relaciones cruzadas entre conceptos principales"""
        try:
            relations = []
            
            for i, concept1 in enumerate(main_concepts):
                for j, concept2 in enumerate(main_concepts[i+1:], i+1):
                    # Buscar relaciones semánticas entre conceptos
                    relation_strength = self._calculate_concept_similarity(
                        concept1['name'], concept2['name']
                    )
                    
                    if relation_strength > 0.3:  # Umbral de relación
                        relations.append({
                            'from': f"concept_{i}",
                            'to': f"concept_{j}",
                            'relation': 'relacionado',
                            'strength': relation_strength
                        })
            
            return relations
            
        except Exception as e:
            logger.error(f"Error identificando relaciones cruzadas: {e}")
            return []
    
    def _identify_branch_connections(self, branches: List[Dict]) -> List[Dict]:
        """Identificar conexiones entre ramas del mapa mental"""
        try:
            connections = []
            
            for i, branch1 in enumerate(branches):
                for j, branch2 in enumerate(branches[i+1:], i+1):
                    # Calcular similitud entre ramas
                    similarity = self._calculate_concept_similarity(
                        branch1['name'], branch2['name']
                    )
                    
                    if similarity > 0.25:  # Umbral para conexiones cruzadas
                        connections.append({
                            'from': f"branch_{i}",
                            'to': f"branch_{j}",
                            'relation': 'conectado',
                            'strength': similarity
                        })
            
            return connections
            
        except Exception as e:
            logger.error(f"Error identificando conexiones entre ramas: {e}")
            return []
    
    def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calcular similitud básica entre dos conceptos"""
        try:
            # Similitud básica basada en palabras comunes
            words1 = set(concept1.lower().split())
            words2 = set(concept2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.error(f"Error calculando similitud de conceptos: {e}")
            return 0.0
    
    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Aclarar un color hexadecimal"""
        try:
            # Remover el # si está presente
            hex_color = hex_color.lstrip('#')
            
            # Convertir a RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Aclarar cada componente
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            
            # Convertir de vuelta a hexadecimal
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception as e:
            logger.error(f"Error aclarando color: {e}")
            return "#95a5a6"  # Color por defecto
    
    def _get_concept_color(self, score: float) -> str:
        """Obtener color basado en la puntuación del concepto"""
        if score > 0.1:
            return "#e74c3c"  # Rojo para alta relevancia
        elif score > 0.05:
            return "#f39c12"  # Naranja para relevancia media
        elif score > 0.02:
            return "#3498db"  # Azul para relevancia baja
        else:
            return "#95a5a6"  # Gris para muy baja relevancia
    
    def _calculate_topic_coherence(self, words: List[str], texts: List[str]) -> float:
        """Calcular coherencia de un tema"""
        try:
            # Implementación básica de coherencia
            word_counts = Counter()
            for text in texts:
                text_words = set(text.lower().split())
                for word in words:
                    if word in text_words:
                        word_counts[word] += 1
            
            if not word_counts:
                return 0.0
            
            # Coherencia basada en co-ocurrencia
            coherence = sum(word_counts.values()) / len(words)
            return min(coherence / len(texts), 1.0)
        except:
            return 0.0
    
    
    def _extract_advanced_themes_detailed(self, chunks: List[Dict], n_topics: int = 10) -> Dict:
        """Extracción avanzada de temas usando LDA con optimización (método legacy detallado)"""
        if not self._validate_chunks(chunks) or not ADVANCED_ANALYSIS_AVAILABLE:
            return self._basic_theme_extraction(chunks)
        
        try:
            # Crear clave de cache
            content_hash = hash(str(sorted([chunk.get('content', '')[:100] for chunk in chunks])))
            cache_key = f"themes_{content_hash}_{n_topics}"
            
            # Verificar cache
            if cache_key in self._concept_analysis_cache:
                logger.info("Usando extracción de temas desde cache")
                return self._concept_analysis_cache[cache_key]
            
            # Preparar textos con validación mejorada
            texts = []
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                if content:
                    processed = self.preprocess_text(content)
                    if len(processed) > 50:  # Filtrar textos muy cortos
                        texts.append(processed)
            
            if len(texts) < 2:
                logger.warning("Insuficientes textos válidos para LDA, usando método básico")
                return self._basic_theme_extraction(chunks)
            
            # Vectorización TF-IDF optimizada
            num_docs = len(texts)
            adjusted_min_df = max(1, min(2, num_docs // 5))
            adjusted_max_df = min(0.95, max(0.5, 1.0 - (2.0 / num_docs)))
            max_features = min(1000, max(100, num_docs * 10))
            
            try:
                # Usar cache de vectorizador
                vectorizer_key = f"theme_vectorizer_{num_docs}_{adjusted_min_df}_{adjusted_max_df}"
                if vectorizer_key in self._tfidf_vectorizers_cache:
                    vectorizer = self._tfidf_vectorizers_cache[vectorizer_key]
                else:
                    vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        stop_words=self._get_spanish_stopwords(),
                        ngram_range=(1, 2),
                        min_df=adjusted_min_df,
                        max_df=adjusted_max_df,
                        token_pattern=r'\b[a-záéíóúñ]{3,}\b'
                    )
                    self._tfidf_vectorizers_cache[vectorizer_key] = vectorizer
                
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                if len(feature_names) == 0:
                    logger.warning("No se extrajeron características, usando método básico")
                    return self._basic_theme_extraction(chunks)
                
                # LDA optimizado para extracción de temas
                n_components = min(n_topics, len(texts), len(feature_names) // 2)
                lda = LatentDirichletAllocation(
                    n_components=n_components,
                    random_state=42,
                    max_iter=20,  # Aumentar iteraciones para mejor convergencia
                    learning_method='batch',  # Más estable para datasets pequeños
                    n_jobs=1  # Evitar problemas de concurrencia
                )
                
                lda.fit(tfidf_matrix)
                
                # Extraer temas optimizado
                topics = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topic_weights = topic[top_words_idx]
                    
                    topics.append({
                        'id': topic_idx,
                        'words': top_words,
                        'weights': topic_weights.tolist(),
                        'weight': float(topic_weights.sum()),  # Agregar clave 'weight' para compatibilidad
                        'total_weight': float(topic_weights.sum()),
                        'coherence': self._calculate_topic_coherence(top_words, texts)
                    })
                
                # Asignar documentos a temas de forma eficiente
                doc_topic_probs = lda.transform(tfidf_matrix)
                
                result = {
                    'topics': topics,
                    'document_topics': doc_topic_probs.tolist(),
                    'feature_names': feature_names.tolist(),
                    'method': 'LDA (Optimizado)',
                    'n_topics': len(topics),
                    'documents_analyzed': num_docs,
                    'perplexity': float(lda.perplexity(tfidf_matrix))
                }
                
                # Guardar en cache
                self._concept_analysis_cache[cache_key] = result
                return result
                
            except Exception as tfidf_error:
                logger.warning(f"Error en TF-IDF para temas avanzados: {tfidf_error}")
                return self._basic_theme_extraction(chunks)
            
        except Exception as e:
            logger.error(f"Error en análisis avanzado de temas: {e}")
            return self._basic_theme_extraction(chunks)
    
    def _basic_theme_extraction(self, chunks: List[Dict]) -> Dict:
        """Extracción básica de temas por frecuencia con caché"""
        # Crear clave de caché basada en el contenido
        content_hash = hashlib.md5(
            str([chunk.get('content', '') for chunk in chunks]).encode()
        ).hexdigest()
        cache_key = f"basic_themes_{content_hash}"
        
        # Verificar caché
        if cache_key in self._concept_analysis_cache:
            return self._concept_analysis_cache[cache_key]
        
        # Validar entrada
        if not chunks:
            return {'topics': [], 'themes': {}, 'method': 'frequency', 'total_words': 0}
        
        # Procesar texto de forma eficiente
        all_text = " ".join([chunk.get('content', '') for chunk in chunks if chunk.get('content')])
        if not all_text.strip():
            return {'topics': [], 'themes': {}, 'method': 'frequency', 'total_words': 0}
        
        # Usar texto preprocesado si está disponible
        processed_text = self.preprocess_text(all_text)
        words_list = processed_text.split()
        
        if not words_list:
            return {'topics': [], 'themes': {}, 'method': 'frequency', 'total_words': 0}
        
        word_freq = Counter(words_list)
        
        # Filtrar palabras usando stopwords cacheadas
        stop_words = set(self._get_spanish_stopwords())
        themes_dict = {
            word: count for word, count in word_freq.most_common(30) 
            if len(word) > 3 and word not in stop_words and word.isalpha()
        }
        
        # Convertir a formato de topics para compatibilidad
        topics = []
        for i, (word, count) in enumerate(list(themes_dict.items())[:10]):
            topic = {
                'id': i,
                'name': f"Tema {i+1}",
                'words': [word],
                'keywords': [word],
                'weight': count / len(words_list),
                'weights': [count / len(words_list)],
                'coherence': count / len(words_list),
                'description': f"Tema relacionado con: {word}"
            }
            topics.append(topic)
        
        result = {
            'topics': topics,  # Formato nuevo para compatibilidad
            'themes': themes_dict,  # Formato legacy
            'method': 'frequency',
            'total_words': len(words_list),
            'unique_words': len(word_freq),
            'documents_analyzed': len(chunks)
        }
        
        # Guardar en caché
        self._concept_analysis_cache[cache_key] = result
        return result
    
    
# =============================================================================
# 9. CLUSTERING Y AGRUPACIÓN
# =============================================================================
    
    def perform_clustering(self, chunks: List[Dict], n_clusters: int = 5) -> Dict:
        """Realizar clustering de documentos"""
        if not chunks or not ADVANCED_ANALYSIS_AVAILABLE:
            return {'error': 'Clustering no disponible'}
        
        try:
            # Preparar textos
            texts = [self.preprocess_text(chunk.get('content', '')) for chunk in chunks]
            texts = [text for text in texts if len(text) > 50]
            
            if len(texts) < 2:
                return {'error': 'Insuficientes documentos para clustering'}
            
            # Vectorización con manejo de errores
            # Ajustar parámetros según el número de documentos
            num_docs = len(texts)
            adjusted_min_df = max(1, min(2, num_docs // 4))
            adjusted_max_df = min(0.95, max(0.5, 1.0 - (1.0 / num_docs)))
            
            try:
                vectorizer = TfidfVectorizer(
                    max_features=500,
                    stop_words=self._get_spanish_stopwords(),
                    ngram_range=(1, 2),
                    min_df=adjusted_min_df,
                    max_df=adjusted_max_df
                )
                
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                # K-means clustering
                n_clusters = min(n_clusters, len(texts))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                
                # Reducción de dimensionalidad para visualización
                if len(texts) > 2:
                    try:
                        from umap import UMAP
                        # Configurar UMAP sin n_jobs para evitar warning con random_state
                        reducer = UMAP(
                            n_components=2, 
                            random_state=42,
                            n_jobs=1,  # Explícitamente establecer n_jobs=1 para evitar warning
                            verbose=False  # Reducir verbosidad
                        )
                        embedding = reducer.fit_transform(tfidf_matrix.toarray())
                    except ImportError:
                        # Fallback a PCA
                        pca = PCA(n_components=2, random_state=42)
                        embedding = pca.fit_transform(tfidf_matrix.toarray())
                else:
                    embedding = np.random.rand(len(texts), 2)
                
                # Preparar resultados
                clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    
                    source_file = chunks[i].get('metadata', {}).get('source_file', 'Desconocido')
                    clusters[label].append({
                        'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                        'source': source_file,
                        'embedding': embedding[i].tolist()
                    })
                
                return {
                    'clusters': clusters,
                    'embeddings': embedding.tolist(),
                    'labels': cluster_labels.tolist(),
                    'n_clusters': n_clusters,
                    'method': 'K-means + UMAP'
                }
                
            except Exception as clustering_error:
                logger.warning(f"Error en clustering TF-IDF: {clustering_error}")
                return {'error': f'Error en clustering: {str(clustering_error)}'}
            
        except Exception as e:
            logger.error(f"Error en clustering: {e}")
            return {'error': str(e)}
    
    def _advanced_sentiment_analysis_detailed(self, chunks: List[Dict]) -> Dict:
        """Análisis avanzado de sentimientos con optimización (método legacy detallado para compatibilidad)"""
        if not self._validate_chunks(chunks):
            return {'error': 'No hay datos válidos disponibles'}
        
        try:
            # Crear clave de cache
            content_hash = hash(str(sorted([chunk.get('content', '')[:50] for chunk in chunks])))
            cache_key = f"sentiment_{content_hash}"
            
            # Verificar cache
            if cache_key in self._concept_analysis_cache:
                logger.info("Usando análisis de sentimientos desde cache")
                return self._concept_analysis_cache[cache_key]
            
            # Determinar método de análisis
            if ADVANCED_ANALYSIS_AVAILABLE:
                try:
                    from nltk.sentiment import SentimentIntensityAnalyzer
                    sia = SentimentIntensityAnalyzer()
                    method = 'VADER (NLTK)'
                    use_vader = True
                except Exception as e:
                    logger.warning(f"Error cargando VADER: {e}")
                    method = 'TextBlob'
                    use_vader = False
            else:
                method = 'TextBlob'
                use_vader = False
            
            results = {
                'by_source': defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0, 'scores': []}),
                'overall_stats': {'distribution': {}, 'mean_score': 0},
                'method': method
            }
            
            all_scores = []
            
            # Procesar chunks en lotes para mejor rendimiento
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                for chunk in batch:
                    content = chunk.get('content', '').strip()
                    source = chunk.get('metadata', {}).get('source_file', 'Desconocido')
                    
                    if not content or len(content) < 10:
                        continue
                    
                    # Análisis de sentimiento optimizado
                    if use_vader and 'sia' in locals():
                        scores = sia.polarity_scores(content)
                        compound_score = scores['compound']
                    else:
                        # Fallback con TextBlob
                        blob = TextBlob(content)
                        compound_score = blob.sentiment.polarity
                    
                    all_scores.append(compound_score)
                    results['by_source'][source]['scores'].append(compound_score)
                    
                    # Clasificar sentimiento con umbrales optimizados
                    if compound_score >= 0.05:
                        results['by_source'][source]['positive'] += 1
                    elif compound_score <= -0.05:
                        results['by_source'][source]['negative'] += 1
                    else:
                        results['by_source'][source]['neutral'] += 1
            
            # Estadísticas generales optimizadas
            if all_scores:
                total_positive = sum(1 for score in all_scores if score >= 0.05)
                total_negative = sum(1 for score in all_scores if score <= -0.05)
                total_neutral = len(all_scores) - total_positive - total_negative
                
                results['overall_stats'] = {
                    'distribution': {
                        'positive': total_positive,
                        'neutral': total_neutral,
                        'negative': total_negative
                    },
                    'mean_score': float(np.mean(all_scores)) if all_scores else 0.0,
                    'total_analyzed': len(all_scores)
                }
            
            # Guardar en cache
            self._concept_analysis_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis de sentimientos: {e}")
            return {'error': str(e)}
    
# =============================================================================
# 14. NUBES DE PALABRAS
# =============================================================================
    
    def generate_word_cloud(self, chunks: List[Dict], source_filter: Optional[str] = None) -> Optional[str]:
        """Generar nube de palabras con cache optimizado"""
        if not self._validate_chunks(chunks) or not ADVANCED_ANALYSIS_AVAILABLE:
            return None
        
        try:
            # Filtrar por fuente si se especifica
            if source_filter:
                filtered_chunks = [
                    chunk for chunk in chunks 
                    if chunk.get('metadata', {}).get('source_file', 'Desconocido') == source_filter
                ]
            else:
                filtered_chunks = chunks
            
            if not filtered_chunks:
                return None
            
            # Crear clave de cache
            content_hash = hash(str(sorted([chunk.get('content', '')[:100] for chunk in filtered_chunks])))
            cache_key = f"wordcloud_{content_hash}_{source_filter or 'all'}"
            
            # Verificar si ya existe en cache
            if hasattr(self, '_wordcloud_cache') and cache_key in self._wordcloud_cache:
                cached_path = self._wordcloud_cache[cache_key]
                if Path(cached_path).exists():
                    logger.info("Usando nube de palabras desde cache")
                    return cached_path
            
            # Combinar y procesar texto
            all_text = " ".join([chunk.get('content', '').strip() for chunk in filtered_chunks])
            processed_text = self.preprocess_text(all_text)
            
            if len(processed_text) < 100:
                logger.warning("Texto insuficiente para generar nube de palabras")
                return None
            
            # Generar nube de palabras con configuración optimizada
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                stopwords=self._get_spanish_stopwords(),
                max_words=150,
                colormap='viridis',
                relative_scaling=0.5,
                random_state=42,
                collocations=False,  # Evitar repeticiones
                max_font_size=100,
                min_font_size=10
            ).generate(processed_text)
            
            # Crear directorio de cache si no existe
            from config.settings import config as global_config
            cache_dir = Path(global_config.CACHE_DIR) / "wordclouds"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar imagen con nombre único pero predecible
            filename = f"wordcloud_{cache_key[:16]}.png"
            temp_path = cache_dir / filename
            
            # Generar imagen optimizada
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(temp_path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            
            # Guardar en cache
            if not hasattr(self, '_wordcloud_cache'):
                self._wordcloud_cache = {}
            self._wordcloud_cache[cache_key] = str(temp_path)
            
            # Limpiar cache si es muy grande
            if len(self._wordcloud_cache) > 20:
                oldest_keys = list(self._wordcloud_cache.keys())[:10]
                for old_key in oldest_keys:
                    old_path = Path(self._wordcloud_cache[old_key])
                    if old_path.exists():
                        old_path.unlink()
                    del self._wordcloud_cache[old_key]
            
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Error generando nube de palabras: {e}")
            return None
    
    def generate_advanced_concept_map(self, chunks: List[Dict]) -> Dict:
        """Generar mapa conceptual avanzado con optimización"""
        if not self._validate_chunks(chunks):
            return {'error': 'No hay datos válidos disponibles'}
        
        try:
            # Crear clave de cache
            content_hash = hash(str(sorted([chunk.get('content', '')[:100] for chunk in chunks])))
            cache_key = f"concept_map_{content_hash}"
            
            # Verificar cache
            if cache_key in self._concept_analysis_cache:
                logger.info("Usando mapa conceptual desde cache")
                return self._concept_analysis_cache[cache_key]
            
            # Preparar textos con validación mejorada
            texts = []
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                if content:
                    processed = self.preprocess_text(content)
                    if len(processed) > 50:
                        texts.append(processed)
            
            if len(texts) < 2:
                error_msg = 'Insuficientes documentos válidos para generar mapa conceptual'
                logger.warning(error_msg)
                return {'error': error_msg}
            
            # Extraer términos importantes con optimización
            if ADVANCED_ANALYSIS_AVAILABLE:
                try:
                    # Ajustar parámetros dinámicamente según el corpus
                    num_docs = len(texts)
                    adjusted_min_df = max(1, min(2, num_docs // 4))
                    adjusted_max_df = min(0.90, max(0.3, 1.0 - (3.0 / num_docs)))
                    max_features = min(60, max(20, num_docs * 3))
                    
                    # Usar cache de vectorizador si existe
                    vectorizer_key = f"concept_vectorizer_{num_docs}_{adjusted_min_df}_{adjusted_max_df}"
                    if vectorizer_key in self._tfidf_vectorizers_cache:
                        vectorizer = self._tfidf_vectorizers_cache[vectorizer_key]
                    else:
                        vectorizer = TfidfVectorizer(
                            max_features=max_features,
                            stop_words=self._get_spanish_stopwords(),
                            ngram_range=(1, 2),
                            min_df=adjusted_min_df,
                            max_df=adjusted_max_df,
                            token_pattern=r'\b[a-záéíóúñ]{3,}\b'  # Solo palabras de 3+ caracteres
                        )
                        self._tfidf_vectorizers_cache[vectorizer_key] = vectorizer
                    
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    if len(feature_names) == 0:
                        return {'error': 'No se pudieron extraer términos significativos'}
                    
                    # Calcular similitudes entre términos de forma eficiente
                    similarity_matrix = cosine_similarity(tfidf_matrix.T)
                    
                    # Crear nodos optimizados con colores jerárquicos
                    nodes = []
                    term_importance = tfidf_matrix.sum(axis=0).A1  # Convertir a array 1D
                    
                    # Ordenar términos por importancia para asignar colores jerárquicos
                    term_importance_pairs = [(term, importance) for term, importance in zip(feature_names, term_importance)]
                    term_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Definir umbrales jerárquicos
                    total_terms = len(term_importance_pairs)
                    tier1_threshold = int(total_terms * 0.2)  # Top 20%
                    tier2_threshold = int(total_terms * 0.5)  # Top 50%
                    tier3_threshold = int(total_terms * 0.8)  # Top 80%
                    
                    for i, (term, importance) in enumerate(term_importance_pairs):
                        # Asignar color jerárquico basado en importancia
                        if i < tier1_threshold:
                            color = '#1a237e'  # Azul profundo para términos más importantes
                            size_multiplier = 35
                        elif i < tier2_threshold:
                            color = '#2196f3'  # Azul medio para términos importantes
                            size_multiplier = 30
                        elif i < tier3_threshold:
                            color = '#64b5f6'  # Azul claro para términos moderados
                            size_multiplier = 25
                        else:
                            color = '#bbdefb'  # Azul muy claro para términos menores
                            size_multiplier = 20
                        
                        nodes.append({
                            'id': term,
                            'label': term.title(),  # Capitalizar para mejor presentación
                            'size': min(max(float(importance) * size_multiplier, 15), 70),
                            'color': color,
                            'importance': float(importance),
                            'tier': i // (total_terms // 4) + 1  # Nivel jerárquico 1-4
                        })
                    
                    # Crear aristas con umbral adaptativo
                    edges = []
                    # Umbral adaptativo basado en la distribución de similitudes
                    similarities = similarity_matrix[similarity_matrix > 0]
                    if len(similarities) > 0:
                        threshold = max(0.2, np.percentile(similarities, 75))
                    else:
                        threshold = 0.3
                    
                    edge_count = 0
                    max_edges = min(100, len(feature_names) * 3)  # Limitar número de aristas
                    
                    for i in range(len(feature_names)):
                        if edge_count >= max_edges:
                            break
                        for j in range(i + 1, len(feature_names)):
                            if edge_count >= max_edges:
                                break
                            similarity = similarity_matrix[i, j]
                            if similarity > threshold:
                                edges.append({
                                    'from': feature_names[i],
                                    'to': feature_names[j],
                                    'weight': float(similarity),
                                    'width': min(max(similarity * 5, 1), 8)
                                })
                                edge_count += 1
                    
                    result = {
                        'nodes': nodes,
                        'edges': edges,
                        'stats': {
                            'total_terms': len(feature_names),
                            'total_connections': len(edges),
                            'documents_analyzed': num_docs,
                            'similarity_threshold': threshold
                        },
                        'method': 'TF-IDF + Cosine Similarity'
                    }
                    
                    # Guardar en cache
                    self._concept_analysis_cache[cache_key] = result
                    return result
                    
                except Exception as tfidf_error:
                    logger.warning(f"Error en TF-IDF para mapa conceptual: {tfidf_error}")
                    # Continuar con fallback básico
                    pass
            
            # Fallback básico si TF-IDF falla
            all_text = " ".join(texts)
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = Counter(words)
            
            stop_words = set(self._get_spanish_stopwords())
            important_words = [
                word for word, count in word_freq.most_common(20)
                if len(word) > 3 and word not in stop_words
            ]
            
            nodes = []
            for word in important_words:
                nodes.append({
                    'id': word,
                    'label': word,
                    'size': min(word_freq[word] * 2, 50),
                    'color': f'hsl({hash(word) % 360}, 70%, 60%)'
                })
                
                # Conexiones básicas (palabras que aparecen juntas)
                edges = []
                for i, word1 in enumerate(important_words):
                    for j, word2 in enumerate(important_words[i+1:], i+1):
                        # Contar co-ocurrencias
                        cooccurrence = sum(1 for text in texts if word1 in text and word2 in text)
                        if cooccurrence > 1:
                            edges.append({
                                'from': word1,
                                'to': word2,
                                'weight': cooccurrence,
                                'width': min(cooccurrence, 5)
                            })
                
                stats = {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'method': 'Basic frequency + co-occurrence'
                }
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'stats': stats
                }
                
        except Exception as e:
            logger.error(f"Error generando mapa conceptual: {e}")
            return {'error': str(e)}
    
    def _create_basic_concept_map(self, chunks: List[Dict], layout_type: str = "spring") -> Dict:
        """Crear mapa conceptual básico sin IA"""
        try:
            if not chunks:
                return {'error': 'No hay datos disponibles'}
            
            # Extraer texto de todos los chunks
            texts = [chunk.get('content', '') for chunk in chunks if chunk.get('content')]
            combined_text = ' '.join(texts)
            
            # Análisis básico de frecuencia de palabras
            words = re.findall(r'\b[a-záéíóúñü]{4,}\b', combined_text.lower())
            word_freq = Counter(words)
            
            # Filtrar stopwords básicas
            basic_stopwords = {
                'para', 'con', 'por', 'como', 'que', 'una', 'del', 'las', 'los', 'este', 'esta',
                'son', 'ser', 'está', 'están', 'tiene', 'tienen', 'puede', 'pueden', 'debe',
                'deben', 'hace', 'hacen', 'desde', 'hasta', 'entre', 'sobre', 'bajo', 'ante',
                'tras', 'durante', 'mediante', 'según', 'sin', 'hacia', 'contra'
            }
            
            # Obtener palabras más frecuentes (excluyendo stopwords)
            important_words = [
                word for word, freq in word_freq.most_common(20)
                if word not in basic_stopwords and len(word) > 3
            ][:15]
            
            # Crear nodos
            nodes = []
            for i, word in enumerate(important_words):
                freq = word_freq[word]
                nodes.append({
                    'id': word,
                    'label': word.title(),
                    'size': min(freq * 3 + 10, 50),
                    'color': f'hsl({(i * 137) % 360}, 70%, 60%)',
                    'font': {'size': min(freq + 12, 20)}
                })
            
            # Crear conexiones básicas (co-ocurrencia)
            edges = []
            for i, word1 in enumerate(important_words):
                for j, word2 in enumerate(important_words[i+1:], i+1):
                    # Contar co-ocurrencias en oraciones
                    cooccurrence = 0
                    for text in texts:
                        sentences = re.split(r'[.!?]+', text.lower())
                        for sentence in sentences:
                            if word1 in sentence and word2 in sentence:
                                cooccurrence += 1
                    
                    if cooccurrence > 0:
                        edges.append({
                            'from': word1,
                            'to': word2,
                            'width': min(cooccurrence * 2, 8),
                            'color': {'color': '#848484', 'opacity': 0.6}
                        })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'method': 'Análisis básico de frecuencia'
                }
            }
            
        except Exception as e:
            logger.error(f"Error creando mapa conceptual básico: {e}")
            return {'error': str(e)}
    
    def _create_basic_mind_map(self, chunks: List[Dict], node_spacing: int = 250, return_data: bool = False) -> Dict:
        """Crear mapa mental básico sin IA"""
        try:
            if not chunks:
                return {'error': 'No hay datos disponibles'}
            
            # Extraer texto y agrupar por documentos
            doc_groups = defaultdict(list)
            for chunk in chunks:
                source = chunk.get('metadata', {}).get('source_file', 'Documento')
                content = chunk.get('content', '')
                if content:
                    doc_groups[source].append(content)
            
            # Crear nodo central
            nodes = [{
                'id': 'centro',
                'label': 'Análisis de Contenido',
                'size': 40,
                'color': '#ff6b6b',
                'font': {'size': 18, 'color': 'white'},
                'shape': 'circle'
            }]
            
            edges = []
            
            # Crear ramas principales por documento
            for i, (doc_name, contents) in enumerate(doc_groups.items()):
                doc_id = f'doc_{i}'
                doc_label = doc_name.replace('.txt', '').replace('.pdf', '')[:20]
                
                # Nodo del documento
                nodes.append({
                    'id': doc_id,
                    'label': doc_label,
                    'size': 25,
                    'color': f'hsl({(i * 60) % 360}, 70%, 60%)',
                    'font': {'size': 14}
                })
                
                # Conexión al centro
                edges.append({
                    'from': 'centro',
                    'to': doc_id,
                    'width': 4,
                    'color': {'color': '#333333'}
                })
                
                # Extraer conceptos clave del documento
                combined_content = ' '.join(contents)
                words = re.findall(r'\b[a-záéíóúñü]{5,}\b', combined_content.lower())
                word_freq = Counter(words)
                
                # Filtrar y obtener top conceptos
                basic_stopwords = {
                    'para', 'con', 'por', 'como', 'que', 'una', 'del', 'las', 'los',
                    'este', 'esta', 'son', 'ser', 'está', 'están', 'tiene', 'tienen'
                }
                
                top_concepts = [
                    word for word, freq in word_freq.most_common(8)
                    if word not in basic_stopwords and len(word) > 4
                ][:5]
                
                # Crear nodos de conceptos
                for j, concept in enumerate(top_concepts):
                    concept_id = f'{doc_id}_concept_{j}'
                    nodes.append({
                        'id': concept_id,
                        'label': concept.title(),
                        'size': 15,
                        'color': f'hsl({(i * 60 + j * 20) % 360}, 50%, 70%)',
                        'font': {'size': 10}
                    })
                    
                    # Conexión al documento
                    edges.append({
                        'from': doc_id,
                        'to': concept_id,
                        'width': 2,
                        'color': {'color': '#666666', 'opacity': 0.7}
                    })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'documents': len(doc_groups),
                    'method': 'Estructura jerárquica básica'
                }
            }
            
        except Exception as e:
            logger.error(f"Error creando mapa mental básico: {e}")
            return {'error': str(e)}
    
    def generate_basic_summary(self, chunks: List[Dict], max_sentences: int = 5) -> str:
        """Generar resumen básico sin IA"""
        try:
            if not chunks:
                return "No hay contenido disponible para resumir."
            
            # Extraer todo el texto
            texts = [chunk.get('content', '') for chunk in chunks if chunk.get('content')]
            combined_text = ' '.join(texts)
            
            # Dividir en oraciones
            sentences = re.split(r'[.!?]+', combined_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if not sentences:
                return "No se encontraron oraciones válidas para resumir."
            
            # Análisis básico de frecuencia de palabras
            words = re.findall(r'\b[a-záéíóúñü]{4,}\b', combined_text.lower())
            word_freq = Counter(words)
            
            # Filtrar stopwords básicas
            basic_stopwords = {
                'para', 'con', 'por', 'como', 'que', 'una', 'del', 'las', 'los',
                'este', 'esta', 'son', 'ser', 'está', 'están', 'tiene', 'tienen',
                'puede', 'pueden', 'debe', 'deben', 'hace', 'hacen', 'muy', 'más',
                'también', 'así', 'todo', 'todos', 'toda', 'todas'
            }
            
            # Calcular palabras importantes
            important_words = {
                word: freq for word, freq in word_freq.most_common(50)
                if word not in basic_stopwords and len(word) > 3
            }
            
            # Puntuar oraciones basado en palabras importantes
            sentence_scores = []
            for sentence in sentences:
                score = 0
                sentence_words = re.findall(r'\b[a-záéíóúñü]+\b', sentence.lower())
                
                for word in sentence_words:
                    if word in important_words:
                        score += important_words[word]
                
                # Bonus por posición (primeras y últimas oraciones)
                position_bonus = 0
                sentence_index = sentences.index(sentence)
                if sentence_index < 3:  # Primeras 3 oraciones
                    position_bonus = 2
                elif sentence_index >= len(sentences) - 3:  # Últimas 3 oraciones
                    position_bonus = 1
                
                sentence_scores.append({
                    'sentence': sentence,
                    'score': score + position_bonus,
                    'length': len(sentence)
                })
            
            # Seleccionar mejores oraciones
            sentence_scores.sort(key=lambda x: x['score'], reverse=True)
            selected_sentences = sentence_scores[:max_sentences]
            
            # Ordenar por aparición original
            selected_sentences.sort(key=lambda x: sentences.index(x['sentence']))
            
            # Construir resumen
            summary_sentences = [s['sentence'] for s in selected_sentences]
            summary = '. '.join(summary_sentences)
            
            # Agregar punto final si no lo tiene
            if not summary.endswith('.'):
                summary += '.'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generando resumen básico: {e}")
            return f"Error al generar resumen: {str(e)}"

# =============================================================================
# 15. VISUALIZACIONES Y GRÁFICOS
# =============================================================================

def render_advanced_dashboard(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Dashboard avanzado con métricas clave"""
    st.header("📊 Dashboard de Análisis")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Métricas generales
    st.subheader("📈 Métricas Generales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documentos", len(set(chunk.get('metadata', {}).get('source_file', 'Desconocido') for chunk in chunks)))
    
    with col2:
        st.metric("📝 Total Chunks", len(chunks))
    
    with col3:
        total_chars = sum(len(chunk.get('content', '')) for chunk in chunks)
        st.metric("🔤 Total Caracteres", f"{total_chars:,}")
    
    with col4:
        avg_chunk_size = total_chars // len(chunks) if chunks else 0
        st.metric("📏 Tamaño Promedio Chunk", avg_chunk_size)
    
    # Distribución por fuente
    st.subheader("📊 Distribución por Documento")
    
    source_stats = defaultdict(lambda: {'chunks': 0, 'characters': 0})
    for chunk in chunks:
        source = chunk.get('metadata', {}).get('source_file', 'Desconocido')
        source_stats[source]['chunks'] += 1
        source_stats[source]['characters'] += len(chunk.get('content', ''))
    
    # Crear DataFrame para visualización
    df_sources = pd.DataFrame([
        {
            'Documento': source,
            'Chunks': stats['chunks'],
            'Caracteres': stats['characters'],
            '% del Total': (stats['characters'] / total_chars) * 100 if total_chars > 0 else 0
        }
        for source, stats in source_stats.items()
    ])
    
    # Gráfico de barras
    fig = px.bar(
        df_sources,
        x='Documento',
        y='Caracteres',
        title="Distribución de Contenido por Documento",
        color='% del Total',
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de estadísticas
    st.dataframe(df_sources, use_container_width=True)

def render_advanced_themes(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Análisis avanzado de temas"""
    st.header("🎯 Análisis Avanzado de Temas")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Configuración
    col1, col2 = st.columns([3, 1])
    
    with col1:
        n_topics = st.slider("Número de temas a extraer:", 3, 15, 8)
    
    with col2:
        if st.button("🔄 Analizar Temas", type="primary", key="analyze_themes"):
            if 'theme_analysis_cache' in st.session_state:
                del st.session_state.theme_analysis_cache
    
    # Realizar análisis
    if 'theme_analysis_cache' not in st.session_state:
        with st.spinner("Analizando temas..."):
            st.session_state.theme_analysis_cache = analyzer.extract_advanced_themes(chunks, n_topics)
    
    theme_analysis = st.session_state.theme_analysis_cache
    
    if 'topics' in theme_analysis:
        st.subheader("📊 Temas Identificados")
        
        # Mostrar temas en tarjetas
        for i, topic in enumerate(theme_analysis['topics']):
            with st.expander(f"**Tema {i+1}** - Coherencia: {topic.get('coherence', 0):.3f}"):
                st.write("**Palabras clave:**")
                # Compatibilidad con ambos formatos: 'words' y 'keywords'
                keywords = topic.get('words', topic.get('keywords', []))
                st.write(", ".join([str(w) for w in keywords[:8]]))
                st.write(f"**Peso:** {topic.get('weight', 0):.3f}")
        
        # Visualización de temas
        if len(theme_analysis['topics']) > 1:
            # Preparar datos para visualización
            topic_data = []
            for topic in theme_analysis['topics']:
                # Compatibilidad con ambos formatos
                keywords = topic.get('words', topic.get('keywords', []))
                topic_data.append({
                    'Tema': f"Tema {topic.get('id', 0)+1}",
                    'Peso': topic.get('weight', 0),
                    'Coherencia': topic.get('coherence', 0),
                    'Palabras': ', '.join([str(w) for w in keywords[:5]])
                })
            
            df_topics = pd.DataFrame(topic_data)
            
            # Gráfico de burbujas
            fig = px.scatter(
                df_topics,
                x='Peso',
                y='Coherencia',
                size='Peso',
                hover_data=['Palabras'],
                title="Temas por Peso y Coherencia",
                labels={'Peso': 'Peso del Tema', 'Coherencia': 'Coherencia del Tema'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif 'themes' in theme_analysis:
        # Análisis básico
        st.subheader("📊 Temas Frecuentes")
        
        themes_data = [{'Tema': theme, 'Frecuencia': freq} 
                      for theme, freq in theme_analysis['themes'].items()]
        
        df_themes = pd.DataFrame(themes_data)
        
        fig = px.bar(
            df_themes.head(15),
            x='Frecuencia',
            y='Tema',
            orientation='h',
            title="Top 15 Temas por Frecuencia"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_clustering_analysis(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Análisis de clustering"""
    st.header("🔍 Análisis de Clustering")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    if not ADVANCED_ANALYSIS_AVAILABLE:
        st.error("Las funcionalidades de clustering requieren dependencias adicionales.")
        st.code("pip install scikit-learn")
        return
    
    # Configuración
    col1, col2 = st.columns([3, 1])
    
    with col1:
        n_clusters = st.slider("Número de clusters:", 2, min(10, len(chunks)//2), 5)
    
    with col2:
        if st.button("🔄 Ejecutar Clustering", type="primary", key="execute_clustering"):
            if 'clustering_cache' in st.session_state:
                del st.session_state.clustering_cache
    
    # Realizar clustering
    if 'clustering_cache' not in st.session_state:
        with st.spinner("Ejecutando análisis de clustering..."):
            st.session_state.clustering_cache = analyzer.perform_clustering(chunks, n_clusters)
    
    clustering_result = st.session_state.clustering_cache
    
    if 'error' in clustering_result:
        st.error(f"Error en clustering: {clustering_result['error']}")
        return
    
    # Mostrar resultados
    st.subheader("📊 Resultados del Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Clusters", clustering_result['n_clusters'])
    
    with col2:
        method = clustering_result.get('method', 'K-means')
        st.metric("🔧 Método", method)
    
    # Visualización de clustering si hay embeddings
    if 'embeddings' in clustering_result and 'labels' in clustering_result:
        st.subheader("📊 Visualización de Clusters")
        
        embeddings = np.array(clustering_result['embeddings'])
        labels = clustering_result['labels']
        
        # Crear DataFrame para Plotly
        df_cluster = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'cluster': [f'Cluster {label}' for label in labels],
            'documento': [f'Doc {i+1}' for i in range(len(labels))]
        })
        
        fig = px.scatter(
            df_cluster,
            x='x', y='y',
            color='cluster',
            hover_data=['documento'],
            title="Distribución de Documentos en Clusters"
        )
        fig.update_layout(
            xaxis_title="Dimensión 1",
            yaxis_title="Dimensión 2"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detalles de clusters
    st.subheader("🔍 Detalles de Clusters")
    
    for cluster_id, cluster_docs in clustering_result['clusters'].items():
        with st.expander(f"**Cluster {cluster_id}** ({len(cluster_docs)} documentos)"):
            # Documentos en el cluster
            st.write("**Documentos:**")
            for doc in cluster_docs[:5]:  # Mostrar solo los primeros 5
                st.write(f"- **{doc['source']}**: {doc['text']}")
                
            if len(cluster_docs) > 5:
                st.write(f"... y {len(cluster_docs) - 5} documentos más")

def render_advanced_concept_map(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Mapa conceptual avanzado"""
    st.header("🗺️ Mapa Conceptual Avanzado")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Generar mapa
    if 'concept_map_cache' not in st.session_state:
        with st.spinner("Generando mapa conceptual avanzado..."):
            st.session_state.concept_map_cache = analyzer.generate_advanced_concept_map(chunks)
    
    map_data = st.session_state.concept_map_cache
    
    if 'error' in map_data:
        st.error(f"Error generando mapa: {map_data['error']}")
        return
    
    if not map_data["nodes"]:
        st.warning("No se pudieron generar datos para el mapa conceptual.")
        return
    
    # Estadísticas del mapa
    stats = map_data.get('stats', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔗 Nodos", stats.get('total_nodes', 0))
    
    with col2:
        st.metric("↔️ Conexiones", stats.get('total_edges', 0))
    
    with col3:
        density = (stats.get('total_edges', 0) * 2) / (stats.get('total_nodes', 1) * (stats.get('total_nodes', 1) - 1)) if stats.get('total_nodes', 0) > 1 else 0
        st.metric("📊 Densidad", f"{density:.2%}")
    
    # Crear visualización
    G = nx.Graph()
    
    # Agregar nodos
    for node in map_data["nodes"]:
        G.add_node(node["id"], **node)
    
    # Agregar aristas
    for edge in map_data["edges"]:
        G.add_edge(edge["from"], edge["to"], **edge)
    
    # Calcular layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.random_layout(G, seed=42)
    
    # Preparar trazas
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        text=[],
        textposition="middle center",
        marker=dict(
            size=[],
            color=[],
            line=dict(width=2)
        )
    )
    
    # Aristas
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    # Nodos
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node[1].get('label', node[0])])
        node_trace['marker']['size'] += tuple([node[1].get('size', 20)])
        node_trace['marker']['color'] += tuple([node[1].get('color', '#888')])
    
    # Crear figura
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Mapa Conceptual Avanzado',
            title_font_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Red de conceptos y similitudes entre documentos",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#888", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_sentiment_analysis(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Análisis de sentimientos avanzado con optimización"""
    st.header("😊 Análisis de Sentimientos Avanzado")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Crear clave de cache única para la sesión
    content_hash = hash(str(sorted([chunk.get('content', '')[:50] for chunk in chunks])))
    cache_key = f"sentiment_ui_{content_hash}"
    
    # Realizar análisis con cache optimizado
    if cache_key not in st.session_state:
        with st.spinner("Analizando sentimientos..."):
            st.session_state[cache_key] = analyzer.advanced_sentiment_analysis(chunks)
    
    sentiment_result = st.session_state[cache_key]
    
    if 'error' in sentiment_result:
        st.error(f"Error en análisis de sentimientos: {sentiment_result['error']}")
        return
    
    # Mostrar método utilizado y estadísticas de cache
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"**Método de análisis:** {sentiment_result.get('method', 'Desconocido')}")
    with col_info2:
        total_analyzed = sentiment_result.get('overall_stats', {}).get('total_analyzed', 0)
        st.info(f"**Documentos analizados:** {total_analyzed}")
    
    # Estadísticas generales optimizadas
    if 'overall_stats' in sentiment_result:
        st.subheader("📊 Estadísticas Generales")
        
        stats = sentiment_result['overall_stats']
        distribution = stats.get('distribution', {})
        
        if distribution:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("😊 Positivo", distribution.get('positive', 0))
            
            with col2:
                st.metric("😐 Neutral", distribution.get('neutral', 0))
            
            with col3:
                st.metric("😞 Negativo", distribution.get('negative', 0))
            
            with col4:
                mean_score = stats.get('mean_score', 0)
                st.metric("📈 Score Promedio", f"{mean_score:.3f}")
            
            # Gráfico de distribución optimizado
            try:
                fig = px.pie(
                    values=list(distribution.values()),
                    names=list(distribution.keys()),
                    title="Distribución de Sentimientos",
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'neutral': '#95a5a6',
                        'negative': '#e74c3c'
                    }
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generando gráfico: {e}")
    
    # Análisis por fuente optimizado
    st.subheader("📋 Análisis por Documento")
    
    try:
        source_data = []
        for source, data in sentiment_result['by_source'].items():
            total = data['positive'] + data['neutral'] + data['negative']
            if total > 0:
                scores = data.get('scores', [])
                avg_score = float(np.mean(scores)) if scores else 0.0
                source_data.append({
                    'Documento': source,
                    'Positivo': data['positive'],
                    'Neutral': data['neutral'],
                    'Negativo': data['negative'],
                    'Total': total,
                    'Score Promedio': round(avg_score, 3),
                    '% Positivo': round((data['positive'] / total) * 100, 1)
                })
        
        if source_data:
            df_sentiment = pd.DataFrame(source_data)
            st.dataframe(df_sentiment, use_container_width=True)
            
            # Gráfico comparativo optimizado
            if len(source_data) <= 10:  # Limitar para mejor visualización
                fig = px.bar(
                    df_sentiment,
                    x='Documento',
                    y=['Positivo', 'Neutral', 'Negativo'],
                    title="Sentimientos por Documento",
                    color_discrete_map={
                        'Positivo': '#2ecc71',
                        'Neutral': '#95a5a6',
                        'Negativo': '#e74c3c'
                    }
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Demasiados documentos para mostrar gráfico. Mostrando solo tabla.")
        else:
            st.warning("No se encontraron datos de sentimientos por documento.")
    
    except Exception as e:
        st.error(f"Error procesando análisis por documento: {e}")
        logger.error(f"Error en render_sentiment_analysis: {e}")

def render_word_cloud(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Nube de palabras"""
    st.header("☁️ Nube de Palabras")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    if not ADVANCED_ANALYSIS_AVAILABLE:
        st.error("La nube de palabras requiere dependencias adicionales.")
        st.code("pip install wordcloud matplotlib")
        return
    
    # Filtro por fuente
    sources = ['Todos'] + list(set(chunk.get('metadata', {}).get('source_file', 'Desconocido') for chunk in chunks))
    selected_source = st.selectbox("Filtrar por documento:", sources)
    
    source_filter = None if selected_source == 'Todos' else selected_source
    
    if st.button("🎨 Generar Nube de Palabras", type="primary", key="generate_wordcloud"):
        with st.spinner("Generando nube de palabras..."):
            wordcloud_path = analyzer.generate_word_cloud(chunks, source_filter)
            
            if wordcloud_path and Path(wordcloud_path).exists():
                st.image(wordcloud_path, caption=f"Nube de palabras - {selected_source}")
                
                # Limpiar archivo temporal después de un tiempo
                try:
                    Path(wordcloud_path).unlink()
                except:
                    pass
            else:
                st.error("No se pudo generar la nube de palabras.")

def render_settings_tab(analyzer: AdvancedQualitativeAnalyzer):
    """Pestaña de configuración del análisis cualitativo"""
    st.header("⚙️ Configuración del Análisis")
    
    st.subheader("🎛️ Parámetros de Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Configuración de Temas")
        
        n_topics = st.slider(
            "Número de temas a extraer",
            min_value=3,
            max_value=20,
            value=st.session_state.get('qualitative_n_topics', 10),
            help="Número de temas principales a identificar en el análisis LDA"
        )
        st.session_state.qualitative_n_topics = n_topics
        
        min_topic_words = st.slider(
            "Palabras mínimas por tema",
            min_value=5,
            max_value=20,
            value=st.session_state.get('qualitative_min_topic_words', 10),
            help="Número mínimo de palabras representativas por tema"
        )
        st.session_state.qualitative_min_topic_words = min_topic_words
    
    with col2:
        st.markdown("### 🔍 Configuración de Clustering")
        
        n_clusters = st.slider(
            "Número de clusters",
            min_value=2,
            max_value=15,
            value=st.session_state.get('qualitative_n_clusters', 5),
            help="Número de grupos para el análisis de clustering"
        )
        st.session_state.qualitative_n_clusters = n_clusters
        
        clustering_method = st.selectbox(
            "Método de clustering",
            options=['K-means', 'DBSCAN'],
            index=0,
            help="Algoritmo de clustering a utilizar"
        )
        st.session_state.qualitative_clustering_method = clustering_method
    
    st.divider()
    
    # Configuración de cache
    st.subheader("💾 Gestión de Cache")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Limpiar Cache de Análisis", type="secondary", key="clear_analysis_cache"):
            # Limpiar cache específico del análisis cualitativo
            cache_keys = [key for key in st.session_state.keys() if key.startswith('qualitative_') or key.endswith('_cache')]
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Cache de análisis limpiado")
            st.rerun()
    
    with col2:
        if st.button("📊 Recargar Datos RAG", type="secondary", key="reload_rag_data"):
            if 'qualitative_chunks' in st.session_state:
                del st.session_state['qualitative_chunks']
            st.success("✅ Datos RAG recargados")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Limpiar Todo", type="secondary", key="clear_all_analysis"):
            # Limpiar todo el cache relacionado
            cache_keys = [key for key in st.session_state.keys() if 'qualitative' in key or 'cache' in key]
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Todo el cache limpiado")
            st.rerun()
    
    st.divider()
    
    # Información del sistema
    st.subheader("ℹ️ Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Estado de Dependencias:**")
        if ADVANCED_ANALYSIS_AVAILABLE:
            st.success("✅ Análisis avanzado disponible")
            st.info("📦 Dependencias instaladas: scikit-learn, nltk, textblob, wordcloud")
        else:
            st.error("❌ Análisis avanzado no disponible")
            st.warning("⚠️ Instala las dependencias adicionales")
    
    with col2:
        st.markdown("**Cache del Sistema:**")
        try:
            cache_path = analyzer.cache_path
            if cache_path.exists():
                cache_size = cache_path.stat().st_size
                st.info(f"📁 Tamaño del cache: {cache_size:,} bytes")
                
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if 'chunks' in cache_data:
                        total_chunks = sum(len(chunk_list) for chunk_list in cache_data['chunks'].values())
                        st.info(f"📄 Chunks en cache: {total_chunks}")
            else:
                st.warning("⚠️ No hay cache disponible")
        except Exception as e:
            st.error(f"❌ Error leyendo cache: {e}")
    
    st.divider()
    
    # Configuración de exportación
    st.subheader("📤 Exportación de Resultados")
    
    export_format = st.selectbox(
        "Formato de exportación",
        options=['JSON', 'CSV', 'Excel'],
        help="Formato para exportar los resultados del análisis"
    )
    
    if st.button("💾 Exportar Análisis Actual", type="primary", key="export_current_analysis"):
        st.info("🚧 Funcionalidad de exportación en desarrollo")

def render_interactive_concept_map(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar mapa conceptual interactivo con opciones duales (normal/IA) y control de pausa"""
    st.header("🗺️ Mapa Conceptual Interactivo")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    if not PYVIS_AVAILABLE:
        st.error("PyVis no está disponible. Instala con: pip install pyvis")
        return
    
    # Configuración avanzada con opciones duales
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        generation_mode = st.selectbox(
            "🎯 Modo de Generación:",
            options=["normal", "ai"],  # Cambio: normal primero (por defecto)
            format_func=lambda x: {
                "normal": "⚡ Generación Normal (Recomendado)",
                "ai": "🧠 Generación con IA (Experimental)"
            }[x],
            help="Modo Normal: Rápido y coherente | Modo IA: Más lento pero con análisis semántico profundo",
            index=0  # Normal por defecto
        )
    
    with col2:
        layout_type = st.selectbox(
            "Tipo de Layout:",
            ["spring", "circular", "random", "shell"],
            index=0,
            help="Selecciona el tipo de distribución de los nodos"
        )
    
    with col3:
        show_full_text = st.checkbox(
            "Mostrar texto completo",
            value=True,
            key="concept_map_show_text",
            help="Incluir el texto completo de los documentos debajo del mapa"
        )
    
    with col4:
        if st.button("🔄 Generar", type="primary", key="concept_map_generate"):
            # Limpiar cache para regenerar
            cache_keys = ['concept_map_html', 'concept_map_structure', 'generation_paused']
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
    
    # Control de pausa para generación con IA
    if generation_mode == "ai":
        st.info("🧠 **Modo IA Activado**: Generación más inteligente pero puede tomar más tiempo")
        
        # Botón de pausa/continuar
        col_pause, col_status = st.columns([1, 3])
        with col_pause:
            if st.session_state.get('generation_in_progress', False):
                if st.button("⏸️ Pausar", key="pause_concept_generation"):
                    st.session_state.generation_paused = True
                    st.session_state.generation_in_progress = False
                    st.rerun()
            elif st.session_state.get('generation_paused', False):
                if st.button("▶️ Continuar", key="resume_concept_generation"):
                    st.session_state.generation_paused = False
                    st.rerun()
        
        with col_status:
            if st.session_state.get('generation_paused', False):
                st.warning("⏸️ Generación pausada. Haz clic en 'Continuar' para reanudar.")
            elif st.session_state.get('generation_in_progress', False):
                st.info("🔄 Generación en progreso... Puedes pausar en cualquier momento.")
    
    # Verificar si la generación está pausada
    if st.session_state.get('generation_paused', False):
        st.stop()
    
    # Usar todos los chunks disponibles
    selected_chunks = chunks
    
    # Generar mapa y estructura
    cache_key = f"concept_map_{generation_mode}_{layout_type}"
    
    if cache_key not in st.session_state:
        # Marcar inicio de generación para modo IA
        if generation_mode == "ai":
            st.session_state.generation_in_progress = True
        
        spinner_text = "🧠 Generando mapa conceptual inteligente..." if generation_mode == "ai" else "⚡ Generando mapa conceptual..."
        
        with st.spinner(spinner_text):
            if generation_mode == "ai":
                # Generación inteligente con IA (usa LLM para análisis semántico)
                try:
                    # Intentar análisis con IA usando el modelo LLM
                    concept_structure = analyzer._analyze_concept_hierarchy_with_ai(selected_chunks)
                    
                    # Si falla (retorna None), usar modo normal
                    if concept_structure is None:
                        logger.info("Análisis con IA retornó None, usando modo normal")
                        concept_structure = analyzer._analyze_concept_hierarchy(selected_chunks)
                    
                    html_file = analyzer.create_interactive_concept_map(selected_chunks, layout_type)
                except Exception as e:
                    logger.warning(f"Error en generación con IA: {e}. Usando modo normal.")
                    # Fallback a modo normal
                    concept_structure = analyzer._analyze_concept_hierarchy(selected_chunks)
                    html_file = analyzer.create_interactive_concept_map(selected_chunks, layout_type)
            else:
                # Generación normal/rápida - análisis mejorado con n-gramas
                concept_structure = analyzer._analyze_concept_hierarchy(selected_chunks)
                html_file = analyzer.create_interactive_concept_map(selected_chunks, layout_type)
            
            st.session_state[cache_key] = {
                'html_file': html_file,
                'concept_structure': concept_structure
            }
        
        # Marcar fin de generación
        if generation_mode == "ai":
            st.session_state.generation_in_progress = False
    
    result = st.session_state[cache_key]
    html_file = result['html_file']
    concept_structure = result['concept_structure']
    
    if html_file and os.path.exists(html_file):
        # Mostrar estadísticas del mapa
        if concept_structure:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Tema Central", "1")
            with col2:
                st.metric("📊 Conceptos Principales", len(concept_structure.get('main_concepts', [])))
            with col3:
                total_sub = sum(len(c.get('sub_concepts', [])) for c in concept_structure.get('main_concepts', []))
                st.metric("🔗 Sub-conceptos", total_sub)
            with col4:
                st.metric("↔️ Relaciones Cruzadas", len(concept_structure.get('cross_relations', [])))
        
        # Leer y mostrar el HTML con tamaño amplio
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Contenedor amplio para mejor visualización
        st.subheader("📈 Visualización del Mapa Conceptual")
        with st.container():
            st.components.v1.html(html_content, height=600, scrolling=True)  # Altura reducida
        
        # Información adicional
        st.info("💡 **Instrucciones:** Puedes hacer clic y arrastrar los nodos, hacer zoom, y pasar el cursor sobre los elementos para ver más información. El mapa se ajusta automáticamente para una mejor visualización.")
        
        # Mostrar texto completo de los documentos si está habilitado
        if show_full_text:
            st.subheader("📄 Texto Completo de los Documentos")
            
            # Organizar por fuente si está disponible
            sources = {}
            for chunk in selected_chunks:
                source = chunk.get('source', 'Documento sin fuente')
                if source not in sources:
                    sources[source] = []
                sources[source].append(chunk)
            
            for source, source_chunks in sources.items():
                with st.expander(f"📖 {source} ({len(source_chunks)} secciones)", expanded=False):
                    for i, chunk in enumerate(source_chunks, 1):
                        st.markdown(f"**Sección {i}:**")
                        content = chunk.get('content', 'Sin contenido')
                        st.markdown(content)
                        if i < len(source_chunks):
                            st.divider()
        
        # Opciones de descarga mejoradas
        st.subheader("📥 Opciones de Descarga")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Descargar HTML del mapa
            with open(html_file, 'rb') as f:
                st.download_button(
                    label="🗺️ Descargar Mapa HTML",
                    data=f.read(),
                    file_name=f"mapa_conceptual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    help="Descarga el mapa conceptual interactivo"
                )
        
        with col2:
            # Descargar estructura JSON
            if concept_structure:
                json_data = json.dumps(concept_structure, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📊 Descargar Estructura JSON",
                    data=json_data,
                    file_name=f"estructura_conceptual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Descarga la estructura de datos del mapa"
                )
        
        with col3:
            # Generar y descargar reporte completo
            full_text = "\n\n".join([chunk.get('content', '') for chunk in selected_chunks])
            report_content = f"""# Reporte de Mapa Conceptual

**Generado el:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## Resumen del Análisis

- **Tema Central:** {concept_structure.get('main_theme', {}).get('name', 'No identificado') if concept_structure else 'No disponible'}
- **Conceptos Principales:** {len(concept_structure.get('main_concepts', [])) if concept_structure else 0}
- **Total de Documentos:** {len(set(chunk.get('source', 'Sin fuente') for chunk in selected_chunks))}
- **Total de Secciones:** {len(selected_chunks)}

## Estructura Conceptual

{json.dumps(concept_structure, indent=2, ensure_ascii=False) if concept_structure else 'No disponible'}

## Texto Completo de los Documentos

{full_text}
"""
            
            st.download_button(
                label="📋 Descargar Reporte Completo",
                data=report_content,
                file_name=f"reporte_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                help="Descarga un reporte completo con mapa, estructura y texto"
            )
    
    else:
        st.error("No se pudo generar el mapa conceptual.")
        
        # Información de debug
        with st.expander("🔧 Información de Debug"):
            st.write("Verificando disponibilidad de PyVis...")
            st.write(f"PyVis disponible: {PYVIS_AVAILABLE}")
            st.write(f"Número de chunks: {len(selected_chunks)}")
            if selected_chunks:
                st.write("Muestra del primer chunk:")
                st.json(selected_chunks[0])

def render_interactive_mind_map(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar mapa mental interactivo con opciones duales (normal/IA) y control de pausa"""
    st.header("🧠 Mapa Mental Interactivo")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    if not AGRAPH_AVAILABLE:
        st.error("streamlit-agraph no está disponible. Instala con: pip install streamlit-agraph")
        return
    
    # Configuración avanzada con opciones duales
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
    
    with col1:
        generation_mode = st.selectbox(
            "🎯 Modo de Generación:",
            options=["normal", "ai"],  # CAMBIO: Normal primero (por defecto)
            format_func=lambda x: {
                "normal": "⚡ Generación Normal (Recomendado)",
                "ai": "🧠 Generación con IA (Experimental)"
            }[x],
            help="Modo Normal: Rápido y coherente | Modo IA: Más lento pero con análisis semántico profundo",
            key="mind_map_generation_mode",
            index=0  # Normal por defecto
        )
    
    with col2:
        node_spacing = st.slider(
            "Espaciado entre Nodos:",
            min_value=200,  # AUMENTADO: de 150 a 200
            max_value=800,  # AUMENTADO: de 600 a 800
            value=450,  # AUMENTADO: de 350 a 450 para mejor separación
            step=50,
            help="Controla la distancia entre los nodos del mapa mental (mayor = más separado)"
        )
    
    with col3:
        physics_strength = st.slider(
            "Fuerza de Física:",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Controla la intensidad de las fuerzas físicas"
        )
    
    with col4:
        show_full_text = st.checkbox(
            "Mostrar texto completo",
            value=True,
            key="mind_map_show_text",
            help="Incluir el texto completo de los documentos debajo del mapa"
        )
    
    with col5:
        if st.button("🔄 Generar", type="primary", key="mind_map_generate"):
            # Limpiar cache para regenerar
            cache_keys = ['mind_map_data', 'mind_map_structure', 'mind_generation_paused']
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
    
    # Control de pausa para generación con IA
    if generation_mode == "ai":
        st.info("🧠 **Modo IA Activado**: Generación más inteligente pero puede tomar más tiempo")
        
        # Botón de pausa/continuar
        col_pause, col_status = st.columns([1, 3])
        with col_pause:
            if st.session_state.get('mind_generation_in_progress', False):
                if st.button("⏸️ Pausar", key="pause_mind_generation"):
                    st.session_state.mind_generation_paused = True
                    st.session_state.mind_generation_in_progress = False
                    st.rerun()
            elif st.session_state.get('mind_generation_paused', False):
                if st.button("▶️ Continuar", key="resume_mind_generation"):
                    st.session_state.mind_generation_paused = False
                    st.rerun()
        
        with col_status:
            if st.session_state.get('mind_generation_paused', False):
                st.warning("⏸️ Generación pausada. Haz clic en 'Continuar' para reanudar.")
            elif st.session_state.get('mind_generation_in_progress', False):
                st.info("🔄 Generación en progreso... Puedes pausar en cualquier momento.")
    
    # Verificar si la generación está pausada
    if st.session_state.get('mind_generation_paused', False):
        st.stop()
    
    # Usar todos los chunks disponibles
    selected_chunks = chunks
    
    # Generar mapa mental
    cache_key = f"mind_map_{generation_mode}_{node_spacing}_{physics_strength}"
    
    if cache_key not in st.session_state:
        # Marcar inicio de generación para modo IA
        if generation_mode == "ai":
            st.session_state.mind_generation_in_progress = True
        
        spinner_text = "🧠 Generando mapa mental inteligente..." if generation_mode == "ai" else "⚡ Generando mapa mental..."
        
        with st.spinner(spinner_text):
            if generation_mode == "ai":
                # Generación inteligente con IA
                mind_map_data = analyzer.create_interactive_mind_map(selected_chunks, node_spacing, return_data=True)
                mind_structure = analyzer._analyze_intelligent_mind_map_structure(selected_chunks)
            else:
                # Generación normal/rápida
                mind_map_data = analyzer._create_basic_mind_map(selected_chunks, node_spacing, return_data=True)
                mind_structure = analyzer._fallback_mind_map_structure(selected_chunks)
            
            st.session_state[cache_key] = {
                'mind_map_data': mind_map_data,
                'mind_structure': mind_structure
            }
        
        # Marcar fin de generación
        if generation_mode == "ai":
            st.session_state.mind_generation_in_progress = False
    
    # Mostrar resultado si existe
    if cache_key in st.session_state:
        result = st.session_state[cache_key]
        mind_map_data = result['mind_map_data']
        mind_structure = result['mind_structure']
        
        if mind_map_data and 'nodes' in mind_map_data and 'edges' in mind_map_data:
            # Mostrar estadísticas
            if 'stats' in mind_map_data:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Nodos", mind_map_data['stats']['total_nodes'])
                with col2:
                    st.metric("🔗 Conexiones", mind_map_data['stats']['total_edges'])
                with col3:
                    st.metric("🎯 Ramas Principales", mind_map_data['stats'].get('main_branches', 0))
                with col4:
                    st.metric("💡 Conceptos Detallados", mind_map_data['stats'].get('detailed_concepts', 0))
        
        # Verificar disponibilidad de streamlit-agraph
        if AGRAPH_AVAILABLE:
            try:
                # Crear nodos con configuración visual optimizada para fondo blanco
                nodes = []
                for node_data in mind_map_data['nodes']:
                    # Colores optimizados para fondo blanco
                    node_color = node_data.get('color', '#2E86AB')  # Azul por defecto
                    if node_data.get('level') == 0:  # Nodo central
                        node_color = '#A23B72'  # Rosa/magenta
                    elif node_data.get('level') == 1:  # Ramas principales
                        node_color = '#F18F01'  # Naranja
                    elif node_data.get('level') == 2:  # Sub-ramas
                        node_color = '#C73E1D'  # Rojo
                    
                    node = Node(
                        id=node_data['id'],
                        label=node_data['label'],
                        size=node_data.get('size', 25),  # Tamaño aumentado
                        color=node_color,
                        font={'size': node_data.get('font_size', 16), 'color': '#2c3e50'},  # Fuente más grande y legible
                        borderWidth=3,
                        shadow=True
                    )
                    nodes.append(node)
                
                # Crear aristas con configuración mejorada
                edges = []
                for edge_data in mind_map_data['edges']:
                    edge = Edge(
                        source=edge_data['from'],
                        target=edge_data['to'],
                        label=edge_data.get('label', ''),
                        color=edge_data.get('color', '#666666'),  # Color más visible
                        width=edge_data.get('width', 3),  # Líneas más gruesas
                        smooth=True
                    )
                    edges.append(edge)
                
                # Configuración mejorada del grafo con tamaño optimizado para pantalla completa
                config = Config(
                    width=1400,  # AUMENTADO: de 1200 a 1400 para cubrir más ancho
                    height=700,  # AUMENTADO: de 500 a 700 para mejor altura
                    directed=False,
                    physics=True,
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",
                    collapsible=False,
                    node={
                        'labelProperty': 'label',
                        'size': 400,  # Tamaño aumentado
                        'highlightStrokeColor': '#A23B72',  # Color de resaltado mejorado
                        'fontSize': 16,  # Fuente más grande
                        'fontColor': '#ffffff'  # CAMBIO: Texto blanco para mejor contraste en fondo oscuro
                    },
                    link={
                        'labelProperty': 'label',
                        'renderLabel': True,
                        'fontSize': 14,  # Fuente más grande para etiquetas
                        'fontColor': '#e0e0e0'  # CAMBIO: Texto gris claro para mejor legibilidad
                    },
                    d3={
                        'alphaTarget': 0.05,
                        'gravity': -physics_strength * 80,  # REDUCIDO: de 100 a 80 para menos atracción
                        'linkDistance': node_spacing * 1.5,  # AUMENTADO: 50% más distancia entre nodos
                        'linkStrength': physics_strength * 0.7,  # REDUCIDO: de 1.0 a 0.7 para menos fuerza
                        'disableLinkForce': False
                    }
                )
                
                # Renderizar el grafo en un contenedor amplio
                st.subheader("🧠 Visualización del Mapa Mental")
                with st.container():
                    return_value = agraph(nodes=nodes, edges=edges, config=config)
                
                # Información adicional
                st.info("💡 **Instrucciones:** Haz clic en los nodos para explorar, arrastra para reorganizar, y usa la rueda del mouse para hacer zoom. El mapa se ajusta automáticamente para una mejor visualización.")
            
            except Exception as e:
                st.error(f"Error renderizando el mapa mental: {e}")
                return
        else:
            st.error("streamlit-agraph no está disponible. Instala con: pip install streamlit-agraph")
            return
        
        # Mostrar texto completo de los documentos si está habilitado
        if show_full_text:
            st.subheader("📄 Texto Completo de los Documentos")
            
            # Organizar por fuente si está disponible
            sources = {}
            for chunk in selected_chunks:
                source = chunk.get('source', 'Documento sin fuente')
                if source not in sources:
                    sources[source] = []
                sources[source].append(chunk)
            
            for source, source_chunks in sources.items():
                with st.expander(f"📖 {source} ({len(source_chunks)} secciones)", expanded=False):
                    for i, chunk in enumerate(source_chunks, 1):
                        st.markdown(f"**Sección {i}:**")
                        content = chunk.get('content', 'Sin contenido')
                        st.markdown(content)
                        if i < len(source_chunks):
                            st.divider()
        
        # Opciones de descarga mejoradas
        st.subheader("📥 Opciones de Descarga")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Descargar datos JSON del mapa
            import json
            json_data = json.dumps(mind_map_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="🧠 Descargar Mapa JSON",
                data=json_data,
                file_name=f"mapa_mental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Descarga los datos del mapa mental en formato JSON"
            )
        
        with col2:
            # Descargar estructura JSON
            if mind_structure:
                structure_json = json.dumps(mind_structure, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📊 Descargar Estructura JSON",
                    data=structure_json,
                    file_name=f"estructura_mental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Descarga la estructura de datos del mapa mental"
                )
        
        with col3:
            # Generar y descargar reporte completo
            full_text = "\n\n".join([chunk.get('content', '') for chunk in selected_chunks])
            report_content = f"""# Reporte de Mapa Mental Interactivo

**Generado el:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## Resumen del Análisis

- **Tema Central:** {mind_structure.get('central_theme', {}).get('name', 'No identificado') if mind_structure else 'No disponible'}
- **Ramas Principales:** {len(mind_structure.get('main_branches', [])) if mind_structure else 0}
- **Total de Nodos:** {mind_map_data['stats']['total_nodes']}
- **Total de Conexiones:** {mind_map_data['stats']['total_edges']}
- **Total de Documentos:** {len(set(chunk.get('source', 'Sin fuente') for chunk in selected_chunks))}
- **Total de Secciones:** {len(selected_chunks)}

## Estructura del Mapa Mental

{json.dumps(mind_structure, indent=2, ensure_ascii=False) if mind_structure else 'No disponible'}

## Datos del Mapa

{json.dumps(mind_map_data, indent=2, ensure_ascii=False)}

## Texto Completo de los Documentos

{full_text}
"""
            
            st.download_button(
                label="📋 Descargar Reporte Completo",
                data=report_content,
                file_name=f"reporte_mental_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                help="Descarga un reporte completo con mapa, estructura y texto"
            )
    
    else:
        st.error("❌ No se pudo generar el mapa mental. Verifica que los datos sean válidos.")
        
        # Información de debug mejorada
        with st.expander("🔧 Información de Debug"):
            st.write("Verificando disponibilidad de streamlit-agraph...")
            st.write(f"streamlit-agraph disponible: {AGRAPH_AVAILABLE}")
            st.write(f"Número de chunks: {len(selected_chunks)}")
            
            if selected_chunks:
                st.write("Muestra del primer chunk:")
                st.json(selected_chunks[0])
            
            st.write("Intentando generar estructura de mapa mental...")
            try:
                test_structure = analyzer._analyze_intelligent_mind_map_structure(chunks[:5])
                st.write("✅ Estructura del mapa mental generada:")
                st.json(test_structure)
            except Exception as e:
                st.write(f"❌ Error analizando estructura: {e}")
                
            # Verificar dependencias adicionales
            if AGRAPH_AVAILABLE:
                st.write("✅ Componentes de streamlit-agraph disponibles")
            else:
                st.write("❌ streamlit-agraph no está disponible")
                st.write("Instala con: pip install streamlit-agraph")

def render_automatic_summary(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar resumen automático mejorado con opciones duales (normal/IA) y control de pausa"""
    st.header("📝 Resumen Inteligente")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Configuración avanzada con opciones duales
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        generation_mode = st.selectbox(
            "🎯 Modo de Generación:",
            options=["ai", "normal"],
            format_func=lambda x: {
                "normal": "⚡ Generación Normal (Rápida)",
                "ai": "🧠 Generación con IA (Inteligente)"
            }[x],
            help="Elige entre generación inteligente con IA o procesamiento tradicional rápido",
            key="summary_generation_mode"
        )
    
    with col2:
        if generation_mode == "ai":
            summary_type = st.selectbox(
                "Tipo de Resumen:",
                options=["comprehensive", "executive", "analytical", "thematic"],
                format_func=lambda x: {
                    "comprehensive": "📋 Comprehensivo",
                    "executive": "🎯 Ejecutivo", 
                    "analytical": "🔍 Analítico",
                    "thematic": "🏷️ Temático"
                }[x],
                help="Selecciona el tipo de análisis que mejor se adapte a tus necesidades"
            )
        else:
            max_sentences = st.slider(
                "Máximo de Oraciones:",
                min_value=3,
                max_value=15,
                value=5,
                help="Número máximo de oraciones en el resumen básico"
            )
    
    # Inicializar estado de generación
    if 'summary_generating' not in st.session_state:
        st.session_state.summary_generating = False
    if 'summary_paused' not in st.session_state:
        st.session_state.summary_paused = False
    
    with col3:
        if not st.session_state.summary_generating:
            if st.button("🔄 Generar", type="primary", key="summary_generate"):
                # Limpiar cache para regenerar
                cache_key = f"summary_{generation_mode}_{summary_type if generation_mode == 'ai' else max_sentences}"
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                st.session_state.summary_generating = True
                st.session_state.summary_paused = False
                st.rerun()
        else:
            if st.button("⏸️ Pausar", type="secondary", key="pause_summary_generation"):
                st.session_state.summary_paused = True
                st.session_state.summary_generating = False
                st.warning("⏸️ Generación pausada por el usuario")
    
    with col4:
        if st.session_state.summary_generating and generation_mode == "ai":
            if st.button("⏹️ Detener", type="secondary", key="stop_summary_generation"):
                st.session_state.summary_generating = False
                st.session_state.summary_paused = False
                st.error("⏹️ Generación detenida por el usuario")
    
    # Generar resumen
    cache_key = f"summary_{generation_mode}_{summary_type if generation_mode == 'ai' else max_sentences}"
    
    if cache_key not in st.session_state and st.session_state.summary_generating and not st.session_state.summary_paused:
        if generation_mode == "ai":
            with st.spinner("🤖 Generando resumen inteligente con IA..."):
                try:
                    result = analyzer.generate_intelligent_summary(chunks, summary_type)
                    st.session_state[cache_key] = result
                    st.session_state.summary_generating = False
                    st.success("✅ Resumen inteligente generado exitosamente")
                except Exception as e:
                    st.error(f"❌ Error generando resumen con IA: {str(e)}")
                    st.session_state.summary_generating = False
        else:
            with st.spinner("📝 Generando resumen básico..."):
                try:
                    basic_summary = analyzer.generate_basic_summary(chunks, max_sentences)
                    result = {
                        'summary': basic_summary,
                        'type': 'basic',
                        'metadata': {
                            'sources_count': len(set(chunk.get('metadata', {}).get('source_file', 'Documento') for chunk in chunks)),
                            'chunks_processed': len(chunks),
                            'max_sentences': max_sentences,
                            'generation_method': 'Análisis básico de frecuencia'
                        }
                    }
                    st.session_state[cache_key] = result
                    st.session_state.summary_generating = False
                    st.success("✅ Resumen básico generado exitosamente")
                except Exception as e:
                    st.error(f"❌ Error generando resumen básico: {str(e)}")
                    st.session_state.summary_generating = False
    
    # Mostrar resumen si existe
    if cache_key in st.session_state:
        result = st.session_state[cache_key]
        summary = result.get('summary', '')
        metadata = result.get('metadata', {})
        
        # Mostrar resumen en contenedor expandible
        with st.container():
            st.subheader("📄 Resumen Generado")
        
        # Crear tabs para mejor organización
        tab1, tab2, tab3 = st.tabs(["📖 Resumen", "📊 Estadísticas", "🔧 Acciones"])
        
        with tab1:
            # Mostrar resumen en área de texto expandible
            st.markdown(summary)
            
            # Mostrar en área de código para fácil copia
            with st.expander("📋 Ver texto plano (para copiar)"):
                st.text_area(
                    "Resumen completo:",
                    value=summary,
                    height=300,
                    help="Puedes copiar todo el texto desde aquí"
                )
        
        with tab2:
            # Estadísticas detalladas
            if metadata:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📄 Fuentes", 
                        metadata.get('sources_count', 'N/A'),
                        help="Número de documentos fuente analizados"
                    )
                
                with col2:
                    st.metric(
                        "🧩 Fragmentos", 
                        metadata.get('chunks_processed', 'N/A'),
                        help="Fragmentos de texto procesados"
                    )
                
                with col3:
                    summary_words = metadata.get('summary_words', len(summary.split()))
                    st.metric(
                        "📝 Palabras", 
                        summary_words,
                        help="Palabras en el resumen generado"
                    )
                
                with col4:
                    if 'compression_ratio' in metadata:
                        st.metric(
                            "📊 Compresión", 
                            f"{metadata['compression_ratio']}%",
                            help="Ratio de compresión del contenido original"
                        )
                    else:
                        st.metric("📊 Caracteres", len(summary))
                
                # Información adicional
                if 'model_used' in metadata:
                    st.info(f"🤖 **Modelo utilizado:** {metadata['model_used']}")
                
                if 'sources' in metadata and metadata['sources']:
                    with st.expander("📚 Fuentes analizadas"):
                        for i, source in enumerate(metadata['sources'], 1):
                            st.write(f"{i}. {source}")
                
                if 'total_words_input' in metadata:
                    st.success(f"✅ Se procesaron **{metadata['total_words_input']:,}** palabras del contenido original")
        
        with tab3:
            # Acciones disponibles
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Descargar TXT", key="download_summary_txt"):
                    # Crear archivo de descarga
                    txt_content = f"""RESUMEN GENERADO - {summary_type.upper()}
{'='*50}

{summary}

{'='*50}
Generado por CogniChat
Fuentes: {metadata.get('sources_count', 'N/A')}
Fragmentos procesados: {metadata.get('chunks_processed', 'N/A')}
"""
                    st.download_button(
                        label="💾 Descargar",
                        data=txt_content,
                        file_name=f"resumen_{summary_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("📊 Analizar Resumen", key="analyze_summary"):
                    # Análisis adicional del resumen
                    with st.spinner("Analizando resumen..."):
                        # Análisis de sentimiento del resumen
                        from textblob import TextBlob
                        blob = TextBlob(summary)
                        sentiment = blob.sentiment
                        
                        st.write("**Análisis del Resumen:**")
                        st.write(f"- **Polaridad:** {sentiment.polarity:.2f} ({'Positivo' if sentiment.polarity > 0 else 'Negativo' if sentiment.polarity < 0 else 'Neutral'})")
                        st.write(f"- **Subjetividad:** {sentiment.subjectivity:.2f} ({'Alto' if sentiment.subjectivity > 0.5 else 'Bajo'})")
                        st.write(f"- **Oraciones:** {len([s for s in summary.split('.') if s.strip()])}")
                        # Corregir el f-string que no puede contener backslashes
                        paragraph_count = len([p for p in summary.split('\n\n') if p.strip()])
                        st.write(f"- **Párrafos:** {paragraph_count}")
            
            with col3:
                if st.button("🔄 Regenerar", key="regenerate_summary"):
                    # Forzar regeneración
                    if cache_key in st.session_state:
                        del st.session_state[cache_key]
                    st.rerun()
    
    # Mostrar información de ayuda
    with st.expander("ℹ️ Información sobre tipos de resumen"):
        st.markdown("""
        **📋 Comprehensivo:** Análisis detallado que cubre todos los aspectos importantes del contenido.
        
        **🎯 Ejecutivo:** Resumen enfocado en puntos clave para toma de decisiones estratégicas.
        
        **🔍 Analítico:** Análisis profundo con enfoque en patrones, relaciones y insights académicos.
        
        **🏷️ Temático:** Organización del contenido por temas principales y sus interrelaciones.
        """)

def render_triangulation_analysis(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar análisis de triangulación (funciona con 1 o más fuentes)"""
    st.header("🔺 Análisis de Triangulación")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Realizar triangulación
    if 'triangulation_data' not in st.session_state:
        with st.spinner("Realizando análisis de triangulación..."):
            triangulation = analyzer.perform_triangulation_analysis(chunks)
            st.session_state.triangulation_data = triangulation
    
    triangulation = st.session_state.triangulation_data
    
    if 'error' in triangulation:
        st.error(f"Error en triangulación: {triangulation['error']}")
        return
    
    # Detectar modo de análisis
    analysis_mode = triangulation.get('analysis_mode', 'multi-source')
    
    # Mostrar información según el modo
    if analysis_mode == 'single-source-internal':
        st.info(f"ℹ️ **Modo: Triangulación Interna** - {triangulation.get('info', 'Análisis de una sola fuente')}")
    else:
        st.info(f"ℹ️ **Modo: Triangulación Multi-Fuente** - Validación cruzada entre {triangulation['total_sources']} fuentes")
    
    # Estadísticas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if analysis_mode == 'single-source-internal':
            st.metric("📚 Fuentes", triangulation['total_sources'])
            st.caption(f"📑 {triangulation.get('total_sections', 0)} secciones")
        else:
            st.metric("📚 Fuentes Totales", triangulation['total_sources'])
    
    with col2:
        st.metric("🔍 Conceptos Totales", triangulation['total_concepts'])
    
    with col3:
        st.metric("✅ Conceptos Validados", triangulation['validated_concepts'])
        if analysis_mode == 'single-source-internal' and 'unique_concepts' in triangulation:
            st.caption(f"🔸 {triangulation['unique_concepts']} únicos")
    
    with col4:
        validation_rate = (triangulation['validated_concepts'] / triangulation['total_concepts']) * 100 if triangulation['total_concepts'] > 0 else 0
        st.metric("📊 Tasa de Validación", f"{validation_rate:.1f}%")
    
    # Lista de fuentes
    st.subheader("📚 Fuentes Analizadas")
    for i, source in enumerate(triangulation['sources'], 1):
        st.write(f"{i}. {source}")
    
    # Título dinámico según modo
    if analysis_mode == 'single-source-internal':
        st.subheader("🔺 Conceptos por Frecuencia de Aparición en Secciones")
        st.caption("Los conceptos que aparecen en múltiples secciones tienen mayor confiabilidad")
    else:
        st.subheader("🔺 Conceptos Triangulados Entre Fuentes")
        st.caption("Los conceptos que aparecen en múltiples fuentes están validados")
    
    if triangulation['triangulated_concepts']:
        # Crear DataFrame para visualización
        concepts_data = []
        label_fuentes = "Secciones" if analysis_mode == 'single-source-internal' else "Fuentes"
        
        for concept in triangulation['triangulated_concepts']:
            # Agregar información de tipo de validación
            validation_type = concept.get('validation_type', 'multi-source')
            validation_icon = {
                'internal-triangulation': '🔄',
                'single-section': '📍',
                'multi-source': '✅',
                'single-source': '⚠️'
            }.get(validation_type, '📊')
            
            concepts_data.append({
                'Concepto': f"{validation_icon} {concept['concept']}",
                label_fuentes: concept['source_count'],
                'Confiabilidad': f"{concept['reliability']:.2%}",
                'Puntuación Promedio': f"{concept['avg_score']:.3f}",
                'Ubicaciones': concept.get('distribution', ', '.join(concept['sources'][:3]) + ('...' if len(concept['sources']) > 3 else ''))
            })
        
        df_concepts = pd.DataFrame(concepts_data)
        st.dataframe(df_concepts, use_container_width=True)
        
        # Leyenda de iconos
        if analysis_mode == 'single-source-internal':
            st.caption("🔄 Aparece en múltiples secciones | 📍 Aparece en una sola sección")
        
        # Gráfico de confiabilidad
        if len(concepts_data) > 1:
            fig = px.scatter(
                df_concepts.head(15),
                x=label_fuentes,
                y='Puntuación Promedio',
                size=label_fuentes,
                hover_data=['Concepto', 'Confiabilidad'],
                title=f"Conceptos por {label_fuentes} y Puntuación",
                labels={
                    label_fuentes: label_fuentes,
                    'Puntuación Promedio': 'Relevancia'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        if analysis_mode == 'single-source-internal':
            st.info("No se encontraron conceptos que aparezcan en múltiples secciones del documento.")
        else:
            st.info("No se encontraron conceptos que aparezcan en múltiples fuentes.")
    
    # Información adicional para fuente única (mostrar siempre)
    if analysis_mode == 'single-source-internal':
        st.divider()
        st.markdown("### 💡 Interpretación de Resultados")
        st.markdown("""
        **Triangulación Interna** analiza la consistencia de conceptos dentro de un mismo documento:
        
        - **Alta confiabilidad** (>50%): El concepto aparece en más de la mitad de las secciones → concepto central
        - **Media confiabilidad** (25-50%): El concepto aparece regularmente → concepto importante
        - **Baja confiabilidad** (<25%): El concepto es específico de ciertas secciones → concepto especializado
        
        Esta técnica es útil para:
        - ✅ Identificar temas centrales vs. específicos
        - ✅ Detectar la estructura del documento
        - ✅ Validar la coherencia del contenido
        """)

# =============================================================================
# 19. FUNCIÓN PRINCIPAL DE RENDERIZADO
# =============================================================================

def render():
    """Función principal para renderizar el módulo de análisis cualitativo avanzado"""
    st.title("🔬 Análisis Cualitativo Avanzado")
    st.markdown("*Análisis profundo de contenido RAG con técnicas de NLP y mapas interactivos*")
    
    # Verificar disponibilidad de funcionalidades avanzadas
    if not ADVANCED_ANALYSIS_AVAILABLE:
        st.warning("⚠️ Algunas funcionalidades avanzadas no están disponibles.")
        st.info("Para habilitar todas las funcionalidades, instala las dependencias adicionales:")
        st.code("pip install -r requirements.txt")
    
    # Inicializar analizador
    analyzer = AdvancedQualitativeAnalyzer()
    
    # Botón para refrescar datos
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Actualizar Datos", type="secondary", key="update_analysis_data"):
            # Limpiar cache de análisis cualitativo
            for key in list(st.session_state.keys()):
                if key.startswith(('qualitative_', 'theme_analysis_', 'clustering_', 'concept_map_', 
                                 'mind_map_', 'auto_summary', 'triangulation_', 'sentiment_', 'wordcloud_')):
                    del st.session_state[key]
            st.rerun()
    
    # Cargar datos del RAG
    if 'qualitative_chunks' not in st.session_state:
        with st.spinner("Cargando datos del sistema RAG..."):
            st.session_state.qualitative_chunks = analyzer.load_rag_data()
    
    chunks = st.session_state.qualitative_chunks
    
    if not chunks:
        st.warning("⚠️ No hay datos disponibles en el sistema RAG.")
        st.info("Asegúrate de haber procesado documentos en la pestaña 'Procesamiento RAG' primero.")
        return
    
    # Mostrar información de datos cargados
    with col1:
        st.info(f"📊 Datos cargados: {len(chunks)} chunks de {len(set(chunk.get('metadata', {}).get('source_file', 'Desconocido') for chunk in chunks))} documentos")
    
    # Pestañas del análisis cualitativo avanzado - ACTUALIZADAS
    tabs = st.tabs([
        "📊 Dashboard",
        "🎯 Temas Avanzados",
        "🔍 Clustering", 
        "🗺️ Mapa Conceptual",
        "🧠 Mapa Mental",
        "📝 Resumen Automático",
        "🔺 Triangulación",
        "😊 Sentimientos",
        "☁️ Nube de Palabras",
        "⚙️ Configuración"
    ])
    
    with tabs[0]:
        render_advanced_dashboard(analyzer, chunks)
    
    with tabs[1]:
        render_advanced_themes(analyzer, chunks)
    
    with tabs[2]:
        render_clustering_analysis(analyzer, chunks)
    
    with tabs[3]:
        render_interactive_concept_map(analyzer, chunks)
    
    with tabs[4]:
        render_interactive_mind_map(analyzer, chunks)
    
    with tabs[5]:
        render_automatic_summary(analyzer, chunks)
    
    with tabs[6]:
        render_triangulation_analysis(analyzer, chunks)
    
    with tabs[7]:
        render_sentiment_analysis(analyzer, chunks)
    
    with tabs[8]:
        render_word_cloud(analyzer, chunks)
    
    with tabs[9]:
        render_settings_tab(analyzer)

if __name__ == "__main__":
    render()