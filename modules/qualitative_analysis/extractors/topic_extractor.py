"""
Extractor de Temas para Análisis Cualitativo
Implementa LDA, BERTopic y clustering semántico para identificación de temas
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Importaciones para procesamiento de texto
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Importaciones para NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Importaciones para visualización
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Importaciones del sistema
from ..core.config import AnalysisConfig
from ..core.citation_manager import CitationManager

logger = logging.getLogger(__name__)

@dataclass
class TopicData:
    """Datos de un tema identificado"""
    topic_id: int
    topic_name: str
    keywords: List[str]
    coherence_score: float
    frequency: int
    documents: List[str]
    citations: List[Dict[str, Any]]
    description: str
    confidence: float

@dataclass
class TopicAnalysisResult:
    """Resultado del análisis de temas"""
    topics: List[TopicData]
    algorithm_used: str
    total_topics: int
    coherence_avg: float
    processing_time: float
    metadata: Dict[str, Any]
    citations: List[Dict[str, Any]]

class TopicExtractor:
    """
    Extractor de temas especializado para análisis cualitativo
    
    Características:
    - Múltiples algoritmos (LDA, BERTopic, Clustering)
    - Sistema de citación integrado
    - Análisis de coherencia
    - Generación de descripciones contextuales
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.citation_manager = CitationManager()
        self.stemmer = SnowballStemmer('spanish')
        self.stopwords = self._get_spanish_stopwords()
        
        # Inicializar NLTK si es necesario
        self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Inicializar recursos de NLTK"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def _get_spanish_stopwords(self) -> List[str]:
        """Obtener stopwords en español"""
        try:
            spanish_stopwords = set(stopwords.words('spanish'))
            # Agregar stopwords personalizados
            custom_stopwords = {
                'también', 'puede', 'ser', 'está', 'están', 'hacer', 'hace',
                'tiene', 'tienen', 'dice', 'dice', 'según', 'mismo', 'misma',
                'mismos', 'mismas', 'otro', 'otra', 'otros', 'otras', 'cada',
                'todo', 'toda', 'todos', 'todas', 'muy', 'más', 'menos',
                'mucho', 'muchos', 'poco', 'pocos', 'algunos', 'algunas'
            }
            return list(spanish_stopwords.union(custom_stopwords))
        except:
            return []
    
    def extract_topics_lda(self, chunks: List[Dict[str, Any]], n_topics: int = 10) -> TopicAnalysisResult:
        """
        Extraer temas usando LDA (Latent Dirichlet Allocation)
        
        Args:
            chunks: Lista de chunks de texto
            n_topics: Número de temas a identificar
            
        Returns:
            Resultado del análisis de temas
        """
        start_time = datetime.now()
        
        try:
            # Preprocesar textos
            texts = self._preprocess_texts(chunks)
            
            if len(texts) < n_topics:
                raise ValueError(f"Insuficientes documentos para {n_topics} temas. Mínimo requerido: {n_topics}")
            
            # Vectorización TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=self.stopwords,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Verificar que la matriz no esté vacía
            if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
                raise ValueError("La matriz TF-IDF está vacía. Verifica que hay suficiente contenido de texto.")
            
            # Aplicar LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100,
                learning_decay=0.7
            )
            
            lda.fit(tfidf_matrix)
            
            # Extraer temas
            topics = self._extract_lda_topics(lda, feature_names, n_topics)
            
            # Calcular coherencia
            coherence_scores = self._calculate_topic_coherence(topics, texts)
            
            # Enriquecer con citaciones
            enriched_topics = self._enrich_topics_with_citations(topics, chunks, coherence_scores)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TopicAnalysisResult(
                topics=enriched_topics,
                algorithm_used="LDA",
                total_topics=len(enriched_topics),
                coherence_avg=np.mean(coherence_scores),
                processing_time=processing_time,
                metadata={
                    'n_topics': n_topics,
                    'n_documents': len(chunks),
                    'n_features': len(feature_names)
                },
                citations=self.citation_manager.export_citations()
            )
            
        except Exception as e:
            logger.error(f"Error en extracción LDA: {str(e)}")
            raise
    
    def extract_topics_clustering(self, chunks: List[Dict[str, Any]], n_clusters: int = 10) -> TopicAnalysisResult:
        """
        Extraer temas usando clustering semántico
        
        Args:
            chunks: Lista de chunks de texto
            n_clusters: Número de clusters/temas
            
        Returns:
            Resultado del análisis de temas
        """
        start_time = datetime.now()
        
        try:
            # Preprocesar textos
            texts = self._preprocess_texts(chunks)
            
            if len(texts) < n_clusters:
                raise ValueError(f"Insuficientes documentos para {n_clusters} clusters. Mínimo requerido: {n_clusters}")
            
            # Vectorización TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=self.stopwords,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Verificar que la matriz no esté vacía
            if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
                raise ValueError("La matriz TF-IDF está vacía. Verifica que hay suficiente contenido de texto.")
            
            # Aplicar K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Extraer temas de clusters
            topics = self._extract_cluster_topics(
                kmeans, tfidf_matrix, feature_names, cluster_labels, texts
            )
            
            # Calcular coherencia
            coherence_scores = self._calculate_topic_coherence(topics, texts)
            
            # Enriquecer con citaciones
            enriched_topics = self._enrich_topics_with_citations(topics, chunks, coherence_scores)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TopicAnalysisResult(
                topics=enriched_topics,
                algorithm_used="Clustering Semántico",
                total_topics=len(enriched_topics),
                coherence_avg=np.mean(coherence_scores),
                processing_time=processing_time,
                metadata={
                    'n_clusters': n_clusters,
                    'n_documents': len(chunks),
                    'n_features': len(feature_names)
                },
                citations=self.citation_manager.export_citations()
            )
            
        except Exception as e:
            logger.error(f"Error en clustering semántico: {str(e)}")
            raise
    
    def _preprocess_texts(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Preprocesar textos para análisis de temas"""
        texts = []
        
        for chunk in chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                text = chunk['content']
                
                # Limpiar texto
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip().lower()
                
                if len(text) > 50:  # Solo textos significativos
                    texts.append(text)
        
        return texts
    
    def _extract_lda_topics(self, lda_model, feature_names: List[str], n_topics: int) -> List[TopicData]:
        """Extraer temas del modelo LDA"""
        topics = []
        
        for topic_idx, topic in enumerate(lda_model.components_):
            # Obtener palabras más importantes
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_scores = [topic[i] for i in top_words_idx]
            
            # Crear nombre del tema
            topic_name = f"Tema {topic_idx + 1}: {' '.join(top_words[:3])}"
            
            # Calcular frecuencia
            frequency = int(np.sum(topic))
            
            # Crear descripción
            description = self._generate_topic_description(top_words, top_scores)
            
            topics.append(TopicData(
                topic_id=topic_idx,
                topic_name=topic_name,
                keywords=top_words,
                coherence_score=0.0,  # Se calculará después
                frequency=frequency,
                documents=[],  # Se llenará después
                citations=[],  # Se llenará después
                description=description,
                confidence=float(np.mean(top_scores))
            ))
        
        return topics
    
    def _extract_cluster_topics(self, kmeans_model, tfidf_matrix, feature_names: List[str], 
                               cluster_labels: np.ndarray, texts: List[str]) -> List[TopicData]:
        """Extraer temas de clusters"""
        topics = []
        
        for cluster_id in range(kmeans_model.n_clusters):
            # Obtener documentos del cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_docs = tfidf_matrix[cluster_mask]
            
            if cluster_docs.shape[0] == 0:
                continue
            
            # Calcular centroide del cluster
            centroid = np.mean(cluster_docs, axis=0).A1
            
            # Obtener palabras más importantes
            top_words_idx = centroid.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_scores = [centroid[i] for i in top_words_idx]
            
            # Crear nombre del tema
            topic_name = f"Cluster {cluster_id + 1}: {' '.join(top_words[:3])}"
            
            # Calcular frecuencia
            frequency = int(np.sum(cluster_mask))
            
            # Crear descripción
            description = self._generate_topic_description(top_words, top_scores)
            
            topics.append(TopicData(
                topic_id=cluster_id,
                topic_name=topic_name,
                keywords=top_words,
                coherence_score=0.0,  # Se calculará después
                frequency=frequency,
                documents=[],  # Se llenará después
                citations=[],  # Se llenará después
                description=description,
                confidence=float(np.mean(top_scores))
            ))
        
        return topics
    
    def _calculate_topic_coherence(self, topics: List[TopicData], texts: List[str]) -> List[float]:
        """Calcular coherencia de temas"""
        coherence_scores = []
        
        for topic in topics:
            # Calcular coherencia basada en co-ocurrencia de palabras
            coherence = self._calculate_topic_coherence_score(topic.keywords, texts)
            coherence_scores.append(coherence)
        
        return coherence_scores
    
    def _calculate_topic_coherence_score(self, keywords: List[str], texts: List[str]) -> float:
        """Calcular score de coherencia para un tema"""
        try:
            # Contar co-ocurrencias de palabras clave
            co_occurrences = 0
            total_pairs = 0
            
            for text in texts:
                words = text.split()
                for i, word1 in enumerate(keywords[:5]):  # Solo primeras 5 palabras
                    for j, word2 in enumerate(keywords[:5]):
                        if i != j and word1 in words and word2 in words:
                            co_occurrences += 1
                        total_pairs += 1
            
            if total_pairs > 0:
                return co_occurrences / total_pairs
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _generate_topic_description(self, keywords: List[str], scores: List[float]) -> str:
        """Generar descripción contextual del tema"""
        try:
            # Crear descripción basada en palabras clave
            top_keywords = keywords[:5]
            description = f"Tema caracterizado por: {', '.join(top_keywords)}"
            
            # Agregar contexto basado en palabras
            if any(word in ['educación', 'formación', 'aprendizaje'] for word in top_keywords):
                description += ". Este tema se relaciona con aspectos educativos y formativos."
            elif any(word in ['trabajo', 'empleo', 'laboral'] for word in top_keywords):
                description += ". Este tema se relaciona con aspectos laborales y profesionales."
            elif any(word in ['social', 'comunidad', 'sociedad'] for word in top_keywords):
                description += ". Este tema se relaciona con aspectos sociales y comunitarios."
            
            return description
            
        except Exception:
            return f"Tema caracterizado por: {', '.join(keywords[:3])}"
    
    def _enrich_topics_with_citations(self, topics: List[TopicData], chunks: List[Dict[str, Any]], 
                                    coherence_scores: List[float]) -> List[TopicData]:
        """Enriquecer temas con citaciones y documentos"""
        enriched_topics = []
        
        for i, topic in enumerate(topics):
            # Actualizar coherencia
            topic.coherence_score = coherence_scores[i] if i < len(coherence_scores) else 0.0
            
            # Buscar documentos relacionados
            related_docs = self._find_related_documents(topic, chunks)
            topic.documents = related_docs
            
            # Generar citaciones
            citations = self._generate_topic_citations(topic, chunks)
            topic.citations = citations
            
            enriched_topics.append(topic)
        
        return enriched_topics
    
    def _find_related_documents(self, topic: TopicData, chunks: List[Dict[str, Any]]) -> List[str]:
        """Encontrar documentos relacionados con el tema"""
        related_docs = []
        
        for chunk in chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                content = chunk['content'].lower()
                
                # Verificar si el chunk contiene palabras clave del tema
                keyword_matches = sum(1 for keyword in topic.keywords[:5] 
                                    if keyword.lower() in content)
                
                if keyword_matches >= 2:  # Al menos 2 palabras clave
                    source = chunk.get('metadata', {}).get('source_file', 'unknown')
                    if source not in related_docs:
                        related_docs.append(source)
        
        return related_docs
    
    def _generate_topic_citations(self, topic: TopicData, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generar citaciones para el tema"""
        citations = []
        
        for chunk in chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                content = chunk['content']
                
                # Verificar si el chunk contiene palabras clave del tema
                keyword_matches = sum(1 for keyword in topic.keywords[:5] 
                                    if keyword.lower() in content.lower())
                
                if keyword_matches >= 2:
                    citation = self.citation_manager.add_citation(
                        source_file=chunk.get('metadata', {}).get('source_file', 'unknown'),
                        chunk_id=chunk.get('metadata', {}).get('chunk_index', 0),
                        content=content,
                        context_before="",  # Se puede mejorar después
                        context_after="",  # Se puede mejorar después
                        relevance_score=keyword_matches / len(topic.keywords[:5])
                    )
                    citations.append(citation.to_dict())
        
        return citations
    
    def generate_topic_visualization(self, result: TopicAnalysisResult) -> Dict[str, Any]:
        """Generar visualizaciones para los temas"""
        try:
            # Preparar datos para visualización
            topics_data = []
            for topic in result.topics:
                topics_data.append({
                    'topic_id': topic.topic_id,
                    'topic_name': topic.topic_name,
                    'frequency': topic.frequency,
                    'coherence': topic.coherence_score,
                    'confidence': topic.confidence,
                    'keywords': ', '.join(topic.keywords[:5])
                })
            
            # Crear gráfico de distribución de temas
            fig_distribution = px.bar(
                topics_data,
                x='topic_name',
                y='frequency',
                title='Distribución de Frecuencia por Tema',
                labels={'frequency': 'Frecuencia', 'topic_name': 'Tema'}
            )
            
            # Crear gráfico de coherencia
            fig_coherence = px.bar(
                topics_data,
                x='topic_name',
                y='coherence',
                title='Coherencia de Temas',
                labels={'coherence': 'Coherencia', 'topic_name': 'Tema'},
                color='coherence',
                color_continuous_scale='Viridis'
            )
            
            return {
                'distribution_chart': fig_distribution,
                'coherence_chart': fig_coherence,
                'topics_data': topics_data,
                'algorithm': result.algorithm_used,
                'total_topics': result.total_topics,
                'avg_coherence': result.coherence_avg
            }
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {str(e)}")
            return {}
    
    def export_topic_analysis(self, result: TopicAnalysisResult, format: str = 'json') -> Dict[str, Any]:
        """Exportar análisis de temas"""
        try:
            export_data = {
                'metadata': {
                    'algorithm': result.algorithm_used,
                    'total_topics': result.total_topics,
                    'avg_coherence': result.coherence_avg,
                    'processing_time': result.processing_time,
                    'export_date': datetime.now().isoformat()
                },
                'topics': []
            }
            
            for topic in result.topics:
                topic_data = {
                    'topic_id': topic.topic_id,
                    'topic_name': topic.topic_name,
                    'keywords': topic.keywords,
                    'coherence_score': topic.coherence_score,
                    'frequency': topic.frequency,
                    'confidence': topic.confidence,
                    'description': topic.description,
                    'documents': topic.documents,
                    'citations_count': len(topic.citations)
                }
                export_data['topics'].append(topic_data)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exportando análisis: {str(e)}")
            return {}
