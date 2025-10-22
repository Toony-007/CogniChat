"""
Extractor Inteligente de Conceptos Clave
Sistema diseñado para identificar y analizar conceptos principales en documentos de investigación

Este módulo NO copia y pega información, sino que:
1. Analiza el contenido completo
2. Identifica patrones y frecuencias
3. Sintetiza conceptos clave
4. Proporciona contexto y fundamentación
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import re
from datetime import datetime

# Importaciones para NLP
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from ..core.citation_manager import CitationManager, Citation
from ..core.config import AnalysisConfig


@dataclass
class ExtractedConcept:
    """
    Representa un concepto extraído del análisis
    
    Cada concepto incluye:
    - El término o frase identificada
    - Métricas de relevancia
    - Citas a las fuentes originales
    - Contexto de aparición
    """
    
    concept: str
    """El concepto identificado (palabra o frase)"""
    
    frequency: int
    """Número de veces que aparece en los documentos"""
    
    relevance_score: float
    """Score de relevancia (0.0 a 1.0) calculado por TF-IDF"""
    
    sources: List[str] = field(default_factory=list)
    """Lista de fuentes donde aparece el concepto"""
    
    citations: List[Citation] = field(default_factory=list)
    """Citas específicas donde aparece"""
    
    context_examples: List[str] = field(default_factory=list)
    """Ejemplos de contexto donde aparece el concepto"""
    
    related_concepts: List[str] = field(default_factory=list)
    """Conceptos relacionados que co-ocurren"""
    
    category: Optional[str] = None
    """Categoría del concepto (si se ha clasificado)"""
    
    extraction_method: str = "tfidf"
    """Método utilizado para extraer el concepto"""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """Momento de extracción"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'concept': self.concept,
            'frequency': self.frequency,
            'relevance_score': self.relevance_score,
            'sources': self.sources,
            'num_citations': len(self.citations),
            'context_examples': self.context_examples,
            'related_concepts': self.related_concepts,
            'category': self.category,
            'extraction_method': self.extraction_method,
            'timestamp': self.timestamp
        }
    
    def get_first_citation(self) -> Optional[Citation]:
        """Obtener la primera cita (más relevante)"""
        return self.citations[0] if self.citations else None
    
    def get_source_distribution(self) -> Dict[str, int]:
        """Obtener distribución del concepto por fuente"""
        distribution = Counter(self.sources)
        return dict(distribution)
    
    def __repr__(self) -> str:
        return f"ExtractedConcept('{self.concept}', freq={self.frequency}, score={self.relevance_score:.3f})"


class ConceptExtractor:
    """
    Extractor Inteligente de Conceptos Clave
    
    Este sistema analiza documentos para identificar los conceptos más importantes,
    proporcionando fundamentación completa con citas a las fuentes originales.
    
    Características principales:
    1. Procesamiento inteligente con TF-IDF
    2. Detección de n-gramas (frases completas)
    3. Sistema de citación integrado
    4. Análisis de co-ocurrencia
    5. Contexto de cada concepto
    
    Ejemplo de uso:
        extractor = ConceptExtractor(config)
        concepts = extractor.extract_concepts(chunks)
        
        # Ver conceptos con sus fuentes
        for concept in concepts[:10]:
            print(f"{concept.concept}: {concept.frequency} ocurrencias")
            print(f"  Fuentes: {', '.join(concept.sources)}")
            if concept.citations:
                citation = concept.get_first_citation()
                print(f"  Cita: {citation.format_citation()}")
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Inicializar el extractor
        
        Args:
            config: Configuración del análisis (usa valores por defecto si no se proporciona)
        """
        self.config = config or AnalysisConfig()
        self.citation_manager = CitationManager()
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """
        Cargar stopwords en español
        
        Returns:
            Conjunto de palabras vacías a ignorar
        """
        if NLTK_AVAILABLE:
            try:
                return set(stopwords.words('spanish'))
            except LookupError:
                # Descargar stopwords si no están disponibles
                try:
                    nltk.download('stopwords', quiet=True)
                    nltk.download('punkt', quiet=True)
                    return set(stopwords.words('spanish'))
                except:
                    pass
        
        # Stopwords básicas en español si NLTK no está disponible
        return {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
            'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
            'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
            'la', 'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy',
            'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo',
            'yo', 'también', 'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero',
            'desde', 'grande', 'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella',
            'sí', 'día', 'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
            'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde', 'ahora', 'parte',
            'después', 'vida', 'quedar', 'siempre', 'creer', 'hablar', 'llevar', 'dejar',
            'nada', 'cada', 'seguir', 'menos', 'nuevo', 'encontrar', 'algo', 'solo',
            'decir', 'mundo', 'casa', 'usar', 'salir', 'volver', 'tomar', 'conocer',
            'durante', 'último', 'llamar', 'empezar', 'menos', 'dios', 'hecho', 'casi',
            'momento', 'través', 'ser', 'estar', 'haber', 'hacer', 'poder', 'tener'
        }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesar texto para análisis
        
        Args:
            text: Texto original
            
        Returns:
            Texto preprocesado
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Eliminar emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Eliminar números standalone (pero mantener en palabras)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_with_tfidf(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[List[ExtractedConcept], List[str]]:
        """
        Extraer conceptos usando TF-IDF (Term Frequency-Inverse Document Frequency)
        
        TF-IDF identifica términos que son:
        - Frecuentes en un documento específico (TF)
        - Raros en el corpus general (IDF)
        
        Esto nos permite encontrar conceptos que son importantes y distintivos.
        
        Args:
            chunks: Lista de chunks de documentos
            
        Returns:
            Tupla de (conceptos extraídos, textos procesados)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn es requerido para extracción con TF-IDF")
        
        # Preparar textos
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '')
            processed = self._preprocess_text(content)
            texts.append(processed)
        
        if not texts:
            return [], []
        
        # Configurar vectorizador TF-IDF
        ngram_min, ngram_max = self.config.ngram_range if self.config.use_ngrams else (1, 1)
        
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_concepts * 2,  # Extraer más para luego filtrar
            stop_words=list(self.stopwords),
            ngram_range=(ngram_min, ngram_max),
            min_df=self.config.min_concept_frequency,
            max_df=0.85,  # Ignorar términos que aparecen en más del 85% de docs
            token_pattern=r'(?u)\b[a-záéíóúñü]+\b'  # Solo palabras en español
        )
        
        try:
            # Calcular TF-IDF
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calcular scores promedio para cada término
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            
            # Crear conceptos
            concepts = []
            term_locations = defaultdict(list)  # Para rastrear dónde aparece cada término
            
            for idx, (term, score) in enumerate(zip(feature_names, mean_scores)):
                if score > 0:
                    # Encontrar en qué chunks aparece
                    chunk_indices = tfidf_matrix[:, idx].nonzero()[0]
                    
                    # Calcular frecuencia real
                    frequency = sum(
                        chunks[chunk_idx]['content'].lower().count(term)
                        for chunk_idx in chunk_indices
                    )
                    
                    # Recopilar fuentes
                    sources = list(set(
                        chunks[chunk_idx]['metadata'].get('source_file', 'unknown')
                        for chunk_idx in chunk_indices
                    ))
                    
                    # Crear citas
                    citations = []
                    context_examples = []
                    
                    for chunk_idx in chunk_indices[:5]:  # Máximo 5 citas por concepto
                        chunk = chunks[chunk_idx]
                        content = chunk['content']
                        source = chunk['metadata'].get('source_file', 'unknown')
                        
                        # Encontrar posición del término
                        term_pos = content.lower().find(term)
                        if term_pos != -1:
                            # Extraer contexto
                            context_start = max(0, term_pos - self.config.citation_context_chars)
                            context_end = min(len(content), term_pos + len(term) + self.config.citation_context_chars)
                            
                            context_before = content[context_start:term_pos]
                            term_exact = content[term_pos:term_pos + len(term)]
                            context_after = content[term_pos + len(term):context_end]
                            
                            # Crear cita
                            if self.config.enable_citations:
                                citation = self.citation_manager.add_citation(
                                    source_file=source,
                                    chunk_id=chunk_idx,
                                    content=term_exact,
                                    context_before=context_before,
                                    context_after=context_after,
                                    page_number=chunk['metadata'].get('page_number'),
                                    relevance_score=float(score)
                                )
                                citations.append(citation)
                            
                            # Agregar ejemplo de contexto
                            context_example = f"...{context_before}[{term_exact}]{context_after}..."
                            context_examples.append(context_example)
                    
                    # Crear concepto
                    concept = ExtractedConcept(
                        concept=term,
                        frequency=frequency,
                        relevance_score=float(score),
                        sources=sources,
                        citations=citations,
                        context_examples=context_examples[:3],  # Máximo 3 ejemplos
                        extraction_method='tfidf'
                    )
                    
                    concepts.append(concept)
                    term_locations[term] = [int(i) for i in chunk_indices]
            
            # Ordenar por relevancia
            concepts.sort(key=lambda c: c.relevance_score, reverse=True)
            
            # Limitar a max_concepts
            concepts = concepts[:self.config.max_concepts]
            
            # Identificar conceptos relacionados (co-ocurrencia)
            self._identify_related_concepts(concepts, term_locations)
            
            return concepts, texts
            
        except Exception as e:
            raise Exception(f"Error en extracción TF-IDF: {str(e)}")
    
    def _identify_related_concepts(
        self,
        concepts: List[ExtractedConcept],
        term_locations: Dict[str, List[int]]
    ):
        """
        Identificar conceptos relacionados basándose en co-ocurrencia
        
        Args:
            concepts: Lista de conceptos extraídos
            term_locations: Diccionario de término -> lista de chunk indices
        """
        for concept in concepts:
            term = concept.concept
            locations = set(term_locations.get(term, []))
            
            if not locations:
                continue
            
            # Encontrar conceptos que co-ocurren
            related = []
            for other_concept in concepts:
                if other_concept.concept == term:
                    continue
                
                other_locations = set(term_locations.get(other_concept.concept, []))
                
                # Calcular co-ocurrencia
                intersection = locations & other_locations
                if len(intersection) >= 2:  # Co-ocurren en al menos 2 chunks
                    jaccard = len(intersection) / len(locations | other_locations)
                    if jaccard > 0.3:  # Umbral de similitud
                        related.append(other_concept.concept)
            
            # Guardar los 5 más relacionados
            concept.related_concepts = related[:5]
    
    def extract_concepts(
        self,
        chunks: List[Dict[str, Any]],
        method: str = "tfidf"
    ) -> List[ExtractedConcept]:
        """
        Extraer conceptos clave de los documentos
        
        Este es el método principal que un investigador utilizaría.
        
        Proceso:
        1. Preprocesa el texto (limpieza, normalización)
        2. Aplica algoritmo de extracción (TF-IDF por defecto)
        3. Identifica fuentes y crea citas
        4. Encuentra conceptos relacionados
        5. Ordena por relevancia
        
        Args:
            chunks: Lista de chunks de documentos con estructura:
                    {
                        'content': str,
                        'metadata': {
                            'source_file': str,
                            'page_number': int (opcional)
                        }
                    }
            method: Método de extracción ('tfidf' por defecto)
            
        Returns:
            Lista de ExtractedConcept ordenados por relevancia
            
        Raises:
            ValueError: Si chunks está vacío o mal formado
            ImportError: Si faltan dependencias necesarias
        """
        # Validar entrada
        if not chunks:
            raise ValueError("La lista de chunks no puede estar vacía")
        
        if not all('content' in chunk and 'metadata' in chunk for chunk in chunks):
            raise ValueError("Cada chunk debe tener 'content' y 'metadata'")
        
        # Limpiar citas anteriores
        self.citation_manager.clear()
        
        # Extraer conceptos según método
        if method == "tfidf":
            concepts, _ = self._extract_with_tfidf(chunks)
        else:
            raise ValueError(f"Método desconocido: {method}")
        
        return concepts
    
    def get_concept_summary(self, concepts: List[ExtractedConcept]) -> Dict[str, Any]:
        """
        Generar resumen estadístico de los conceptos extraídos
        
        Args:
            concepts: Lista de conceptos extraídos
            
        Returns:
            Diccionario con estadísticas
        """
        if not concepts:
            return {
                'total_concepts': 0,
                'total_frequency': 0,
                'avg_relevance': 0.0,
                'unique_sources': 0,
                'total_citations': 0
            }
        
        total_freq = sum(c.frequency for c in concepts)
        avg_relevance = sum(c.relevance_score for c in concepts) / len(concepts)
        all_sources = set()
        total_citations = 0
        
        for concept in concepts:
            all_sources.update(concept.sources)
            total_citations += len(concept.citations)
        
        return {
            'total_concepts': len(concepts),
            'total_frequency': total_freq,
            'avg_relevance': avg_relevance,
            'unique_sources': len(all_sources),
            'total_citations': total_citations,
            'top_concept': concepts[0].concept if concepts else None,
            'citation_stats': self.citation_manager.get_statistics()
        }
    
    def export_concepts(
        self,
        concepts: List[ExtractedConcept],
        include_citations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Exportar conceptos a formato serializable
        
        Args:
            concepts: Lista de conceptos
            include_citations: Si incluir información de citas
            
        Returns:
            Lista de diccionarios con conceptos
        """
        exported = []
        
        for concept in concepts:
            data = concept.to_dict()
            
            if include_citations and concept.citations:
                data['citations'] = [c.to_dict() for c in concept.citations]
            
            exported.append(data)
        
        return exported
    
    def __repr__(self) -> str:
        return f"ConceptExtractor(citations={len(self.citation_manager)}, config={self.config})"

