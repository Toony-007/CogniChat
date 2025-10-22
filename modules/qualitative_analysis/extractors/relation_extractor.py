"""
Extractor de Relaciones entre Conceptos
Identifica y analiza relaciones semánticas entre conceptos en documentos
"""

import re
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.config import AnalysisConfig
from ..core.citation_manager import CitationManager


@dataclass
class Relation:
    """Representa una relación entre dos conceptos"""
    source_concept: str
    target_concept: str
    relation_type: str  # 'co-occurrence', 'semantic', 'causal', 'hierarchical'
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    context: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationAnalysisResult:
    """Resultado del análisis de relaciones"""
    relations: List[Relation]
    concepts: List[str]
    relation_matrix: np.ndarray
    total_relations: int
    analysis_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class RelationExtractor:
    """
    Extractor de relaciones entre conceptos
    
    Identifica y analiza relaciones semánticas, co-ocurrencias y 
    conexiones causales entre conceptos en documentos.
    """
    
    def __init__(self, config: AnalysisConfig):
        """
        Inicializar extractor de relaciones
        
        Args:
            config: Configuración del análisis
        """
        self.config = config
        self.citation_manager = CitationManager()
        
        # Patrones para identificar relaciones causales
        self.causal_patterns = [
            r'\bcausa\b', r'\bprovoca\b', r'\bgenera\b', r'\bproduce\b',
            r'\bresulta en\b', r'\blleva a\b', r'\bconduce a\b',
            r'\borigina\b', r'\bdesencadena\b', r'\bdetermina\b',
            r'\bporque\b', r'\bdebido a\b', r'\bgracias a\b'
        ]
        
        # Patrones para relaciones jerárquicas
        self.hierarchical_patterns = [
            r'\bes parte de\b', r'\bincluye\b', r'\bcontiene\b',
            r'\bpertenece a\b', r'\bse divide en\b', r'\bcompuesto por\b',
            r'\bes un tipo de\b', r'\bes una forma de\b'
        ]
        
        # Stopwords en español para filtrado
        self.stopwords = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se',
            'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con',
            'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más'
        }
    
    def analyze_relations_cooccurrence(
        self, 
        chunks: List[Dict[str, Any]],
        window_size: int = 50,
        min_cooccurrence: int = 2
    ) -> RelationAnalysisResult:
        """
        Analizar relaciones basadas en co-ocurrencias
        
        Args:
            chunks: Lista de chunks de texto
            window_size: Tamaño de la ventana para co-ocurrencias (en palabras)
            min_cooccurrence: Número mínimo de co-ocurrencias
            
        Returns:
            RelationAnalysisResult con relaciones identificadas
        """
        if not chunks:
            return RelationAnalysisResult(
                relations=[],
                concepts=[],
                relation_matrix=np.array([[]]),
                total_relations=0,
                analysis_type='co-occurrence',
                metadata={'error': 'No chunks provided'}
            )
        
        # Extraer conceptos clave primero
        concepts = self._extract_key_concepts(chunks)
        
        if len(concepts) < 2:
            return RelationAnalysisResult(
                relations=[],
                concepts=concepts,
                relation_matrix=np.zeros((len(concepts), len(concepts))),
                total_relations=0,
                analysis_type='co-occurrence',
                metadata={'warning': 'Insufficient concepts for relation analysis'}
            )
        
        # Analizar co-ocurrencias
        cooccurrence_matrix = np.zeros((len(concepts), len(concepts)))
        relation_contexts = defaultdict(list)
        
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            words = content.split()
            
            # Buscar co-ocurrencias dentro de ventanas
            for i, concept1 in enumerate(concepts):
                for j, concept2 in enumerate(concepts):
                    if i >= j:  # Evitar duplicados y auto-relaciones
                        continue
                    
                    # Buscar ambos conceptos en ventanas de texto
                    cooccurrences = self._find_cooccurrences(
                        content, concept1, concept2, window_size
                    )
                    
                    if cooccurrences > 0:
                        cooccurrence_matrix[i, j] += cooccurrences
                        cooccurrence_matrix[j, i] += cooccurrences
                        
                        # Guardar contexto
                        context = self._extract_context(content, concept1, concept2)
                        if context:
                            relation_contexts[(concept1, concept2)].append({
                                'context': context,
                                'chunk': chunk
                            })
        
        # Crear relaciones a partir de co-ocurrencias significativas
        relations = []
        max_cooccurrence = cooccurrence_matrix.max() if cooccurrence_matrix.max() > 0 else 1
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:
                    continue
                
                cooccurrence_count = cooccurrence_matrix[i, j]
                
                if cooccurrence_count >= min_cooccurrence:
                    # Normalizar strength
                    strength = float(cooccurrence_count / max_cooccurrence)
                    
                    # Calcular confidence basado en frecuencia
                    confidence = min(1.0, cooccurrence_count / (min_cooccurrence * 3))
                    
                    # Obtener contexto y citaciones
                    contexts_data = relation_contexts.get((concept1, concept2), [])
                    context = contexts_data[0]['context'] if contexts_data else f"Co-ocurren {int(cooccurrence_count)} veces"
                    
                    # Crear citaciones
                    citations = []
                    for ctx_data in contexts_data[:3]:  # Máximo 3 citaciones
                        chunk = ctx_data['chunk']
                        citation = self.citation_manager.add_citation(
                            source_file=chunk.get('metadata', {}).get('source_file', 'unknown'),
                            chunk_id=chunk.get('metadata', {}).get('chunk_id', 0),
                            content=ctx_data['context'],
                            relevance_score=strength
                        )
                        citations.append({
                            'citation_id': citation.citation_id,
                            'source_file': chunk.get('metadata', {}).get('source_file', 'unknown'),
                            'content': ctx_data['context'][:200]
                        })
                    
                    relation = Relation(
                        source_concept=concept1,
                        target_concept=concept2,
                        relation_type='co-occurrence',
                        strength=strength,
                        confidence=confidence,
                        context=context,
                        citations=citations,
                        metadata={
                            'cooccurrence_count': int(cooccurrence_count),
                            'window_size': window_size
                        }
                    )
                    relations.append(relation)
        
        return RelationAnalysisResult(
            relations=relations,
            concepts=concepts,
            relation_matrix=cooccurrence_matrix,
            total_relations=len(relations),
            analysis_type='co-occurrence',
            metadata={
                'window_size': window_size,
                'min_cooccurrence': min_cooccurrence,
                'total_concepts': len(concepts),
                'chunks_analyzed': len(chunks)
            }
        )
    
    def analyze_relations_semantic(
        self,
        chunks: List[Dict[str, Any]],
        similarity_threshold: float = 0.3
    ) -> RelationAnalysisResult:
        """
        Analizar relaciones semánticas usando TF-IDF y similitud coseno
        
        Args:
            chunks: Lista de chunks de texto
            similarity_threshold: Umbral mínimo de similitud (0.0 - 1.0)
            
        Returns:
            RelationAnalysisResult con relaciones semánticas
        """
        if not chunks:
            return RelationAnalysisResult(
                relations=[],
                concepts=[],
                relation_matrix=np.array([[]]),
                total_relations=0,
                analysis_type='semantic',
                metadata={'error': 'No chunks provided'}
            )
        
        # Extraer conceptos
        concepts = self._extract_key_concepts(chunks)
        
        if len(concepts) < 2:
            return RelationAnalysisResult(
                relations=[],
                concepts=concepts,
                relation_matrix=np.zeros((len(concepts), len(concepts))),
                total_relations=0,
                analysis_type='semantic',
                metadata={'warning': 'Insufficient concepts for semantic analysis'}
            )
        
        # Preparar textos para cada concepto (contextos donde aparece)
        concept_contexts = {concept: [] for concept in concepts}
        
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            for concept in concepts:
                if concept.lower() in content:
                    concept_contexts[concept].append(content)
        
        # Crear vectorizador TF-IDF
        concept_texts = [' '.join(contexts) for contexts in concept_contexts.values()]
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words=list(self.stopwords),
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(concept_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
        except Exception as e:
            return RelationAnalysisResult(
                relations=[],
                concepts=concepts,
                relation_matrix=np.zeros((len(concepts), len(concepts))),
                total_relations=0,
                analysis_type='semantic',
                metadata={'error': f'TF-IDF error: {str(e)}'}
            )
        
        # Crear relaciones a partir de similitudes significativas
        relations = []
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:
                    continue
                
                similarity = similarity_matrix[i, j]
                
                if similarity >= similarity_threshold:
                    # Encontrar contexto compartido
                    context = self._find_semantic_context(
                        concept1, concept2, chunks
                    )
                    
                    # Crear citaciones
                    citations = self._create_semantic_citations(
                        concept1, concept2, chunks, similarity
                    )
                    
                    relation = Relation(
                        source_concept=concept1,
                        target_concept=concept2,
                        relation_type='semantic',
                        strength=float(similarity),
                        confidence=float(similarity),
                        context=context,
                        citations=citations,
                        metadata={
                            'similarity_score': float(similarity),
                            'similarity_threshold': similarity_threshold
                        }
                    )
                    relations.append(relation)
        
        return RelationAnalysisResult(
            relations=relations,
            concepts=concepts,
            relation_matrix=similarity_matrix,
            total_relations=len(relations),
            analysis_type='semantic',
            metadata={
                'similarity_threshold': similarity_threshold,
                'total_concepts': len(concepts),
                'chunks_analyzed': len(chunks)
            }
        )
    
    def analyze_relations_causal(
        self,
        chunks: List[Dict[str, Any]]
    ) -> RelationAnalysisResult:
        """
        Analizar relaciones causales entre conceptos
        
        Args:
            chunks: Lista de chunks de texto
            
        Returns:
            RelationAnalysisResult con relaciones causales identificadas
        """
        if not chunks:
            return RelationAnalysisResult(
                relations=[],
                concepts=[],
                relation_matrix=np.array([[]]),
                total_relations=0,
                analysis_type='causal',
                metadata={'error': 'No chunks provided'}
            )
        
        # Extraer conceptos
        concepts = self._extract_key_concepts(chunks)
        
        if len(concepts) < 2:
            return RelationAnalysisResult(
                relations=[],
                concepts=concepts,
                relation_matrix=np.zeros((len(concepts), len(concepts))),
                total_relations=0,
                analysis_type='causal',
                metadata={'warning': 'Insufficient concepts for causal analysis'}
            )
        
        # Buscar relaciones causales en el texto
        relations = []
        causal_matrix = np.zeros((len(concepts), len(concepts)))
        
        for chunk in chunks:
            content = chunk.get('content', '')
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Verificar si hay patrón causal
                has_causal = any(
                    re.search(pattern, sentence_lower)
                    for pattern in self.causal_patterns
                )
                
                if not has_causal:
                    continue
                
                # Buscar conceptos en la oración
                concepts_in_sentence = [
                    c for c in concepts 
                    if c.lower() in sentence_lower
                ]
                
                # Crear relaciones entre conceptos encontrados
                for i, concept1 in enumerate(concepts_in_sentence):
                    for concept2 in concepts_in_sentence[i+1:]:
                        idx1 = concepts.index(concept1)
                        idx2 = concepts.index(concept2)
                        
                        causal_matrix[idx1, idx2] += 1
                        
                        # Crear citación
                        citation = self.citation_manager.add_citation(
                            source_file=chunk.get('metadata', {}).get('source_file', 'unknown'),
                            chunk_id=chunk.get('metadata', {}).get('chunk_id', 0),
                            content=sentence.strip(),
                            relevance_score=0.8
                        )
                        
                        relation = Relation(
                            source_concept=concept1,
                            target_concept=concept2,
                            relation_type='causal',
                            strength=0.8,
                            confidence=0.7,
                            context=sentence.strip(),
                            citations=[{
                                'citation_id': citation.citation_id,
                                'source_file': chunk.get('metadata', {}).get('source_file', 'unknown'),
                                'content': sentence.strip()[:200]
                            }],
                            metadata={'pattern_type': 'causal'}
                        )
                        relations.append(relation)
        
        return RelationAnalysisResult(
            relations=relations,
            concepts=concepts,
            relation_matrix=causal_matrix,
            total_relations=len(relations),
            analysis_type='causal',
            metadata={
                'total_concepts': len(concepts),
                'chunks_analyzed': len(chunks),
                'patterns_used': len(self.causal_patterns)
            }
        )
    
    def _extract_key_concepts(
        self,
        chunks: List[Dict[str, Any]],
        max_concepts: int = 30
    ) -> List[str]:
        """
        Extraer conceptos clave de los chunks
        
        Args:
            chunks: Lista de chunks
            max_concepts: Número máximo de conceptos
            
        Returns:
            Lista de conceptos clave
        """
        # Combinar todo el texto
        all_text = ' '.join([
            chunk.get('content', '') for chunk in chunks
        ]).lower()
        
        # Extraer palabras
        words = re.findall(r'\b[a-záéíóúñü]+\b', all_text)
        
        # Filtrar stopwords y palabras cortas
        filtered_words = [
            word for word in words
            if word not in self.stopwords and len(word) > 3
        ]
        
        # Contar frecuencias
        word_freq = Counter(filtered_words)
        
        # Obtener top conceptos
        top_concepts = [
            word for word, _ in word_freq.most_common(max_concepts)
        ]
        
        return top_concepts
    
    def _find_cooccurrences(
        self,
        text: str,
        concept1: str,
        concept2: str,
        window_size: int
    ) -> int:
        """
        Encontrar co-ocurrencias de dos conceptos dentro de una ventana
        
        Args:
            text: Texto donde buscar
            concept1: Primer concepto
            concept2: Segundo concepto
            window_size: Tamaño de la ventana
            
        Returns:
            Número de co-ocurrencias
        """
        words = text.split()
        cooccurrences = 0
        
        # Buscar posiciones de ambos conceptos
        positions1 = [i for i, word in enumerate(words) if concept1.lower() in word.lower()]
        positions2 = [i for i, word in enumerate(words) if concept2.lower() in word.lower()]
        
        # Contar co-ocurrencias dentro de la ventana
        for pos1 in positions1:
            for pos2 in positions2:
                if abs(pos1 - pos2) <= window_size:
                    cooccurrences += 1
        
        return cooccurrences
    
    def _extract_context(
        self,
        text: str,
        concept1: str,
        concept2: str,
        context_size: int = 100
    ) -> str:
        """
        Extraer contexto donde aparecen ambos conceptos
        
        Args:
            text: Texto completo
            concept1: Primer concepto
            concept2: Segundo concepto
            context_size: Tamaño del contexto en caracteres
            
        Returns:
            Contexto extraído
        """
        # Buscar primera aparición de ambos conceptos
        pos1 = text.find(concept1.lower())
        pos2 = text.find(concept2.lower())
        
        if pos1 == -1 or pos2 == -1:
            return f"Relación entre {concept1} y {concept2}"
        
        # Tomar el contexto alrededor de ambos conceptos
        start = max(0, min(pos1, pos2) - context_size)
        end = min(len(text), max(pos1, pos2) + context_size)
        
        context = text[start:end].strip()
        
        return f"...{context}..."
    
    def _find_semantic_context(
        self,
        concept1: str,
        concept2: str,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Encontrar contexto semántico compartido entre conceptos
        
        Args:
            concept1: Primer concepto
            concept2: Segundo concepto
            chunks: Lista de chunks
            
        Returns:
            Contexto semántico
        """
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            if concept1.lower() in content and concept2.lower() in content:
                # Extraer oración que contiene ambos
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    if concept1.lower() in sentence and concept2.lower() in sentence:
                        return sentence.strip()[:200]
        
        return f"Relación semántica entre {concept1} y {concept2}"
    
    def _create_semantic_citations(
        self,
        concept1: str,
        concept2: str,
        chunks: List[Dict[str, Any]],
        similarity: float
    ) -> List[Dict[str, Any]]:
        """
        Crear citaciones para relación semántica
        
        Args:
            concept1: Primer concepto
            concept2: Segundo concepto
            chunks: Lista de chunks
            similarity: Similitud semántica
            
        Returns:
            Lista de citaciones
        """
        citations = []
        
        for chunk in chunks[:3]:  # Máximo 3 citaciones
            content = chunk.get('content', '').lower()
            if concept1.lower() in content and concept2.lower() in content:
                citation = self.citation_manager.add_citation(
                    source_file=chunk.get('metadata', {}).get('source_file', 'unknown'),
                    chunk_id=chunk.get('metadata', {}).get('chunk_id', 0),
                    content=content[:300],
                    relevance_score=similarity
                )
                
                citations.append({
                    'citation_id': citation.citation_id,
                    'source_file': chunk.get('metadata', {}).get('source_file', 'unknown'),
                    'content': content[:200]
                })
        
        return citations
    
    def export_relations(
        self,
        result: RelationAnalysisResult,
        format: str = 'json'
    ) -> Dict[str, Any]:
        """
        Exportar resultados del análisis de relaciones
        
        Args:
            result: Resultado del análisis
            format: Formato de exportación ('json', 'csv', 'graphml')
            
        Returns:
            Datos exportados
        """
        export_data = {
            'metadata': {
                'timestamp': result.timestamp.isoformat(),
                'analysis_type': result.analysis_type,
                'total_relations': result.total_relations,
                'total_concepts': len(result.concepts),
                **result.metadata
            },
            'concepts': result.concepts,
            'relations': [
                {
                    'source': rel.source_concept,
                    'target': rel.target_concept,
                    'type': rel.relation_type,
                    'strength': rel.strength,
                    'confidence': rel.confidence,
                    'context': rel.context,
                    'citations': rel.citations,
                    'metadata': rel.metadata
                }
                for rel in result.relations
            ],
            'citations': self.citation_manager.export_citations()
        }
        
        return export_data

