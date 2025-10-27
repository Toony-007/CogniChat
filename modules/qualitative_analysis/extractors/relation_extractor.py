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
from utils.ollama_client import ollama_client


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
    
    def analyze_relations_hybrid(self, chunks: List[Dict[str, Any]]) -> RelationAnalysisResult:
        """
        Análisis híbrido de relaciones: algoritmo + IA
        
        Este método combina múltiples algoritmos para identificar relaciones
        y luego usa IA para refinarlas y generar explicaciones profundas.
        
        Args:
            chunks: Lista de chunks de documentos
            
        Returns:
            Resultado del análisis híbrido de relaciones
        """
        print("🔗 Iniciando análisis híbrido de relaciones...")
        
        # FASE 1: Extracción inicial con algoritmos
        print(f"🔍 Fase 1: Extracción inicial con {self.config.relation_algorithm.upper()}...")
        
        if self.config.relation_algorithm == "cooccurrence":
            raw_result = self.analyze_relations_cooccurrence(
                chunks, 
                window_size=self.config.relation_window_size,
                min_cooccurrence=self.config.min_cooccurrence
            )
        elif self.config.relation_algorithm == "semantic":
            raw_result = self.analyze_relations_semantic(
                chunks,
                similarity_threshold=self.config.min_relation_strength
            )
        elif self.config.relation_algorithm == "causal":
            raw_result = self.analyze_relations_causal(chunks)
        elif self.config.relation_algorithm == "hybrid":
            # Combinar múltiples algoritmos
            result_cooc = self.analyze_relations_cooccurrence(
                chunks, 
                window_size=self.config.relation_window_size,
                min_cooccurrence=self.config.min_cooccurrence
            )
            result_sem = self.analyze_relations_semantic(
                chunks,
                similarity_threshold=self.config.min_relation_strength
            )
            result_caus = self.analyze_relations_causal(chunks)
            
            # Combinar resultados
            all_relations = result_cooc.relations + result_sem.relations + result_caus.relations
            raw_result = RelationAnalysisResult(
                relations=all_relations,
                concepts=result_cooc.concepts,
                relation_matrix=result_cooc.relation_matrix,
                total_relations=len(all_relations),
                analysis_type='hybrid',
                metadata={
                    'cooccurrence_count': len(result_cooc.relations),
                    'semantic_count': len(result_sem.relations),
                    'causal_count': len(result_caus.relations)
                }
            )
        else:
            raise ValueError(f"Algoritmo desconocido: {self.config.relation_algorithm}")
        
        # FASE 2: Refinamiento con IA (si está habilitado)
        if self.config.enable_relation_refinement:
            print(f"🤖 Fase 2: Refinamiento con {self.config.llm_model}...")
            refined_result = self._refine_relations_with_llm(raw_result, chunks)
        else:
            print("⚠️ Refinamiento con IA deshabilitado, usando resultados algorítmicos")
            refined_result = raw_result
        
        print(f"✅ Análisis híbrido completado: {len(refined_result.relations)} relaciones identificadas")
        return refined_result
    
    def _refine_relations_with_llm(
        self, 
        raw_result: RelationAnalysisResult, 
        chunks: List[Dict[str, Any]]
    ) -> RelationAnalysisResult:
        """
        Refinar relaciones usando modelo LLM
        
        Args:
            raw_result: Resultado del análisis algorítmico
            chunks: Chunks originales para contexto
            
        Returns:
            Resultado refinado con IA
        """
        try:
            # Preparar datos para el LLM
            relations_data = []
            for relation in raw_result.relations[:20]:  # Máximo 20 relaciones para análisis profundo
                relations_data.append({
                    'source': relation.source_concept,
                    'target': relation.target_concept,
                    'type': relation.relation_type,
                    'strength': relation.strength,
                    'context': relation.context
                })
            
            if not relations_data:
                print("⚠️ No hay relaciones para refinar")
                return raw_result
            
            # Crear contexto del documento
            document_context = " ".join([chunk.get('content', '') for chunk in chunks[:5]])
            context_preview = document_context[:2000] + "..." if len(document_context) > 2000 else document_context
            
            # Crear prompt para el LLM
            prompt = self._create_relation_refinement_prompt(relations_data, context_preview)
            
            # Llamar al LLM
            response = ollama_client.generate_response(
                model=self.config.llm_model,
                prompt=prompt
            )
            
            # Parsear respuesta del LLM
            refined_relations = self._parse_llm_relation_response(response, raw_result.relations)
            
            # Crear resultado refinado
            refined_result = RelationAnalysisResult(
                relations=refined_relations,
                concepts=raw_result.concepts,
                relation_matrix=raw_result.relation_matrix,
                total_relations=len(refined_relations),
                analysis_type=f"{raw_result.analysis_type}_refined",
                metadata={
                    **raw_result.metadata,
                    'llm_refinement': True,
                    'original_relations': len(raw_result.relations),
                    'refined_relations': len(refined_relations)
                }
            )
            
            return refined_result
            
        except Exception as e:
            print(f"Error en refinamiento LLM: {e}")
            return raw_result
    
    def _create_relation_refinement_prompt(self, relations_data: List[Dict], context_preview: str) -> str:
        """
        Crear prompt para refinamiento de relaciones con LLM
        
        Args:
            relations_data: Datos de relaciones a refinar
            context_preview: Contexto del documento
            
        Returns:
            Prompt formateado para el LLM
        """
        relations_text = ""
        for i, rel in enumerate(relations_data, 1):
            relations_text += f"{i}. {rel['source']} ↔ {rel['target']} (Tipo: {rel['type']}, Fuerza: {rel['strength']:.3f})\n"
        
        return f"""Eres un experto en análisis cualitativo de relaciones semánticas. Tu tarea es refinar y mejorar las relaciones identificadas entre conceptos, generando explicaciones profundas y académicas.

IMPORTANTE: Responde ÚNICAMENTE en español. Todas las explicaciones y análisis deben estar en español.

CONTEXTO DEL DOCUMENTO:
{context_preview}

RELACIONES IDENTIFICADAS:
{relations_text}

DEFINICIÓN DE RELACIÓN SEMÁNTICA:
Una relación semántica en investigación cualitativa es una conexión significativa entre conceptos que revela:
- Patrones de co-ocurrencia y proximidad contextual
- Conexiones causales, jerárquicas o asociativas
- Relaciones de dependencia, influencia o correlación
- Estructuras conceptuales emergentes del contenido

INSTRUCCIONES ESPECÍFICAS:
1. ANALIZA cada relación identificada en su contexto específico
2. REFINA las relaciones eliminando conexiones superficiales o irrelevantes
3. GENERA explicaciones profundas que expliquen "por qué" y "cómo" se relacionan los conceptos
4. IDENTIFICA patrones complejos que van más allá de la simple co-ocurrencia
5. PRIORIZA relaciones que revelen estructuras conceptuales significativas
6. CADA explicación debe ser ÚNICA y específica a la relación particular
7. NO uses plantillas genéricas - cada explicación debe ser diferente

FORMATO DE RESPUESTA (JSON estricto):
{{
    "relaciones_refinadas": [
        {{
            "concepto_origen": "nombre_del_concepto_origen",
            "concepto_destino": "nombre_del_concepto_destino",
            "tipo_relacion": "coocurrencia|semantica|causal|jerarquica|asociativa",
            "fuerza": 0.0-1.0,
            "confianza": 0.0-1.0,
            "explicacion": "Explicación detallada y ÚNICA de por qué estos conceptos están relacionados y qué patrón conceptual revelan.",
            "patron_conceptual": "Descripción del patrón o estructura conceptual que emerge de esta relación",
            "relevancia_investigacion": "Por qué esta relación es importante para la investigación"
        }}
    ]
}}

CRITERIOS DE CALIDAD:
- Las relaciones deben ser significativas y no superficiales
- Las explicaciones deben ser específicas y únicas
- Deben revelar patrones conceptuales profundos
- Deben ser útiles para la comprensión del fenómeno estudiado
- NO deben ser relaciones obvias o triviales
- Cada explicación debe aportar valor analítico específico

EJEMPLOS DE RELACIONES VÁLIDAS:
✅ "Resiliencia comunitaria" ↔ "Redes de apoyo" (patrón de interdependencia social)
✅ "Identidad profesional" ↔ "Transición laboral" (patrón de transformación identitaria)
✅ "Alfabetización digital" ↔ "Brecha generacional" (patrón de desigualdad tecnológica)

EJEMPLOS DE RELACIONES INVÁLIDAS:
❌ "texto" ↔ "documento" (relación superficial)
❌ "análisis" ↔ "investigación" (relación genérica)
❌ "resultado" ↔ "conclusión" (relación obvia)

REGLAS CRÍTICAS:
- Cada explicación debe ser completamente diferente
- Las explicaciones deben ser específicas al contexto del documento
- NO uses frases genéricas o plantillas repetidas
- Si una relación no es significativa, es mejor excluirla
- Prioriza calidad sobre cantidad

IMPORTANTE:
- Responde SOLO con JSON válido
- Máximo {self.config.max_relations} relaciones refinadas
- Cada relación debe tener una explicación única y específica
- TODO debe estar en español
- Si no encuentras relaciones significativas, es mejor devolver menos relaciones pero de mayor calidad"""

    def _parse_llm_relation_response(self, response: str, original_relations: List[Relation]) -> List[Relation]:
        """
        Parsear respuesta del LLM para relaciones refinadas
        
        Args:
            response: Respuesta del LLM
            original_relations: Relaciones originales para preservar metadatos
            
        Returns:
            Lista de relaciones refinadas
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
            
            # Intentar parsear JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ Error JSON inicial: {e}")
                json_str = self._aggressive_json_clean(json_str)
                try:
                    data = json.loads(json_str)
                    print("✅ JSON parseado exitosamente después de limpieza")
                except json.JSONDecodeError as e2:
                    print(f"❌ Error JSON persistente: {e2}")
                    raise ValueError(f"No se pudo parsear JSON: {e2}")
            
            refined_relations = []
            
            for item in data.get('relaciones_refinadas', []):
                # Buscar relación original más similar para preservar metadatos
                original_relation = self._find_most_similar_original_relation(
                    item['concepto_origen'], item['concepto_destino'], original_relations
                )
                
                # Crear relación refinada
                refined_relation = Relation(
                    source_concept=item['concepto_origen'],
                    target_concept=item['concepto_destino'],
                    relation_type=item['tipo_relacion'],
                    strength=float(item['fuerza']),
                    confidence=float(item['confianza']),
                    context=item.get('explicacion', ''),
                    citations=original_relation.citations if original_relation else [],
                    metadata={
                        'pattern_conceptual': item.get('patron_conceptual', ''),
                        'relevancia_investigacion': item.get('relevancia_investigacion', ''),
                        'llm_refined': True,
                        'original_strength': original_relation.strength if original_relation else 0.0
                    }
                )
                
                refined_relations.append(refined_relation)
            
            return refined_relations
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parseando respuesta LLM: {e}")
            return original_relations[:self.config.max_relations]
    
    def _clean_json_response(self, response: str) -> str:
        """Limpiar caracteres de control inválidos de la respuesta del LLM"""
        import re
        
        # Remover caracteres de control excepto \n, \r, \t
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', response)
        
        # Normalizar saltos de línea
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remover espacios múltiples
        cleaned = re.sub(r' +', ' ', cleaned)
        
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
        
        # Escapar caracteres problemáticos en strings
        def escape_string(match):
            string_content = match.group(1)
            string_content = string_content.replace('\\', '\\\\')
            string_content = string_content.replace('"', '\\"')
            string_content = string_content.replace('\n', '\\n')
            string_content = string_content.replace('\t', '\\t')
            string_content = string_content.replace('\r', '\\r')
            return f'"{string_content}"'
        
        json_str = re.sub(r'"([^"]*)"', escape_string, json_str)
        
        # Corregir comas faltantes
        json_str = re.sub(r'"\s*}', '",\n}', json_str)
        json_str = re.sub(r'"\s*]', '",\n]', json_str)
        json_str = re.sub(r'}\s*{', '},\n{', json_str)
        
        # Limpiar comas duplicadas
        json_str = re.sub(r',\s*,', ',', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _find_most_similar_original_relation(
        self, 
        source: str, 
        target: str, 
        original_relations: List[Relation]
    ) -> Optional[Relation]:
        """Encontrar la relación original más similar"""
        if not original_relations:
            return None
        
        source_lower = source.lower()
        target_lower = target.lower()
        
        for original in original_relations:
            if (source_lower in original.source_concept.lower() or 
                original.source_concept.lower() in source_lower) and \
               (target_lower in original.target_concept.lower() or 
                original.target_concept.lower() in target_lower):
                return original
        
        return original_relations[0] if original_relations else None
    
    def get_relation_summary(self, result: RelationAnalysisResult) -> Dict[str, Any]:
        """
        Generar resumen estadístico de las relaciones analizadas
        
        Args:
            result: Resultado del análisis de relaciones
            
        Returns:
            Diccionario con estadísticas
        """
        if not result.relations:
            return {
                'total_relations': 0,
                'total_concepts': 0,
                'avg_strength': 0.0,
                'avg_confidence': 0.0,
                'unique_sources': 0,
                'total_citations': 0
            }
        
        total_strength = sum(r.strength for r in result.relations)
        total_confidence = sum(r.confidence for r in result.relations)
        avg_strength = total_strength / len(result.relations)
        avg_confidence = total_confidence / len(result.relations)
        
        all_sources = set()
        total_citations = 0
        
        for relation in result.relations:
            for citation in relation.citations:
                if isinstance(citation, dict):
                    all_sources.add(citation.get('source_file', 'unknown'))
                else:
                    all_sources.add(citation.source_file)
            total_citations += len(relation.citations)
        
        return {
            'total_relations': len(result.relations),
            'total_concepts': len(result.concepts),
            'avg_strength': avg_strength,
            'avg_confidence': avg_confidence,
            'unique_sources': len(all_sources),
            'total_citations': total_citations,
            'top_relation': result.relations[0] if result.relations else None,
            'analysis_type': result.analysis_type,
            'metadata': result.metadata
        }
    
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

