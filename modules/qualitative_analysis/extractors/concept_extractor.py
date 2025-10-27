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
from utils.ollama_client import ollama_client


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
            
            # Calcular scores máximos para cada término (más diferenciados que el promedio)
            max_scores = tfidf_matrix.max(axis=0).toarray().flatten()
            
            # Crear conceptos
            concepts = []
            term_locations = defaultdict(list)  # Para rastrear dónde aparece cada término
            
            for idx, (term, score) in enumerate(zip(feature_names, max_scores)):
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
                    
                    # Calcular score de relevancia mejorado
                    # Combinar TF-IDF con frecuencia y distribución
                    frequency_score = min(1.0, frequency / 10.0)  # Normalizar frecuencia
                    distribution_score = len(sources) / len(set(chunk['metadata'].get('source_file', 'unknown') for chunk in chunks))
                    combined_score = (score * 0.6 + frequency_score * 0.3 + distribution_score * 0.1)
                    
                    # Crear concepto
                    concept = ExtractedConcept(
                        concept=term,
                        frequency=frequency,
                        relevance_score=combined_score,
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
    
    def _refine_concepts_with_llm(
        self,
        raw_concepts: List[ExtractedConcept],
        document_context: str
    ) -> List[ExtractedConcept]:
        """
        Refinar conceptos usando modelo LLM (DeepSeek R1)
        
        Este método toma los conceptos candidatos extraídos por TF-IDF
        y los envía al LLM para:
        1. Filtrar conceptos irrelevantes
        2. Consolidar conceptos relacionados
        3. Generar nombres más precisos
        4. Añadir explicaciones semánticas
        
        Args:
            raw_concepts: Conceptos candidatos extraídos por TF-IDF
            document_context: Contexto del documento para el LLM
            
        Returns:
            Lista de conceptos refinados y mejorados
        """
        if not self.config.enable_llm_refinement:
            return raw_concepts[:self.config.max_final_concepts]
        
        try:
            # Filtrar conceptos candidatos más prometedores
            # Priorizar conceptos con mayor relevancia y frecuencia
            filtered_concepts = [
                c for c in raw_concepts 
                if len(c.concept) > 3 and  # Más de 3 caracteres
                c.relevance_score > 0.1 and  # Relevancia mínima
                c.frequency > 1 and  # Frecuencia mínima
                not c.concept.lower() in ['texto', 'documento', 'anexo', 'página', 'capítulo']  # Filtrar genéricos
            ]
            
            # Tomar los mejores candidatos (máximo 15 para análisis profundo)
            concept_list = [c.concept for c in filtered_concepts[:15]]
            
            if not concept_list:
                print("⚠️ No se encontraron conceptos candidatos de calidad suficiente")
                return raw_concepts[:self.config.max_final_concepts]
            
            # Crear prompt para el LLM
            prompt = self._create_refinement_prompt(concept_list, document_context)
            
            # Llamar al LLM sin límites de tokens para evitar JSON truncado
            response = ollama_client.generate_response(
                model=self.config.llm_model,
                prompt=prompt
                # Sin max_tokens para permitir respuestas completas
            )
            
            # Parsear respuesta del LLM
            refined_concepts = self._parse_llm_response(response, raw_concepts)
            
            # Validar calidad de conceptos refinados con criterios más flexibles
            quality_concepts = []
            for concept in refined_concepts:
                # Criterios más flexibles y realistas
                concept_words = concept.concept.split()
                is_valid = (
                    len(concept_words) >= 1 and  # Al menos 1 palabra (más flexible)
                    len(concept.concept) > 3 and  # Más de 3 caracteres (más flexible)
                    not concept.concept.lower() in ['texto', 'documento', 'anexo', 'página', 'capítulo', 'sección', 'parte'] and  # Filtrar genéricos
                    not concept.concept.lower().strip() in ['a', 'e', 'i', 'o', 'u', 'y', 'de', 'la', 'el', 'en', 'un', 'es', 'se', 'no', 'si']  # Filtrar palabras muy cortas
                )
                
                if is_valid:
                    quality_concepts.append(concept)
                    print(f"✅ Concepto válido: '{concept.concept}' (palabras: {len(concept_words)}, chars: {len(concept.concept)})")
                else:
                    print(f"❌ Concepto rechazado: '{concept.concept}' (palabras: {len(concept_words)}, chars: {len(concept.concept)})")
            
            if not quality_concepts:
                print("⚠️ Los conceptos refinados no cumplen criterios de calidad")
                print("📊 Conceptos recibidos del LLM:")
                for i, concept in enumerate(refined_concepts, 1):
                    print(f"  {i}. '{concept.concept}' (palabras: {len(concept.concept.split())}, chars: {len(concept.concept)})")
                
                # Fallback más inteligente: usar conceptos originales pero mejorados
                fallback_concepts = []
                for c in raw_concepts:
                    if len(c.concept) > 3 and not c.concept.lower() in ['texto', 'documento', 'anexo', 'página']:
                        fallback_concepts.append(c)
                
                print(f"🔄 Usando {len(fallback_concepts)} conceptos originales como fallback")
                return fallback_concepts[:self.config.max_final_concepts]
            
            print(f"✅ {len(quality_concepts)} conceptos pasaron los criterios de calidad")
            return quality_concepts[:self.config.max_final_concepts]
            
        except Exception as e:
            print(f"Error en refinamiento LLM: {e}")
            # Fallback: devolver conceptos originales limitados
            return raw_concepts[:self.config.max_final_concepts]
    
    def _get_category_options(self) -> str:
        """
        Obtener opciones de categorías para el prompt
        
        Returns:
            String con las opciones de categorías disponibles
        """
        if self.config.use_custom_categories and self.config.custom_categories:
            # Usar categorías personalizadas del usuario
            return "|".join(self.config.custom_categories.keys())
        else:
            # Usar categorías por defecto
            return "metodologia|teoria|resultado|proceso|herramienta|fenomeno_social"
    
    def _create_refinement_prompt(self, concept_list: List[str], document_context: str) -> str:
        """
        Crear prompt optimizado para DeepSeek R1 que genere conceptos académicos profundos
        
        Args:
            concept_list: Lista de conceptos candidatos
            document_context: Contexto del documento
            
        Returns:
            Prompt formateado para el LLM
        """
        context_preview = document_context[:2000] + "..." if len(document_context) > 2000 else document_context
        
        # Usar prompt especializado según si hay categorías personalizadas
        if self.config.use_custom_categories and self.config.custom_categories:
            return self._create_custom_categories_prompt(concept_list, context_preview)
        else:
            return self._create_general_analysis_prompt(concept_list, context_preview)
    
    def _create_custom_categories_prompt(self, concept_list: List[str], context_preview: str) -> str:
        """
        Crear prompt especializado para análisis con categorías personalizadas del usuario
        
        Args:
            concept_list: Lista de conceptos candidatos
            context_preview: Contexto del documento
            
        Returns:
            Prompt especializado para categorías personalizadas
        """
        categories_section = ""
        for category, definition in self.config.custom_categories.items():
            categories_section += f"- **{category}**: {definition}\n"
        
        return f"""Eres un experto en análisis cualitativo de documentos académicos. Tu tarea es identificar conceptos clave que se ajusten EXACTAMENTE a las categorías específicas definidas por el usuario.

IMPORTANTE: Responde ÚNICAMENTE en español. Todos los conceptos, explicaciones y categorías deben estar en español.

CONTEXTO DEL DOCUMENTO:
{context_preview}

TÉRMINOS CANDIDATOS EXTRAÍDOS:
{', '.join(concept_list)}

CATEGORÍAS ESPECÍFICAS REQUERIDAS POR EL USUARIO:
{categories_section}

INSTRUCCIONES CRÍTICAS:
1. SOLO genera conceptos que encajen PERFECTAMENTE en las categorías definidas arriba
2. Si un concepto no encaja en ninguna categoría, NO lo incluyas
3. Cada concepto DEBE ser clasificado en una de las categorías del usuario
4. Prioriza conceptos que sean específicos y relevantes para cada categoría
5. Genera conceptos que sean útiles para la investigación del usuario
6. Cada concepto debe tener una explicación ÚNICA y específica
7. NO uses plantillas genéricas - cada explicación debe ser única

FORMATO DE RESPUESTA (JSON estricto):
{{
    "conceptos_refinados": [
        {{
            "concepto": "nombre_del_concepto_específico_en_español",
            "relevancia": "alta|media|baja",
            "categoria": "{'|'.join(self.config.custom_categories.keys())}",
            "explicacion": "Explicación detallada y ÚNICA de por qué este concepto específico es importante para esta categoría particular y qué fenómeno captura.",
            "conceptos_relacionados": ["concepto_relacionado1", "concepto_relacionado2"]
        }}
    ]
}}

CRITERIOS DE CALIDAD PARA CATEGORÍAS PERSONALIZADAS:
- Los conceptos deben ser específicos a las categorías definidas
- Deben ser útiles para la investigación del usuario
- Deben tener explicaciones únicas y específicas
- NO deben ser conceptos genéricos o irrelevantes
- Cada concepto debe aportar valor específico a su categoría
- Los conceptos deben tener AL MENOS 2 palabras y más de 5 caracteres
- NO uses palabras sueltas como "a", "e", "i", "o", "u", "de", "la", "el"
- NO uses términos genéricos como "texto", "documento", "página"

EJEMPLOS DE CONCEPTOS VÁLIDOS:
✅ "Análisis cualitativo de datos" (específico, útil)
✅ "Metodología de investigación" (claro, académico)
✅ "Teoría del aprendizaje" (específico, teórico)

EJEMPLOS DE CONCEPTOS INVÁLIDOS:
❌ "texto" (genérico)
❌ "análisis" (muy general)
❌ "a" (palabra suelta)
❌ "documento" (genérico)

REGLAS CRÍTICAS:
- SOLO incluye conceptos que encajen en las categorías del usuario
- Cada explicación debe ser completamente diferente
- Las explicaciones deben ser específicas al concepto y su categoría
- NO uses frases genéricas o plantillas repetidas
- Si no encuentras conceptos que encajen, devuelve menos conceptos pero de mayor calidad

IMPORTANTE:
- Responde SOLO con JSON válido
- Máximo {self.config.max_final_concepts} conceptos
- Cada concepto debe ser clasificado en una de las categorías del usuario
- TODO debe estar en español
- Si no encuentras conceptos relevantes para las categorías, es mejor devolver menos conceptos pero de mayor calidad"""

    def _create_general_analysis_prompt(self, concept_list: List[str], context_preview: str) -> str:
        """
        Crear prompt para análisis general sin categorías específicas
        
        Args:
            concept_list: Lista de conceptos candidatos
            context_preview: Contexto del documento
            
        Returns:
            Prompt para análisis general
        """
        return f"""Eres un experto en análisis cualitativo de documentos académicos. Tu tarea es identificar y desarrollar CONCEPTOS CLAVE profundos y significativos que emergen del documento.

IMPORTANTE: Responde ÚNICAMENTE en español. Todos los conceptos, explicaciones y categorías deben estar en español.

CONTEXTO DEL DOCUMENTO:
{context_preview}

TÉRMINOS CANDIDATOS EXTRAÍDOS:
{', '.join(concept_list)}

DEFINICIÓN DE CONCEPTO CLAVE:
Un concepto clave en investigación cualitativa es una idea o categoría fundamental que emerge de los datos para explicar y dar sentido al fenómeno estudiado. Los conceptos clave permiten:
- Identificar patrones o temas recurrentes
- Capturar motivaciones, experiencias, creencias y emociones
- Ofrecer comprensión profunda y matizada de fenómenos complejos
- Desarrollar teorías o marcos conceptuales

EJEMPLOS DE CONCEPTOS CLAVE ACADÉMICOS EN ESPAÑOL:
- "Resiliencia comunitaria" (no "comunidad")
- "Identidad profesional en transición" (no "trabajo")
- "Alfabetización digital generacional" (no "tecnología")
- "Redes de apoyo vecinal" (no "vecinos")
- "Memoria colectiva del trauma" (no "recuerdos")

INSTRUCCIONES ESPECÍFICAS:
1. ANALIZA el contexto completo del documento para identificar fenómenos complejos
2. DESARROLLA conceptos que capturen la esencia de procesos, relaciones o fenómenos
3. EVITA palabras simples o genéricas - busca conceptos que expliquen "cómo" y "por qué"
4. COMBINA términos relacionados en conceptos más ricos y significativos
5. PRIORIZA conceptos que revelen patrones, procesos o relaciones sociales
6. ELIMINA completamente letras sueltas, palabras genéricas y términos irrelevantes
7. CADA CONCEPTO DEBE TENER UNA EXPLICACIÓN ÚNICA Y ESPECÍFICA
8. NO REPITAS explicaciones entre conceptos diferentes

FORMATO DE RESPUESTA (JSON estricto):
{{
    "conceptos_refinados": [
        {{
            "concepto": "nombre_del_concepto_académico_profundo_en_español",
            "relevancia": "alta|media|baja",
            "categoria": "metodologia|teoria|resultado|proceso|herramienta|fenomeno_social",
            "explicacion": "Explicación detallada y ÚNICA de por qué este concepto específico es importante y qué fenómeno particular captura. Debe ser diferente a cualquier otra explicación.",
            "conceptos_relacionados": ["concepto_relacionado1", "concepto_relacionado2"]
        }}
    ]
}}

CRITERIOS DE CALIDAD:
- Los conceptos deben ser sustantivos y descriptivos (2-4 palabras)
- Deben capturar procesos, relaciones o fenómenos complejos
- Deben ser útiles para explicar el fenómeno estudiado
- Deben ser específicos al contexto del documento
- NO deben ser palabras simples o genéricas
- CADA explicación debe ser ÚNICA y específica al concepto
- Los conceptos deben tener AL MENOS 2 palabras y más de 5 caracteres
- NO uses palabras sueltas como "a", "e", "i", "o", "u", "de", "la", "el"
- NO uses términos genéricos como "texto", "documento", "página"

EJEMPLOS DE CONCEPTOS VÁLIDOS:
✅ "Resiliencia comunitaria" (específico, académico)
✅ "Identidad profesional en transición" (complejo, significativo)
✅ "Alfabetización digital generacional" (específico, moderno)

EJEMPLOS DE CONCEPTOS INVÁLIDOS:
❌ "comunidad" (muy general)
❌ "trabajo" (genérico)
❌ "texto" (irrelevante)
❌ "a" (palabra suelta)

REGLAS CRÍTICAS:
- Cada concepto debe tener una explicación completamente diferente
- Las explicaciones deben ser específicas al concepto, no genéricas
- Si dos conceptos son similares, sus explicaciones deben destacar las diferencias
- NO uses plantillas o frases genéricas repetidas

IMPORTANTE:
- Responde SOLO con JSON válido
- Máximo {self.config.max_final_concepts} conceptos
- Cada concepto debe ser una idea compleja y significativa
- Si no encuentras conceptos profundos, es mejor devolver menos conceptos pero de mayor calidad
- TODO debe estar en español"""
    
    def _parse_llm_response(self, response: str, original_concepts: List[ExtractedConcept]) -> List[ExtractedConcept]:
        """
        Parsear respuesta del LLM y crear conceptos refinados
        
        Args:
            response: Respuesta del LLM
            original_concepts: Conceptos originales para preservar metadatos
            
        Returns:
            Lista de conceptos refinados
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
            
            refined_concepts = []
            
            for item in data.get('conceptos_refinados', []):
                # Buscar concepto original más similar para preservar metadatos
                original_concept = self._find_most_similar_original(
                    item['concepto'], original_concepts
                )
                
                # Calcular score de relevancia mejorado para conceptos refinados
                base_score = original_concept.relevance_score if original_concept else 0.5
                
                # Ajustar score basado en relevancia del LLM
                relevancia_llm = item.get('relevancia', 'media')
                if relevancia_llm == 'alta':
                    llm_multiplier = 1.2
                elif relevancia_llm == 'media':
                    llm_multiplier = 1.0
                else:  # baja
                    llm_multiplier = 0.8
                
                # Ajustar por longitud del concepto (conceptos más largos suelen ser más específicos)
                length_bonus = min(0.1, len(item['concepto'].split()) * 0.02)
                
                final_score = min(1.0, (base_score * llm_multiplier) + length_bonus)
                
                # Crear concepto refinado
                refined_concept = ExtractedConcept(
                    concept=item['concepto'],
                    frequency=original_concept.frequency if original_concept else 1,
                    relevance_score=final_score,
                    sources=original_concept.sources if original_concept else [],
                    citations=original_concept.citations if original_concept else [],
                    context_examples=original_concept.context_examples if original_concept else [],
                    related_concepts=item.get('conceptos_relacionados', []),
                    category=item.get('categoria'),
                    extraction_method='llm_refined',
                    timestamp=datetime.now().isoformat()
                )
                
                # Añadir explicación si está disponible
                if self.config.include_concept_explanations and 'explicacion' in item:
                    explanation_text = f"Explicación: {item['explicacion']}"
                    refined_concept.context_examples.append(explanation_text)
                    print(f"✅ Explicación agregada para '{item['concepto']}': {item['explicacion'][:100]}...")
                else:
                    print(f"⚠️ No se agregó explicación para '{item['concepto']}' - include_explanations: {self.config.include_concept_explanations}, tiene_explicacion: {'explicacion' in item}")
                
                print(f"📊 Concepto refinado creado: '{refined_concept.concept}' - Método: {refined_concept.extraction_method}, Categoría: {refined_concept.category}")
                refined_concepts.append(refined_concept)
            
            return refined_concepts
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parseando respuesta LLM: {e}")
            print(f"Respuesta original: {response[:500]}...")
            # Fallback: devolver conceptos originales
            return original_concepts[:self.config.max_final_concepts]
    
    def _clean_json_response(self, response: str) -> str:
        """
        Limpiar caracteres de control inválidos de la respuesta del LLM
        
        Args:
            response: Respuesta original del LLM
            
        Returns:
            Respuesta limpia sin caracteres de control inválidos
        """
        import re
        
        # Remover caracteres de control excepto \n, \r, \t
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', response)
        
        # Normalizar saltos de línea
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remover espacios múltiples
        cleaned = re.sub(r' +', ' ', cleaned)
        
        return cleaned
    
    def _aggressive_json_clean(self, json_str: str) -> str:
        """
        Limpieza agresiva de JSON para manejar casos problemáticos
        
        Args:
            json_str: String JSON a limpiar
            
        Returns:
            JSON limpio y válido
        """
        import re
        
        # Paso 1: Escapar caracteres problemáticos en strings
        def escape_string(match):
            string_content = match.group(1)
            # Escapar caracteres problemáticos
            string_content = string_content.replace('\\', '\\\\')
            string_content = string_content.replace('"', '\\"')
            string_content = string_content.replace('\n', '\\n')
            string_content = string_content.replace('\t', '\\t')
            string_content = string_content.replace('\r', '\\r')
            return f'"{string_content}"'
        
        # Aplicar escape a strings
        json_str = re.sub(r'"([^"]*)"', escape_string, json_str)
        
        # Paso 2: Corregir comas faltantes antes de llaves de cierre
        # Buscar patrones como: "valor" } o "valor" ] y agregar coma
        json_str = re.sub(r'"\s*}\s*', '",\n}', json_str)
        json_str = re.sub(r'"\s*]\s*', '",\n]', json_str)
        
        # Paso 3: Corregir comas faltantes después de valores
        # Buscar patrones como: "valor" "siguiente" y agregar coma
        json_str = re.sub(r'"\s*"', '",\n"', json_str)
        
        # Paso 4: Limpiar comas múltiples
        json_str = re.sub(r',\s*,', ',', json_str)
        
        # Paso 5: Corregir comas antes de llaves de cierre (casos específicos)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Paso 6: Remover caracteres de control restantes
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', json_str)
        
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
        
        # Reparaciones básicas comunes
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
        
        # 5. Asegurar que el JSON esté bien formado
        if not json_str.strip().startswith('{'):
            json_str = '{' + json_str
        if not json_str.strip().endswith('}'):
            json_str = json_str + '}'
        
        # 6. Reparación específica para el caso del usuario
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
        
        print(f"📄 JSON reparado manualmente (últimos 200 chars): ...{json_str[-200:]}")
        return json_str
    
    def _find_most_similar_original(self, refined_name: str, original_concepts: List[ExtractedConcept]) -> Optional[ExtractedConcept]:
        """
        Encontrar el concepto original más similar al refinado
        
        Args:
            refined_name: Nombre del concepto refinado
            original_concepts: Lista de conceptos originales
            
        Returns:
            Concepto original más similar o None
        """
        if not original_concepts:
            return None
        
        # Búsqueda simple por similitud de texto
        refined_lower = refined_name.lower()
        
        for original in original_concepts:
            if refined_lower in original.concept.lower() or original.concept.lower() in refined_lower:
                return original
        
        # Si no hay coincidencia, devolver el primero
        return original_concepts[0]
    
    def extract_concepts(
        self,
        chunks: List[Dict[str, Any]],
        method: str = "tfidf"
    ) -> List[ExtractedConcept]:
        """
        Extraer conceptos clave de los documentos usando enfoque híbrido
        
        Este es el método principal que un investigador utilizaría.
        
        Proceso híbrido:
        1. Preprocesa el texto (limpieza, normalización)
        2. Aplica algoritmo de extracción inicial (TF-IDF por defecto)
        3. Identifica fuentes y crea citas
        4. [NUEVO] Refina conceptos con modelo LLM (DeepSeek R1)
        5. Encuentra conceptos relacionados
        6. Ordena por relevancia
        
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
        
        # FASE 1: Extracción inicial con TF-IDF
        print("🔍 Extrayendo conceptos candidatos con TF-IDF...")
        if method == "tfidf":
            raw_concepts, processed_texts = self._extract_with_tfidf(chunks)
        else:
            raise ValueError(f"Método desconocido: {method}")
        
        # Crear contexto del documento para el LLM
        document_context = " ".join(processed_texts)
        
        # FASE 2: Refinamiento con LLM (si está habilitado)
        if self.config.enable_llm_refinement:
            print(f"🤖 Refinando conceptos con {self.config.llm_model}...")
            refined_concepts = self._refine_concepts_with_llm(raw_concepts, document_context)
            
            # Identificar conceptos relacionados en los refinados
            term_locations = defaultdict(list)
            for i, concept in enumerate(refined_concepts):
                term_locations[concept.concept] = [i]
            
            self._identify_related_concepts(refined_concepts, term_locations)
            
            print(f"✅ Análisis completado: {len(refined_concepts)} conceptos refinados")
            return refined_concepts
        else:
            # Solo TF-IDF, sin refinamiento LLM
            print(f"✅ Análisis completado: {len(raw_concepts)} conceptos extraídos")
            return raw_concepts
    
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

