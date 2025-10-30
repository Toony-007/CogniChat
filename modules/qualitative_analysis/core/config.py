"""
Configuración del Sistema de Análisis Cualitativo

PARA MODIFICAR PARÁMETROS:
==========================
Los parámetros marcados con "MODIFICAR AQUÍ" pueden ser cambiados fácilmente.
Los valores están optimizados para la mayoría de casos de uso.

CONFIGURACIONES RECOMENDADAS:
- Para documentos pequeños (< 10 páginas): max_concepts = 20, max_final_concepts = 10
- Para documentos medianos (10-50 páginas): max_concepts = 30, max_final_concepts = 15  
- Para documentos grandes (> 50 páginas): max_concepts = 50, max_final_concepts = 25

MODELOS LLM DISPONIBLES:
- deepseek-r1:7b: Mejor calidad, más lento
- llama3.2:3b: Balance calidad/velocidad
- qwen2.5:3b: Rápido, buena calidad
- phi3:mini: Muy rápido, calidad básica
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class AnalysisConfig:
    """
    Configuración centralizada para el análisis cualitativo
    
    Esta clase permite personalizar todos los parámetros del análisis
    según las necesidades específicas de cada investigación.
    """
    
    # Configuración de extracción de conceptos (OPTIMIZADA)
    min_concept_frequency: int = 2
    """Frecuencia mínima para considerar un concepto relevante - MODIFICAR AQUÍ"""
    
    max_concepts: int = 30
    """Número máximo de conceptos candidatos a extraer con TF-IDF - MODIFICAR AQUÍ"""
    
    use_ngrams: bool = True
    """Si se deben detectar frases completas (bigrams, trigrams) además de palabras - MODIFICAR AQUÍ"""
    
    ngram_range: tuple = (1, 3)
    """Rango de n-gramas a considerar (1=palabras, 2=frases de 2 palabras, 3=frases de 3) - MODIFICAR AQUÍ"""
    
    # Configuración de categorías iniciales
    use_custom_categories: bool = False
    """Activar categorías iniciales personalizadas del usuario"""
    
    custom_categories: Dict[str, str] = field(default_factory=dict)
    """Categorías personalizadas: {categoria: definicion}"""
    
    # Configuración de refinamiento con LLM (OPTIMIZADA)
    enable_llm_refinement: bool = True
    """Activar refinamiento de conceptos con modelo LLM - MODIFICAR AQUÍ"""
    
    llm_model: str = "deepseek-r1:7b"
    """Modelo LLM a utilizar para refinamiento - MODIFICAR AQUÍ: deepseek-r1:7b, llama3.2:3b, qwen2.5:3b, phi3:mini"""
    
    llm_temperature: float = 0.3
    """Temperatura del LLM (0.1=preciso, 1.0=creativo) - MODIFICAR AQUÍ"""
    
    max_final_concepts: int = 15
    """Número máximo de conceptos finales después del refinamiento LLM - MODIFICAR AQUÍ"""
    
    include_concept_explanations: bool = True
    """Incluir explicaciones generadas por el LLM para cada concepto - MODIFICAR AQUÍ"""
    
    llm_max_tokens: int = 999999
    """Máximo número de tokens para la respuesta del LLM (999999 = sin límite) - MODIFICAR AQUÍ"""
    
    # Configuración de análisis de temas (OPTIMIZADA)
    enable_topic_analysis: bool = True
    """Activar análisis de temas - MODIFICAR AQUÍ"""
    
    topic_algorithm: str = "lda"
    """Algoritmo para análisis de temas: 'lda' o 'clustering' - MODIFICAR AQUÍ"""
    
    max_topics: int = 8
    """Número máximo de temas a identificar - MODIFICAR AQUÍ"""
    
    min_topic_frequency: int = 3
    """Frecuencia mínima para considerar un tema válido - MODIFICAR AQUÍ"""
    
    enable_topic_refinement: bool = True
    """Activar refinamiento de temas con IA - MODIFICAR AQUÍ"""
    
    include_topic_explanations: bool = True
    """Incluir explicaciones detalladas de cada tema - MODIFICAR AQUÍ"""
    

    chunk_size: int = 2000
    """Tamaño de fragmentos de texto para procesamiento"""
    
    chunk_overlap: int = 200
    """Solapamiento entre fragmentos para mantener contexto"""
    
    # Configuración de citación
    enable_citations: bool = True
    """Activar sistema de citación de fuentes"""
    
    citation_context_chars: int = 150
    """Caracteres de contexto a incluir en cada cita"""
    
    # Configuración de asistencia al investigador
    show_explanations: bool = True
    """Mostrar explicaciones educativas en la interfaz"""
    
    show_methodology: bool = True
    """Mostrar detalles sobre la metodología utilizada"""
    
    show_interpretation_guide: bool = True
    """Mostrar guías de interpretación de resultados"""
    
    # Configuración de rendimiento
    enable_cache: bool = True
    """Activar sistema de caché para mejorar rendimiento"""
    
    parallel_processing: bool = True
    """Procesar múltiples análisis en paralelo"""
    
    max_workers: int = 4
    """Número máximo de workers para procesamiento paralelo"""

    # Configuración del asistente de codificación cualitativa
    enable_coding_assistant: bool = True
    """Activar asistente de codificación cualitativa - MODIFICAR AQUÍ"""

    max_code_suggestions: int = 25
    """Número máximo de códigos sugeridos inicialmente - MODIFICAR AQUÍ"""

    coding_use_ngrams: bool = True
    """Detectar n-gramas (1-3) para candidatos de códigos - MODIFICAR AQUÍ"""

    coding_ngram_range: tuple = (1, 3)
    """Rango de n-gramas para candidatos de códigos - MODIFICAR AQUÍ"""

    enable_coding_llm_refinement: bool = True
    """Refinar nombres/definiciones/ejemplos de códigos con LLM - MODIFICAR AQUÍ"""

    coding_llm_model: str = "deepseek-r1:7b"
    """Modelo LLM para refinamiento de códigos - MODIFICAR AQUÍ"""

    coding_llm_temperature: float = 0.3
    """Temperatura LLM para codificación - MODIFICAR AQUÍ"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        return {
            'min_concept_frequency': self.min_concept_frequency,
            'max_concepts': self.max_concepts,
            'use_ngrams': self.use_ngrams,
            'ngram_range': self.ngram_range,
            'use_custom_categories': self.use_custom_categories,
            'custom_categories': self.custom_categories,
            'enable_llm_refinement': self.enable_llm_refinement,
            'llm_model': self.llm_model,
            'llm_temperature': self.llm_temperature,
            'max_final_concepts': self.max_final_concepts,
            'include_concept_explanations': self.include_concept_explanations,
            'llm_max_tokens': self.llm_max_tokens,
            'enable_topic_analysis': self.enable_topic_analysis,
            'topic_algorithm': self.topic_algorithm,
            'max_topics': self.max_topics,
            'min_topic_frequency': self.min_topic_frequency,
            'enable_topic_refinement': self.enable_topic_refinement,
            'include_topic_explanations': self.include_topic_explanations,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'enable_citations': self.enable_citations,
            'citation_context_chars': self.citation_context_chars,
            'show_explanations': self.show_explanations,
            'show_methodology': self.show_methodology,
            'show_interpretation_guide': self.show_interpretation_guide,
            'enable_cache': self.enable_cache,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers
            , 'enable_coding_assistant': self.enable_coding_assistant
            , 'max_code_suggestions': self.max_code_suggestions
            , 'coding_use_ngrams': self.coding_use_ngrams
            , 'coding_ngram_range': self.coding_ngram_range
            , 'enable_coding_llm_refinement': self.enable_coding_llm_refinement
            , 'coding_llm_model': self.coding_llm_model
            , 'coding_llm_temperature': self.coding_llm_temperature
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """Crear configuración desde diccionario"""
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return f"AnalysisConfig(concepts={self.max_concepts}, ngrams={self.use_ngrams}, citations={self.enable_citations})"

