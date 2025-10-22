"""
Configuración del Sistema de Análisis Cualitativo
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
    
    # Configuración de extracción de conceptos
    min_concept_frequency: int = 2
    """Frecuencia mínima para considerar un concepto relevante"""
    
    max_concepts: int = 30
    """Número máximo de conceptos a extraer"""
    
    use_ngrams: bool = True
    """Si se deben detectar frases completas (bigrams, trigrams) además de palabras"""
    
    ngram_range: tuple = (1, 3)
    """Rango de n-gramas a considerar (1=palabras, 2=frases de 2 palabras, 3=frases de 3)"""
    
    # Configuración de procesamiento
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        return {
            'min_concept_frequency': self.min_concept_frequency,
            'max_concepts': self.max_concepts,
            'use_ngrams': self.use_ngrams,
            'ngram_range': self.ngram_range,
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
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """Crear configuración desde diccionario"""
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        return f"AnalysisConfig(concepts={self.max_concepts}, ngrams={self.use_ngrams}, citations={self.enable_citations})"

