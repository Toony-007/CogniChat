"""
Analizador Cualitativo Principal
Orquestador central del sistema de análisis cualitativo
"""

from typing import List, Dict, Any, Optional
from .config import AnalysisConfig
from ..extractors.concept_extractor import ConceptExtractor, ExtractedConcept


class QualitativeAnalyzer:
    """
    Analizador Cualitativo Principal
    
    Clase orquestadora que coordina todos los sub-módulos de análisis cualitativo.
    Por ahora solo incluye extracción de conceptos, pero se expandirá con:
    - Análisis de temas
    - Análisis de sentimientos
    - Triangulación
    - Mapas conceptuales
    - etc.
    
    Ejemplo de uso:
        config = AnalysisConfig(max_concepts=30)
        analyzer = QualitativeAnalyzer(config)
        
        # Extraer conceptos
        concepts = analyzer.extract_concepts(chunks)
        
        # Ver resultados
        for concept in concepts[:10]:
            print(f"{concept.concept}: {concept.relevance_score:.3f}")
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Inicializar analizador
        
        Args:
            config: Configuración del análisis (usa valores por defecto si no se proporciona)
        """
        self.config = config or AnalysisConfig()
        
        # Inicializar sub-módulos
        self.concept_extractor = ConceptExtractor(self.config)
    
    def extract_concepts(
        self,
        chunks: List[Dict[str, Any]],
        method: str = "tfidf"
    ) -> List[ExtractedConcept]:
        """
        Extraer conceptos clave de los documentos
        
        Args:
            chunks: Lista de chunks de documentos
            method: Método de extracción ('tfidf' por defecto)
            
        Returns:
            Lista de conceptos extraídos ordenados por relevancia
        """
        return self.concept_extractor.extract_concepts(chunks, method=method)
    
    def get_concept_summary(self, concepts: List[ExtractedConcept]) -> Dict[str, Any]:
        """
        Obtener resumen estadístico de conceptos
        
        Args:
            concepts: Lista de conceptos extraídos
            
        Returns:
            Diccionario con estadísticas
        """
        return self.concept_extractor.get_concept_summary(concepts)
    
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
        return self.concept_extractor.export_concepts(concepts, include_citations)
    
    def __repr__(self) -> str:
        return f"QualitativeAnalyzer(config={self.config})"

