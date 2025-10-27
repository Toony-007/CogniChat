"""
Módulo de Análisis Cualitativo Avanzado para Investigación
Sistema diseñado para asistir a investigadores en el análisis profundo de contenido

Principios Fundamentales:
1. Asistencia al Investigador: Guiar en cada paso del análisis
2. Procesamiento Inteligente: Analizar, sintetizar y contextualizar (no copiar/pegar)
3. Fundamentación: Citar todas las fuentes de información
4. Transparencia: Explicar qué hace cada análisis y cómo interpretar resultados

Estructura:
- core: Configuración y componentes base
- extractors: Extracción inteligente de información
- analyzers: Análisis avanzados (temas, sentimientos, relaciones)
- visualizations: Visualizaciones interactivas
- ui: Interfaz de usuario educativa
"""

# Exportar función principal para mantener compatibilidad con app.py
from .ui.main_render import render

# Exportar clases principales para uso programático
from .core.analyzer import QualitativeAnalyzer
from .core.config import AnalysisConfig

__version__ = "2.0.0"

__all__ = [
    'render',
    'QualitativeAnalyzer', 
    'AnalysisConfig'
]
