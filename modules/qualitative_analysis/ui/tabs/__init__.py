"""
Tabs: Interfaces de cada sub-módulo de análisis
"""

from .concepts_tab import render_concepts_tab
from .topics_tab import render_topics_tab
from .relations_tab import render_relations_tab

__all__ = ['render_concepts_tab', 'render_topics_tab', 'render_relations_tab']

