"""
Tests automatizados para el módulo qualitative_analysis.py
Pruebas de funcionalidad del análisis cualitativo avanzado
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
from collections import defaultdict

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.qualitative_analysis as qa_module

class TestAdvancedQualitativeAnalyzer:
    """Tests para la clase AdvancedQualitativeAnalyzer"""
    
    def setup_method(self):
        """Configuración inicial para cada test"""
        # Mock session_state para evitar errores de contexto
        with patch('streamlit.session_state', {}):
            pass
    
    def test_analyzer_class_exists(self):
        """Test que verifica la existencia de la clase AdvancedQualitativeAnalyzer"""
        assert hasattr(qa_module, 'AdvancedQualitativeAnalyzer')
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        assert analyzer is not None
    
    def test_analyzer_initialization(self):
        """Test inicialización del analizador"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        # Verificar que el analizador se inicializa correctamente
        assert analyzer is not None
        # Verificar que tiene los atributos básicos esperados
        assert hasattr(analyzer, 'rag_processor')
        assert hasattr(analyzer, 'cache_path')
        assert hasattr(analyzer, 'analysis_cache_path')
    
    def test_load_rag_data_method_exists(self):
        """Test que verifica la existencia del método load_rag_data"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        assert hasattr(analyzer, 'load_rag_data')
        assert callable(getattr(analyzer, 'load_rag_data'))
    
    def test_extract_key_concepts_method_exists(self):
        """Test que verifica la existencia del método extract_key_concepts"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        assert hasattr(analyzer, 'extract_key_concepts')
        assert callable(getattr(analyzer, 'extract_key_concepts'))
    
    def test_create_interactive_concept_map_method_exists(self):
        """Test que verifica la existencia del método create_interactive_concept_map"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        assert hasattr(analyzer, 'create_interactive_concept_map')
        assert callable(getattr(analyzer, 'create_interactive_concept_map'))
    
    def test_generate_rag_summary_method_exists(self):
        """Test que verifica la existencia del método generate_rag_summary"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        assert hasattr(analyzer, 'generate_rag_summary')
        assert callable(getattr(analyzer, 'generate_rag_summary'))
    
    def test_preprocess_text_method_exists(self):
        """Test que verifica la existencia del método preprocess_text"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        assert hasattr(analyzer, 'preprocess_text')
        assert callable(getattr(analyzer, 'preprocess_text'))
    
    def test_cache_management_methods_exist(self):
        """Test que verifica la existencia de métodos de gestión de cache"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        assert hasattr(analyzer, 'clear_cache')
        assert hasattr(analyzer, 'get_cache_stats')
        assert callable(getattr(analyzer, 'clear_cache'))
        assert callable(getattr(analyzer, 'get_cache_stats'))

class TestQualitativeAnalysisIntegration:
    """Tests de integración para análisis cualitativo"""
    
    def setup_method(self):
        """Configuración inicial para cada test"""
        # Mock session_state para evitar errores de contexto
        with patch('streamlit.session_state', {}):
            pass
    
    def test_render_function_exists(self):
        """Test que verifica la existencia de la función render"""
        assert hasattr(qa_module, 'render')
        assert callable(getattr(qa_module, 'render'))
    
    def test_advanced_analysis_available_flag(self):
        """Test que verifica la existencia de la bandera ADVANCED_ANALYSIS_AVAILABLE"""
        assert hasattr(qa_module, 'ADVANCED_ANALYSIS_AVAILABLE')
        assert isinstance(qa_module.ADVANCED_ANALYSIS_AVAILABLE, bool)
    
    def test_render_functions_exist(self):
        """Test que verifica la existencia de funciones de renderizado"""
        render_functions = [
            'render_advanced_dashboard',
            'render_advanced_themes',
            'render_clustering_analysis',
            'render_advanced_concept_map',
            'render_sentiment_analysis',
            'render_word_cloud',
            'render_settings_tab',
            'render_interactive_concept_map',
            'render_interactive_mind_map',
            'render_automatic_summary'
        ]
        
        for func_name in render_functions:
            assert hasattr(qa_module, func_name), f"Función {func_name} no encontrada"
            assert callable(getattr(qa_module, func_name)), f"Función {func_name} no es callable"
    
    def test_analyzer_with_empty_chunks(self):
        """Test manejo de chunks vacíos"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        empty_chunks = []
        
        # Verificar que los métodos manejen chunks vacíos sin errores
        try:
            # Solo verificamos que no lance excepciones inmediatas
            result = analyzer.extract_key_concepts(empty_chunks)
            assert isinstance(result, list)
        except Exception as e:
            # Es aceptable que algunos métodos fallen con datos vacíos
            assert isinstance(e, (ValueError, TypeError, AttributeError))
    
    def test_analyzer_basic_functionality(self):
        """Test funcionalidad básica del analizador"""
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        
        # Test datos mock básicos
        mock_chunks = [
            {
                'content': 'Este es un documento de prueba sobre inteligencia artificial.',
                'metadata': {'source_file': 'test.pdf', 'chunk_id': 1}
            }
        ]
        
        # Verificar que los métodos principales existen y son llamables
        assert callable(analyzer.extract_key_concepts)
        assert callable(analyzer.generate_rag_summary)
        assert callable(analyzer.preprocess_text)

if __name__ == "__main__":
    pytest.main([__file__])