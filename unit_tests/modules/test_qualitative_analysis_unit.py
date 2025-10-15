"""
Pruebas unitarias para el módulo qualitative_analysis.py
Cubre análisis cualitativo avanzado, mapas conceptuales, resúmenes automáticos y visualizaciones
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import tempfile
import os

# Agregar el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent.parent))

# Importar el módulo a testear
import modules.qualitative_analysis as qa_module


class TestQualitativeAnalysisBasic:
    """Pruebas básicas del módulo de análisis cualitativo"""
    
    def test_module_imports_correctly(self):
        """Test que verifica que el módulo se importa correctamente"""
        assert qa_module is not None
        assert hasattr(qa_module, 'AdvancedQualitativeAnalyzer')
    
    def test_render_function_exists(self):
        """Test que verifica la existencia de la función render principal"""
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
            'render_automatic_summary',
            'render_triangulation_analysis'
        ]
        
        for func_name in render_functions:
            assert hasattr(qa_module, func_name), f"Función {func_name} no encontrada"
            assert callable(getattr(qa_module, func_name)), f"Función {func_name} no es callable"


class TestAdvancedQualitativeAnalyzer:
    """Pruebas para la clase AdvancedQualitativeAnalyzer"""
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    def test_analyzer_initialization(self, mock_config, mock_logger, mock_rag):
        """Test de inicialización del analizador"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        
        assert analyzer is not None
        assert hasattr(analyzer, 'rag_processor')
        assert hasattr(analyzer, 'cache_path')
        assert hasattr(analyzer, 'analysis_cache_path')
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    def test_validate_chunks_empty_list(self, mock_config, mock_logger, mock_rag):
        """Test de validación con lista vacía"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        result = analyzer._validate_chunks([])
        
        assert result is False
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    def test_validate_chunks_valid_data(self, mock_config, mock_logger, mock_rag):
        """Test de validación con datos válidos"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        valid_chunks = [
            {'content': 'Este es un contenido válido con suficiente texto para ser procesado'},
            {'content': 'Otro contenido válido con información relevante para el análisis'}
        ]
        
        result = analyzer._validate_chunks(valid_chunks)
        assert result is True
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    def test_clear_cache(self, mock_config, mock_logger, mock_rag):
        """Test de limpieza de cache"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        analyzer._spanish_stopwords_cache = {'test': 'data'}
        analyzer._tfidf_vectorizers_cache = {'test': 'vectorizer'}
        
        analyzer.clear_cache()
        
        assert analyzer._spanish_stopwords_cache is None
        assert len(analyzer._tfidf_vectorizers_cache) == 0
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    def test_get_cache_stats(self, mock_config, mock_logger, mock_rag):
        """Test de estadísticas del cache"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        stats = analyzer.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'stopwords_cached' in stats
        assert 'tfidf_vectorizers' in stats
        assert 'processed_texts' in stats
        assert 'concept_analyses' in stats


class TestDataLoading:
    """Pruebas para carga de datos"""
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_load_rag_data_success(self, mock_exists, mock_file, mock_config, mock_logger, mock_rag):
        """Test de carga exitosa de datos RAG"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        mock_exists.return_value = True
        
        test_data = {
            'chunks': {
                'file1.txt': [
                    {'content': 'Contenido del primer chunk'},
                    {'content': 'Contenido del segundo chunk'}
                ]
            }
        }
        mock_file.return_value.read.return_value = json.dumps(test_data)
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        result = analyzer.load_rag_data()
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    @patch('pathlib.Path.exists')
    def test_load_rag_data_no_file(self, mock_exists, mock_config, mock_logger, mock_rag):
        """Test de carga cuando no existe el archivo"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        mock_exists.return_value = False
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        result = analyzer.load_rag_data()
        
        assert result == []


class TestSummaryGeneration:
    """Pruebas para generación de resúmenes"""
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    def test_generate_rag_summary_empty_chunks(self, mock_config, mock_logger, mock_rag):
        """Test de generación de resumen con chunks vacíos"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        result = analyzer.generate_rag_summary([])
        
        assert "No hay contenido disponible" in result
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    @patch('modules.qualitative_analysis.TextBlob')
    def test_generate_rag_summary_valid_chunks(self, mock_textblob, mock_config, mock_logger, mock_rag):
        """Test de generación de resumen con chunks válidos"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        # Mock TextBlob
        mock_blob = Mock()
        mock_blob.sentences = [
            Mock(__str__=lambda: "Primera oración del contenido."),
            Mock(__str__=lambda: "Segunda oración con información relevante."),
            Mock(__str__=lambda: "Tercera oración para completar el resumen.")
        ]
        mock_textblob.return_value = mock_blob
        
        chunks = [
            {'content': 'Este es un contenido válido con suficiente información para generar un resumen completo', 'source': 'doc1.txt'},
            {'content': 'Contenido adicional que complementa la información anterior', 'source': 'doc2.txt'}
        ]
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        result = analyzer.generate_rag_summary(chunks)
        
        assert "Resumen Básico" in result
        assert isinstance(result, str)
        assert len(result) > 0
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    def test_generate_intelligent_summary_empty_chunks(self, mock_config, mock_logger, mock_rag):
        """Test de generación de resumen inteligente con chunks vacíos"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        result = analyzer.generate_intelligent_summary([])
        
        assert isinstance(result, dict)
        assert result['type'] == 'error'
        assert "No hay contenido disponible" in result['summary']


class TestRenderFunctions:
    """Pruebas para funciones de renderizado"""
    
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    @patch('streamlit.code')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.spinner')
    @patch('streamlit.session_state', new_callable=MagicMock)
    @patch('streamlit.tabs')
    @patch('streamlit.rerun')
    @patch('modules.qualitative_analysis.AdvancedQualitativeAnalyzer')
    def test_render_main_function(self, mock_analyzer_class, mock_rerun, mock_tabs,
                                 mock_session_state, mock_spinner, mock_button, 
                                 mock_columns, mock_code, mock_info, mock_warning, 
                                 mock_markdown, mock_title):
        """Test de la función render principal"""
        # Configurar mocks
        mock_analyzer = Mock()
        mock_analyzer.load_rag_data.return_value = []
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_columns.return_value = []
        # Configurar columns mock con context manager
        col_mocks = []
        for i in range(2):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks.append(col_mock)
        mock_columns.return_value = col_mocks
        mock_button.return_value = False
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Configurar tabs mock - NO necesitan context manager
        tab_mocks = [Mock() for _ in range(10)]
        mock_tabs.return_value = tab_mocks
        
        # Ejecutar función sin errores
        qa_module.render()
        
        # Verificar que se llamaron las funciones principales
        mock_title.assert_called_once()
        mock_analyzer_class.assert_called_once()
        mock_analyzer.load_rag_data.assert_called_once()
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.plotly_chart')
    @patch('streamlit.dataframe')
    def test_render_advanced_dashboard(self, mock_dataframe, mock_plotly_chart, mock_metric, mock_columns, mock_subheader):
        """Test de renderizado del dashboard avanzado"""
        mock_analyzer = Mock()
        
        # Configurar mocks de columnas para soportar context manager
        col_mocks = []
        for i in range(4):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks.append(col_mock)
        
        # Mock para st.columns que devuelve exactamente 4 columnas
        def mock_columns_side_effect(spec):
            if isinstance(spec, list) and len(spec) == 4:
                return col_mocks
            elif isinstance(spec, int) and spec == 4:
                return col_mocks
            else:
                return col_mocks[:spec] if isinstance(spec, int) else col_mocks
        
        mock_columns.side_effect = mock_columns_side_effect
        
        chunks = [
            {'content': 'Contenido de prueba', 'metadata': {'source_file': 'test.txt'}}
        ]
        
        # Ejecutar función sin errores
        qa_module.render_advanced_dashboard(mock_analyzer, chunks)
        
        # Verificar que se llamaron las funciones principales
        mock_subheader.assert_called()
        mock_columns.assert_called()
        mock_metric.assert_called()
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_render_advanced_themes_empty_chunks(self, mock_warning, mock_info, mock_subheader):
        """Test de renderizado de temas con chunks vacíos"""
        mock_analyzer = Mock()
        
        try:
            qa_module.render_advanced_themes(mock_analyzer, [])
            test_passed = True
        except Exception:
            test_passed = False
        
        assert test_passed
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_render_clustering_analysis_empty_chunks(self, mock_warning, mock_info, mock_subheader):
        """Test de renderizado de análisis de clustering con chunks vacíos"""
        mock_analyzer = Mock()
        
        try:
            qa_module.render_clustering_analysis(mock_analyzer, [])
            test_passed = True
        except Exception:
            test_passed = False
        
        assert test_passed
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_render_sentiment_analysis_empty_chunks(self, mock_warning, mock_info, mock_subheader):
        """Test de renderizado de análisis de sentimientos con chunks vacíos"""
        mock_analyzer = Mock()
        
        try:
            qa_module.render_sentiment_analysis(mock_analyzer, [])
            test_passed = True
        except Exception:
            test_passed = False
        
        assert test_passed
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_render_word_cloud_empty_chunks(self, mock_warning, mock_info, mock_subheader):
        """Test de renderizado de nube de palabras con chunks vacíos"""
        mock_analyzer = Mock()
        
        try:
            qa_module.render_word_cloud(mock_analyzer, [])
            test_passed = True
        except Exception:
            test_passed = False
        
        assert test_passed


class TestUtilityFunctions:
    """Pruebas para funciones utilitarias"""
    
    @patch('modules.qualitative_analysis.RAGProcessor')
    @patch('modules.qualitative_analysis.setup_logger')
    @patch('modules.qualitative_analysis.config')
    def test_get_term_color(self, mock_config, mock_logger, mock_rag):
        """Test de generación de colores para términos"""
        mock_config.CACHE_DIR = "/tmp/cache"
        mock_logger.return_value = Mock()
        mock_rag.return_value = Mock()
        
        analyzer = qa_module.AdvancedQualitativeAnalyzer()
        
        # Test con diferentes términos
        color1 = analyzer._get_term_color("término_largo_complejo")
        color2 = analyzer._get_term_color("término_medio")
        color3 = analyzer._get_term_color("corto")
        
        assert color1.startswith('hsl(')
        assert color2.startswith('hsl(')
        assert color3.startswith('hsl(')
        
        # El mismo término debe generar el mismo color
        color1_repeat = analyzer._get_term_color("término_largo_complejo")
        assert color1 == color1_repeat


class TestConceptMapGeneration:
    """Pruebas para generación de mapas conceptuales"""
    
    @patch('modules.qualitative_analysis.PYVIS_AVAILABLE', True)
    @patch('streamlit.stop')
    @patch('streamlit.spinner')
    @patch('streamlit.checkbox')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.selectbox')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    @patch('streamlit.header')
    def test_render_interactive_concept_map_basic(self, mock_header, mock_info, mock_warning,
                                                 mock_selectbox, mock_button, mock_columns,
                                                 mock_checkbox, mock_spinner, mock_stop):
        """Test básico de renderizado de mapa conceptual interactivo"""
        mock_analyzer = Mock()
        mock_analyzer.create_interactive_concept_map.return_value = "test_map.html"
        mock_analyzer._analyze_concept_hierarchy.return_value = {"test": "structure"}
        
        # Configurar mocks de columnas para soportar context manager
        col_mocks_4 = []
        for i in range(4):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks_4.append(col_mock)
        
        col_mocks_2 = []
        for i in range(2):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks_2.append(col_mock)
        
        # Mock para st.columns que maneja múltiples llamadas
        call_count = 0
        def mock_columns_side_effect(spec):
            nonlocal call_count
            call_count += 1
            if isinstance(spec, list) and len(spec) == 4:
                return col_mocks_4
            elif isinstance(spec, int) and spec == 4:
                return col_mocks_4
            elif isinstance(spec, list) and len(spec) == 2:
                return col_mocks_2
            elif isinstance(spec, int) and spec == 2:
                return col_mocks_2
            else:
                return col_mocks_4  # Default
        
        mock_columns.side_effect = mock_columns_side_effect
        
        # Configurar spinner como context manager
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        mock_selectbox.return_value = "ai"
        mock_button.return_value = False
        
        chunks = [
            {'content': 'Contenido de prueba para mapa conceptual', 'source': 'test.txt'}
        ]
        
        # Ejecutar función sin errores
        qa_module.render_interactive_concept_map(mock_analyzer, chunks)
        
        # Verificar que se llamaron las funciones principales
        mock_header.assert_called()
        mock_columns.assert_called()
        mock_selectbox.assert_called()
    
    @patch('modules.qualitative_analysis.AGRAPH_AVAILABLE', True)
    @patch('streamlit.stop')
    @patch('streamlit.spinner')
    @patch('streamlit.checkbox')
    @patch('streamlit.slider')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.selectbox')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    @patch('streamlit.header')
    def test_render_interactive_mind_map_basic(self, mock_header, mock_info, mock_warning,
                                              mock_selectbox, mock_button, mock_columns,
                                              mock_slider, mock_checkbox, mock_spinner, mock_stop):
        """Test básico de renderizado de mapa mental interactivo"""
        # Configurar mock del analizador con datos serializables
        mock_analyzer = MagicMock()
        mock_analyzer.create_interactive_mind_map.return_value = {
            'mind_map_data': {
                'nodes': [
                    {'id': 'node1', 'label': 'Test Node', 'level': 0, 'size': 25, 'color': '#A23B72'},
                    {'id': 'node2', 'label': 'Test Node 2', 'level': 1, 'size': 20, 'color': '#F18F01'}
                ],
                'edges': [
                    {'from': 'node1', 'to': 'node2', 'label': 'connection', 'color': '#666666', 'width': 3}
                ],
                'stats': {
                    'total_nodes': 2,
                    'total_edges': 1,
                    'main_branches': 1,
                    'detailed_concepts': 2
                }
            },
            'mind_structure': {
                'central_topic': 'Test Topic',
                'branches': ['Branch 1', 'Branch 2']
            }
        }
        mock_analyzer._analyze_intelligent_mind_map_structure.return_value = {
            'central_topic': 'Test Topic',
            'branches': ['Branch 1', 'Branch 2']
        }
        
        # Configurar mocks de columnas para soportar context manager
        col_mocks_5 = []
        for i in range(5):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks_5.append(col_mock)
        
        col_mocks_4 = []
        for i in range(4):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks_4.append(col_mock)
        
        col_mocks_3 = []
        for i in range(3):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks_3.append(col_mock)
        
        col_mocks_2 = []
        for i in range(2):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks_2.append(col_mock)
        
        # Mock para st.columns que maneja múltiples llamadas
        call_count = 0
        def mock_columns_side_effect(spec):
            nonlocal call_count
            call_count += 1
            if isinstance(spec, list) and len(spec) == 5:
                return col_mocks_5
            elif isinstance(spec, int) and spec == 5:
                return col_mocks_5
            elif isinstance(spec, list) and len(spec) == 4:
                return col_mocks_4
            elif isinstance(spec, int) and spec == 4:
                return col_mocks_4
            elif isinstance(spec, list) and len(spec) == 3:
                return col_mocks_3
            elif isinstance(spec, int) and spec == 3:
                return col_mocks_3
            elif isinstance(spec, list) and len(spec) == 2:
                return col_mocks_2
            elif isinstance(spec, int) and spec == 2:
                return col_mocks_2
            else:
                return col_mocks_5  # Default
        
        mock_columns.side_effect = mock_columns_side_effect
        
        # Configurar spinner como context manager
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        mock_selectbox.return_value = "ai"
        mock_button.return_value = False
        mock_slider.return_value = 250
        
        chunks = [
            {'content': 'Contenido de prueba para mapa mental', 'source': 'test.txt'}
        ]
        
        # Ejecutar función sin errores
        qa_module.render_interactive_mind_map(mock_analyzer, chunks)
        
        # Verificar que se llamaron las funciones principales
        mock_header.assert_called()
        mock_columns.assert_called()
        mock_selectbox.assert_called()


class TestAutomaticSummary:
    """Pruebas para resúmenes automáticos"""
    
    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.slider')
    @patch('streamlit.selectbox')
    @patch('streamlit.header')
    def test_render_automatic_summary_basic(self, mock_header, mock_selectbox, mock_slider,
                                           mock_button, mock_columns, mock_info, mock_warning,
                                           mock_markdown, mock_expander):
        """Test básico de renderizado de resumen automático"""
        mock_analyzer = Mock()
        mock_analyzer.generate_rag_summary.return_value = "Resumen de prueba"
        
        # Configurar mocks de columnas para soportar context manager
        col_mocks = []
        for i in range(4):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks.append(col_mock)
        
        # Mock para st.columns que devuelve exactamente 4 columnas
        def mock_columns_side_effect(spec):
            if isinstance(spec, list) and len(spec) == 4:
                return col_mocks
            elif isinstance(spec, int) and spec == 4:
                return col_mocks
            else:
                return col_mocks[:spec] if isinstance(spec, int) else col_mocks
        
        mock_columns.side_effect = mock_columns_side_effect
        
        mock_selectbox.return_value = "comprehensive"
        mock_button.return_value = False
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        chunks = [
            {'content': 'Contenido de prueba para resumen', 'source': 'test.txt'}
        ]
        
        # Ejecutar función sin errores
        qa_module.render_automatic_summary(mock_analyzer, chunks)
        
        # Verificar que se llamaron las funciones principales
        mock_header.assert_called()
        mock_columns.assert_called()
        mock_selectbox.assert_called()


class TestSettingsAndConfiguration:
    """Pruebas para configuración y ajustes"""
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.button')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.slider')
    @patch('streamlit.selectbox')
    @patch('streamlit.divider')
    @patch('streamlit.markdown')
    @patch('streamlit.rerun')
    @patch('streamlit.session_state', new_callable=MagicMock)
    def test_render_settings_tab(self, mock_session_state, mock_rerun, mock_markdown,
                                mock_divider, mock_selectbox, mock_slider, mock_info, 
                                mock_success, mock_button, mock_metric, mock_columns, 
                                mock_subheader):
        """Test de renderizado de la pestaña de configuración"""
        mock_analyzer = Mock()
        mock_analyzer.get_cache_stats.return_value = {
            'stopwords_cached': True,
            'tfidf_vectorizers': 2,
            'processed_texts': 5,
            'concept_analyses': 3
        }
        mock_analyzer.cache_path = Mock()
        mock_analyzer.cache_path.exists.return_value = False
        
        # Configurar mocks de columnas para soportar context manager (múltiples llamadas)
        col_mocks_2 = []
        for i in range(2):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks_2.append(col_mock)
            
        col_mocks_3 = []
        for i in range(3):
            col_mock = MagicMock()
            col_mock.__enter__ = Mock(return_value=col_mock)
            col_mock.__exit__ = Mock(return_value=None)
            col_mocks_3.append(col_mock)
        
        # Mock para st.columns que maneja múltiples llamadas
        call_count = 0
        def mock_columns_side_effect(spec):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and spec == 2:
                return col_mocks_2
            elif call_count == 2 and spec == 3:
                return col_mocks_3
            else:
                return col_mocks_2 if spec == 2 else col_mocks_3
        
        mock_columns.side_effect = mock_columns_side_effect
        
        mock_button.return_value = False
        mock_slider.return_value = 10
        mock_selectbox.return_value = "K-means"
        
        # Ejecutar función sin errores
        qa_module.render_settings_tab(mock_analyzer)
        
        # Verificar que se llamaron las funciones principales
        mock_subheader.assert_called()
        mock_columns.assert_called()
        mock_slider.assert_called()


class TestTriangulationAnalysis:
    """Pruebas para análisis de triangulación"""
    
    @patch('streamlit.subheader')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    def test_render_triangulation_analysis_empty_chunks(self, mock_warning, mock_info, mock_subheader):
        """Test de renderizado de análisis de triangulación con chunks vacíos"""
        mock_analyzer = Mock()
        
        try:
            qa_module.render_triangulation_analysis(mock_analyzer, [])
            test_passed = True
        except Exception:
            test_passed = False
        
        assert test_passed


class TestIntegration:
    """Pruebas de integración del módulo"""
    
    def test_all_render_functions_are_callable(self):
        """Test que verifica que todas las funciones de renderizado son llamables"""
        render_functions = [
            'render',
            'render_advanced_dashboard',
            'render_advanced_themes',
            'render_clustering_analysis', 
            'render_advanced_concept_map',
            'render_sentiment_analysis',
            'render_word_cloud',
            'render_settings_tab',
            'render_interactive_concept_map',
            'render_interactive_mind_map',
            'render_automatic_summary',
            'render_triangulation_analysis'
        ]
        
        for func_name in render_functions:
            func = getattr(qa_module, func_name)
            assert callable(func)
    
    def test_advanced_qualitative_analyzer_class_exists(self):
        """Test que verifica la existencia de la clase principal"""
        assert hasattr(qa_module, 'AdvancedQualitativeAnalyzer')
        assert callable(qa_module.AdvancedQualitativeAnalyzer)
    
    def test_module_constants_exist(self):
        """Test que verifica la existencia de constantes del módulo"""
        constants = [
            'ADVANCED_ANALYSIS_AVAILABLE',
            'PYVIS_AVAILABLE',
            'AGRAPH_AVAILABLE',
            'GRAPHVIZ_AVAILABLE'
        ]
        
        for constant in constants:
            assert hasattr(qa_module, constant)
            assert isinstance(getattr(qa_module, constant), bool)