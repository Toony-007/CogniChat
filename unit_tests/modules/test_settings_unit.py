"""
Pruebas unitarias para el módulo settings.py
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import sys
import os
import json
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar el módulo a probar
import modules.settings as settings_module


class TestSettingsBasic(unittest.TestCase):
    """Pruebas básicas del módulo settings"""
    
    def test_module_imports(self):
        """Test que verifica las importaciones del módulo"""
        self.assertTrue(hasattr(settings_module, 'st'))
        self.assertTrue(hasattr(settings_module, 'json'))
        self.assertTrue(hasattr(settings_module, 'os'))
        self.assertTrue(hasattr(settings_module, 'sys'))
        self.assertTrue(hasattr(settings_module, 'time'))
        self.assertTrue(hasattr(settings_module, 'datetime'))
        self.assertTrue(hasattr(settings_module, 'Path'))
        self.assertTrue(hasattr(settings_module, 'ErrorHandler'))
        self.assertTrue(hasattr(settings_module, 'setup_logger'))
        self.assertTrue(hasattr(settings_module, 'OllamaClient'))
        self.assertTrue(hasattr(settings_module, 'config'))
    
    def test_logger_initialization(self):
        """Test que verifica la inicialización del logger"""
        self.assertTrue(hasattr(settings_module, 'logger'))
        self.assertIsNotNone(settings_module.logger)
    
    def test_error_handler_initialization(self):
        """Test que verifica la inicialización del error handler"""
        self.assertTrue(hasattr(settings_module, 'error_handler'))
        self.assertIsNotNone(settings_module.error_handler)


class TestSettingsRenderFunction(unittest.TestCase):
    """Pruebas para la función render principal"""
    
    def test_render_function_exists(self):
        """Test que verifica la existencia de la función render"""
        self.assertTrue(hasattr(settings_module, 'render'))
        self.assertTrue(callable(getattr(settings_module, 'render')))
    
    @patch('modules.settings.render_models_config')
    @patch('modules.settings.render_processing_config')
    @patch('modules.settings.render_system_config')
    @patch('modules.settings.render_export_import')
    @patch('modules.settings.st')
    @patch('modules.settings.OllamaClient')
    def test_render_function_basic_execution(self, mock_ollama_client, mock_st, 
                                           mock_export_import, mock_system_config, 
                                           mock_processing_config, mock_models_config):
        """Test que verifica la ejecución básica de render"""
        # Configurar mocks
        mock_client_instance = Mock()
        mock_client_instance.is_available.return_value = True
        mock_ollama_client.return_value = mock_client_instance
        
        # Crear mocks para tabs que actúen como context managers
        tab_mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = tab_mocks
        
        # Ejecutar función
        settings_module.render()
        
        # Verificar llamadas básicas
        mock_st.header.assert_called()
        mock_ollama_client.assert_called()
        mock_client_instance.is_available.assert_called()
    
    @patch('modules.settings.st')
    @patch('modules.settings.OllamaClient')
    def test_render_ollama_status_check(self, mock_ollama_client, mock_st):
        """Test que verifica la verificación del estado de Ollama"""
        mock_client_instance = Mock()
        mock_client_instance.is_available.return_value = True
        mock_ollama_client.return_value = mock_client_instance
        
        # Crear mocks para tabs
        tab_mocks = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = tab_mocks
        
        with patch('modules.settings.render_models_config'), \
             patch('modules.settings.render_processing_config'), \
             patch('modules.settings.render_system_config'), \
             patch('modules.settings.render_export_import'):
            
            settings_module.render()
            mock_client_instance.is_available.assert_called()


class TestSettingsModelsConfig(unittest.TestCase):
    """Pruebas para la función render_models_config"""
    
    def test_render_models_config_exists(self):
        """Test que verifica la existencia de la función render_models_config"""
        self.assertTrue(hasattr(settings_module, 'render_models_config'))
        self.assertTrue(callable(getattr(settings_module, 'render_models_config')))
    
    @patch('modules.settings.st')
    @patch('modules.settings.OllamaClient')
    def test_render_models_config_ollama_unavailable(self, mock_ollama_client, mock_st):
        """Test que verifica el comportamiento cuando Ollama no está disponible"""
        mock_client_instance = Mock()
        mock_client_instance.is_available.return_value = False
        mock_ollama_client.return_value = mock_client_instance
        
        settings_module.render_models_config()
        
        mock_st.error.assert_called()
        mock_st.markdown.assert_called()
    
    @patch('modules.settings.st')
    @patch('modules.settings.OllamaClient')
    @patch('modules.settings.time')
    def test_render_models_config_basic_execution(self, mock_time, mock_ollama_client, mock_st):
        """Test que verifica la ejecución básica con modelos disponibles"""
        # Configurar mocks
        mock_client_instance = Mock()
        mock_client_instance.is_available.return_value = True
        mock_client_instance.get_available_models.return_value = [
            {'name': 'llama3.1:8b'},
            {'name': 'nomic-embed-text'}
        ]
        mock_ollama_client.return_value = mock_client_instance
        
        mock_time.time.return_value = 1000
        mock_session_state = MagicMock()
        mock_st.session_state = mock_session_state
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        
        settings_module.render_models_config()
        
        mock_client_instance.get_available_models.assert_called()
        mock_st.columns.assert_called()


class TestSettingsProcessingConfig(unittest.TestCase):
    """Pruebas para la función render_processing_config"""
    
    def test_render_processing_config_exists(self):
        """Test que verifica la existencia de la función render_processing_config"""
        self.assertTrue(hasattr(settings_module, 'render_processing_config'))
        self.assertTrue(callable(getattr(settings_module, 'render_processing_config')))
    
    @patch('modules.settings.st')
    @patch('modules.settings.config')
    def test_render_processing_config_basic_execution(self, mock_config, mock_st):
        """Test que verifica la ejecución básica de processing config"""
        # Configurar valores por defecto
        mock_config.CHUNK_SIZE = 1000
        mock_config.CHUNK_OVERLAP = 200
        mock_config.MAX_RETRIEVAL_DOCS = 10
        mock_config.SIMILARITY_THRESHOLD = 0.7
        mock_config.MAX_FILE_SIZE_MB = 50
        mock_config.SUPPORTED_FORMATS = ['.txt', '.pdf', '.docx']
        
        mock_st.session_state = MagicMock()
        # Configurar columns para devolver suficientes columnas
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        mock_st.number_input.return_value = 500  # Valor específico para number_input
        mock_st.slider.return_value = 100  # Valor específico para slider
        
        settings_module.render_processing_config()
        
        # Verificar que se llamaron funciones básicas
        mock_st.subheader.assert_called()
        mock_st.columns.assert_called()


class TestSettingsSystemConfig(unittest.TestCase):
    """Pruebas para la función render_system_config"""
    
    def test_render_system_config_exists(self):
        """Test que verifica la existencia de la función render_system_config"""
        self.assertTrue(hasattr(settings_module, 'render_system_config'))
        self.assertTrue(callable(getattr(settings_module, 'render_system_config')))
    
    @patch('modules.settings.st')
    def test_render_system_config_basic_execution(self, mock_st):
        """Test que verifica la ejecución básica de system config"""
        # Configurar mocks básicos para Streamlit
        mock_st.subheader = MagicMock()
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        mock_st.selectbox.return_value = "INFO"
        mock_st.text_input.return_value = "http://localhost:11434"
        mock_st.number_input.return_value = 500
        mock_st.slider.return_value = 100
        mock_st.__version__ = "1.28.0"
        mock_st.metric = MagicMock()
        mock_st.info = MagicMock()
        mock_st.markdown = MagicMock()
        
        # Configurar session_state
        mock_session_state = MagicMock()
        mock_session_state.get.return_value = "INFO"
        mock_st.session_state = mock_session_state
        
        with patch('modules.settings.config') as mock_config, \
             patch('modules.settings.os.path.exists', return_value=True), \
             patch('modules.settings.os.listdir', return_value=[]), \
             patch('modules.settings.sys'):
            
            # Configurar config mock
            mock_config.LOG_LEVEL = "INFO"
            mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_config.OLLAMA_TIMEOUT = 120
            mock_config.DATA_DIR = "/data"
            mock_config.UPLOADS_DIR = "/uploads"
            mock_config.CACHE_DIR = "/cache"
            
            # Ejecutar la función
            settings_module.render_system_config()
            
            # Verificar que se llamaron las funciones básicas
            mock_st.subheader.assert_called()
            mock_st.columns.assert_called()
    
    @patch('modules.settings.st')
    @patch('modules.settings.OllamaClient')
    def test_render_system_config_ollama_test(self, mock_ollama_client, mock_st):
        """Test que verifica la funcionalidad básica de test de conexión"""
        mock_client_instance = Mock()
        mock_client_instance.is_available.return_value = True
        mock_ollama_client.return_value = mock_client_instance
        
        # Configurar mocks básicos para Streamlit
        mock_st.subheader = MagicMock()
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        mock_st.selectbox.return_value = "INFO"
        mock_st.text_input.return_value = "http://localhost:11434"
        mock_st.number_input.return_value = 500
        mock_st.slider.return_value = 100
        mock_st.__version__ = "1.28.0"
        mock_st.metric = MagicMock()
        mock_st.info = MagicMock()
        mock_st.markdown = MagicMock()
        
        # Configurar session_state
        mock_session_state = MagicMock()
        mock_session_state.get.return_value = "INFO"
        mock_st.session_state = mock_session_state
        
        with patch('modules.settings.config') as mock_config, \
             patch('modules.settings.os.path.exists', return_value=True), \
             patch('modules.settings.os.listdir', return_value=[]), \
             patch('modules.settings.sys'):
            
            # Configurar config mock
            mock_config.LOG_LEVEL = "INFO"
            mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_config.OLLAMA_TIMEOUT = 120
            mock_config.DATA_DIR = "/data"
            mock_config.UPLOADS_DIR = "/uploads"
            mock_config.CACHE_DIR = "/cache"
            
            # Ejecutar la función
            settings_module.render_system_config()
            
            # Verificar que se puede crear el cliente
            mock_ollama_client.assert_called()


class TestSettingsExportImport(unittest.TestCase):
    """Pruebas para la función render_export_import"""
    
    def test_render_export_import_exists(self):
        """Test que verifica la existencia de la función render_export_import"""
        self.assertTrue(hasattr(settings_module, 'render_export_import'))
        self.assertTrue(callable(getattr(settings_module, 'render_export_import')))
    
    @patch('modules.settings.st')
    @patch('modules.settings.config')
    def test_render_export_import_basic_execution(self, mock_config, mock_st):
        """Test que verifica la ejecución básica de export/import"""
        mock_st.session_state = MagicMock()
        # Crear mocks que soporten context manager
        col1_mock = MagicMock()
        col1_mock.__enter__ = MagicMock(return_value=col1_mock)
        col1_mock.__exit__ = MagicMock(return_value=None)
        col2_mock = MagicMock()
        col2_mock.__enter__ = MagicMock(return_value=col2_mock)
        col2_mock.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [col1_mock, col2_mock]
        
        settings_module.render_export_import()
        
        # Verificar llamadas básicas
        mock_st.subheader.assert_called()
        mock_st.columns.assert_called()
    
    @patch('modules.settings.st')
    @patch('modules.settings.json')
    def test_render_export_import_json_handling(self, mock_json, mock_st):
        """Test que verifica el manejo básico de JSON"""
        mock_st.session_state = MagicMock()
        # Crear mocks que soporten context manager
        col1_mock = MagicMock()
        col1_mock.__enter__ = MagicMock(return_value=col1_mock)
        col1_mock.__exit__ = MagicMock(return_value=None)
        col2_mock = MagicMock()
        col2_mock.__enter__ = MagicMock(return_value=col2_mock)
        col2_mock.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [col1_mock, col2_mock]
        mock_json.dumps.return_value = '{"test": "data"}'
        
        with patch('modules.settings.config'):
            settings_module.render_export_import()
        
        # Verificar que se puede usar json
        self.assertTrue(hasattr(settings_module, 'json'))
    
    @patch('modules.settings.st')
    @patch('modules.settings.config')
    @patch('modules.settings.json')
    @patch('modules.settings.datetime')
    def test_render_export_import_export_functionality(self, mock_datetime, mock_json, mock_config, mock_st):
        """Test que verifica la funcionalidad de exportación"""
        mock_st.session_state = MagicMock()
        # Crear mocks que soporten context manager
        col1_mock = MagicMock()
        col1_mock.__enter__ = MagicMock(return_value=col1_mock)
        col1_mock.__exit__ = MagicMock(return_value=None)
        col2_mock = MagicMock()
        col2_mock.__enter__ = MagicMock(return_value=col2_mock)
        col2_mock.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [col1_mock, col2_mock]
        mock_st.button.return_value = True
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
        
        settings_module.render_export_import()
        
        # Verificar que se llamaron las funciones de UI
        mock_st.subheader.assert_called()
        mock_st.columns.assert_called()
        mock_st.button.assert_called()
    
    @patch('modules.settings.st')
    @patch('modules.settings.json')
    def test_render_export_import_import_functionality(self, mock_json, mock_st):
        """Test que verifica la funcionalidad de importación"""
        # Configurar mock de archivo subido
        mock_uploaded_file = Mock()
        mock_uploaded_file.name = "test_config.json"
        
        mock_config_data = {
            'selected_llm_model': 'llama3.1:8b',
            'selected_embedding_model': 'nomic-embed-text',
            'chunk_size': 1000,
            '_metadata': {
                'export_date': '2024-01-01T00:00:00',
                'version': '1.0',
                'app_name': 'CogniChat'
            }
        }
        
        mock_json.load.return_value = mock_config_data
        mock_st.session_state = MagicMock()
        # Crear mocks que soporten context manager
        col1_mock = MagicMock()
        col1_mock.__enter__ = MagicMock(return_value=col1_mock)
        col1_mock.__exit__ = MagicMock(return_value=None)
        col2_mock = MagicMock()
        col2_mock.__enter__ = MagicMock(return_value=col2_mock)
        col2_mock.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [col1_mock, col2_mock]
        mock_st.file_uploader.return_value = mock_uploaded_file
        
        settings_module.render_export_import()
        
        # Verificar llamadas
        mock_st.file_uploader.assert_called()
        mock_json.load.assert_called_with(mock_uploaded_file)
    
    @patch('modules.settings.st')
    def test_render_export_import_reset_functionality(self, mock_st):
        """Test que verifica la funcionalidad de reset"""
        mock_session_state = MagicMock()
        mock_session_state.__contains__ = MagicMock(return_value=True)
        mock_session_state.__delitem__ = MagicMock()
        mock_session_state.selected_llm_model = 'test_model'
        mock_session_state.chunk_size = 1000
        mock_session_state.confirm_reset = True
        mock_st.session_state = mock_session_state
        # Crear mocks que soporten context manager
        col1_mock = MagicMock()
        col1_mock.__enter__ = MagicMock(return_value=col1_mock)
        col1_mock.__exit__ = MagicMock(return_value=None)
        col2_mock = MagicMock()
        col2_mock.__enter__ = MagicMock(return_value=col2_mock)
        col2_mock.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [col1_mock, col2_mock]
        mock_st.button.return_value = True
        
        settings_module.render_export_import()
        
        # Verificar que se llamó el botón
        mock_st.button.assert_called()


class TestSettingsIntegration(unittest.TestCase):
    """Pruebas de integración para el módulo settings"""
    
    def test_all_render_functions_callable(self):
        """Test que verifica que todas las funciones render son llamables"""
        render_functions = [
            'render',
            'render_models_config',
            'render_processing_config',
            'render_system_config',
            'render_export_import'
        ]
        
        for func_name in render_functions:
            with self.subTest(function=func_name):
                self.assertTrue(hasattr(settings_module, func_name))
                self.assertTrue(callable(getattr(settings_module, func_name)))
    
    def test_module_constants(self):
        """Test que verifica la existencia de constantes del módulo"""
        # Verificar que el módulo tiene las importaciones necesarias
        self.assertTrue(hasattr(settings_module, 'st'))
        self.assertTrue(hasattr(settings_module, 'json'))
        self.assertTrue(hasattr(settings_module, 'os'))


class TestSettingsConstants(unittest.TestCase):
    """Pruebas para constantes y configuraciones del módulo"""
    
    def test_module_constants(self):
        """Test que verifica las constantes del módulo"""
        # Verificar que las constantes existen
        self.assertTrue(hasattr(settings_module, 'logger'))
        self.assertTrue(hasattr(settings_module, 'error_handler'))
        self.assertTrue(hasattr(settings_module, 'config'))
    
    def test_module_docstring(self):
        """Test que verifica la documentación del módulo"""
        self.assertIsNotNone(settings_module.__doc__)
        self.assertIn("Pestaña de configuraciones", settings_module.__doc__)


if __name__ == '__main__':
    unittest.main()