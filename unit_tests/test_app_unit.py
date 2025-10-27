"""
Pruebas unitarias para app.py
Archivo principal de la aplicación CogniChat
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

# Agregar el directorio raíz al path para las importaciones
sys.path.append(str(Path(__file__).parent.parent))

import app


class TestAppFunctions:
    """Pruebas para las funciones principales de app.py"""
    
    @patch('streamlit.markdown')
    def test_load_custom_css(self, mock_markdown):
        """Test de carga de estilos CSS personalizados"""
        app.load_custom_css()
        
        # Verificar que se llamó st.markdown con CSS
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args
        assert 'unsafe_allow_html=True' in str(call_args)
        assert '.main-header' in call_args[0][0]
        assert 'background: linear-gradient' in call_args[0][0]
    
    @patch('requests.get')
    def test_check_ollama_status_success(self, mock_get):
        """Test de verificación exitosa del estado de Ollama"""
        # Configurar mock para respuesta exitosa
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = app.check_ollama_status()
        
        assert result is True
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)
    
    @patch('requests.get')
    def test_check_ollama_status_failure(self, mock_get):
        """Test de verificación fallida del estado de Ollama"""
        # Configurar mock para respuesta de error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = app.check_ollama_status()
        
        assert result is False
    
    @patch('requests.get')
    @patch('app.logger')
    def test_check_ollama_status_exception(self, mock_logger, mock_get):
        """Test de manejo de excepciones en verificación de Ollama"""
        # Configurar mock para lanzar excepción
        mock_get.side_effect = Exception("Connection error")
        
        result = app.check_ollama_status()
        
        assert result is False
        mock_logger.error.assert_called_once()
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_initialize_session_state_new_session(self, mock_session_state):
        """Test de inicialización de session_state con valores por defecto"""
        # Crear un mock que simule el comportamiento de session_state
        class MockSessionState(dict):
            def __getattr__(self, name):
                return self.get(name)
            
            def __setattr__(self, name, value):
                self[name] = value
            
            def __contains__(self, name):
                return dict.__contains__(self, name)
        
        mock_session_state_obj = MockSessionState()
        
        with patch('streamlit.session_state', mock_session_state_obj):
            app.initialize_session_state()
            
            # Verificar que se establecieron los valores por defecto
            assert 'selected_llm_model' in mock_session_state_obj
            assert 'selected_embedding_model' in mock_session_state_obj
            assert 'chunk_size' in mock_session_state_obj
            assert 'chunk_overlap' in mock_session_state_obj
            assert 'max_retrieval_docs' in mock_session_state_obj
            assert 'similarity_threshold' in mock_session_state_obj
    
    def test_initialize_session_state_existing_values(self):
        """Test de inicialización cuando ya existen valores en session_state"""
        existing_values = {
            'selected_llm_model': 'existing_model',
            'selected_embedding_model': 'existing_embedding',
            'chunk_size': 1000,
            'chunk_overlap': 100,
            'max_retrieval_docs': 5,
            'similarity_threshold': 0.8
        }
        
        with patch('streamlit.session_state', existing_values) as mock_session_state:
            app.initialize_session_state()
            
            # Verificar que los valores existentes no se sobrescribieron
            assert mock_session_state['selected_llm_model'] == 'existing_model'
            assert mock_session_state['chunk_size'] == 1000


class TestRenderSidebarConfig:
    """Pruebas para la función render_sidebar_config"""
    
    @patch('streamlit.sidebar')
    @patch('app.check_ollama_status')
    @patch('modules.document_upload.get_valid_uploaded_files')
    def test_render_sidebar_config_ollama_online(self, mock_get_files, mock_ollama_status, mock_sidebar):
        """Test de renderizado de sidebar con Ollama conectado"""
        # Configurar mocks
        mock_ollama_status.return_value = True
        mock_get_files.return_value = ['file1.pdf', 'file2.docx']
        
        # Configurar session_state mock
        session_state_mock = {
            'selected_llm_model': 'llama2',
            'selected_embedding_model': 'nomic-embed-text',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_retrieval_docs': 10,
            'similarity_threshold': 0.7
        }
        
        with patch('streamlit.session_state', session_state_mock):
            app.render_sidebar_config()
        
        # Verificar que se llamaron las funciones de sidebar
        assert mock_sidebar.markdown.called
        mock_ollama_status.assert_called_once()
    
    @patch('streamlit.sidebar')
    @patch('app.check_ollama_status')
    def test_render_sidebar_config_ollama_offline(self, mock_ollama_status, mock_sidebar):
        """Test de renderizado de sidebar con Ollama desconectado"""
        # Configurar mock para Ollama desconectado
        mock_ollama_status.return_value = False
        
        app.render_sidebar_config()
        
        # Verificar que se mostró el error
        mock_sidebar.error.assert_called_once_with("⚠️ Ollama no disponible")
        mock_ollama_status.assert_called_once()


class TestMainFunction:
    """Pruebas para la función principal main()"""
    
    @patch('streamlit.tabs')
    @patch('streamlit.markdown')
    @patch('app.render_sidebar_config')
    @patch('app.load_custom_css')
    @patch('app.initialize_session_state')
    @patch('modules.document_upload.render')
    @patch('modules.document_processor.main')
    @patch('modules.chatbot.render')
    @patch('modules.qualitative_analysis.render')
    @patch('modules.alerts.render')
    @patch('modules.settings.render')
    def test_main_function_success(self, mock_settings, mock_alerts, mock_qualitative, 
                                 mock_chatbot, mock_processor, mock_upload,
                                 mock_init_session, mock_load_css, mock_sidebar, 
                                 mock_markdown, mock_tabs):
        """Test de ejecución exitosa de la función main"""
        # Configurar mocks de tabs
        tab_mocks = []
        for i in range(6):
            tab_mock = Mock()
            tab_mock.__enter__ = Mock(return_value=tab_mock)
            tab_mock.__exit__ = Mock(return_value=None)
            tab_mocks.append(tab_mock)
        
        mock_tabs.return_value = tab_mocks
        
        app.main()
        
        # Verificar que se llamaron todas las funciones de inicialización
        mock_init_session.assert_called_once()
        mock_load_css.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_markdown.assert_called()
        mock_tabs.assert_called_once()
        
        # Verificar que se renderizaron todos los módulos
        mock_upload.assert_called_once()
        mock_processor.assert_called_once()
        mock_chatbot.assert_called_once()
        mock_qualitative.assert_called_once()
        mock_alerts.assert_called_once()
        mock_settings.assert_called_once()
    
    @patch('streamlit.error')
    @patch('app.error_handler')
    @patch('app.initialize_session_state')
    def test_main_function_exception_handling(self, mock_init_session, mock_error_handler, mock_st_error):
        """Test de manejo de excepciones en la función main"""
        # Configurar mock para lanzar excepción
        mock_init_session.side_effect = Exception("Test error")
        
        app.main()
        
        # Verificar que se manejó el error
        mock_error_handler.handle_error.assert_called_once()
        mock_st_error.assert_called_once_with(
            "Ha ocurrido un error inesperado. Consulta la pestaña de Alertas para más detalles."
        )


class TestAppImports:
    """Pruebas para verificar las importaciones del módulo"""
    
    def test_required_imports_exist(self):
        """Test de que todas las importaciones requeridas existen"""
        # Verificar que los módulos principales están disponibles
        assert hasattr(app, 'st')
        assert hasattr(app, 'config')
        assert hasattr(app, 'logger')
        assert hasattr(app, 'error_handler')
    
    def test_module_functions_exist(self):
        """Test de que todas las funciones principales existen"""
        required_functions = [
            'load_custom_css',
            'check_ollama_status', 
            'render_sidebar_config',
            'initialize_session_state',
            'main'
        ]
        
        for func_name in required_functions:
            assert hasattr(app, func_name), f"Función {func_name} no encontrada"
            assert callable(getattr(app, func_name)), f"Función {func_name} no es callable"


class TestAppConfiguration:
    """Pruebas para la configuración de la aplicación"""
    
    @patch('streamlit.set_page_config')
    def test_streamlit_page_config(self, mock_set_page_config):
        """Test de configuración de página de Streamlit"""
        # Reimportar el módulo para ejecutar la configuración
        import importlib
        importlib.reload(app)
        
        # Verificar que se configuró la página correctamente
        mock_set_page_config.assert_called_with(
            page_title="CogniChat - Sistema RAG Avanzado",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )


if __name__ == "__main__":
    pytest.main([__file__])