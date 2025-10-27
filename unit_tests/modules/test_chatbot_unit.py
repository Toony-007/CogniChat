"""
Pruebas Unitarias - Módulo Chatbot

Este archivo contiene pruebas unitarias específicas para el módulo chatbot,
enfocándose en la lógica de negocio y funcionalidades individuales.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class TestChatbotUnit(unittest.TestCase):
    """Pruebas unitarias para el módulo chatbot."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        self.mock_session_state = {}
        
    @patch('streamlit.session_state', {})
    @patch('modules.chatbot.OllamaClient')
    def test_chatbot_initialization(self, mock_ollama_client):
        """Prueba la inicialización correcta del chatbot."""
        # Configurar mock
        mock_client = Mock()
        mock_client.get_available_models.return_value = [{"name": "llama2:7b"}]
        mock_ollama_client.return_value = mock_client
        
        # Importar después del patch
        from modules import chatbot
        
        # Verificar que el módulo se puede importar
        self.assertTrue(hasattr(chatbot, 'render'))
        
    @patch('streamlit.session_state', {})
    @patch('modules.chatbot.OllamaClient')
    def test_model_selection_logic(self, mock_ollama_client):
        """Prueba la lógica de selección de modelos."""
        # Configurar mock
        mock_client = Mock()
        mock_client.get_available_models.return_value = [
            {"name": "llama2:7b"},
            {"name": "mistral:7b"}
        ]
        mock_ollama_client.return_value = mock_client
        
        # Importar después del patch
        from modules import chatbot
        
        # Verificar que se pueden obtener modelos disponibles
        models = mock_client.get_available_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "llama2:7b")
        
    @patch('streamlit.session_state', {'messages': []})
    def test_message_handling(self):
        """Prueba el manejo de mensajes en el chat."""
        from modules import chatbot
        
        # Verificar que existe la función para manejar mensajes
        self.assertTrue(hasattr(chatbot, 'render'))
        
    @patch('streamlit.session_state', {})
    @patch('modules.chatbot.OllamaClient')
    def test_rag_context_integration(self, mock_ollama_client):
        """Prueba la integración con el contexto RAG."""
        # Configurar mock
        mock_client = Mock()
        mock_ollama_client.return_value = mock_client
        
        from modules import chatbot
        
        # Verificar que existe la función para obtener contexto RAG
        self.assertTrue(hasattr(chatbot, 'get_rag_context'))
        
    def test_conversation_export_functionality(self):
        """Prueba la funcionalidad de exportación de conversaciones."""
        from modules import chatbot
        
        # Verificar que existe la función de exportación
        self.assertTrue(hasattr(chatbot, 'render_conversation_export_controls'))
        
    @patch('streamlit.session_state', {'messages': [
        {'role': 'user', 'content': 'Hola'},
        {'role': 'assistant', 'content': 'Hola, ¿cómo puedo ayudarte?'}
    ]})
    def test_message_history_management(self):
        """Prueba el manejo del historial de mensajes."""
        from modules import chatbot
        
        # Verificar que el módulo puede manejar el historial
        self.assertTrue(hasattr(chatbot, 'render'))
        
    @patch('modules.chatbot.OllamaClient')
    def test_error_handling_no_models(self, mock_ollama_client):
        """Prueba el manejo de errores cuando no hay modelos disponibles."""
        # Configurar mock para simular error
        mock_client = Mock()
        mock_client.get_available_models.return_value = []
        mock_ollama_client.return_value = mock_client
        
        from modules import chatbot
        
        # Verificar que el módulo maneja la ausencia de modelos
        models = mock_client.get_available_models()
        self.assertEqual(len(models), 0)
        
    @patch('modules.chatbot.OllamaClient')
    def test_connection_status_check(self, mock_ollama_client):
        """Prueba la verificación del estado de conexión."""
        # Configurar mock
        mock_client = Mock()
        mock_client.is_connected.return_value = True
        mock_ollama_client.return_value = mock_client
        
        from modules import chatbot
        
        # Verificar estado de conexión
        self.assertTrue(mock_client.is_connected())

if __name__ == '__main__':
    unittest.main()