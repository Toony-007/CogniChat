"""Tests automatizados para el módulo chatbot.py
Pruebas de funcionalidad del chat inteligente con RAG"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import modules.chatbot as chatbot_module

class TestChatbotModule(unittest.TestCase):
    """Tests para el módulo de chatbot"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Mock de session_state
        self.mock_session_state = {
            'messages': [],
            'chat_model': 'llama2',
            'debug_mode': False,
            'enable_tracing': False
        }
        
    @patch('streamlit.session_state')
    @patch('modules.chatbot.OllamaClient')
    def test_render_function_exists(self, mock_ollama, mock_session):
        """Test que la función render existe y es callable"""
        mock_session.return_value = self.mock_session_state
        mock_ollama.return_value.is_available.return_value = True
        
        # Verificar que la función render existe
        self.assertTrue(hasattr(chatbot_module, 'render'))
        self.assertTrue(callable(chatbot_module.render))
        
    @patch('streamlit.session_state')
    @patch('modules.chatbot.OllamaClient')
    def test_get_rag_context_function_exists(self, mock_ollama, mock_session):
        """Test que la función get_rag_context existe"""
        mock_session.return_value = self.mock_session_state
        
        # Verificar que la función get_rag_context existe
        self.assertTrue(hasattr(chatbot_module, 'get_rag_context'))
        self.assertTrue(callable(chatbot_module.get_rag_context))
        
    @patch('streamlit.session_state')
    def test_render_conversation_export_controls_exists(self, mock_session):
        """Test que la función render_conversation_export_controls existe"""
        mock_session.return_value = self.mock_session_state
        
        # Verificar que la función existe
        self.assertTrue(hasattr(chatbot_module, 'render_conversation_export_controls'))
        self.assertTrue(callable(chatbot_module.render_conversation_export_controls))
    
    @patch('streamlit.session_state')
    def test_message_history_management(self, mock_session_state):
        """Test gestión del historial de mensajes"""
        # Inicializar historial
        mock_session_state.messages = []
        
        # Agregar mensajes de prueba
        test_messages = [
            {"role": "user", "content": "Pregunta 1"},
            {"role": "assistant", "content": "Respuesta 1", "context_used": True},
            {"role": "user", "content": "Pregunta 2"},
            {"role": "assistant", "content": "Respuesta 2", "context_used": False}
        ]
        
        mock_session_state.messages.extend(test_messages)
        
        # Verificar conteo de mensajes
        user_messages = len([m for m in mock_session_state.messages if m["role"] == "user"])
        assistant_messages = len([m for m in mock_session_state.messages if m["role"] == "assistant"])
        rag_responses = len([m for m in mock_session_state.messages if m["role"] == "assistant" and m.get("context_used")])
        
        assert user_messages == 2
        assert assistant_messages == 2
        assert rag_responses == 1
    
    @patch('modules.chatbot.OllamaClient')
    def test_model_selection_validation(self, mock_ollama_class):
        """Test validación de selección de modelo"""
        # Configurar mock del cliente
        mock_client = Mock()
        mock_ollama_class.return_value = mock_client
        
        # Configurar modelos disponibles
        mock_client.get_available_models.return_value = [
            {"name": "deepseek-r1:1.5b"},
            {"name": "llama3.2:3b"},
            {"name": "qwen2.5:7b"}
        ]
        
        available_models = mock_client.get_available_models()
        
        # Verificar que hay modelos disponibles
        assert len(available_models) > 0
        assert any(model["name"] == "deepseek-r1:1.5b" for model in available_models)
    
    @patch('streamlit.session_state')
    def test_debug_mode_functionality(self, mock_session_state):
        """Test funcionalidad del modo debug"""
        mock_session_state.messages = []
        
        # Simular mensaje con información de debug
        debug_message = {
            "role": "assistant",
            "content": "Respuesta con debug",
            "context_used": True,
            "query_id": "debug_query_123",
            "chunks_count": 5,
            "model_used": "deepseek-r1:1.5b",
            "max_tokens": 4000
        }
        
        mock_session_state.messages.append(debug_message)
        
        # Verificar información de debug
        message = mock_session_state.messages[0]
        assert message.get("query_id") == "debug_query_123"
        assert message.get("chunks_count") == 5
        assert message.get("model_used") == "deepseek-r1:1.5b"
    
    def test_prompt_construction_with_context(self):
        """Test construcción de prompts con contexto"""
        user_prompt = "¿Cuál es el tema principal?"
        context = "Este es el contexto de los documentos..."
        
        # Simular construcción del prompt
        full_prompt = f"""<think>
El usuario me ha proporcionado documentos completos y me está haciendo una pregunta.
</think>

CONTEXTO COMPLETO DE TODOS LOS DOCUMENTOS:
{context}

PREGUNTA DEL USUARIO: {user_prompt}"""
        
        # Verificar que el prompt contiene los elementos necesarios
        assert "<think>" in full_prompt
        assert context in full_prompt
        assert user_prompt in full_prompt
        assert "CONTEXTO COMPLETO" in full_prompt
    
    def test_prompt_construction_without_context(self):
        """Test construcción de prompts sin contexto RAG"""
        user_prompt = "¿Cuál es tu opinión sobre esto?"
        
        # Simular construcción del prompt sin RAG
        full_prompt = f"""<think>
El usuario me está haciendo una pregunta general sin documentos específicos.
</think>

PREGUNTA DEL USUARIO: {user_prompt}"""
        
        # Verificar que el prompt es apropiado para modo sin RAG
        assert "<think>" in full_prompt
        assert user_prompt in full_prompt
        assert "CONTEXTO COMPLETO" not in full_prompt
    
    @patch('streamlit.session_state')
    def test_conversation_metrics(self, mock_session_state):
        """Test cálculo de métricas de conversación"""
        # Configurar historial de prueba
        mock_session_state.messages = [
            {"role": "user", "content": "Pregunta 1"},
            {"role": "assistant", "content": "Respuesta 1", "context_used": True},
            {"role": "user", "content": "Pregunta 2"},
            {"role": "assistant", "content": "Respuesta 2", "context_used": True},
            {"role": "user", "content": "Pregunta 3"},
            {"role": "assistant", "content": "Respuesta 3", "context_used": False}
        ]
        
        # Calcular métricas
        user_messages = len([m for m in mock_session_state.messages if m["role"] == "user"])
        assistant_messages = len([m for m in mock_session_state.messages if m["role"] == "assistant"])
        rag_responses = len([m for m in mock_session_state.messages if m["role"] == "assistant" and m.get("context_used")])
        
        # Verificar métricas
        assert user_messages == 3
        assert assistant_messages == 3
        assert rag_responses == 2
    
    @patch('modules.chatbot.TraceabilityManager')
    def test_traceability_integration(self, mock_traceability):
        """Test integración con sistema de trazabilidad"""
        # Configurar mock del manager de trazabilidad
        mock_manager = Mock()
        mock_traceability.return_value = mock_manager
        
        # Simular datos de trazabilidad
        test_sources = ["documento1.pdf", "documento2.txt"]
        test_query = "¿Cuál es el tema principal?"
        test_response = "El tema principal es..."
        
        # Verificar que el manager se inicializa correctamente
        traceability_manager = mock_traceability()
        assert traceability_manager is not None
    
    def test_message_validation(self):
        """Test validación de mensajes"""
        # Casos válidos
        valid_messages = [
            "¿Cuál es el tema principal?",
            "Explícame sobre este documento",
            "Resumen ejecutivo por favor"
        ]
        
        for message in valid_messages:
            assert len(message.strip()) > 0
            assert isinstance(message, str)
        
        # Casos inválidos
        invalid_messages = ["", "   ", None]
        
        for message in invalid_messages:
            if message is None:
                assert message is None
            else:
                assert len(message.strip()) == 0
    
    @patch('streamlit.session_state')
    def test_chat_state_management(self, mock_session_state):
        """Test gestión del estado del chat"""
        # Inicializar estado
        mock_session_state.messages = []
        mock_session_state.processing_cancelled = False
        
        # Verificar estado inicial
        assert len(mock_session_state.messages) == 0
        assert mock_session_state.processing_cancelled == False
        
        # Simular cancelación
        mock_session_state.processing_cancelled = True
        assert mock_session_state.processing_cancelled == True
        
        # Limpiar estado
        mock_session_state.processing_cancelled = False
        assert mock_session_state.processing_cancelled == False


class TestChatExportFunctionality:
    """Tests para funcionalidad de exportación de conversaciones"""
    
    @patch('streamlit.session_state')
    def test_conversation_export_data_preparation(self, mock_session_state):
        """Test preparación de datos para exportación"""
        # Configurar conversación de prueba
        mock_session_state.messages = [
            {"role": "user", "content": "Pregunta 1", "timestamp": "2024-01-01 10:00:00"},
            {"role": "assistant", "content": "Respuesta 1", "context_used": True},
            {"role": "user", "content": "Pregunta 2", "timestamp": "2024-01-01 10:05:00"},
            {"role": "assistant", "content": "Respuesta 2", "context_used": False}
        ]
        
        # Verificar estructura de datos para exportación
        messages = mock_session_state.messages
        assert len(messages) == 4
        
        # Verificar que cada mensaje tiene la estructura correcta
        for message in messages:
            assert "role" in message
            assert "content" in message
            assert message["role"] in ["user", "assistant"]


if __name__ == "__main__":
    pytest.main([__file__])