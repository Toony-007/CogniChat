"""
Pruebas unitarias para OllamaClient
Cobertura completa de funcionalidades de conexión con Ollama
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import requests
from datetime import datetime, timedelta

# Configurar el path antes de importar
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TestOllamaClient(unittest.TestCase):
    """Pruebas para la clase OllamaClient"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        # Mock de configuración
        self.mock_config = Mock()
        self.mock_config.OLLAMA_BASE_URL = "http://localhost:11434"
        self.mock_config.OLLAMA_TIMEOUT = 30
        self.mock_config.MAX_RESPONSE_TOKENS = 2048
        
        # Mock de logger y error_handler
        self.mock_logger = Mock()
        self.mock_error_handler = Mock()
        
        # Patches para evitar importaciones circulares
        self.config_patch = patch('utils.ollama_client.config', self.mock_config)
        self.logger_patch = patch('utils.ollama_client.logger', self.mock_logger)
        self.error_handler_patch = patch('utils.ollama_client.error_handler', self.mock_error_handler)
        self.setup_logger_patch = patch('utils.ollama_client.setup_logger', return_value=self.mock_logger)
        self.error_handler_class_patch = patch('utils.ollama_client.ErrorHandler', return_value=self.mock_error_handler)
        
        # Iniciar patches
        self.config_patch.start()
        self.logger_patch.start()
        self.error_handler_patch.start()
        self.setup_logger_patch.start()
        self.error_handler_class_patch.start()
        
        # Importar después de configurar mocks
        from utils.ollama_client import OllamaClient
        self.client = OllamaClient()
    
    def tearDown(self):
        """Limpieza después de cada prueba"""
        self.config_patch.stop()
        self.logger_patch.stop()
        self.error_handler_patch.stop()
        self.setup_logger_patch.stop()
        self.error_handler_class_patch.stop()
    
    @patch('utils.ollama_client.requests.get')
    def test_is_available_success(self, mock_get):
        """Probar verificación exitosa de disponibilidad"""
        # Configurar mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Ejecutar
        result = self.client.is_available()
        
        # Verificar
        self.assertTrue(result)
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)
    
    @patch('utils.ollama_client.requests.get')
    def test_is_available_failure(self, mock_get):
        """Probar fallo en verificación de disponibilidad"""
        # Configurar mock para lanzar excepción
        mock_get.side_effect = requests.RequestException("Connection error")
        
        # Ejecutar
        result = self.client.is_available()
        
        # Verificar
        self.assertFalse(result)
        self.mock_logger.error.assert_called_once()
    
    @patch('utils.ollama_client.requests.get')
    def test_get_available_models_success(self, mock_get):
        """Probar obtención exitosa de modelos disponibles"""
        # Configurar mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {
                    'name': 'llama3.2:3b',
                    'size': 2048000000,
                    'modified_at': '2024-01-01T00:00:00Z',
                    'digest': 'abc123'
                },
                {
                    'name': 'nomic-embed-text',
                    'size': 1024000000,
                    'modified_at': '2024-01-01T00:00:00Z',
                    'digest': 'def456'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Ejecutar
        result = self.client.get_available_models()
        
        # Verificar
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'llama3.2:3b')
        self.assertEqual(result[1]['name'], 'nomic-embed-text')
        self.mock_logger.info.assert_called_once()
    
    @patch('utils.ollama_client.requests.get')
    def test_get_available_models_with_cache(self, mock_get):
        """Probar uso de caché en obtención de modelos"""
        # Configurar caché
        self.client._models_cache = [{'name': 'cached_model'}]
        self.client._models_cache_time = datetime.now()
        
        # Ejecutar
        result = self.client.get_available_models()
        
        # Verificar que usa caché y no hace request
        self.assertEqual(result, [{'name': 'cached_model'}])
        mock_get.assert_not_called()
    
    @patch('utils.ollama_client.requests.get')
    def test_get_available_models_force_refresh(self, mock_get):
        """Probar actualización forzada de modelos"""
        # Configurar caché
        self.client._models_cache = [{'name': 'cached_model'}]
        self.client._models_cache_time = datetime.now()
        
        # Configurar mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'models': [{'name': 'new_model'}]}
        mock_get.return_value = mock_response
        
        # Ejecutar con force_refresh
        result = self.client.get_available_models(force_refresh=True)
        
        # Verificar que hace request a pesar del caché
        self.assertEqual(result[0]['name'], 'new_model')
        mock_get.assert_called_once()
    
    @patch('utils.ollama_client.requests.post')
    def test_generate_response_success(self, mock_post):
        """Probar generación exitosa de respuesta"""
        # Configurar mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Esta es la respuesta generada'}
        mock_post.return_value = mock_response
        
        # Ejecutar
        result = self.client.generate_response(
            model="llama3.2:3b",
            prompt="¿Cuál es la capital de Colombia?",
            context="Información sobre geografía"
        )
        
        # Verificar
        self.assertEqual(result, 'Esta es la respuesta generada')
        mock_post.assert_called_once()
        
        # Verificar payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        self.assertEqual(payload['model'], 'llama3.2:3b')
        self.assertIn('Contexto: Información sobre geografía', payload['prompt'])
        self.assertIn('¿Cuál es la capital de Colombia?', payload['prompt'])
    
    @patch('utils.ollama_client.requests.post')
    def test_generate_response_with_deepseek_optimization(self, mock_post):
        """Probar optimizaciones específicas para DeepSeek"""
        # Configurar mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Respuesta optimizada'}
        mock_post.return_value = mock_response
        
        # Ejecutar con modelo DeepSeek
        result = self.client.generate_response(
            model="deepseek-coder:6.7b",
            prompt="Escribe código Python"
        )
        
        # Verificar
        self.assertEqual(result, 'Respuesta optimizada')
        
        # Verificar configuraciones específicas de DeepSeek
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        options = payload['options']
        self.assertEqual(options['temperature'], 0.7)
        self.assertEqual(options['top_p'], 0.9)
        self.assertEqual(options['num_ctx'], 8192)
    
    @patch('utils.ollama_client.requests.post')
    def test_generate_response_streaming(self, mock_post):
        """Probar generación de respuesta con streaming"""
        # Configurar mock para streaming
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Hola", "done": false}',
            b'{"response": " mundo", "done": false}',
            b'{"response": "!", "done": true}'
        ]
        mock_post.return_value = mock_response
        
        # Ejecutar
        result = self.client.generate_response(
            model="llama3.2:3b",
            prompt="Saluda",
            stream=True
        )
        
        # Verificar
        self.assertEqual(result, 'Hola mundo!')
    
    @patch('utils.ollama_client.requests.post')
    def test_generate_response_error(self, mock_post):
        """Probar manejo de errores en generación de respuesta"""
        # Configurar mock para error
        mock_post.side_effect = requests.RequestException("Connection error")
        
        # Ejecutar
        result = self.client.generate_response(
            model="llama3.2:3b",
            prompt="Test prompt"
        )
        
        # Verificar
        self.assertEqual(result, "Lo siento, ha ocurrido un error al generar la respuesta.")
        self.mock_error_handler.handle_error.assert_called_once()
    
    @patch('utils.ollama_client.requests.post')
    def test_generate_embeddings_success(self, mock_post):
        """Probar generación exitosa de embeddings"""
        # Configurar mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'embedding': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        mock_post.return_value = mock_response
        
        # Ejecutar
        result = self.client.generate_embeddings(
            model="nomic-embed-text",
            text="Texto para generar embedding"
        )
        
        # Verificar
        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4, 0.5])
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": "Texto para generar embedding"
            },
            timeout=None
        )
    
    @patch('utils.ollama_client.requests.post')
    def test_generate_embeddings_error(self, mock_post):
        """Probar manejo de errores en generación de embeddings"""
        # Configurar mock para error
        mock_post.side_effect = requests.RequestException("Connection error")
        
        # Ejecutar
        result = self.client.generate_embeddings(
            model="nomic-embed-text",
            text="Texto de prueba"
        )
        
        # Verificar
        self.assertEqual(result, [])
        self.mock_error_handler.handle_error.assert_called_once()
    
    @patch('utils.ollama_client.requests.post')
    def test_pull_model_success(self, mock_post):
        """Probar descarga exitosa de modelo"""
        # Configurar mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Ejecutar
        result = self.client.pull_model("llama3.2:3b")
        
        # Verificar
        self.assertTrue(result)
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/pull",
            json={"name": "llama3.2:3b"},
            timeout=300
        )
        self.mock_logger.info.assert_called_once()
    
    @patch('utils.ollama_client.requests.post')
    def test_pull_model_error(self, mock_post):
        """Probar manejo de errores en descarga de modelo"""
        # Configurar mock para error
        mock_post.side_effect = requests.RequestException("Download failed")
        
        # Ejecutar
        result = self.client.pull_model("llama3.2:3b")
        
        # Verificar
        self.assertFalse(result)
        self.mock_error_handler.handle_error.assert_called_once()
    
    def test_initialization(self):
        """Probar inicialización correcta del cliente"""
        self.assertEqual(self.client.base_url, "http://localhost:11434")
        self.assertEqual(self.client.timeout, 30)
        self.assertIsNone(self.client._models_cache)
        self.assertIsNone(self.client._models_cache_time)
        self.assertEqual(self.client._cache_duration, timedelta(minutes=5))


class TestOllamaClientUtilities(unittest.TestCase):
    """Pruebas para funciones utilitarias de OllamaClient"""
    
    def setUp(self):
        """Configuración inicial"""
        # Mock de configuración
        self.mock_config = Mock()
        self.mock_logger = Mock()
        
        # Patches
        self.config_patch = patch('utils.ollama_client.config', self.mock_config)
        self.logger_patch = patch('utils.ollama_client.logger', self.mock_logger)
        
        self.config_patch.start()
        self.logger_patch.start()
    
    def tearDown(self):
        """Limpieza"""
        self.config_patch.stop()
        self.logger_patch.stop()
    
    def test_get_default_models(self):
        """Probar obtención de modelos predefinidos"""
        from utils.ollama_client import get_default_models
        
        result = get_default_models()
        
        # Verificar estructura
        self.assertIn('chat', result)
        self.assertIn('embeddings', result)
        self.assertIsInstance(result['chat'], list)
        self.assertIsInstance(result['embeddings'], list)
        
        # Verificar contenido
        self.assertIn('llama3.2:3b', result['chat'])
        self.assertIn('nomic-embed-text', result['embeddings'])
    
    @patch('utils.ollama_client.OllamaClient')
    def test_get_available_or_default_models_with_available_client(self, mock_client_class):
        """Probar obtención de modelos con cliente disponible"""
        from utils.ollama_client import get_available_or_default_models
        
        # Configurar mock
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.get_available_models.return_value = [
            {'name': 'model1'}, {'name': 'model2'}
        ]
        
        # Ejecutar
        result = get_available_or_default_models(mock_client)
        
        # Verificar
        self.assertEqual(result, ['model1', 'model2'])
    
    @patch('utils.ollama_client.OllamaClient')
    def test_get_available_or_default_models_fallback(self, mock_client_class):
        """Probar fallback a modelos predefinidos"""
        from utils.ollama_client import get_available_or_default_models
        
        # Configurar mock para cliente no disponible
        mock_client = Mock()
        mock_client.is_available.return_value = False
        
        # Ejecutar
        result = get_available_or_default_models(mock_client)
        
        # Verificar que usa modelos predefinidos
        self.assertIn('llama3.2:3b', result)
        self.assertIn('nomic-embed-text', result)
        self.mock_logger.warning.assert_called_once()


if __name__ == '__main__':
    unittest.main()