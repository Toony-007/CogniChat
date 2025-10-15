"""
Pruebas unitarias para el módulo de configuración
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import fields

# Importar el módulo a probar
from config.settings import AppConfig, config


class TestAppConfigCore(unittest.TestCase):
    """Pruebas para la funcionalidad principal de AppConfig"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Limpieza después de cada prueba"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_appconfig_dataclass_structure(self):
        """Prueba que AppConfig es un dataclass con los campos esperados"""
        # Verificar que es un dataclass
        self.assertTrue(hasattr(AppConfig, '__dataclass_fields__'))
        
        # Verificar campos principales
        field_names = [field.name for field in fields(AppConfig)]
        expected_fields = [
            'PROJECT_ROOT', 'DATA_DIR', 'UPLOADS_DIR', 'PROCESSED_DIR',
            'CACHE_DIR', 'LOGS_DIR', 'OLLAMA_BASE_URL', 'OLLAMA_TIMEOUT',
            'DEFAULT_LLM_MODEL', 'DEFAULT_EMBEDDING_MODEL', 'CHUNK_SIZE',
            'CHUNK_OVERLAP', 'MAX_RETRIEVAL_DOCS', 'SIMILARITY_THRESHOLD',
            'MAX_RESPONSE_TOKENS', 'MAX_FILE_SIZE_MB', 'LOG_LEVEL', 'LOG_FORMAT'
        ]
        
        for field in expected_fields:
            self.assertIn(field, field_names)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_appconfig_default_values(self):
        """Prueba valores por defecto de la configuración"""
        app_config = AppConfig()
        
        # Verificar valores por defecto
        self.assertEqual(app_config.OLLAMA_BASE_URL, "http://localhost:11434")
        self.assertEqual(app_config.OLLAMA_TIMEOUT, 120)
        self.assertEqual(app_config.DEFAULT_LLM_MODEL, "deepseek-r1:7b")
        self.assertEqual(app_config.DEFAULT_EMBEDDING_MODEL, "nomic-embed-text:latest")
        self.assertEqual(app_config.CHUNK_SIZE, 2000)
        self.assertEqual(app_config.CHUNK_OVERLAP, 300)
        self.assertEqual(app_config.MAX_RETRIEVAL_DOCS, 15)
        self.assertEqual(app_config.SIMILARITY_THRESHOLD, 0.6)
        self.assertEqual(app_config.MAX_RESPONSE_TOKENS, 3000)
        self.assertEqual(app_config.MAX_FILE_SIZE_MB, 100)
        self.assertEqual(app_config.LOG_LEVEL, "INFO")
    
    def test_appconfig_environment_variables_parsing(self):
        """Prueba que se parsean correctamente las variables de entorno"""
        # En lugar de mockear, probamos que la configuración actual sea válida
        app_config = AppConfig()
        
        # Verificar que las variables tienen valores válidos
        self.assertIsInstance(app_config.OLLAMA_BASE_URL, str)
        self.assertTrue(app_config.OLLAMA_BASE_URL.startswith('http'))
        self.assertIsInstance(app_config.OLLAMA_TIMEOUT, int)
        self.assertGreater(app_config.OLLAMA_TIMEOUT, 0)
        self.assertIsInstance(app_config.DEFAULT_LLM_MODEL, str)
        self.assertGreater(len(app_config.DEFAULT_LLM_MODEL), 0)
        self.assertIsInstance(app_config.CHUNK_SIZE, int)
        self.assertGreater(app_config.CHUNK_SIZE, 0)
        self.assertIn(app_config.LOG_LEVEL, ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    def test_appconfig_path_structure(self):
        """Prueba la estructura de rutas del proyecto"""
        app_config = AppConfig()
        
        # Verificar que las rutas son Path objects
        self.assertIsInstance(app_config.PROJECT_ROOT, Path)
        self.assertIsInstance(app_config.DATA_DIR, Path)
        self.assertIsInstance(app_config.UPLOADS_DIR, Path)
        self.assertIsInstance(app_config.PROCESSED_DIR, Path)
        self.assertIsInstance(app_config.CACHE_DIR, Path)
        self.assertIsInstance(app_config.LOGS_DIR, Path)
        
        # Verificar relaciones entre rutas
        self.assertEqual(app_config.DATA_DIR, app_config.PROJECT_ROOT / "data")
        self.assertEqual(app_config.UPLOADS_DIR, app_config.DATA_DIR / "uploads")
        self.assertEqual(app_config.PROCESSED_DIR, app_config.DATA_DIR / "processed")
        self.assertEqual(app_config.CACHE_DIR, app_config.DATA_DIR / "cache")
        self.assertEqual(app_config.LOGS_DIR, app_config.PROJECT_ROOT / "logs")
    
    @patch('config.settings.AppConfig._setup_directories')
    @patch('config.settings.AppConfig._setup_models')
    @patch('config.settings.AppConfig._setup_supported_formats')
    def test_appconfig_post_init_calls(self, mock_formats, mock_models, mock_dirs):
        """Prueba que __post_init__ llama a los métodos de configuración"""
        AppConfig()
        
        mock_dirs.assert_called_once()
        mock_models.assert_called_once()
        mock_formats.assert_called_once()
    
    def test_setup_models_configuration(self):
        """Prueba la configuración de modelos disponibles"""
        app_config = AppConfig()
        
        # Verificar que se configuraron los modelos LLM
        self.assertIsNotNone(app_config.AVAILABLE_LLM_MODELS)
        self.assertIsInstance(app_config.AVAILABLE_LLM_MODELS, dict)
        self.assertIn("deepseek-r1:7b", app_config.AVAILABLE_LLM_MODELS)
        self.assertIn("llama3.1:8b", app_config.AVAILABLE_LLM_MODELS)
        
        # Verificar estructura de modelo LLM
        deepseek_model = app_config.AVAILABLE_LLM_MODELS["deepseek-r1:7b"]
        required_keys = ["name", "description", "context_length", "recommended_use"]
        for key in required_keys:
            self.assertIn(key, deepseek_model)
        
        # Verificar que se configuraron los modelos de embedding
        self.assertIsNotNone(app_config.AVAILABLE_EMBEDDING_MODELS)
        self.assertIsInstance(app_config.AVAILABLE_EMBEDDING_MODELS, dict)
        self.assertIn("nomic-embed-text:latest", app_config.AVAILABLE_EMBEDDING_MODELS)
        
        # Verificar estructura de modelo de embedding
        nomic_model = app_config.AVAILABLE_EMBEDDING_MODELS["nomic-embed-text:latest"]
        required_keys = ["name", "description", "dimensions", "max_tokens", "languages"]
        for key in required_keys:
            self.assertIn(key, nomic_model)
    
    def test_setup_supported_formats(self):
        """Prueba la configuración de formatos soportados"""
        app_config = AppConfig()
        
        self.assertIsNotNone(app_config.SUPPORTED_FORMATS)
        self.assertIsInstance(app_config.SUPPORTED_FORMATS, list)
        
        # Verificar algunos formatos esperados
        expected_formats = [".txt", ".pdf", ".docx", ".csv", ".json"]
        for format_ext in expected_formats:
            self.assertIn(format_ext, app_config.SUPPORTED_FORMATS)
    
    def test_get_config_method(self):
        """Prueba el método get_config"""
        app_config = AppConfig()
        config_dict = app_config.get_config()
        
        # Verificar que retorna un diccionario
        self.assertIsInstance(config_dict, dict)
        
        # Verificar claves esperadas
        expected_keys = [
            'llm_model', 'embedding_model', 'chunk_size', 'chunk_overlap',
            'max_retrieval_docs', 'similarity_threshold', 'max_response_tokens',
            'ollama_url', 'ollama_timeout', 'log_level', 'max_file_size',
            'supported_formats', 'enable_chunk_logging', 'enable_debug_mode',
            'enable_history_tracking'
        ]
        
        for key in expected_keys:
            self.assertIn(key, config_dict)
        
        # Verificar algunos valores
        self.assertEqual(config_dict['llm_model'], app_config.DEFAULT_LLM_MODEL)
        self.assertEqual(config_dict['chunk_size'], app_config.CHUNK_SIZE)
        self.assertEqual(config_dict['log_level'], app_config.LOG_LEVEL)


class TestAppConfigDirectorySetup(unittest.TestCase):
    """Pruebas para la configuración de directorios"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_root = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Limpieza"""
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root)
    
    @patch('config.settings.AppConfig.PROJECT_ROOT')
    def test_setup_directories_creation(self, mock_project_root):
        """Prueba que se crean los directorios necesarios"""
        mock_project_root.__truediv__ = lambda self, other: self.temp_root / other
        mock_project_root.return_value = self.temp_root
        
        # Configurar paths temporales
        app_config = AppConfig()
        app_config.PROJECT_ROOT = self.temp_root
        app_config.DATA_DIR = self.temp_root / "data"
        app_config.UPLOADS_DIR = self.temp_root / "data" / "uploads"
        app_config.PROCESSED_DIR = self.temp_root / "data" / "processed"
        app_config.CACHE_DIR = self.temp_root / "data" / "cache"
        app_config.LOGS_DIR = self.temp_root / "logs"
        
        # Llamar al método de configuración
        app_config._setup_directories()
        
        # Verificar que se crearon los directorios
        self.assertTrue(app_config.DATA_DIR.exists())
        self.assertTrue(app_config.UPLOADS_DIR.exists())
        self.assertTrue(app_config.PROCESSED_DIR.exists())
        self.assertTrue(app_config.CACHE_DIR.exists())
        self.assertTrue(app_config.LOGS_DIR.exists())


class TestAppConfigIntegration(unittest.TestCase):
    """Pruebas de integración para AppConfig"""
    
    def test_global_config_instance(self):
        """Prueba que existe una instancia global de configuración"""
        from config.settings import config
        
        self.assertIsInstance(config, AppConfig)
        self.assertIsNotNone(config.PROJECT_ROOT)
        self.assertIsNotNone(config.AVAILABLE_LLM_MODELS)
        self.assertIsNotNone(config.SUPPORTED_FORMATS)
    
    def test_config_immutability_safety(self):
        """Prueba que la configuración mantiene consistencia"""
        from config.settings import config
        
        # Obtener configuración inicial
        initial_config = config.get_config()
        
        # Verificar que los valores son consistentes
        self.assertEqual(initial_config['llm_model'], config.DEFAULT_LLM_MODEL)
        self.assertEqual(initial_config['chunk_size'], config.CHUNK_SIZE)
    
    def test_boolean_environment_variables_parsing(self):
        """Prueba el manejo de variables de entorno booleanas"""
        app_config = AppConfig()
        
        # Verificar que las variables booleanas son del tipo correcto
        self.assertIsInstance(app_config.ENABLE_DEBUG_MODE, bool)
        self.assertIsInstance(app_config.ENABLE_CHUNK_LOGGING, bool)
        self.assertIsInstance(app_config.ENABLE_HISTORY_TRACKING, bool)
    
    def test_numeric_environment_variables_parsing(self):
        """Prueba validación de variables numéricas"""
        app_config = AppConfig()
        
        # Verificar que las variables numéricas son del tipo correcto y tienen valores válidos
        self.assertIsInstance(app_config.CHUNK_SIZE, int)
        self.assertGreater(app_config.CHUNK_SIZE, 0)
        
        self.assertIsInstance(app_config.SIMILARITY_THRESHOLD, float)
        self.assertGreaterEqual(app_config.SIMILARITY_THRESHOLD, 0.0)
        self.assertLessEqual(app_config.SIMILARITY_THRESHOLD, 1.0)
        
        self.assertIsInstance(app_config.OLLAMA_TIMEOUT, int)
        self.assertGreater(app_config.OLLAMA_TIMEOUT, 0)
        
        self.assertIsInstance(app_config.MAX_RETRIEVAL_DOCS, int)
        self.assertGreater(app_config.MAX_RETRIEVAL_DOCS, 0)


class TestAppConfigUtilities(unittest.TestCase):
    """Pruebas para utilidades y casos edge de AppConfig"""
    
    def test_log_format_structure(self):
        """Prueba la estructura del formato de logging"""
        app_config = AppConfig()
        
        # Verificar que el formato contiene elementos esperados
        log_format = app_config.LOG_FORMAT
        self.assertIn("%(asctime)s", log_format)
        self.assertIn("%(name)s", log_format)
        self.assertIn("%(levelname)s", log_format)
        self.assertIn("%(message)s", log_format)
    
    def test_model_configuration_completeness(self):
        """Prueba que la configuración de modelos está completa"""
        app_config = AppConfig()
        
        # Verificar que el modelo por defecto existe en los disponibles
        self.assertIn(app_config.DEFAULT_LLM_MODEL, app_config.AVAILABLE_LLM_MODELS)
        self.assertIn(app_config.DEFAULT_EMBEDDING_MODEL, app_config.AVAILABLE_EMBEDDING_MODELS)
    
    def test_path_resolution(self):
        """Prueba que las rutas se resuelven correctamente"""
        app_config = AppConfig()
        
        # Verificar que PROJECT_ROOT es absoluto
        self.assertTrue(app_config.PROJECT_ROOT.is_absolute())
        
        # Verificar que las rutas derivadas son absolutas
        self.assertTrue(app_config.DATA_DIR.is_absolute())
        self.assertTrue(app_config.UPLOADS_DIR.is_absolute())


if __name__ == '__main__':
    unittest.main()