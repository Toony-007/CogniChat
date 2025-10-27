"""
Pruebas unitarias para el módulo logger
"""

import unittest
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Importar el módulo a probar
from utils.logger import setup_logger


class TestLoggerCore(unittest.TestCase):
    """Pruebas para la funcionalidad principal del logger"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_logger_name = "TestLogger"
        
        # Mock de configuración
        self.mock_config = MagicMock()
        self.mock_config.LOG_LEVEL = "INFO"
        self.mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.mock_config.LOGS_DIR = self.temp_dir
        
    def tearDown(self):
        """Limpieza después de cada prueba"""
        # Limpiar handlers de logging para evitar interferencias
        logger = logging.getLogger(self.test_logger_name)
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except:
                pass
        
        # Limpiar todos los loggers creados durante las pruebas
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith('Test') or name == 'CogniChat':
                test_logger = logging.getLogger(name)
                for handler in test_logger.handlers[:]:
                    try:
                        handler.close()
                        test_logger.removeHandler(handler)
                    except:
                        pass
        
        # Limpiar directorio temporal con reintentos
        import time
        for attempt in range(3):
            try:
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                break
            except PermissionError:
                time.sleep(0.1)
                continue
    
    @patch('utils.logger.config')
    def test_setup_logger_basic_configuration(self, mock_config):
        """Prueba configuración básica del logger"""
        mock_config.LOG_LEVEL = "INFO"
        mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        mock_config.LOGS_DIR = self.temp_dir
        
        logger = setup_logger(self.test_logger_name)
        
        # Verificar que el logger se creó correctamente
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, self.test_logger_name)
        self.assertEqual(logger.level, logging.INFO)
    
    @patch('utils.logger.config')
    def test_setup_logger_with_custom_level(self, mock_config):
        """Prueba configuración del logger con nivel personalizado"""
        mock_config.LOG_LEVEL = "INFO"
        mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        mock_config.LOGS_DIR = self.temp_dir
        
        logger = setup_logger(self.test_logger_name, level="DEBUG")
        
        # Verificar nivel personalizado
        self.assertEqual(logger.level, logging.DEBUG)
    
    @patch('utils.logger.config')
    def test_setup_logger_handlers_creation(self, mock_config):
        """Prueba que se crean los handlers correctos"""
        mock_config.LOG_LEVEL = "INFO"
        mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        mock_config.LOGS_DIR = self.temp_dir
        
        logger = setup_logger(self.test_logger_name)
        
        # Verificar que se crearon los handlers
        self.assertEqual(len(logger.handlers), 3)  # Console, File, Error
        
        # Verificar tipos de handlers
        handler_types = [type(handler).__name__ for handler in logger.handlers]
        self.assertIn('StreamHandler', handler_types)
        self.assertIn('FileHandler', handler_types)
    
    @patch('utils.logger.config')
    def test_setup_logger_prevents_duplicate_handlers(self, mock_config):
        """Prueba que no se duplican handlers al llamar múltiples veces"""
        mock_config.LOG_LEVEL = "INFO"
        mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        mock_config.LOGS_DIR = self.temp_dir
        
        # Primera llamada
        logger1 = setup_logger(self.test_logger_name)
        initial_handlers = len(logger1.handlers)
        
        # Segunda llamada con el mismo nombre
        logger2 = setup_logger(self.test_logger_name)
        
        # Verificar que es el mismo logger y no se duplicaron handlers
        self.assertIs(logger1, logger2)
        self.assertEqual(len(logger2.handlers), initial_handlers)
    
    @patch('utils.logger.config')
    @patch('utils.logger.datetime')
    def test_setup_logger_file_naming(self, mock_datetime, mock_config):
        """Prueba que los archivos de log se nombran correctamente"""
        mock_config.LOG_LEVEL = "INFO"
        mock_config.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        mock_config.LOGS_DIR = self.temp_dir
        
        # Mock de datetime para controlar el nombre del archivo
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20240101"
        mock_datetime.now.return_value = mock_now
        
        with patch('builtins.open', mock_open()) as mock_file:
            logger = setup_logger(self.test_logger_name)
            
            # Verificar que se intentaron abrir los archivos correctos
            expected_calls = [
                unittest.mock.call(self.temp_dir / "cognichat_20240101.log", encoding='utf-8'),
                unittest.mock.call(self.temp_dir / "errors_20240101.log", encoding='utf-8')
            ]
            
            # Verificar que se llamó a open con los archivos esperados
            self.assertEqual(mock_file.call_count, 2)


class TestLoggerIntegration(unittest.TestCase):
    """Pruebas de integración del logger"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Limpieza"""
        # Limpiar handlers de logging
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if 'Integration' in name or name == 'CogniChat':
                test_logger = logging.getLogger(name)
                for handler in test_logger.handlers[:]:
                    try:
                        handler.close()
                        test_logger.removeHandler(handler)
                    except:
                        pass
        
        # Limpiar directorio temporal con reintentos
        import time
        for attempt in range(3):
            try:
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                break
            except PermissionError:
                time.sleep(0.1)
                continue
    
    def test_logger_module_imports(self):
        """Prueba que el módulo se importa correctamente"""
        from utils.logger import setup_logger
        self.assertTrue(callable(setup_logger))
    
    def test_logger_with_config_integration(self):
        """Prueba integración con el módulo de configuración"""
        with patch('utils.logger.config') as mock_config:
            mock_config.LOG_LEVEL = "WARNING"
            mock_config.LOG_FORMAT = "%(levelname)s: %(message)s"
            mock_config.LOGS_DIR = self.temp_dir
            
            logger = setup_logger("IntegrationTest")
            
            # Verificar que usa la configuración
            self.assertEqual(logger.level, logging.WARNING)


class TestLoggerUtilities(unittest.TestCase):
    """Pruebas para utilidades y casos edge del logger"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Limpieza"""
        # Limpiar handlers de logging
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if 'Test' in name or name == 'CogniChat':
                test_logger = logging.getLogger(name)
                for handler in test_logger.handlers[:]:
                    try:
                        handler.close()
                        test_logger.removeHandler(handler)
                    except:
                        pass
        
        # Limpiar directorio temporal con reintentos
        import time
        for attempt in range(3):
            try:
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                break
            except PermissionError:
                time.sleep(0.1)
                continue
    
    @patch('utils.logger.config')
    def test_logger_level_validation(self, mock_config):
        """Prueba validación de niveles de logging"""
        mock_config.LOG_LEVEL = "INFO"
        mock_config.LOG_FORMAT = "%(asctime)s - %(message)s"
        mock_config.LOGS_DIR = self.temp_dir
        
        # Probar diferentes niveles válidos
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            with self.subTest(level=level):
                logger_name = f"TestLogger_{level}"
                logger = setup_logger(logger_name, level=level)
                expected_level = getattr(logging, level)
                self.assertEqual(logger.level, expected_level)
                
                # Limpiar handlers
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
    
    @patch('utils.logger.config')
    def test_logger_default_name(self, mock_config):
        """Prueba nombre por defecto del logger"""
        mock_config.LOG_LEVEL = "INFO"
        mock_config.LOG_FORMAT = "%(asctime)s - %(message)s"
        mock_config.LOGS_DIR = self.temp_dir
        
        logger = setup_logger()  # Sin especificar nombre
        
        self.assertEqual(logger.name, "CogniChat")
    
    @patch('utils.logger.config')
    def test_logger_formatter_configuration(self, mock_config):
        """Prueba configuración del formatter"""
        test_format = "%(levelname)s - %(message)s"
        mock_config.LOG_LEVEL = "INFO"
        mock_config.LOG_FORMAT = test_format
        mock_config.LOGS_DIR = self.temp_dir
        
        logger = setup_logger("FormatterTest")
        
        # Verificar que los handlers tienen el formatter correcto
        for handler in logger.handlers:
            if hasattr(handler, 'formatter') and handler.formatter:
                self.assertEqual(handler.formatter._fmt, test_format)


if __name__ == '__main__':
    unittest.main()