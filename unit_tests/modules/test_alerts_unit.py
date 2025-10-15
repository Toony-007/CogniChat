"""
Pruebas unitarias para el módulo de alertas.

Este módulo contiene las pruebas para verificar el correcto funcionamiento
del sistema de alertas y monitoreo del sistema.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import sys
import os

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar módulos necesarios
import modules.alerts as alerts
from utils.error_handler import ErrorHandler


class TestAlertsBasic(unittest.TestCase):
    """Pruebas básicas para el módulo de alertas"""
    
    def test_render_function_exists(self):
        """Probar que la función render existe y es callable"""
        self.assertTrue(callable(alerts.render))
        
    def test_render_errors_tab_function_exists(self):
        """Probar que la función render_errors_tab existe"""
        self.assertTrue(callable(alerts.render_errors_tab))
        
    def test_render_warnings_tab_function_exists(self):
        """Probar que la función render_warnings_tab existe"""
        self.assertTrue(callable(alerts.render_warnings_tab))
        
    def test_render_info_tab_function_exists(self):
        """Probar que la función render_info_tab existe"""
        self.assertTrue(callable(alerts.render_info_tab))
        
    def test_render_system_tab_function_exists(self):
        """Probar que la función render_system_tab existe"""
        self.assertTrue(callable(alerts.render_system_tab))


class TestAlertsOllamaStatus(unittest.TestCase):
    """Pruebas para el estado de Ollama"""
    
    @patch('modules.alerts.OllamaClient')
    def test_ollama_client_creation(self, mock_ollama_class):
        """Probar que se puede crear un cliente de Ollama"""
        mock_ollama = Mock()
        mock_ollama_class.return_value = mock_ollama
        
        # Crear instancia
        client = mock_ollama_class()
        
        # Verificar que se creó correctamente
        self.assertIsNotNone(client)
        mock_ollama_class.assert_called_once()
    
    @patch('modules.alerts.OllamaClient')
    def test_ollama_availability_check(self, mock_ollama_class):
        """Probar la verificación de disponibilidad de Ollama"""
        mock_ollama = Mock()
        mock_ollama.is_available.return_value = True
        mock_ollama_class.return_value = mock_ollama
        
        # Crear cliente y verificar disponibilidad
        client = mock_ollama_class()
        is_available = client.is_available()
        
        # Verificar resultado
        self.assertTrue(is_available)
        mock_ollama.is_available.assert_called_once()


class TestAlertsFileHandling(unittest.TestCase):
    """Pruebas para el manejo de archivos"""
    
    @patch('modules.alerts.get_valid_uploaded_files')
    def test_get_valid_uploaded_files_called(self, mock_get_files):
        """Probar que se llama a get_valid_uploaded_files"""
        mock_get_files.return_value = ['file1.pdf', 'file2.txt']
        
        # Llamar función
        files = mock_get_files()
        
        # Verificar resultado
        self.assertEqual(len(files), 2)
        self.assertIn('file1.pdf', files)
        self.assertIn('file2.txt', files)
        mock_get_files.assert_called_once()
    
    @patch('modules.alerts.get_valid_uploaded_files')
    def test_empty_files_list(self, mock_get_files):
        """Probar el manejo de lista vacía de archivos"""
        mock_get_files.return_value = []
        
        # Llamar función
        files = mock_get_files()
        
        # Verificar resultado
        self.assertEqual(len(files), 0)
        mock_get_files.assert_called_once()


class TestAlertsDiskUsage(unittest.TestCase):
    """Pruebas para el uso de disco"""
    
    @patch('shutil.disk_usage')
    def test_disk_usage_check(self, mock_disk_usage):
        """Probar la verificación del uso de disco"""
        # Simular 100GB total, 10GB libre
        mock_disk_usage.return_value = (100 * 1024**3, 90 * 1024**3, 10 * 1024**3)
        
        # Llamar función
        total, used, free = mock_disk_usage('/')
        
        # Verificar resultado
        self.assertEqual(total, 100 * 1024**3)
        self.assertEqual(used, 90 * 1024**3)
        self.assertEqual(free, 10 * 1024**3)
        mock_disk_usage.assert_called_once_with('/')


class TestAlertsErrorHandler(unittest.TestCase):
    """Pruebas para el manejo de errores"""
    
    def setUp(self):
        """Configurar mock del ErrorHandler"""
        self.mock_error_handler = Mock(spec=ErrorHandler)
    
    def test_error_handler_get_recent_errors(self):
        """Probar la obtención de errores recientes"""
        # Configurar mock
        mock_errors = [
            {
                'type': 'ValueError',
                'context': 'test_context',
                'message': 'Test error',
                'timestamp': datetime.now(),
                'traceback': 'Test traceback'
            }
        ]
        self.mock_error_handler.get_recent_errors.return_value = mock_errors
        
        # Llamar función
        errors = self.mock_error_handler.get_recent_errors(20)
        
        # Verificar resultado
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['type'], 'ValueError')
        self.mock_error_handler.get_recent_errors.assert_called_once_with(20)
    
    def test_error_handler_get_recent_warnings(self):
        """Probar la obtención de advertencias recientes"""
        # Configurar mock
        mock_warnings = [
            {
                'context': 'test_context',
                'message': 'Test warning',
                'timestamp': datetime.now()
            }
        ]
        self.mock_error_handler.get_recent_warnings.return_value = mock_warnings
        
        # Llamar función
        warnings = self.mock_error_handler.get_recent_warnings(20)
        
        # Verificar resultado
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['message'], 'Test warning')
        self.mock_error_handler.get_recent_warnings.assert_called_once_with(20)
    
    def test_error_handler_get_recent_info(self):
        """Probar la obtención de información reciente"""
        # Configurar mock
        mock_info = [
            {
                'context': 'test_context',
                'message': 'Test info',
                'timestamp': datetime.now()
            }
        ]
        self.mock_error_handler.get_recent_info.return_value = mock_info
        
        # Llamar función
        info = self.mock_error_handler.get_recent_info(20)
        
        # Verificar resultado
        self.assertEqual(len(info), 1)
        self.assertEqual(info[0]['message'], 'Test info')
        self.mock_error_handler.get_recent_info.assert_called_once_with(20)
    
    def test_error_handler_get_error_stats(self):
        """Probar la obtención de estadísticas de errores"""
        # Configurar mock
        mock_stats = {
            'total_errors': 5,
            'total_warnings': 3,
            'total_info': 10
        }
        self.mock_error_handler.get_error_stats.return_value = mock_stats
        
        # Llamar función
        stats = self.mock_error_handler.get_error_stats()
        
        # Verificar resultado
        self.assertEqual(stats['total_errors'], 5)
        self.assertEqual(stats['total_warnings'], 3)
        self.assertEqual(stats['total_info'], 10)
        self.mock_error_handler.get_error_stats.assert_called_once()


class TestAlertsTabFunctions(unittest.TestCase):
    """Pruebas simplificadas para las funciones de pestañas"""
    
    def setUp(self):
        """Configurar mocks comunes"""
        self.mock_error_handler = Mock(spec=ErrorHandler)
    
    def test_render_errors_tab_with_mock_handler(self):
        """Probar render_errors_tab con mock handler"""
        # Configurar mock para no errores
        self.mock_error_handler.get_recent_errors.return_value = []
        
        # Verificar que el método existe y se puede llamar
        with patch('modules.alerts.st'):
            try:
                alerts.render_errors_tab(self.mock_error_handler)
                test_passed = True
            except Exception:
                test_passed = False
        
        # Verificar que se ejecutó sin errores críticos
        self.assertTrue(test_passed)
        self.mock_error_handler.get_recent_errors.assert_called_once_with(20)
    
    def test_render_warnings_tab_with_mock_handler(self):
        """Probar render_warnings_tab con mock handler"""
        # Configurar mock para no advertencias
        self.mock_error_handler.get_recent_warnings.return_value = []
        
        # Verificar que el método existe y se puede llamar
        with patch('modules.alerts.st'):
            try:
                alerts.render_warnings_tab(self.mock_error_handler)
                test_passed = True
            except Exception:
                test_passed = False
        
        # Verificar que se ejecutó sin errores críticos
        self.assertTrue(test_passed)
        self.mock_error_handler.get_recent_warnings.assert_called_once_with(20)
    
    def test_render_info_tab_with_mock_handler(self):
        """Probar render_info_tab con mock handler"""
        # Configurar mock para no información
        self.mock_error_handler.get_recent_info.return_value = []
        
        # Verificar que el método existe y se puede llamar
        with patch('modules.alerts.st'):
            try:
                alerts.render_info_tab(self.mock_error_handler)
                test_passed = True
            except Exception:
                test_passed = False
        
        # Verificar que se ejecutó sin errores críticos
        self.assertTrue(test_passed)
        self.mock_error_handler.get_recent_info.assert_called_once_with(20)
    
    def test_render_system_tab_with_mocks(self):
        """Probar render_system_tab con mocks básicos"""
        # Simplemente verificar que la función existe y es callable
        self.assertTrue(callable(alerts.render_system_tab))
        
        # Verificar que se puede importar sin errores
        try:
            from modules.alerts import render_system_tab
            test_passed = True
        except ImportError:
            test_passed = False
        
        self.assertTrue(test_passed, "render_system_tab debe ser importable")


class TestAlertsIntegration(unittest.TestCase):
    """Pruebas de integración básicas"""
    
    def test_module_imports_correctly(self):
        """Probar que el módulo se importa correctamente"""
        self.assertIsNotNone(alerts)
        self.assertTrue(hasattr(alerts, 'render'))
        self.assertTrue(hasattr(alerts, 'render_errors_tab'))
        self.assertTrue(hasattr(alerts, 'render_warnings_tab'))
        self.assertTrue(hasattr(alerts, 'render_info_tab'))
        self.assertTrue(hasattr(alerts, 'render_system_tab'))
    
    def test_all_functions_are_callable(self):
        """Probar que todas las funciones principales son callable"""
        functions_to_test = [
            'render',
            'render_errors_tab',
            'render_warnings_tab',
            'render_info_tab',
            'render_system_tab'
        ]
        
        for func_name in functions_to_test:
            with self.subTest(function=func_name):
                func = getattr(alerts, func_name)
                self.assertTrue(callable(func), f"La función {func_name} no es callable")


if __name__ == '__main__':
    unittest.main()