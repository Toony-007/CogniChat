"""
Pruebas Unitarias - Módulo Error Handler

Este archivo contiene pruebas unitarias específicas para el módulo error_handler,
enfocándose en el manejo centralizado de errores, advertencias y mensajes informativos.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.error_handler import ErrorHandler

class TestErrorHandlerUnit(unittest.TestCase):
    """Pruebas unitarias para el módulo error_handler."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        self.error_handler = ErrorHandler()
        
    def test_error_handler_initialization(self):
        """Prueba la inicialización correcta del error handler."""
        self.assertIsInstance(self.error_handler.errors, list)
        self.assertIsInstance(self.error_handler.warnings, list)
        self.assertIsInstance(self.error_handler.info_messages, list)
        self.assertEqual(len(self.error_handler.errors), 0)
        self.assertEqual(len(self.error_handler.warnings), 0)
        self.assertEqual(len(self.error_handler.info_messages), 0)
    
    @patch('streamlit.error')
    @patch('streamlit.expander')
    @patch('utils.error_handler.logger')
    def test_handle_error_basic(self, mock_logger, mock_expander, mock_st_error):
        """Prueba el manejo básico de errores."""
        # Configurar mocks
        mock_expander_obj = Mock()
        mock_expander.return_value.__enter__ = Mock(return_value=mock_expander_obj)
        mock_expander.return_value.__exit__ = Mock(return_value=None)
        
        # Crear excepción de prueba
        test_exception = ValueError("Error de prueba")
        context = "test_context"
        user_message = "Mensaje personalizado"
        
        # Ejecutar método
        result = self.error_handler.handle_error(
            test_exception, 
            context=context, 
            user_message=user_message,
            show_in_ui=True
        )
        
        # Verificar resultado
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'ValueError')
        self.assertEqual(result['message'], 'Error de prueba')
        self.assertEqual(result['context'], context)
        self.assertEqual(result['user_message'], user_message)
        self.assertIsInstance(result['timestamp'], datetime)
        
        # Verificar que se almacenó el error
        self.assertEqual(len(self.error_handler.errors), 1)
        
        # Verificar llamadas a logger
        mock_logger.error.assert_called_once()
        mock_logger.debug.assert_called_once()
        
        # Verificar llamadas a Streamlit
        mock_st_error.assert_called_once_with(user_message)
        mock_expander.assert_called_once()
    
    @patch('streamlit.error')
    @patch('utils.error_handler.logger')
    def test_handle_error_no_ui(self, mock_logger, mock_st_error):
        """Prueba el manejo de errores sin mostrar en UI."""
        test_exception = RuntimeError("Error sin UI")
        
        result = self.error_handler.handle_error(
            test_exception, 
            context="test", 
            show_in_ui=False
        )
        
        # Verificar que no se llamó a Streamlit
        mock_st_error.assert_not_called()
        
        # Verificar que se registró en logs
        mock_logger.error.assert_called_once()
        
        # Verificar que se almacenó
        self.assertEqual(len(self.error_handler.errors), 1)
        self.assertEqual(result['user_message'], "Ha ocurrido un error inesperado")
    
    @patch('streamlit.warning')
    @patch('utils.error_handler.logger')
    def test_handle_warning(self, mock_logger, mock_st_warning):
        """Prueba el manejo de advertencias."""
        message = "Advertencia de prueba"
        context = "test_warning"
        
        result = self.error_handler.handle_warning(
            message, 
            context=context, 
            show_in_ui=True
        )
        
        # Verificar resultado
        self.assertIsInstance(result, dict)
        self.assertEqual(result['message'], message)
        self.assertEqual(result['context'], context)
        self.assertEqual(result['type'], 'warning')
        self.assertIsInstance(result['timestamp'], datetime)
        
        # Verificar almacenamiento
        self.assertEqual(len(self.error_handler.warnings), 1)
        
        # Verificar llamadas
        mock_logger.warning.assert_called_once()
        mock_st_warning.assert_called_once_with(message)
    
    @patch('streamlit.warning')
    @patch('utils.error_handler.logger')
    def test_handle_warning_no_ui(self, mock_logger, mock_st_warning):
        """Prueba el manejo de advertencias sin mostrar en UI."""
        message = "Advertencia sin UI"
        
        result = self.error_handler.handle_warning(
            message, 
            show_in_ui=False
        )
        
        # Verificar que no se mostró en UI
        mock_st_warning.assert_not_called()
        
        # Verificar que se registró
        mock_logger.warning.assert_called_once()
        self.assertEqual(len(self.error_handler.warnings), 1)
    
    @patch('streamlit.info')
    @patch('utils.error_handler.logger')
    def test_handle_info(self, mock_logger, mock_st_info):
        """Prueba el manejo de mensajes informativos."""
        message = "Mensaje informativo"
        context = "test_info"
        
        result = self.error_handler.handle_info(
            message, 
            context=context, 
            show_in_ui=True
        )
        
        # Verificar resultado
        self.assertIsInstance(result, dict)
        self.assertEqual(result['message'], message)
        self.assertEqual(result['context'], context)
        self.assertEqual(result['type'], 'info')
        self.assertIsInstance(result['timestamp'], datetime)
        
        # Verificar almacenamiento
        self.assertEqual(len(self.error_handler.info_messages), 1)
        
        # Verificar llamadas
        mock_logger.info.assert_called_once()
        mock_st_info.assert_called_once_with(message)
    
    def test_get_recent_errors(self):
        """Prueba la obtención de errores recientes."""
        # Agregar varios errores
        for i in range(5):
            self.error_handler.errors.append({
                'timestamp': datetime.now(),
                'message': f'Error {i}',
                'type': 'TestError'
            })
        
        # Obtener errores recientes
        recent = self.error_handler.get_recent_errors(limit=3)
        
        self.assertEqual(len(recent), 3)
        self.assertIsInstance(recent, list)
    
    def test_get_recent_warnings(self):
        """Prueba la obtención de advertencias recientes."""
        # Agregar varias advertencias
        for i in range(3):
            self.error_handler.warnings.append({
                'timestamp': datetime.now(),
                'message': f'Warning {i}',
                'type': 'warning'
            })
        
        recent = self.error_handler.get_recent_warnings(limit=2)
        
        self.assertEqual(len(recent), 2)
        self.assertIsInstance(recent, list)
    
    def test_get_recent_info(self):
        """Prueba la obtención de mensajes informativos recientes."""
        # Agregar varios mensajes
        for i in range(4):
            self.error_handler.info_messages.append({
                'timestamp': datetime.now(),
                'message': f'Info {i}',
                'type': 'info'
            })
        
        recent = self.error_handler.get_recent_info(limit=2)
        
        self.assertEqual(len(recent), 2)
        self.assertIsInstance(recent, list)
    
    @patch('utils.error_handler.logger')
    def test_clear_all(self, mock_logger):
        """Prueba la limpieza de todos los mensajes."""
        # Agregar algunos mensajes
        self.error_handler.errors.append({'test': 'error'})
        self.error_handler.warnings.append({'test': 'warning'})
        self.error_handler.info_messages.append({'test': 'info'})
        
        # Verificar que hay mensajes
        self.assertEqual(len(self.error_handler.errors), 1)
        self.assertEqual(len(self.error_handler.warnings), 1)
        self.assertEqual(len(self.error_handler.info_messages), 1)
        
        # Limpiar
        self.error_handler.clear_all()
        
        # Verificar que se limpiaron
        self.assertEqual(len(self.error_handler.errors), 0)
        self.assertEqual(len(self.error_handler.warnings), 0)
        self.assertEqual(len(self.error_handler.info_messages), 0)
        
        # Verificar log
        mock_logger.info.assert_called_once_with("Alertas limpiadas por el usuario")
    
    def test_get_error_stats(self):
        """Prueba la obtención de estadísticas de errores."""
        # Agregar algunos mensajes
        self.error_handler.errors.extend([{'test': 'error1'}, {'test': 'error2'}])
        self.error_handler.warnings.append({'test': 'warning'})
        self.error_handler.info_messages.extend([{'test': 'info1'}, {'test': 'info2'}, {'test': 'info3'}])
        
        stats = self.error_handler.get_error_stats()
        
        expected_stats = {
            "total_errors": 2,
            "total_warnings": 1,
            "total_info": 3
        }
        
        self.assertEqual(stats, expected_stats)
    
    def test_multiple_errors_storage(self):
        """Prueba el almacenamiento de múltiples errores."""
        # Agregar múltiples errores
        errors = [
            ValueError("Error 1"),
            RuntimeError("Error 2"),
            TypeError("Error 3")
        ]
        
        for i, error in enumerate(errors):
            with patch('streamlit.error'), patch('streamlit.expander'), patch('utils.error_handler.logger'):
                self.error_handler.handle_error(error, context=f"context_{i}", show_in_ui=False)
        
        # Verificar que se almacenaron todos
        self.assertEqual(len(self.error_handler.errors), 3)
        
        # Verificar que los tipos son correctos
        stored_types = [err['type'] for err in self.error_handler.errors]
        expected_types = ['ValueError', 'RuntimeError', 'TypeError']
        self.assertEqual(stored_types, expected_types)
    
    def test_error_with_traceback(self):
        """Prueba que se capture el traceback correctamente."""
        try:
            # Generar un error con traceback
            raise ValueError("Error con traceback")
        except ValueError as e:
            with patch('streamlit.error'), patch('streamlit.expander'), patch('utils.error_handler.logger'):
                result = self.error_handler.handle_error(e, show_in_ui=False)
        
        # Verificar que se capturó el traceback
        self.assertIn('traceback', result)
        self.assertIsInstance(result['traceback'], str)
        self.assertIn('ValueError', result['traceback'])
        self.assertIn('Error con traceback', result['traceback'])

if __name__ == '__main__':
    unittest.main()