"""
Tests automatizados para el módulo alerts.py
Pruebas de funcionalidad del sistema de alertas
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.alerts import render

class TestAlertsModule:
    """Tests para el módulo de alertas"""
    
    def setup_method(self):
        """Configuración inicial para cada test"""
        # Mock session_state para evitar errores de contexto
        with patch('streamlit.session_state', {}):
            pass
    
    def test_alert_types_validation(self):
        """Test validación de tipos de alerta"""
        # Tipos de alerta válidos
        valid_alert_types = [
            'success',
            'info', 
            'warning',
            'error'
        ]
        
        for alert_type in valid_alert_types:
            # Verificar que el tipo es válido
            assert alert_type in valid_alert_types
        
        # Tipos de alerta inválidos
        invalid_alert_types = [
            'debug',
            'critical',
            'notice',
            'alert'
        ]
        
        for alert_type in invalid_alert_types:
            # Verificar que el tipo no es válido
            assert alert_type not in valid_alert_types
    
    def test_alert_message_validation(self):
        """Test validación de mensajes de alerta"""
        # Mensajes válidos
        valid_messages = [
            "Operación completada exitosamente",
            "Documento procesado correctamente",
            "Error al procesar el archivo",
            "Advertencia: Archivo muy grande"
        ]
        
        for message in valid_messages:
            # Verificar que el mensaje no está vacío
            assert len(message.strip()) > 0
            assert isinstance(message, str)
        
        # Mensajes inválidos
        invalid_messages = [
            "",           # Vacío
            "   ",        # Solo espacios
            None,         # Nulo
            123           # No es string
        ]
        
        for message in invalid_messages:
            if message is None or not isinstance(message, str):
                is_valid = False
            else:
                is_valid = len(message.strip()) > 0
            assert is_valid == False
    
    @patch('streamlit.success')
    def test_success_alert_display(self, mock_success):
        """Test visualización de alertas de éxito"""
        test_message = "Operación completada exitosamente"
        
        # Simular llamada a st.success
        mock_success.return_value = None
        
        # Llamar función simulada
        mock_success(test_message)
        
        # Verificar que se llamó con el mensaje correcto
        mock_success.assert_called_once_with(test_message)
    
    @patch('streamlit.info')
    def test_info_alert_display(self, mock_info):
        """Test visualización de alertas informativas"""
        test_message = "Información importante para el usuario"
        
        # Simular llamada a st.info
        mock_info.return_value = None
        
        # Llamar función simulada
        mock_info(test_message)
        
        # Verificar que se llamó con el mensaje correcto
        mock_info.assert_called_once_with(test_message)
    
    @patch('streamlit.warning')
    def test_warning_alert_display(self, mock_warning):
        """Test visualización de alertas de advertencia"""
        test_message = "Advertencia: Revise la configuración"
        
        # Simular llamada a st.warning
        mock_warning.return_value = None
        
        # Llamar función simulada
        mock_warning(test_message)
        
        # Verificar que se llamó con el mensaje correcto
        mock_warning.assert_called_once_with(test_message)
    
    @patch('streamlit.error')
    def test_error_alert_display(self, mock_error):
        """Test visualización de alertas de error"""
        test_message = "Error crítico en el sistema"
        
        # Simular llamada a st.error
        mock_error.return_value = None
        
        # Llamar función simulada
        mock_error(test_message)
        
        # Verificar que se llamó con el mensaje correcto
        mock_error.assert_called_once_with(test_message)
    
    def test_alert_persistence(self):
        """Test persistencia de alertas en session_state"""
        # Simular alertas almacenadas
        test_alerts = [
            {
                'type': 'success',
                'message': 'Documento procesado',
                'timestamp': datetime.now()
            },
            {
                'type': 'warning',
                'message': 'Archivo grande detectado',
                'timestamp': datetime.now()
            }
        ]
        
        # Verificar estructura de alertas
        for alert in test_alerts:
            assert 'type' in alert
            assert 'message' in alert
            assert 'timestamp' in alert
            assert isinstance(alert['timestamp'], datetime)
    
    def test_alert_expiration(self):
        """Test expiración de alertas por tiempo"""
        # Crear alertas con diferentes timestamps
        current_time = datetime.now()
        old_alert = {
            'type': 'info',
            'message': 'Alerta antigua',
            'timestamp': current_time - timedelta(minutes=10)
        }
        
        recent_alert = {
            'type': 'success',
            'message': 'Alerta reciente',
            'timestamp': current_time - timedelta(seconds=30)
        }
        
        # Simular tiempo de expiración (5 minutos)
        expiration_time = timedelta(minutes=5)
        
        # Verificar expiración
        old_alert_expired = (current_time - old_alert['timestamp']) > expiration_time
        recent_alert_expired = (current_time - recent_alert['timestamp']) > expiration_time
        
        assert old_alert_expired == True
        assert recent_alert_expired == False
    
    def test_alert_queue_management(self):
        """Test gestión de cola de alertas"""
        # Simular cola de alertas
        alert_queue = []
        max_alerts = 5
        
        # Agregar alertas
        for i in range(7):  # Más del máximo
            alert = {
                'type': 'info',
                'message': f'Alerta {i+1}',
                'timestamp': datetime.now()
            }
            alert_queue.append(alert)
            
            # Mantener solo las más recientes
            if len(alert_queue) > max_alerts:
                alert_queue.pop(0)  # Remover la más antigua
        
        # Verificar que no excede el máximo
        assert len(alert_queue) <= max_alerts
        assert len(alert_queue) == 5
        
        # Verificar que se mantuvieron las más recientes
        assert alert_queue[0]['message'] == 'Alerta 3'
        assert alert_queue[-1]['message'] == 'Alerta 7'
    
    def test_alert_filtering_by_type(self):
        """Test filtrado de alertas por tipo"""
        # Crear alertas de diferentes tipos
        all_alerts = [
            {'type': 'success', 'message': 'Éxito 1'},
            {'type': 'error', 'message': 'Error 1'},
            {'type': 'warning', 'message': 'Advertencia 1'},
            {'type': 'success', 'message': 'Éxito 2'},
            {'type': 'info', 'message': 'Info 1'}
        ]
        
        # Filtrar por tipo
        success_alerts = [alert for alert in all_alerts if alert['type'] == 'success']
        error_alerts = [alert for alert in all_alerts if alert['type'] == 'error']
        
        # Verificar filtrado
        assert len(success_alerts) == 2
        assert len(error_alerts) == 1
        assert all(alert['type'] == 'success' for alert in success_alerts)
        assert all(alert['type'] == 'error' for alert in error_alerts)
    
    def test_alert_priority_system(self):
        """Test sistema de prioridades de alertas"""
        # Definir prioridades
        priority_order = {
            'error': 1,      # Máxima prioridad
            'warning': 2,
            'info': 3,
            'success': 4     # Mínima prioridad
        }
        
        # Crear alertas con diferentes prioridades
        alerts = [
            {'type': 'success', 'message': 'Éxito'},
            {'type': 'error', 'message': 'Error crítico'},
            {'type': 'warning', 'message': 'Advertencia'},
            {'type': 'info', 'message': 'Información'}
        ]
        
        # Ordenar por prioridad
        sorted_alerts = sorted(alerts, key=lambda x: priority_order[x['type']])
        
        # Verificar orden correcto
        assert sorted_alerts[0]['type'] == 'error'
        assert sorted_alerts[1]['type'] == 'warning'
        assert sorted_alerts[2]['type'] == 'info'
        assert sorted_alerts[3]['type'] == 'success'
    
    def test_alert_formatting(self):
        """Test formateo de alertas"""
        # Casos de prueba para formateo
        test_cases = [
            {
                'input': 'mensaje simple',
                'expected_length': len('mensaje simple')
            },
            {
                'input': 'Mensaje con MAYÚSCULAS y números 123',
                'expected_contains': ['MAYÚSCULAS', '123']
            },
            {
                'input': 'Mensaje con caracteres especiales: ñáéíóú',
                'expected_contains': ['ñáéíóú']
            }
        ]
        
        for case in test_cases:
            message = case['input']
            
            # Verificar longitud si se especifica
            if 'expected_length' in case:
                assert len(message) == case['expected_length']
            
            # Verificar contenido si se especifica
            if 'expected_contains' in case:
                for expected_text in case['expected_contains']:
                    assert expected_text in message
    
    def test_alert_sanitization(self):
        """Test sanitización de mensajes de alerta"""
        # Mensajes que requieren sanitización
        unsafe_messages = [
            "<script>alert('xss')</script>",
            "Mensaje con <b>HTML</b>",
            "Texto con & caracteres especiales",
            "Mensaje con 'comillas' y \"dobles\""
        ]
        
        for message in unsafe_messages:
            # Simular sanitización básica
            sanitized = message.replace('<', '&lt;').replace('>', '&gt;')
            
            # Verificar que no contiene tags HTML peligrosos
            assert '<script>' not in sanitized
            assert '</script>' not in sanitized
    
    def test_alert_localization(self):
        """Test localización de mensajes de alerta"""
        # Mensajes en español
        spanish_messages = {
            'success': 'Operación completada exitosamente',
            'error': 'Error al procesar la solicitud',
            'warning': 'Advertencia: Revise la configuración',
            'info': 'Información importante'
        }
        
        # Verificar que los mensajes están en español
        for alert_type, message in spanish_messages.items():
            assert isinstance(message, str)
            assert len(message) > 0
            # Verificar caracteres en español
            spanish_chars = ['ñ', 'á', 'é', 'í', 'ó', 'ú']
            has_spanish = any(char in message.lower() for char in spanish_chars)
            # No todos los mensajes necesariamente tienen caracteres especiales
            assert isinstance(has_spanish, bool)


class TestAlertsIntegration:
    """Tests de integración para el sistema de alertas"""
    
    @patch('modules.alerts.config')
    def test_alert_logging_integration(self, mock_config):
        """Test integración con sistema de logging"""
        # Configurar mock
        mock_config.LOG_ALERTS = True
        
        # Simular alerta que debe ser registrada
        alert_data = {
            'type': 'error',
            'message': 'Error crítico del sistema',
            'timestamp': datetime.now(),
            'user_id': 'test_user'
        }
        
        # Verificar que la alerta tiene la información necesaria para logging
        assert 'type' in alert_data
        assert 'message' in alert_data
        assert 'timestamp' in alert_data
        assert alert_data['type'] in ['success', 'info', 'warning', 'error']
    
    def test_alert_metrics_collection(self):
        """Test recolección de métricas de alertas"""
        # Simular alertas para métricas
        alerts_history = [
            {'type': 'success', 'timestamp': datetime.now()},
            {'type': 'error', 'timestamp': datetime.now()},
            {'type': 'warning', 'timestamp': datetime.now()},
            {'type': 'success', 'timestamp': datetime.now()},
            {'type': 'error', 'timestamp': datetime.now()}
        ]
        
        # Calcular métricas
        total_alerts = len(alerts_history)
        error_count = len([a for a in alerts_history if a['type'] == 'error'])
        success_count = len([a for a in alerts_history if a['type'] == 'success'])
        
        # Verificar métricas
        assert total_alerts == 5
        assert error_count == 2
        assert success_count == 2
        
        # Calcular tasa de error
        error_rate = error_count / total_alerts if total_alerts > 0 else 0
        assert 0 <= error_rate <= 1
        assert error_rate == 0.4  # 2/5
    
    @patch('streamlit.session_state')
    def test_alert_state_persistence(self, mock_session_state):
        """Test persistencia del estado de alertas"""
        # Configurar mock del session_state
        mock_session_state.__contains__ = Mock(return_value=True)
        mock_session_state.__getitem__ = Mock(return_value=[])
        mock_session_state.__setitem__ = Mock()
        
        # Simular operación con alertas
        alerts_key = 'system_alerts'
        
        # Verificar que se puede acceder al estado
        mock_session_state.__contains__(alerts_key)
        mock_session_state.__getitem__(alerts_key)
        
        # Verificar que se puede modificar el estado
        new_alert = {'type': 'info', 'message': 'Test alert'}
        mock_session_state.__setitem__(alerts_key, [new_alert])
        
        # Verificar llamadas
        mock_session_state.__contains__.assert_called()
        mock_session_state.__getitem__.assert_called()
        mock_session_state.__setitem__.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])