"""
Manejo centralizado de errores y excepciones
"""

import traceback
import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any
from utils.logger import setup_logger

logger = setup_logger()

class ErrorHandler:
    """Manejador centralizado de errores"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info_messages = []
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "", 
                    user_message: Optional[str] = None,
                    show_in_ui: bool = True) -> Dict[str, Any]:
        """
        Manejar errores de manera centralizada
        
        Args:
            error: Excepción capturada
            context: Contexto donde ocurrió el error
            user_message: Mensaje personalizado para el usuario
            show_in_ui: Si mostrar el error en la interfaz
        
        Returns:
            Diccionario con información del error
        """
        error_info = {
            "timestamp": datetime.now(),
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "user_message": user_message or "Ha ocurrido un error inesperado"
        }
        
        # Registrar en logs
        logger.error(f"Error en {context}: {error_info['type']} - {error_info['message']}")
        logger.debug(f"Traceback completo: {error_info['traceback']}")
        
        # Almacenar para la pestaña de alertas
        self.errors.append(error_info)
        
        # Mostrar en UI si es necesario
        if show_in_ui:
            st.error(error_info['user_message'])
            with st.expander("Detalles técnicos"):
                st.code(f"Tipo: {error_info['type']}\nMensaje: {error_info['message']}\nContexto: {error_info['context']}")
        
        return error_info
    
    def handle_warning(self, 
                      message: str, 
                      context: str = "",
                      show_in_ui: bool = True) -> Dict[str, Any]:
        """
        Manejar advertencias
        
        Args:
            message: Mensaje de advertencia
            context: Contexto de la advertencia
            show_in_ui: Si mostrar en la interfaz
        
        Returns:
            Diccionario con información de la advertencia
        """
        warning_info = {
            "timestamp": datetime.now(),
            "message": message,
            "context": context,
            "type": "warning"
        }
        
        # Registrar en logs
        logger.warning(f"Advertencia en {context}: {message}")
        
        # Almacenar para la pestaña de alertas
        self.warnings.append(warning_info)
        
        # Mostrar en UI si es necesario
        if show_in_ui:
            st.warning(message)
        
        return warning_info
    
    def handle_info(self, 
                   message: str, 
                   context: str = "",
                   show_in_ui: bool = True) -> Dict[str, Any]:
        """
        Manejar mensajes informativos
        
        Args:
            message: Mensaje informativo
            context: Contexto del mensaje
            show_in_ui: Si mostrar en la interfaz
        
        Returns:
            Diccionario con información del mensaje
        """
        info = {
            "timestamp": datetime.now(),
            "message": message,
            "context": context,
            "type": "info"
        }
        
        # Registrar en logs
        logger.info(f"Info en {context}: {message}")
        
        # Almacenar para la pestaña de alertas
        self.info_messages.append(info)
        
        # Mostrar en UI si es necesario
        if show_in_ui:
            st.info(message)
        
        return info
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Obtener errores recientes"""
        return sorted(self.errors, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_recent_warnings(self, limit: int = 10) -> list:
        """Obtener advertencias recientes"""
        return sorted(self.warnings, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_recent_info(self, limit: int = 10) -> list:
        """Obtener mensajes informativos recientes"""
        return sorted(self.info_messages, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def clear_all(self):
        """Limpiar todos los mensajes almacenados"""
        self.errors.clear()
        self.warnings.clear()
        self.info_messages.clear()
        logger.info("Alertas limpiadas por el usuario")
    
    def get_error_stats(self) -> Dict[str, int]:
        """Obtener estadísticas de errores"""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "total_info": len(self.info_messages)
        }
