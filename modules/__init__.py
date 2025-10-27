# Archivo de inicialización del paquete modules

# Importar todos los módulos disponibles
from . import document_upload
from . import chatbot
from . import alerts
from . import settings
from . import document_processor
from . import qualitative_analysis

# Hacer disponibles los módulos
__all__ = [
    'document_upload',
    'chatbot', 
    'alerts',
    'settings',
    'document_processor',
    'qualitative_analysis'
]