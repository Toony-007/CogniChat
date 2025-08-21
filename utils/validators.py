"""
Validadores para CogniChat
Funciones de validación para archivos, configuraciones y datos
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger()

class FileValidator:
    """Validador de archivos"""
    
    @staticmethod
    def is_valid_file_type(file_path: Path) -> bool:
        """Verificar si el tipo de archivo es válido"""
        try:
            suffix = file_path.suffix.lower()
            return suffix in config.SUPPORTED_FORMATS
        except Exception as e:
            logger.error(f"Error al validar tipo de archivo: {e}")
            return False
    
    @staticmethod
    def is_valid_file_size(file_path: Path, max_size_mb: int = None) -> bool:
        """Verificar si el tamaño del archivo es válido"""
        try:
            max_size = max_size_mb or config.MAX_FILE_SIZE_MB
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            return file_size_mb <= max_size
        except Exception as e:
            logger.error(f"Error al validar tamaño de archivo: {e}")
            return False
    
    @staticmethod
    def validate_file(file_path: Path) -> Tuple[bool, str]:
        """Validación completa de archivo"""
        try:
            # Verificar que el archivo existe
            if not file_path.exists():
                return False, "El archivo no existe"
            
            # Verificar que es un archivo (no directorio)
            if not file_path.is_file():
                return False, "La ruta no corresponde a un archivo"
            
            # Verificar tipo de archivo
            if not FileValidator.is_valid_file_type(file_path):
                return False, f"Tipo de archivo no soportado: {file_path.suffix}"
            
            # Verificar tamaño
            if not FileValidator.is_valid_file_size(file_path):
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                return False, f"Archivo muy grande: {file_size_mb:.1f}MB (máximo: {config.MAX_FILE_SIZE_MB}MB)"
            
            return True, "Archivo válido"
            
        except Exception as e:
            logger.error(f"Error en validación de archivo: {e}")
            return False, f"Error de validación: {str(e)}"

class ConfigValidator:
    """Validador de configuraciones"""
    
    @staticmethod
    def validate_chunk_size(chunk_size: int) -> Tuple[bool, str]:
        """Validar tamaño de chunk"""
        if not isinstance(chunk_size, int):
            return False, "El tamaño de chunk debe ser un número entero"
        
        if chunk_size < 100:
            return False, "El tamaño de chunk debe ser al menos 100 caracteres"
        
        if chunk_size > 10000:
            return False, "El tamaño de chunk no debe exceder 10,000 caracteres"
        
        return True, "Tamaño de chunk válido"
    
    @staticmethod
    def validate_chunk_overlap(chunk_overlap: int, chunk_size: int) -> Tuple[bool, str]:
        """Validar solapamiento de chunks"""
        if not isinstance(chunk_overlap, int):
            return False, "El solapamiento debe ser un número entero"
        
        if chunk_overlap < 0:
            return False, "El solapamiento no puede ser negativo"
        
        if chunk_overlap >= chunk_size:
            return False, "El solapamiento debe ser menor que el tamaño del chunk"
        
        if chunk_overlap > chunk_size * 0.5:
            return False, "El solapamiento no debería exceder el 50% del tamaño del chunk"
        
        return True, "Solapamiento válido"
    
    @staticmethod
    def validate_similarity_threshold(threshold: float) -> Tuple[bool, str]:
        """Validar umbral de similitud"""
        if not isinstance(threshold, (int, float)):
            return False, "El umbral debe ser un número"
        
        if threshold < 0.0 or threshold > 1.0:
            return False, "El umbral debe estar entre 0.0 y 1.0"
        
        return True, "Umbral válido"
    
    @staticmethod
    def validate_max_retrieval_docs(max_docs: int) -> Tuple[bool, str]:
        """Validar número máximo de documentos a recuperar"""
        if not isinstance(max_docs, int):
            return False, "El número máximo de documentos debe ser un entero"
        
        if max_docs < 1:
            return False, "Debe recuperar al menos 1 documento"
        
        if max_docs > 50:
            return False, "No se recomienda recuperar más de 50 documentos"
        
        return True, "Número máximo de documentos válido"

class TextValidator:
    """Validador de texto"""
    
    @staticmethod
    def is_valid_query(query: str) -> Tuple[bool, str]:
        """Validar consulta de texto"""
        if not isinstance(query, str):
            return False, "La consulta debe ser texto"
        
        query = query.strip()
        
        if not query:
            return False, "La consulta no puede estar vacía"
        
        if len(query) < 3:
            return False, "La consulta debe tener al menos 3 caracteres"
        
        if len(query) > 1000:
            return False, "La consulta no debe exceder 1000 caracteres"
        
        return True, "Consulta válida"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitizar nombre de archivo"""
        # Remover caracteres no válidos
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limitar longitud
        if len(sanitized) > 100:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:100-len(ext)] + ext
        
        return sanitized
    
    @staticmethod
    def is_valid_model_name(model_name: str) -> bool:
        """Validar nombre de modelo"""
        if not isinstance(model_name, str):
            return False
        
        # Patrón básico para nombres de modelos de Ollama
        pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$'
        return bool(re.match(pattern, model_name))

class SystemValidator:
    """Validador del sistema"""
    
    @staticmethod
    def validate_directories() -> List[str]:
        """Validar que todos los directorios necesarios existan"""
        issues = []
        
        required_dirs = [
            config.UPLOADS_DIR,
            config.CACHE_DIR,
            config.PROCESSED_DIR,
            config.LOGS_DIR
        ]
        
        for directory in required_dirs:
            try:
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Directorio creado: {directory}")
                elif not directory.is_dir():
                    issues.append(f"La ruta {directory} existe pero no es un directorio")
            except Exception as e:
                issues.append(f"Error con directorio {directory}: {str(e)}")
        
        return issues
    
    @staticmethod
    def validate_environment() -> Dict[str, Any]:
        """Validar el entorno del sistema"""
        validation_results = {
            'directories': SystemValidator.validate_directories(),
            'disk_space': SystemValidator._check_disk_space(),
            'permissions': SystemValidator._check_permissions()
        }
        
        return validation_results
    
    @staticmethod
    def _check_disk_space() -> Dict[str, Any]:
        """Verificar espacio en disco"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(config.BASE_DIR)
            
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            used_percent = (used / total) * 100
            
            return {
                'free_gb': round(free_gb, 2),
                'total_gb': round(total_gb, 2),
                'used_percent': round(used_percent, 2),
                'sufficient': free_gb > 1.0  # Al menos 1GB libre
            }
        except Exception as e:
            logger.error(f"Error al verificar espacio en disco: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _check_permissions() -> Dict[str, bool]:
        """Verificar permisos de escritura"""
        permissions = {}
        
        test_dirs = [
            config.UPLOADS_DIR,
            config.CACHE_DIR,
            config.PROCESSED_DIR,
            config.LOGS_DIR
        ]
        
        for directory in test_dirs:
            try:
                test_file = directory / '.test_write'
                test_file.write_text('test')
                test_file.unlink()
                permissions[str(directory)] = True
            except Exception:
                permissions[str(directory)] = False
        
        return permissions

# Instancias globales
file_validator = FileValidator()
config_validator = ConfigValidator()
text_validator = TextValidator()
system_validator = SystemValidator()