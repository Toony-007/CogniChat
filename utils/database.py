"""
Utilidades para manejo de base de datos
Gestión de conexiones y operaciones con ChromaDB
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger()

class DatabaseManager:
    """Gestor de base de datos para CogniChat"""
    
    def __init__(self):
        self.db_path = config.CACHE_DIR / "chroma_db"
        self.metadata_file = config.CACHE_DIR / "db_metadata.json"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Asegurar que los directorios necesarios existan"""
        self.db_path.mkdir(parents=True, exist_ok=True)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Obtener información de las colecciones"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error al leer metadata de DB: {e}")
            return {}
    
    def save_collection_info(self, info: Dict[str, Any]):
        """Guardar información de las colecciones"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error al guardar metadata de DB: {e}")
    
    def clear_database(self):
        """Limpiar toda la base de datos"""
        try:
            import shutil
            if self.db_path.exists():
                shutil.rmtree(self.db_path)
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            self._ensure_directories()
            logger.info("Base de datos limpiada exitosamente")
        except Exception as e:
            logger.error(f"Error al limpiar base de datos: {e}")
            raise
    
    def get_database_size(self) -> int:
        """Obtener el tamaño de la base de datos en bytes"""
        try:
            total_size = 0
            if self.db_path.exists():
                for file_path in self.db_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            logger.error(f"Error al calcular tamaño de DB: {e}")
            return 0

# Instancia global
db_manager = DatabaseManager()