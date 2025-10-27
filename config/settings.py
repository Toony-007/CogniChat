"""
Configuración centralizada de la aplicación
"""

import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

@dataclass
class AppConfig:
    """Configuración principal de la aplicación"""
    
    # Rutas del proyecto
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CACHE_DIR: Path = DATA_DIR / "cache"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Configuración de Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    # Modelos disponibles (basados en los que tienes instalados)
    AVAILABLE_LLM_MODELS: Dict[str, Dict] = None
    AVAILABLE_EMBEDDING_MODELS: Dict[str, Dict] = None
    
    # Configuración por defecto
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "deepseek-r1:7b")
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text:latest")
    
    # Configuración RAG optimizada
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "300"))
    MAX_RETRIEVAL_DOCS: int = int(os.getenv("MAX_RETRIEVAL_DOCS", "15"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    
    # Configuración de generación de respuestas
    MAX_RESPONSE_TOKENS: int = int(os.getenv("MAX_RESPONSE_TOKENS", "3000"))
    
    # Configuración de la interfaz
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    SUPPORTED_FORMATS: List[str] = None
    
    # Configuración de logs
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configuración de trazabilidad
    ENABLE_CHUNK_LOGGING: bool = os.getenv("ENABLE_CHUNK_LOGGING", "true").lower() == "true"
    ENABLE_DEBUG_MODE: bool = os.getenv("ENABLE_DEBUG_MODE", "true").lower() == "true"
    ENABLE_HISTORY_TRACKING: bool = os.getenv("ENABLE_HISTORY_TRACKING", "true").lower() == "true"
    
    def __post_init__(self):
        """Inicialización posterior"""
        self._setup_directories()
        self._setup_models()
        self._setup_supported_formats()
    
    def _setup_directories(self):
        """Crear directorios necesarios"""
        for directory in [self.DATA_DIR, self.UPLOADS_DIR, self.PROCESSED_DIR, 
                         self.CACHE_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_models(self):
        """Configurar modelos disponibles"""
        self.AVAILABLE_LLM_MODELS = {
            "llama3.1:8b": {
                "name": "Llama 3.1 8B",
                "description": "Modelo balanceado con buen rendimiento",
                "context_length": 8192,
                "recommended_use": "Uso general, análisis de documentos"
            },
            "deepseek-r1:7b": {
                "name": "DeepSeek R1 7B",
                "description": "Modelo con excelente razonamiento",
                "context_length": 8192,
                "recommended_use": "Análisis profundo, razonamiento complejo"
            }
        }
        
        self.AVAILABLE_EMBEDDING_MODELS = {
            "nomic-embed-text:latest": {
                "name": "Nomic Embed Text",
                "description": "Modelo de embeddings multilingüe de alta calidad",
                "dimensions": 768,
                "max_tokens": 8192,
                "languages": ["es", "en", "fr", "de", "it", "pt"]
            },
            "all-minilm:latest": {
                "name": "All-MiniLM L6 v2",
                "description": "Modelo eficiente y rápido",
                "dimensions": 384,
                "max_tokens": 256,
                "languages": ["en", "es", "fr", "de", "it"]
            }
        }
    
    def _setup_supported_formats(self):
        """Configurar formatos de archivo soportados"""
        self.SUPPORTED_FORMATS = [
            ".txt", ".md", ".pdf", ".docx", ".doc", 
            ".xlsx", ".xls", ".csv", ".pptx", ".ppt",
            ".html", ".htm", ".json", ".xml"
        ]
    
    def get_config(self):
        """Obtener configuración actual como diccionario"""
        return {
            'llm_model': self.DEFAULT_LLM_MODEL,
            'embedding_model': self.DEFAULT_EMBEDDING_MODEL,
            'chunk_size': self.CHUNK_SIZE,
            'chunk_overlap': self.CHUNK_OVERLAP,
            'max_retrieval_docs': self.MAX_RETRIEVAL_DOCS,
            'similarity_threshold': self.SIMILARITY_THRESHOLD,
            'max_response_tokens': self.MAX_RESPONSE_TOKENS,
            'ollama_url': self.OLLAMA_BASE_URL,
            'ollama_timeout': self.OLLAMA_TIMEOUT,
            'log_level': self.LOG_LEVEL,
            'max_file_size': self.MAX_FILE_SIZE_MB,
            'supported_formats': self.SUPPORTED_FORMATS,
            'enable_chunk_logging': self.ENABLE_CHUNK_LOGGING,
            'enable_debug_mode': self.ENABLE_DEBUG_MODE,
            'enable_history_tracking': self.ENABLE_HISTORY_TRACKING
        }

# Instancia global de configuración
config = AppConfig()