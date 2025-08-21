"""
Cliente para interactuar con Ollama
"""

import requests
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from config.settings import config
from utils.error_handler import ErrorHandler
from utils.logger import setup_logger

logger = setup_logger("OllamaClient")
error_handler = ErrorHandler()

class OllamaClient:
    """Cliente para interactuar con Ollama"""
    
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.timeout = config.OLLAMA_TIMEOUT
        
        # Caché para modelos disponibles
        self._models_cache = None
        self._models_cache_time = None
        self._cache_duration = timedelta(minutes=5)  # Caché válido por 5 minutos
    
    def is_available(self) -> bool:
        """Verificar si Ollama está disponible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama no disponible: {e}")
            return False
    
    def get_available_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Obtener lista de modelos disponibles con caché
        
        Args:
            force_refresh: Forzar actualización del caché
        
        Returns:
            Lista de modelos disponibles
        """
        # Verificar si el caché es válido
        if (not force_refresh and 
            self._models_cache is not None and 
            self._models_cache_time is not None and 
            datetime.now() - self._models_cache_time < self._cache_duration):
            return self._models_cache
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            
            models_data = response.json()
            models = []
            
            for model in models_data.get('models', []):
                model_name = model.get('name', '')
                models.append({
                    'name': model_name,
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at', ''),
                    'digest': model.get('digest', '')
                })
            
            # Actualizar caché
            self._models_cache = models
            self._models_cache_time = datetime.now()
            
            if not force_refresh:  # Solo mostrar log si no es un refresh forzado
                logger.info(f"Modelos disponibles obtenidos: {len(models)}")
            
            return models
            
        except Exception as e:
            error_handler.handle_error(e, "Error al obtener modelos de Ollama")
            # Si hay error pero tenemos caché, devolverlo
            if self._models_cache is not None:
                logger.warning("Usando caché de modelos debido a error en la consulta")
                return self._models_cache
            return []
    
    def generate_response(self, 
                         model: str, 
                         prompt: str, 
                         context: Optional[str] = None,
                         stream: bool = False,
                         max_tokens: Optional[int] = None) -> str:
        """
        Generar respuesta usando un modelo LLM
        
        Args:
            model: Nombre del modelo
            prompt: Prompt para el modelo
            context: Contexto adicional
            stream: Si usar streaming
            max_tokens: Máximo número de tokens a generar
        
        Returns:
            Respuesta generada
        """
        try:
            full_prompt = prompt
            if context:
                full_prompt = f"Contexto: {context}\n\nPregunta: {prompt}"
            
            # Configurar parámetros del modelo
            options = {}
            
            # Configurar max tokens si se especifica o usar configuración por defecto
            if max_tokens:
                options["num_predict"] = max_tokens
            elif hasattr(config, 'MAX_RESPONSE_TOKENS'):
                options["num_predict"] = config.MAX_RESPONSE_TOKENS
            
            # Configuraciones optimizadas para DeepSeek
            if "deepseek" in model.lower():
                options.update({
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_ctx": 8192,  # Contexto máximo
                })
            
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": stream,
                "options": options
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=None  # Sin timeout - permite que el modelo piense todo lo necesario
            )
            response.raise_for_status()
            
            if stream:
                # Manejar respuesta streaming
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'response' in data:
                            full_response += data['response']
                        if data.get('done', False):
                            break
                return full_response
            else:
                # Respuesta completa
                result = response.json()
                return result.get('response', '')
                
        except Exception as e:
            error_handler.handle_error(e, f"Error al generar respuesta con modelo {model}")
            return "Lo siento, ha ocurrido un error al generar la respuesta."
    
    def generate_embeddings(self, model: str, text: str) -> List[float]:
        """
        Generar embeddings para un texto
        
        Args:
            model: Modelo de embeddings
            text: Texto a procesar
        
        Returns:
            Vector de embeddings
        """
        try:
            payload = {
                "model": model,
                "prompt": text
            }
            
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=None  # Sin timeout para permitir procesamiento completo
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('embedding', [])
            
        except Exception as e:
            error_handler.handle_error(e, f"Error al generar embeddings con modelo {model}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Descargar un modelo
        
        Args:
            model_name: Nombre del modelo a descargar
        
        Returns:
            True si se descargó correctamente
        """
        try:
            payload = {"name": model_name}
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # Timeout más largo para descarga
            )
            response.raise_for_status()
            
            logger.info(f"Modelo {model_name} descargado correctamente")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, f"Error al descargar modelo {model_name}")
            return False

# Modelos predefinidos recomendados
DEFAULT_MODELS = {
    "chat": [
        "llama3.2:3b",
        "llama3.2:1b", 
        "qwen2.5:3b",
        "phi3:mini",
        "gemma2:2b"
    ],
    "embeddings": [
        "nomic-embed-text",
        "mxbai-embed-large",
        "all-minilm:l6-v2",
        "bge-large:en",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
}

def get_default_models():
    """Obtener modelos predefinidos recomendados"""
    return DEFAULT_MODELS

def get_available_or_default_models(client_instance=None):
    """
    Obtener modelos disponibles o usar predefinidos como fallback
    
    Args:
        client_instance: Instancia de OllamaClient (opcional)
    
    Returns:
        Lista de modelos disponibles o predefinidos
    """
    try:
        if client_instance and client_instance.is_available():
            available_models = client_instance.get_available_models()
            if available_models:
                return [model['name'] for model in available_models]
        
        # Fallback a modelos predefinidos
        logger.warning("Usando modelos predefinidos como fallback")
        return DEFAULT_MODELS["chat"] + DEFAULT_MODELS["embeddings"]
        
    except Exception as e:
        logger.error(f"Error al obtener modelos: {e}")
        return DEFAULT_MODELS["chat"] + DEFAULT_MODELS["embeddings"]

# Crear instancia global del cliente
ollama_client = OllamaClient()