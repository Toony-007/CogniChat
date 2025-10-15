# Arquitectura Detallada del Sistema CogniChat

## Tabla de Contenidos
1. [Introducción y Principios Fundamentales](#introducción-y-principios-fundamentales)
2. [Arquitectura en Capas y Patrones de Diseño](#arquitectura-en-capas-y-patrones-de-diseño)
3. [Integración de Componentes de IA](#integración-de-componentes-de-ia)
4. [Sistema de Caché Multinivel](#sistema-de-caché-multinivel)
5. [Ejemplos de Código Detallados](#ejemplos-de-código-detallados)
6. [Especificaciones para Capturas de Pantalla](#especificaciones-para-capturas-de-pantalla)
7. [Diagrama de Arquitectura C4](#diagrama-de-arquitectura-c4)

---

## Introducción y Principios Fundamentales

El diseño de la arquitectura del prototipo CogniChat representa el pilar fundamental sobre el cual se erige toda la estrategia computacional. Esta arquitectura no es simplemente una estructura técnica, sino un marco conceptual y lógico diseñado meticulosamente para asegurar que el sistema sea funcional, sostenible, adaptable y escalable.

### Principios Arquitectónicos Implementados

#### 1. Modularidad
La arquitectura modular se implementa a través de una separación clara de responsabilidades, donde cada módulo tiene interfaces bien definidas:

```python
# Estructura modular del proyecto
modules/
├── __init__.py              # Inicialización del paquete
├── chatbot.py              # Módulo de interacción conversacional
├── document_processor.py   # Procesamiento de documentos
├── document_upload.py      # Gestión de carga de archivos
├── qualitative_analysis.py # Análisis cualitativo avanzado
├── alerts.py              # Sistema de notificaciones
└── settings.py            # Configuraciones de usuario

utils/
├── __init__.py
├── rag_processor.py       # Procesamiento RAG
├── ollama_client.py       # Cliente para Ollama
├── database.py            # Abstracción de base de datos
├── error_handler.py       # Manejo centralizado de errores
├── logger.py              # Sistema de logging
├── metrics.py             # Métricas del sistema
├── traceability.py        # Trazabilidad de operaciones
└── validators.py          # Validaciones de entrada
```

**Ejemplo de implementación modular:**

```python
# modules/__init__.py - Patrón de inicialización modular
"""
Archivo de inicialización del paquete modules
Implementa el patrón de carga dinámica de módulos
"""

# Importar todos los módulos disponibles
from . import document_upload
from . import chatbot
from . import alerts
from . import settings
from . import document_processor
from . import qualitative_analysis

# Hacer disponibles los módulos con lazy loading
__all__ = [
    'document_upload',
    'chatbot', 
    'alerts',
    'settings',
    'document_processor',
    'qualitative_analysis'
]

# Configuración de módulos con metadatos
MODULE_METADATA = {
    'document_upload': {
        'description': 'Gestión de carga y validación de documentos',
        'dependencies': ['streamlit', 'pathlib'],
        'version': '1.0.0'
    },
    'chatbot': {
        'description': 'Interfaz conversacional con IA',
        'dependencies': ['ollama_client', 'rag_processor'],
        'version': '1.2.0'
    },
    # ... más metadatos
}
```

#### 2. Escalabilidad
La escalabilidad se implementa mediante configuraciones dinámicas y arquitectura desacoplada:

```python
# config/settings.py - Configuración escalable
@dataclass
class AppConfig:
    """
    Configuración principal de la aplicación
    Implementa el patrón Singleton para configuración global
    """
    
    # Configuración RAG optimizada para escalabilidad
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "300"))
    MAX_RETRIEVAL_DOCS: int = int(os.getenv("MAX_RETRIEVAL_DOCS", "15"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    
    # Configuración de escalado horizontal
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    def __post_init__(self):
        """Inicialización posterior con validaciones"""
        self._setup_directories()
        self._setup_models()
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validar configuración para escalabilidad"""
        if self.CHUNK_SIZE < 500:
            raise ValueError("CHUNK_SIZE debe ser al menos 500 para rendimiento óptimo")
        
        if self.MAX_RETRIEVAL_DOCS > 50:
            logger.warning("MAX_RETRIEVAL_DOCS alto puede afectar rendimiento")
```

#### 3. Robustez
La robustez se garantiza mediante manejo exhaustivo de errores y logging detallado:

```python
# utils/error_handler.py - Sistema robusto de manejo de errores
class ErrorHandler:
    """
    Manejo centralizado de errores con logging y recuperación
    Implementa el patrón Observer para notificaciones de errores
    """
    
    def __init__(self):
        self.logger = setup_logger("ErrorHandler")
        self.error_observers = []
        self.error_history = []
    
    def handle_error(self, error: Exception, context: str = "", 
                    recovery_action: callable = None):
        """
        Manejo robusto de errores con recuperación automática
        
        Args:
            error: Excepción capturada
            context: Contexto donde ocurrió el error
            recovery_action: Función de recuperación opcional
        """
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        # Logging detallado
        self.logger.error(f"Error en {context}: {error}")
        self.logger.debug(f"Traceback completo: {error_info['traceback']}")
        
        # Almacenar en historial para análisis
        self.error_history.append(error_info)
        
        # Notificar a observadores
        self._notify_observers(error_info)
        
        # Intentar recuperación automática
        if recovery_action:
            try:
                recovery_action()
                self.logger.info(f"Recuperación exitosa para error en {context}")
            except Exception as recovery_error:
                self.logger.error(f"Fallo en recuperación: {recovery_error}")
        
        return error_info
```

---

## Arquitectura en Capas y Patrones de Diseño

### Estructura en Capas

La aplicación implementa una arquitectura de 4 capas bien definidas:

```
┌─────────────────────────────────────┐
│        CAPA DE PRESENTACIÓN         │
│     (Streamlit UI Components)       │
├─────────────────────────────────────┤
│         CAPA DE APLICACIÓN          │
│    (Business Logic & Workflows)     │
├─────────────────────────────────────┤
│         CAPA DE DOMINIO             │
│   (Core Models & Domain Logic)      │
├─────────────────────────────────────┤
│      CAPA DE INFRAESTRUCTURA        │
│  (Database, File System, Ollama)    │
└─────────────────────────────────────┘
```

### Patrones de Diseño Implementados

#### 1. Patrón Repository
Abstrae el acceso a datos proporcionando una interfaz coherente:

```python
# utils/database.py - Implementación del patrón Repository
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class DocumentRepository(ABC):
    """
    Interfaz abstracta para repositorio de documentos
    Implementa el patrón Repository para abstracción de datos
    """
    
    @abstractmethod
    def save_document(self, document: Dict[str, Any]) -> str:
        """Guardar documento y retornar ID"""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Obtener documento por ID"""
        pass
    
    @abstractmethod
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Buscar documentos por consulta"""
        pass

class FileSystemDocumentRepository(DocumentRepository):
    """
    Implementación concreta del repositorio usando sistema de archivos
    Permite cambiar fácilmente a base de datos sin afectar lógica de negocio
    """
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.metadata_file = base_path / "metadata.json"
        self._load_metadata()
    
    def save_document(self, document: Dict[str, Any]) -> str:
        """
        Guardar documento en sistema de archivos
        Implementa versionado y validación automática
        """
        doc_id = self._generate_document_id(document)
        
        # Validar documento antes de guardar
        if not self._validate_document(document):
            raise ValueError("Documento inválido")
        
        # Guardar archivo físico
        doc_path = self.base_path / f"{doc_id}.json"
        with open(doc_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        
        # Actualizar metadatos
        self._update_metadata(doc_id, document)
        
        logger.info(f"Documento guardado: {doc_id}")
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Recuperar documento con caché inteligente"""
        # Verificar caché primero
        if doc_id in self._cache:
            return self._cache[doc_id]
        
        # Cargar desde disco
        doc_path = self.base_path / f"{doc_id}.json"
        if not doc_path.exists():
            return None
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            document = json.load(f)
        
        # Actualizar caché
        self._cache[doc_id] = document
        return document
```

#### 2. Patrón Strategy
Permite seleccionar dinámicamente algoritmos de análisis:

```python
# modules/qualitative_analysis.py - Implementación del patrón Strategy
from abc import ABC, abstractmethod

class AnalysisStrategy(ABC):
    """
    Estrategia abstracta para análisis cualitativo
    Permite intercambiar algoritmos dinámicamente
    """
    
    @abstractmethod
    def analyze(self, data: List[str]) -> Dict[str, Any]:
        """Ejecutar análisis específico"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Nombre de la estrategia"""
        pass

class ThematicAnalysisStrategy(AnalysisStrategy):
    """
    Estrategia de análisis temático usando clustering
    Implementa algoritmos de ML para identificar temas
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    
    def analyze(self, data: List[str]) -> Dict[str, Any]:
        """
        Análisis temático completo con clustering y visualización
        """
        if not data:
            return {"error": "No hay datos para analizar"}
        
        try:
            # Vectorización TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(data)
            
            # Clustering K-means
            clusters = self.clustering_model.fit_predict(tfidf_matrix)
            
            # Extraer términos más importantes por cluster
            feature_names = self.vectorizer.get_feature_names_out()
            cluster_centers = self.clustering_model.cluster_centers_
            
            themes = {}
            for i in range(self.n_clusters):
                # Obtener top términos para cada cluster
                top_indices = cluster_centers[i].argsort()[-10:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                
                themes[f"Tema_{i+1}"] = {
                    "terminos_clave": top_terms,
                    "documentos": [j for j, cluster in enumerate(clusters) if cluster == i],
                    "peso": float(cluster_centers[i].max())
                }
            
            return {
                "strategy": self.get_strategy_name(),
                "themes": themes,
                "total_documents": len(data),
                "clusters_found": self.n_clusters
            }
            
        except Exception as e:
            logger.error(f"Error en análisis temático: {e}")
            return {"error": str(e)}
    
    def get_strategy_name(self) -> str:
        return "Análisis Temático con Clustering"

class SentimentAnalysisStrategy(AnalysisStrategy):
    """
    Estrategia de análisis de sentimientos
    Utiliza TextBlob para análisis de polaridad y subjetividad
    """
    
    def analyze(self, data: List[str]) -> Dict[str, Any]:
        """
        Análisis de sentimientos con métricas detalladas
        """
        sentiments = []
        
        for text in data:
            blob = TextBlob(text)
            sentiment_data = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity,
                "sentiment_label": self._get_sentiment_label(blob.sentiment.polarity)
            }
            sentiments.append(sentiment_data)
        
        # Calcular estadísticas agregadas
        polarities = [s["polarity"] for s in sentiments]
        subjectivities = [s["subjectivity"] for s in sentiments]
        
        return {
            "strategy": self.get_strategy_name(),
            "sentiments": sentiments,
            "statistics": {
                "avg_polarity": np.mean(polarities),
                "avg_subjectivity": np.mean(subjectivities),
                "positive_count": len([p for p in polarities if p > 0.1]),
                "negative_count": len([p for p in polarities if p < -0.1]),
                "neutral_count": len([p for p in polarities if -0.1 <= p <= 0.1])
            }
        }
    
    def _get_sentiment_label(self, polarity: float) -> str:
        """Clasificar sentimiento basado en polaridad"""
        if polarity > 0.1:
            return "Positivo"
        elif polarity < -0.1:
            return "Negativo"
        else:
            return "Neutral"
    
    def get_strategy_name(self) -> str:
        return "Análisis de Sentimientos"

class QualitativeAnalysisContext:
    """
    Contexto que utiliza las estrategias de análisis
    Implementa el patrón Strategy para intercambio dinámico
    """
    
    def __init__(self, strategy: AnalysisStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: AnalysisStrategy):
        """Cambiar estrategia dinámicamente"""
        self._strategy = strategy
        logger.info(f"Estrategia cambiada a: {strategy.get_strategy_name()}")
    
    def execute_analysis(self, data: List[str]) -> Dict[str, Any]:
        """Ejecutar análisis con la estrategia actual"""
        start_time = time.time()
        result = self._strategy.analyze(data)
        execution_time = time.time() - start_time
        
        result["execution_time"] = execution_time
        result["timestamp"] = datetime.now().isoformat()
        
        return result
```

#### 3. Patrón Observer
Gestiona eventos y notificaciones del sistema:

```python
# utils/traceability.py - Implementación del patrón Observer
from typing import List, Callable, Dict, Any
from abc import ABC, abstractmethod

class Observer(ABC):
    """Interfaz para observadores del sistema"""
    
    @abstractmethod
    def update(self, event_type: str, data: Dict[str, Any]):
        """Recibir notificación de evento"""
        pass

class Subject(ABC):
    """Sujeto observable del sistema"""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        """Agregar observador"""
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        """Remover observador"""
        self._observers.remove(observer)
    
    def notify(self, event_type: str, data: Dict[str, Any]):
        """Notificar a todos los observadores"""
        for observer in self._observers:
            observer.update(event_type, data)

class TraceabilityManager(Subject):
    """
    Gestor de trazabilidad que notifica eventos del sistema
    Implementa el patrón Observer para logging distribuido
    """
    
    def __init__(self):
        super().__init__()
        self.trace_history = []
        self.logger = setup_logger("TraceabilityManager")
    
    def log_rag_query(self, query: str, chunks_found: int, model_used: str):
        """Registrar consulta RAG con notificación"""
        trace_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "chunks_found": chunks_found,
            "model_used": model_used,
            "query_id": self._generate_query_id()
        }
        
        self.trace_history.append(trace_data)
        self.logger.info(f"RAG Query: {query[:50]}... -> {chunks_found} chunks")
        
        # Notificar a observadores
        self.notify("rag_query", trace_data)
        
        return trace_data["query_id"]
    
    def log_document_processing(self, filename: str, chunks_created: int, 
                              processing_time: float):
        """Registrar procesamiento de documento"""
        trace_data = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "chunks_created": chunks_created,
            "processing_time": processing_time,
            "status": "completed"
        }
        
        self.trace_history.append(trace_data)
        self.logger.info(f"Document processed: {filename} -> {chunks_created} chunks")
        
        # Notificar a observadores
        self.notify("document_processing", trace_data)

class MetricsObserver(Observer):
    """
    Observador que recopila métricas del sistema
    """
    
    def __init__(self):
        self.metrics = {
            "rag_queries": 0,
            "documents_processed": 0,
            "total_chunks": 0,
            "avg_processing_time": 0
        }
        self.processing_times = []
    
    def update(self, event_type: str, data: Dict[str, Any]):
        """Actualizar métricas basado en eventos"""
        if event_type == "rag_query":
            self.metrics["rag_queries"] += 1
            
        elif event_type == "document_processing":
            self.metrics["documents_processed"] += 1
            self.metrics["total_chunks"] += data["chunks_created"]
            
            # Calcular tiempo promedio de procesamiento
            self.processing_times.append(data["processing_time"])
            self.metrics["avg_processing_time"] = np.mean(self.processing_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales"""
        return self.metrics.copy()

# Instancia global del gestor de trazabilidad
traceability_manager = TraceabilityManager()

# Agregar observador de métricas
metrics_observer = MetricsObserver()
traceability_manager.attach(metrics_observer)
```

#### 4. Patrón Factory
Centraliza la creación de objetos complejos:

```python
# utils/rag_processor.py - Implementación del patrón Factory
from typing import Union, Dict, Any
from pathlib import Path

class DocumentProcessorFactory:
    """
    Factory para crear procesadores de documentos específicos
    Permite agregar nuevos tipos sin modificar código existente
    """
    
    _processors = {}
    
    @classmethod
    def register_processor(cls, file_extension: str, processor_class):
        """Registrar nuevo procesador para extensión"""
        cls._processors[file_extension.lower()] = processor_class
    
    @classmethod
    def create_processor(cls, file_path: Union[str, Path]) -> 'DocumentProcessor':
        """
        Crear procesador apropiado basado en extensión de archivo
        
        Args:
            file_path: Ruta del archivo a procesar
            
        Returns:
            Instancia del procesador apropiado
            
        Raises:
            ValueError: Si no hay procesador para la extensión
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in cls._processors:
            raise ValueError(f"No hay procesador disponible para archivos {extension}")
        
        processor_class = cls._processors[extension]
        return processor_class(file_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Obtener lista de extensiones soportadas"""
        return list(cls._processors.keys())

class DocumentProcessor(ABC):
    """Interfaz base para procesadores de documentos"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.logger = setup_logger(f"Processor_{self.__class__.__name__}")
    
    @abstractmethod
    def extract_text(self) -> str:
        """Extraer texto del documento"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Obtener metadatos del documento"""
        pass

class PDFProcessor(DocumentProcessor):
    """Procesador específico para archivos PDF"""
    
    def extract_text(self) -> str:
        """Extraer texto de PDF usando PyPDF2"""
        try:
            text = ""
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Página {page_num + 1} ---\n{page_text}"
            
            self.logger.info(f"Texto extraído de PDF: {len(text)} caracteres")
            return text
            
        except Exception as e:
            self.logger.error(f"Error extrayendo texto de PDF: {e}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """Obtener metadatos del PDF"""
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                metadata = {
                    "pages": len(pdf_reader.pages),
                    "file_size": self.file_path.stat().st_size,
                    "created": datetime.fromtimestamp(
                        self.file_path.stat().st_ctime
                    ).isoformat()
                }
                
                # Agregar metadatos del PDF si están disponibles
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    metadata.update({
                        "title": pdf_meta.get("/Title", ""),
                        "author": pdf_meta.get("/Author", ""),
                        "subject": pdf_meta.get("/Subject", ""),
                        "creator": pdf_meta.get("/Creator", "")
                    })
                
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error obteniendo metadatos de PDF: {e}")
            return {"error": str(e)}

class DOCXProcessor(DocumentProcessor):
    """Procesador específico para archivos DOCX"""
    
    def extract_text(self) -> str:
        """Extraer texto de DOCX usando python-docx"""
        try:
            doc = docx.Document(self.file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Extraer texto de tablas
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
            
            self.logger.info(f"Texto extraído de DOCX: {len(text)} caracteres")
            return text
            
        except Exception as e:
            self.logger.error(f"Error extrayendo texto de DOCX: {e}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """Obtener metadatos del DOCX"""
        try:
            doc = docx.Document(self.file_path)
            
            metadata = {
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables),
                "file_size": self.file_path.stat().st_size,
                "created": datetime.fromtimestamp(
                    self.file_path.stat().st_ctime
                ).isoformat()
            }
            
            # Agregar propiedades del documento si están disponibles
            if hasattr(doc, 'core_properties'):
                props = doc.core_properties
                metadata.update({
                    "title": props.title or "",
                    "author": props.author or "",
                    "subject": props.subject or "",
                    "created": props.created.isoformat() if props.created else ""
                })
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error obteniendo metadatos de DOCX: {e}")
            return {"error": str(e)}

# Registrar procesadores en el factory
DocumentProcessorFactory.register_processor('.pdf', PDFProcessor)
DocumentProcessorFactory.register_processor('.docx', DOCXProcessor)
DocumentProcessorFactory.register_processor('.doc', DOCXProcessor)
```

---

## Integración de Componentes de IA

### Cliente Ollama Optimizado

La integración con Ollama se realiza a través de un cliente especializado que maneja la comunicación, el estado de los modelos y la optimización de recursos:

```python
# utils/ollama_client.py - Cliente optimizado para Ollama
class OllamaClient:
    """
    Cliente optimizado para interactuar con Ollama
    Incluye gestión de modelos, caché y recuperación de errores
    """
    
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.timeout = config.OLLAMA_TIMEOUT
        self.logger = setup_logger("OllamaClient")
        self.session = requests.Session()
        
        # Configurar adaptadores para reintentos
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Caché de modelos y respuestas
        self._model_cache = {}
        self._response_cache = {}
        self._cache_ttl = 300  # 5 minutos
    
    def is_available(self) -> bool:
        """
        Verificar disponibilidad de Ollama con reintentos
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Ollama no disponible: {e}")
            return False
    
    def get_available_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Obtener modelos disponibles con caché inteligente
        
        Args:
            force_refresh: Forzar actualización del caché
            
        Returns:
            Lista de modelos disponibles con metadatos
        """
        cache_key = "available_models"
        current_time = time.time()
        
        # Verificar caché si no se fuerza actualización
        if not force_refresh and cache_key in self._model_cache:
            cached_data, timestamp = self._model_cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                return cached_data
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            models_data = response.json()
            models = []
            
            for model in models_data.get('models', []):
                model_info = {
                    'name': model['name'],
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at', ''),
                    'digest': model.get('digest', ''),
                    'details': model.get('details', {})
                }
                models.append(model_info)
            
            # Actualizar caché
            self._model_cache[cache_key] = (models, current_time)
            
            self.logger.info(f"Modelos disponibles: {len(models)}")
            return models
            
        except Exception as e:
            self.logger.error(f"Error obteniendo modelos: {e}")
            return []
    
    def generate_response(self, model: str, prompt: str, 
                         context: Optional[str] = None,
                         max_tokens: int = 3000,
                         temperature: float = 0.7,
                         stream: bool = False) -> str:
        """
        Generar respuesta con configuración avanzada
        
        Args:
            model: Nombre del modelo a usar
            prompt: Prompt para el modelo
            context: Contexto adicional opcional
            max_tokens: Máximo número de tokens
            temperature: Temperatura para generación
            stream: Si usar streaming o no
            
        Returns:
            Respuesta generada por el modelo
        """
        # Crear caché key para respuestas similares
        cache_key = hashlib.md5(
            f"{model}:{prompt[:100]}:{context[:100] if context else ''}".encode()
        ).hexdigest()
        
        # Verificar caché de respuestas
        current_time = time.time()
        if cache_key in self._response_cache:
            cached_response, timestamp = self._response_cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                self.logger.info("Respuesta obtenida del caché")
                return cached_response
        
        try:
            # Construir prompt completo
            full_prompt = prompt
            if context:
                full_prompt = f"Contexto: {context}\n\nPregunta: {prompt}"
            
            # Configurar parámetros de generación
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": stream,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
            
            self.logger.info(f"Generando respuesta con {model}")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Procesar respuesta
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
                result = full_response
            else:
                # Respuesta completa
                result = response.json().get('response', '')
            
            generation_time = time.time() - start_time
            self.logger.info(f"Respuesta generada en {generation_time:.2f}s")
            
            # Guardar en caché
            self._response_cache[cache_key] = (result, current_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generando respuesta: {e}")
            raise
    
    def generate_embeddings(self, model: str, text: str) -> List[float]:
        """
        Generar embeddings para texto
        
        Args:
            model: Modelo de embeddings a usar
            text: Texto para generar embeddings
            
        Returns:
            Vector de embeddings
        """
        try:
            payload = {
                "model": model,
                "prompt": text
            }
            
            response = self.session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            embeddings = response.json().get('embedding', [])
            self.logger.debug(f"Embeddings generados: {len(embeddings)} dimensiones")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generando embeddings: {e}")
            raise
    
    def pull_model(self, model_name: str) -> bool:
        """
        Descargar modelo si no está disponible
        
        Args:
            model_name: Nombre del modelo a descargar
            
        Returns:
            True si la descarga fue exitosa
        """
        try:
            payload = {"name": model_name}
            
            self.logger.info(f"Descargando modelo: {model_name}")
            
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=600  # 10 minutos para descarga
            )
            response.raise_for_status()
            
            self.logger.info(f"Modelo {model_name} descargado exitosamente")
            
            # Limpiar caché de modelos para refrescar
            if "available_models" in self._model_cache:
                del self._model_cache["available_models"]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error descargando modelo {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Obtener información detallada de un modelo
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Información del modelo
        """
        try:
            payload = {"name": model_name}
            
            response = self.session.post(
                f"{self.base_url}/api/show",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error obteniendo info del modelo {model_name}: {e}")
            return {}
```

### Procesamiento RAG Avanzado

El sistema RAG implementa técnicas avanzadas de recuperación y generación:

```python
# utils/rag_processor.py - Sistema RAG completo
class RAGProcessor:
    """
    Procesador RAG completo con técnicas avanzadas
    Incluye chunking inteligente, embeddings y recuperación semántica
    """
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.cache_path = Path(config.CACHE_DIR) / "rag_cache.json"
        self.logger = setup_logger("RAGProcessor")
        
        # Configuración de chunking
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        
        # Caché de embeddings y chunks
        self._load_cache()
        
        # Configuración de similitud
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        self.max_retrieval_docs = config.MAX_RETRIEVAL_DOCS
    
    def process_document(self, file_path: Path, 
                        force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Procesar documento completo con chunking y embeddings
        
        Args:
            file_path: Ruta del documento
            force_reprocess: Forzar reprocesamiento
            
        Returns:
            Información del procesamiento
        """
        start_time = time.time()
        
        # Generar hash del archivo para caché
        file_hash = self._get_file_hash(file_path)
        
        # Verificar si ya está procesado
        if not force_reprocess and file_hash in self.cache:
            self.logger.info(f"Documento ya procesado: {file_path.name}")
            return self.cache[file_hash]
        
        try:
            # Crear procesador usando Factory
            processor = DocumentProcessorFactory.create_processor(file_path)
            
            # Extraer texto y metadatos
            text = processor.extract_text()
            metadata = processor.get_metadata()
            
            # Chunking inteligente
            chunks = self._intelligent_chunking(text, file_path.name)
            
            # Generar embeddings para cada chunk
            embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.ollama_client.generate_embeddings(
                        config.DEFAULT_EMBEDDING_MODEL,
                        chunk['text']
                    )
                    chunk['embedding'] = embedding
                    embeddings.append(embedding)
                    
                    self.logger.debug(f"Embedding generado para chunk {i+1}/{len(chunks)}")
                    
                except Exception as e:
                    self.logger.error(f"Error generando embedding para chunk {i}: {e}")
                    chunk['embedding'] = []
            
            # Preparar datos para caché
            processed_data = {
                'file_path': str(file_path),
                'file_hash': file_hash,
                'filename': file_path.name,
                'text': text,
                'chunks': chunks,
                'metadata': metadata,
                'processed_at': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'chunk_count': len(chunks),
                'embedding_model': config.DEFAULT_EMBEDDING_MODEL
            }
            
            # Guardar en caché
            self.cache[file_hash] = processed_data
            self._save_cache()
            
            # Registrar en trazabilidad
            traceability_manager.log_document_processing(
                file_path.name,
                len(chunks),
                processed_data['processing_time']
            )
            
            self.logger.info(
                f"Documento procesado: {file_path.name} -> "
                f"{len(chunks)} chunks en {processed_data['processing_time']:.2f}s"
            )
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error procesando documento {file_path}: {e}")
            raise
    
    def _intelligent_chunking(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Chunking inteligente que preserva contexto semántico
        
        Args:
            text: Texto a dividir
            filename: Nombre del archivo para referencia
            
        Returns:
            Lista de chunks con metadatos
        """
        # Limpiar y normalizar texto
        text = self._clean_text(text)
        
        # Dividir por párrafos primero
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_chunk_start = 0
        
        for i, paragraph in enumerate(paragraphs):
            # Si agregar el párrafo excede el tamaño máximo
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    # Guardar chunk actual
                    chunk_data = {
                        'text': current_chunk.strip(),
                        'chunk_id': len(chunks),
                        'source_file': filename,
                        'start_paragraph': current_chunk_start,
                        'end_paragraph': i - 1,
                        'char_count': len(current_chunk),
                        'word_count': len(current_chunk.split()),
                        'created_at': datetime.now().isoformat()
                    }
                    chunks.append(chunk_data)
                    
                    # Iniciar nuevo chunk con overlap
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    
                    current_chunk_start = i
                else:
                    # Párrafo muy largo, dividir por oraciones
                    sentences = self._split_by_sentences(paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > self.chunk_size:
                            if current_chunk:
                                chunk_data = {
                                    'text': current_chunk.strip(),
                                    'chunk_id': len(chunks),
                                    'source_file': filename,
                                    'start_paragraph': current_chunk_start,
                                    'end_paragraph': i,
                                    'char_count': len(current_chunk),
                                    'word_count': len(current_chunk.split()),
                                    'created_at': datetime.now().isoformat(),
                                    'split_type': 'sentence'
                                }
                                chunks.append(chunk_data)
                            
                            current_chunk = sentence
                            current_chunk_start = i
                        else:
                            current_chunk += " " + sentence
            else:
                current_chunk += "\n\n" + paragraph
        
        # Agregar último chunk si existe
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'chunk_id': len(chunks),
                'source_file': filename,
                'start_paragraph': current_chunk_start,
                'end_paragraph': len(paragraphs) - 1,
                'char_count': len(current_chunk),
                'word_count': len(current_chunk.split()),
                'created_at': datetime.now().isoformat()
            }
            chunks.append(chunk_data)
        
        self.logger.info(f"Chunking completado: {len(chunks)} chunks creados")
        return chunks
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica avanzada en todos los documentos
        
        Args:
            query: Consulta de búsqueda
            top_k: Número máximo de resultados
            
        Returns:
            Lista de chunks relevantes con scores de similitud
        """
        if top_k is None:
            top_k = self.max_retrieval_docs
        
        try:
            # Generar embedding de la consulta
            query_embedding = self.ollama_client.generate_embeddings(
                config.DEFAULT_EMBEDDING_MODEL,
                query
            )
            
            if not query_embedding:
                self.logger.error("No se pudo generar embedding para la consulta")
                return []
            
            # Recopilar todos los chunks con embeddings
            all_chunks = []
            for file_hash, doc_data in self.cache.items():
                for chunk in doc_data.get('chunks', []):
                    if chunk.get('embedding'):
                        all_chunks.append({
                            'chunk': chunk,
                            'document': doc_data['filename'],
                            'file_hash': file_hash
                        })
            
            if not all_chunks:
                self.logger.warning("No hay chunks con embeddings disponibles")
                return []
            
            # Calcular similitudes
            similarities = []
            for item in all_chunks:
                chunk_embedding = item['chunk']['embedding']
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                
                if similarity >= self.similarity_threshold:
                    similarities.append({
                        'chunk': item['chunk'],
                        'document': item['document'],
                        'file_hash': item['file_hash'],
                        'similarity_score': similarity
                    })
            
            # Ordenar por similitud descendente
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Tomar top_k resultados
            results = similarities[:top_k]
            
            # Registrar en trazabilidad
            query_id = traceability_manager.log_rag_query(
                query,
                len(results),
                config.DEFAULT_EMBEDDING_MODEL
            )
            
            # Agregar información adicional a los resultados
            for result in results:
                result['query_id'] = query_id
                result['search_timestamp'] = datetime.now().isoformat()
            
            self.logger.info(
                f"Búsqueda semántica: '{query[:50]}...' -> "
                f"{len(results)} chunks relevantes"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda semántica: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calcular similitud coseno entre dos vectores
        
        Args:
            vec1: Primer vector
            vec2: Segundo vector
            
        Returns:
            Score de similitud coseno (0-1)
        """
        try:
            # Convertir a numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calcular similitud coseno
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            similarity = dot_product / (norm_v1 * norm_v2)
            
            # Normalizar a rango 0-1
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            self.logger.error(f"Error calculando similitud coseno: {e}")
            return 0.0
```

---

## Sistema de Caché Multinivel

El sistema implementa un caché multinivel sofisticado que optimiza el acceso a datos frecuentemente utilizados:

```python
# utils/cache_manager.py - Sistema de caché multinivel
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import pickle
import json
import time
import threading
from pathlib import Path

class CacheLevel(Enum):
    """Niveles de caché disponibles"""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"

@dataclass
class CacheEntry:
    """Entrada de caché con metadatos"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    size_bytes: int = 0
    level: CacheLevel = CacheLevel.MEMORY

class MultiLevelCache:
    """
    Sistema de caché multinivel con políticas de evicción inteligentes
    Implementa LRU, TTL y gestión automática de memoria
    """
    
    def __init__(self, 
                 memory_limit_mb: int = 100,
                 disk_limit_mb: int = 1000,
                 default_ttl: int = 3600):
        
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.disk_limit_bytes = disk_limit_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        # Caché en memoria (L1)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_usage = 0
        
        # Caché en disco (L2)
        self.disk_cache_dir = Path(config.CACHE_DIR) / "multilevel"
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_index: Dict[str, Dict] = {}
        
        # Control de concurrencia
        self.memory_lock = threading.RLock()
        self.disk_lock = threading.RLock()
        
        # Métricas
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'promotions': 0
        }
        
        self.logger = setup_logger("MultiLevelCache")
        
        # Cargar índice de disco
        self._load_disk_index()
        
        # Iniciar limpieza automática
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del caché con promoción automática
        
        Args:
            key: Clave del caché
            
        Returns:
            Valor almacenado o None si no existe
        """
        current_time = time.time()
        
        # Buscar en caché de memoria (L1)
        with self.memory_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Verificar TTL
                if self._is_expired(entry, current_time):
                    del self.memory_cache[key]
                    self.memory_usage -= entry.size_bytes
                else:
                    # Actualizar estadísticas de acceso
                    entry.accessed_at = current_time
                    entry.access_count += 1
                    self.stats['memory_hits'] += 1
                    
                    self.logger.debug(f"Cache hit (memory): {key}")
                    return entry.value
        
        # Buscar en caché de disco (L2)
        with self.disk_lock:
            if key in self.disk_index:
                disk_entry_info = self.disk_index[key]
                
                # Verificar TTL
                if self._is_disk_entry_expired(disk_entry_info, current_time):
                    self._remove_from_disk(key)
                else:
                    # Cargar desde disco
                    value = self._load_from_disk(key)
                    if value is not None:
                        # Promover a memoria si hay espacio
                        self._promote_to_memory(key, value, disk_entry_info)
                        
                        self.stats['disk_hits'] += 1
                        self.stats['promotions'] += 1
                        
                        self.logger.debug(f"Cache hit (disk): {key}")
                        return value
        
        # No encontrado en ningún nivel
        self.stats['misses'] += 1
        self.logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Almacenar valor en caché con gestión automática de niveles
        
        Args:
            key: Clave del caché
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos
            
        Returns:
            True si se almacenó exitosamente
        """
        current_time = time.time()
        
        # Calcular tamaño del valor
        try:
            value_size = len(pickle.dumps(value))
        except Exception:
            value_size = len(str(value).encode('utf-8'))
        
        # Crear entrada de caché
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=current_time,
            accessed_at=current_time,
            access_count=1,
            ttl=ttl or self.default_ttl,
            size_bytes=value_size,
            level=CacheLevel.MEMORY
        )
        
        # Intentar almacenar en memoria primero
        if self._can_fit_in_memory(value_size):
            with self.memory_lock:
                # Hacer espacio si es necesario
                self._ensure_memory_space(value_size)
                
                # Almacenar en memoria
                self.memory_cache[key] = entry
                self.memory_usage += value_size
                
                self.logger.debug(f"Cached in memory: {key} ({value_size} bytes)")
                return True
        
        # Si no cabe en memoria, almacenar en disco
        else:
            entry.level = CacheLevel.DISK
            return self._store_to_disk(key, entry)
    
    def _promote_to_memory(self, key: str, value: Any, disk_info: Dict):
        """
        Promover entrada de disco a memoria
        
        Args:
            key: Clave del caché
            value: Valor a promover
            disk_info: Información de la entrada en disco
        """
        value_size = disk_info.get('size_bytes', 0)
        
        if self._can_fit_in_memory(value_size):
            with self.memory_lock:
                self._ensure_memory_space(value_size)
                
                # Crear entrada de memoria
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=disk_info['created_at'],
                    accessed_at=time.time(),
                    access_count=disk_info.get('access_count', 1) + 1,
                    ttl=disk_info.get('ttl'),
                    size_bytes=value_size,
                    level=CacheLevel.MEMORY
                )
                
                self.memory_cache[key] = entry
                self.memory_usage += value_size
                
                self.logger.debug(f"Promoted to memory: {key}")
    
    def _ensure_memory_space(self, required_bytes: int):
        """
        Asegurar espacio en memoria usando política LRU
        
        Args:
            required_bytes: Bytes requeridos
        """
        while (self.memory_usage + required_bytes > self.memory_limit_bytes 
               and self.memory_cache):
            
            # Encontrar entrada menos recientemente usada
            lru_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].accessed_at
            )
            
            lru_entry = self.memory_cache[lru_key]
            
            # Mover a disco si es valiosa (accedida múltiples veces)
            if lru_entry.access_count > 1:
                self._demote_to_disk(lru_key, lru_entry)
            
            # Remover de memoria
            del self.memory_cache[lru_key]
            self.memory_usage -= lru_entry.size_bytes
            self.stats['evictions'] += 1
            
            self.logger.debug(f"Evicted from memory: {lru_key}")
    
    def _store_to_disk(self, key: str, entry: CacheEntry) -> bool:
        """
        Almacenar entrada en disco
        
        Args:
            key: Clave del caché
            entry: Entrada a almacenar
            
        Returns:
            True si se almacenó exitosamente
        """
        try:
            with self.disk_lock:
                # Crear archivo de caché
                cache_file = self.disk_cache_dir / f"{key}.cache"
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry.value, f)
                
                # Actualizar índice
                self.disk_index[key] = {
                    'file_path': str(cache_file),
                    'created_at': entry.created_at,
                    'accessed_at': entry.accessed_at,
                    'access_count': entry.access_count,
                    'ttl': entry.ttl,
                    'size_bytes': entry.size_bytes
                }
                
                # Guardar índice
                self._save_disk_index()
                
                self.logger.debug(f"Cached to disk: {key}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing to disk: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidar entrada de caché en todos los niveles
        
        Args:
            key: Clave a invalidar
            
        Returns:
            True si se invalidó alguna entrada
        """
        invalidated = False
        
        # Remover de memoria
        with self.memory_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                del self.memory_cache[key]
                self.memory_usage -= entry.size_bytes
                invalidated = True
        
        # Remover de disco
        with self.disk_lock:
            if key in self.disk_index:
                self._remove_from_disk(key)
                invalidated = True
        
        if invalidated:
            self.logger.debug(f"Invalidated: {key}")
        
        return invalidated
    
    def clear(self, level: Optional[CacheLevel] = None):
        """
        Limpiar caché completamente o por nivel
        
        Args:
            level: Nivel específico a limpiar, None para todos
        """
        if level is None or level == CacheLevel.MEMORY:
            with self.memory_lock:
                self.memory_cache.clear()
                self.memory_usage = 0
                self.logger.info("Memory cache cleared")
        
        if level is None or level == CacheLevel.DISK:
            with self.disk_lock:
                # Remover archivos de caché
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        self.logger.error(f"Error removing cache file {cache_file}: {e}")
                
                self.disk_index.clear()
                self._save_disk_index()
                self.logger.info("Disk cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del caché
        
        Returns:
            Diccionario con estadísticas
        """
        total_requests = (self.stats['memory_hits'] + 
                         self.stats['disk_hits'] +
                         self.stats['misses'])
        
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (self.stats['memory_hits'] + self.stats['disk_hits']) / total_requests
        
        return {
            'memory_entries': len(self.memory_cache),
            'memory_usage_mb': self.memory_usage / (1024 * 1024),
            'memory_limit_mb': self.memory_limit_bytes / (1024 * 1024),
            'disk_entries': len(self.disk_index),
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            **self.stats
        }

# Instancia global del caché multinivel
cache_manager = MultiLevelCache()
```

---

## Ejemplos de Código Detallados

### Implementación del Chatbot Principal

```python
# modules/chatbot.py - Implementación completa del chatbot
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

class ChatbotInterface:
    """
    Interfaz principal del chatbot con capacidades RAG
    Integra procesamiento de lenguaje natural y recuperación de información
    """
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.rag_processor = RAGProcessor()
        self.logger = setup_logger("ChatbotInterface")
        
        # Configuración de conversación
        self.max_history_length = 10
        self.context_window_size = 4000
        
        # Inicializar estado de sesión
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializar variables de estado de Streamlit"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'current_model' not in st.session_state:
            st.session_state.current_model = config.DEFAULT_LLM_MODEL
        
        if 'rag_enabled' not in st.session_state:
            st.session_state.rag_enabled = True
        
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = self._generate_conversation_id()
    
    def render_chat_interface(self):
        """
        Renderizar interfaz completa del chat
        Incluye historial, entrada de usuario y configuraciones
        """
        st.header("🤖 Chat Inteligente con RAG")
        
        # Verificar disponibilidad de Ollama
        if not self.ollama_client.is_available():
            st.error("⚠️ Ollama no está disponible. Verifique la conexión.")
            return
        
        # Configuraciones del chat
        self._render_chat_settings()
        
        # Contenedor del historial de chat
        chat_container = st.container()
        
        with chat_container:
            self._render_chat_history()
        
        # Entrada de usuario
        self._render_user_input()
        
        # Botones de control
        self._render_control_buttons()
    
    def _render_chat_settings(self):
        """Renderizar configuraciones del chat"""
        with st.expander("⚙️ Configuraciones del Chat", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Selección de modelo
                available_models = [model['name'] for model in 
                                  self.ollama_client.get_available_models()]
                
                if available_models:
                    selected_model = st.selectbox(
                        "Modelo LLM:",
                        available_models,
                        index=available_models.index(st.session_state.current_model)
                        if st.session_state.current_model in available_models else 0
                    )
                    st.session_state.current_model = selected_model
                else:
                    st.warning("No hay modelos disponibles")
            
            with col2:
                # Configuración RAG
                st.session_state.rag_enabled = st.checkbox(
                    "Habilitar RAG",
                    value=st.session_state.rag_enabled,
                    help="Usar documentos cargados como contexto"
                )
                
                if st.session_state.rag_enabled:
                    max_docs = st.slider(
                        "Máx. documentos:",
                        min_value=1,
                        max_value=20,
                        value=config.MAX_RETRIEVAL_DOCS
                    )
                    config.MAX_RETRIEVAL_DOCS = max_docs
            
            with col3:
                # Parámetros de generación
                temperature = st.slider(
                    "Temperatura:",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.7,
                    step=0.1,
                    help="Controla la creatividad de las respuestas"
                )
                
                max_tokens = st.slider(
                    "Máx. tokens:",
                    min_value=100,
                    max_value=4000,
                    value=config.MAX_RESPONSE_TOKENS
                )
    
    def _render_chat_history(self):
        """Renderizar historial de conversación"""
        if not st.session_state.chat_history:
            st.info("👋 ¡Hola! Soy tu asistente de IA. Puedo ayudarte con preguntas sobre los documentos cargados.")
            return
        
        for i, message in enumerate(st.session_state.chat_history):
            self._render_message(message, i)
    
    def _render_message(self, message: Dict[str, Any], index: int):
        """
        Renderizar mensaje individual del chat
        
        Args:
            message: Datos del mensaje
            index: Índice del mensaje en el historial
        """
        is_user = message['role'] == 'user'
        
        # Contenedor del mensaje con estilo
        with st.container():
            if is_user:
                # Mensaje del usuario (alineado a la derecha)
                col1, col2 = st.columns([1, 4])
                with col2:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #e3f2fd;
                            padding: 10px;
                            border-radius: 10px;
                            margin: 5px 0;
                            border-left: 4px solid #2196f3;
                        ">
                            <strong>👤 Usuario:</strong><br>
                            {message['content']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                # Mensaje del asistente
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f3e5f5;
                            padding: 10px;
                            border-radius: 10px;
                            margin: 5px 0;
                            border-left: 4px solid #9c27b0;
                        ">
                            <strong>🤖 Asistente:</strong><br>
                            {message['content']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Mostrar metadatos si están disponibles
                    if 'metadata' in message:
                        with st.expander("📊 Información adicional", expanded=False):
                            metadata = message['metadata']
                            
                            col_meta1, col_meta2 = st.columns(2)
                            
                            with col_meta1:
                                st.write(f"**Modelo usado:** {metadata.get('model', 'N/A')}")
                                st.write(f"**Tiempo de respuesta:** {metadata.get('response_time', 0):.2f}s")
                                st.write(f"**Tokens generados:** {metadata.get('tokens', 'N/A')}")
                            
                            with col_meta2:
                                if metadata.get('rag_used', False):
                                    st.write(f"**Documentos consultados:** {metadata.get('docs_used', 0)}")
                                    st.write(f"**Chunks relevantes:** {metadata.get('chunks_used', 0)}")
                                    
                                    if 'sources' in metadata:
                                        st.write("**Fuentes:**")
                                        for source in metadata['sources'][:3]:
                                            st.write(f"- {source}")
    
    def _render_user_input(self):
        """Renderizar área de entrada del usuario"""
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "Escribe tu mensaje:",
                    placeholder="Pregunta algo sobre los documentos cargados...",
                    height=100,
                    key="user_message_input"
                )
            
            with col2:
                st.write("")  # Espaciado
                submit_button = st.form_submit_button(
                    "📤 Enviar",
                    use_container_width=True
                )
            
            if submit_button and user_input.strip():
                self._process_user_message(user_input.strip())
                st.rerun()
    
    def _process_user_message(self, user_message: str):
        """
        Procesar mensaje del usuario y generar respuesta
        
        Args:
            user_message: Mensaje del usuario
        """
        start_time = time.time()
        
        try:
            # Agregar mensaje del usuario al historial
            user_msg = {
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.chat_history.append(user_msg)
            
            # Mostrar indicador de carga
            with st.spinner("🤔 Pensando..."):
                # Generar respuesta
                response_data = self._generate_response(user_message)
            
            # Agregar respuesta al historial
            assistant_msg = {
                'role': 'assistant',
                'content': response_data['content'],
                'timestamp': datetime.now().isoformat(),
                'metadata': response_data['metadata']
            }
            st.session_state.chat_history.append(assistant_msg)
            
            # Mantener historial dentro del límite
            if len(st.session_state.chat_history) > self.max_history_length * 2:
                st.session_state.chat_history = st.session_state.chat_history[-self.max_history_length * 2:]
            
            processing_time = time.time() - start_time
            self.logger.info(f"Mensaje procesado en {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error procesando mensaje: {e}")
            st.error(f"Error procesando mensaje: {str(e)}")
    
    def _generate_response(self, user_message: str) -> Dict[str, Any]:
        """
        Generar respuesta usando RAG y LLM
        
        Args:
            user_message: Mensaje del usuario
            
        Returns:
            Diccionario con respuesta y metadatos
        """
        start_time = time.time()
        
        # Inicializar contexto y metadatos
        context = ""
        metadata = {
            'model': st.session_state.current_model,
            'rag_used': False,
            'docs_used': 0,
            'chunks_used': 0,
            'sources': []
        }
        
        # Usar RAG si está habilitado
        if st.session_state.rag_enabled:
            relevant_chunks = self.rag_processor.semantic_search(
                user_message,
                top_k=config.MAX_RETRIEVAL_DOCS
            )
            
            if relevant_chunks:
                metadata['rag_used'] = True
                metadata['chunks_used'] = len(relevant_chunks)
                
                # Construir contexto desde chunks relevantes
                context_parts = []
                sources = set()
                
                for chunk_data in relevant_chunks:
                    chunk = chunk_data['chunk']
                    document = chunk_data['document']
                    similarity = chunk_data['similarity_score']
                    
                    context_parts.append(
                        f"[Fuente: {document} - Similitud: {similarity:.2f}]\n"
                        f"{chunk['text']}\n"
                    )
                    sources.add(document)
                
                context = "\n---\n".join(context_parts)
                metadata['sources'] = list(sources)
                metadata['docs_used'] = len(sources)
        
        # Construir prompt completo
        if context:
            full_prompt = f"""
Contexto de documentos relevantes:
{context}

---

Pregunta del usuario: {user_message}

Instrucciones:
- Responde basándote principalmente en el contexto proporcionado
- Si la información no está en el contexto, indícalo claramente
- Sé preciso y cita las fuentes cuando sea relevante
- Mantén un tono profesional y útil
"""
        else:
            full_prompt = f"""
Pregunta del usuario: {user_message}

Instrucciones:
- Responde de manera útil y precisa
- Si no tienes información específica sobre el tema, indícalo
- Mantén un tono profesional y conversacional
"""
        
        # Generar respuesta con Ollama
        try:
            response = self.ollama_client.generate_response(
                model=st.session_state.current_model,
                prompt=full_prompt,
                max_tokens=config.MAX_RESPONSE_TOKENS,
                temperature=0.7
            )
            
            # Calcular métricas
            response_time = time.time() - start_time
            metadata['response_time'] = response_time
            metadata['tokens'] = len(response.split())
            
            return {
                'content': response,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error generando respuesta: {e}")
            return {
                'content': f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}",
                'metadata': metadata
            }

def render_chatbot():
    """Función principal para renderizar el chatbot"""
    chatbot = ChatbotInterface()
    chatbot.render_chat_interface()
```

---

## Especificaciones para Capturas de Pantalla

### 1. Interfaz Principal de la Aplicación

**Ubicación:** Página principal de CogniChat
**Archivo:** `app.py` - función `main()`

**Instrucciones de captura:**
1. Ejecutar la aplicación con `streamlit run app.py`
2. Esperar a que cargue completamente la interfaz
3. Asegurarse de que la barra lateral esté visible
4. Capturar pantalla completa incluyendo:
   - Header con título "CogniChat - Asistente de IA para Análisis Cualitativo"
   - Barra lateral con configuraciones
   - Área principal con pestañas de navegación
   - Footer con información del sistema

**Descripción técnica:**
Esta captura muestra la arquitectura de interfaz de usuario implementada con Streamlit, destacando:
- **Layout responsivo:** Uso de `st.columns()` y `st.container()` para organización
- **Navegación por pestañas:** Implementación de `st.tabs()` para módulos
- **Estado de sesión:** Variables persistentes en `st.session_state`
- **CSS personalizado:** Estilos inyectados para mejorar la apariencia

### 2. Sistema de Carga de Documentos

**Ubicación:** Pestaña "📁 Gestión de Documentos"
**Archivo:** `modules/document_upload.py`

**Instrucciones de captura:**
1. Navegar a la pestaña de gestión de documentos
2. Mostrar el área de drag & drop para archivos
3. Cargar un documento PDF o DOCX de ejemplo
4. Capturar el proceso de validación y progreso
5. Mostrar la lista de documentos cargados con metadatos

**Descripción técnica:**
Demuestra la implementación del patrón Factory para procesamiento de documentos:
- **Validación de archivos:** Verificación de tipos y tamaños permitidos
- **Procesamiento asíncrono:** Uso de `st.progress()` para feedback visual
- **Almacenamiento estructurado:** Organización en directorios por tipo
- **Metadatos automáticos:** Extracción de información del documento

### 3. Configuración del Sistema RAG

**Ubicación:** Barra lateral - Sección "Configuración RAG"
**Archivo:** `app.py` - función `render_sidebar()`

**Instrucciones de captura:**
1. Expandir la sección de configuración RAG en la barra lateral
2. Mostrar los controles deslizantes para:
   - Tamaño de chunk (chunk_size)
   - Solapamiento (chunk_overlap)
   - Máximo de documentos a recuperar
   - Umbral de similitud
3. Capturar también la selección de modelo de embeddings

**Descripción técnica:**
Ilustra la configuración dinámica del sistema RAG:
- **Chunking configurable:** Parámetros ajustables para optimización
- **Modelos intercambiables:** Selección dinámica de modelos de embedding
- **Umbral de relevancia:** Control de calidad de recuperación
- **Persistencia de configuración:** Almacenamiento en `st.session_state`

### 4. Interfaz del Chat Inteligente

**Ubicación:** Pestaña "💬 Chat Inteligente"
**Archivo:** `modules/chatbot.py`

**Instrucciones de captura:**
1. Navegar a la pestaña de chat inteligente
2. Realizar una consulta sobre documentos cargados
3. Esperar la respuesta del sistema
4. Expandir la sección "Información adicional" para mostrar metadatos
5. Capturar el historial de conversación completo

**Descripción técnica:**
Muestra la integración completa del sistema RAG con LLM:
- **Interfaz conversacional:** Diseño tipo chat con burbujas de mensaje
- **Recuperación semántica:** Búsqueda en documentos indexados
- **Generación contextual:** Respuestas basadas en contexto recuperado
- **Trazabilidad:** Metadatos de fuentes y métricas de rendimiento

### 5. Análisis Cualitativo Avanzado

**Ubicación:** Pestaña "📊 Análisis Cualitativo"
**Archivo:** `modules/qualitative_analysis.py`

**Instrucciones de captura:**
1. Navegar a la pestaña de análisis cualitativo
2. Seleccionar documentos para análisis
3. Elegir tipo de análisis (temático o sentimientos)
4. Ejecutar el análisis y esperar resultados
5. Capturar visualizaciones generadas (gráficos, nubes de palabras)

**Descripción técnica:**
Demuestra la implementación del patrón Strategy para análisis:
- **Algoritmos intercambiables:** Diferentes estrategias de análisis
- **Visualización de datos:** Gráficos interactivos con Plotly
- **Procesamiento ML:** Uso de scikit-learn para clustering
- **Exportación de resultados:** Descarga de análisis en formato JSON

### 6. Panel de Monitoreo y Métricas

**Ubicación:** Pestaña "📈 Monitoreo"
**Archivo:** `modules/alerts.py`

**Instrucciones de captura:**
1. Navegar a la pestaña de monitoreo
2. Mostrar métricas del sistema en tiempo real
3. Capturar gráficos de rendimiento
4. Mostrar alertas y notificaciones del sistema
5. Incluir estadísticas de uso de caché y memoria

**Descripción técnica:**
Ilustra el sistema de observabilidad implementado:
- **Métricas en tiempo real:** Actualización automática de estadísticas
- **Alertas inteligentes:** Sistema de notificaciones basado en umbrales
- **Visualización de rendimiento:** Gráficos de líneas y barras
- **Gestión de recursos:** Monitoreo de memoria y caché

### 7. Configuraciones Avanzadas

**Ubicación:** Pestaña "⚙️ Configuraciones"
**Archivo:** `modules/settings.py`

**Instrucciones de captura:**
1. Navegar a la pestaña de configuraciones
2. Mostrar todas las secciones de configuración expandidas
3. Capturar controles para modelos LLM y embedding
4. Mostrar configuraciones de logging y trazabilidad
5. Incluir botones de exportar/importar configuración

**Descripción técnica:**
Demuestra la gestión centralizada de configuraciones:
- **Configuración modular:** Secciones organizadas por funcionalidad
- **Validación en tiempo real:** Verificación de parámetros
- **Persistencia:** Guardado automático de cambios
- **Exportación/Importación:** Backup y restauración de configuraciones

---

## Diagrama de Arquitectura C4

### Nivel 1: Contexto del Sistema

```
workspace "CogniChat Architecture" "Sistema de Análisis Cualitativo con IA" {

    !identifiers hierarchical

    model {
        // Actores del sistema
        investigador = person "Investigador" "Usuario que realiza análisis cualitativo de documentos académicos y de investigación."
        
        estudiante = person "Estudiante" "Usuario que analiza documentos para trabajos académicos y tesis."
        
        administrador = person "Administrador" "Usuario técnico que gestiona el sistema y sus configuraciones."

        // Sistema principal
        cognichat = softwareSystem "CogniChat" "Sistema de análisis cualitativo asistido por IA que permite procesar documentos, realizar consultas inteligentes y generar análisis automatizados." {
            
            // Contenedores principales
            webApp = container "Aplicación Web Streamlit" "Interfaz de usuario principal del sistema" "Python + Streamlit" {
                tags "WebApp"
            }
            
            aiEngine = container "Motor de IA" "Procesamiento de lenguaje natural y generación de respuestas" "Ollama + LLM" {
                tags "AIEngine"
            }
            
            ragSystem = container "Sistema RAG" "Recuperación y generación aumentada por documentos" "Python + Vector DB" {
                tags "RAGSystem"
            }
            
            documentProcessor = container "Procesador de Documentos" "Extracción y procesamiento de texto de documentos" "Python + PyPDF2 + python-docx" {
                tags "DocumentProcessor"
            }
            
            cacheSystem = container "Sistema de Caché" "Caché multinivel para optimización de rendimiento" "Python + File System" {
                tags "CacheSystem"
            }
            
            configManager = container "Gestor de Configuración" "Gestión centralizada de configuraciones del sistema" "Python + JSON" {
                tags "ConfigManager"
            }
        }

        // Sistemas externos
        ollamaServer = softwareSystem "Servidor Ollama" "Servidor de modelos de lenguaje local" {
            tags "External"
        }
        
        fileSystem = softwareSystem "Sistema de Archivos" "Almacenamiento local de documentos y caché" {
            tags "External"
        }

        // Relaciones principales
        investigador -> cognichat "Analiza documentos y realiza consultas"
        estudiante -> cognichat "Procesa documentos académicos"
        administrador -> cognichat "Configura y administra el sistema"
        
        cognichat -> ollamaServer "Utiliza modelos de IA"
        cognichat -> fileSystem "Almacena documentos y caché"
        
        // Relaciones entre contenedores
        investigador -> webApp "Interactúa con la interfaz"
        estudiante -> webApp "Usa la aplicación web"
        administrador -> webApp "Administra configuraciones"
        
        webApp -> aiEngine "Solicita generación de respuestas"
        webApp -> ragSystem "Realiza búsquedas semánticas"
        webApp -> documentProcessor "Procesa documentos cargados"
        webApp -> configManager "Gestiona configuraciones"
        
        aiEngine -> ollamaServer "Genera respuestas con LLM"
        ragSystem -> aiEngine "Obtiene embeddings"
        ragSystem -> cacheSystem "Almacena y recupera vectores"
        documentProcessor -> fileSystem "Lee documentos"
        cacheSystem -> fileSystem "Persiste caché"
        configManager -> fileSystem "Guarda configuraciones"
    }

    views {
        systemContext cognichat "DiagramaContexto" {
            include *
            autolayout lr
            title "CogniChat - Diagrama de Contexto"
            description "Vista general del sistema CogniChat y sus usuarios"
        }

        container cognichat "DiagramaContenedores" {
            include *
            autolayout tb
            title "CogniChat - Diagrama de Contenedores"
            description "Arquitectura interna del sistema CogniChat"
        }

        styles {
            element "Person" {
                color #ffffff
                background #1f77b4
                stroke #1f77b4
                strokeWidth 2
                shape person
                fontSize 12
            }

            element "Software System" {
                color #ffffff
                background #2ca02c
                stroke #2ca02c
                strokeWidth 2
                shape roundedbox
                fontSize 14
            }
            
            element "External" {
                color #ffffff
                background #ff7f0e
                stroke #ff7f0e
                strokeWidth 2
                shape roundedbox
            }
            
            element "WebApp" {
                color #ffffff
                background #d62728
                stroke #d62728
                strokeWidth 2
                shape roundedbox
                icon "🌐"
            }
            
            element "AIEngine" {
                color #ffffff
                background #9467bd
                stroke #9467bd
                strokeWidth 2
                shape roundedbox
                icon "🤖"
            }
            
            element "RAGSystem" {
                color #ffffff
                background #8c564b
                stroke #8c564b
                strokeWidth 2
                shape roundedbox
                icon "🔍"
            }
            
            element "DocumentProcessor" {
                color #ffffff
                background #e377c2
                stroke #e377c2
                strokeWidth 2
                shape roundedbox
                icon "📄"
            }
            
            element "CacheSystem" {
                color #ffffff
                background #7f7f7f
                stroke #7f7f7f
                strokeWidth 2
                shape roundedbox
                icon "💾"
            }
            
            element "ConfigManager" {
                color #ffffff
                background #bcbd22
                stroke #bcbd22
                strokeWidth 2
                shape roundedbox
                icon "⚙️"
            }
            
            relationship "Relationship" {
                color #666666
                strokeWidth 2
                fontSize 10
            }
        }
    }
}
```

### Nivel 2: Contenedores - Explicación Técnica

#### Aplicación Web Streamlit
- **Responsabilidad:** Interfaz de usuario principal
- **Tecnología:** Python + Streamlit
- **Características:**
  - Renderizado reactivo de componentes
  - Gestión de estado de sesión
  - Navegación por pestañas
  - Formularios interactivos

#### Motor de IA
- **Responsabilidad:** Procesamiento de lenguaje natural
- **Tecnología:** Ollama + Modelos LLM
- **Características:**
  - Generación de respuestas contextuales
  - Creación de embeddings
  - Gestión de modelos múltiples
  - Optimización de prompts

#### Sistema RAG
- **Responsabilidad:** Recuperación y generación aumentada
- **Tecnología:** Python + Algoritmos de similitud
- **Características:**
  - Búsqueda semántica vectorial
  - Chunking inteligente de documentos
  - Ranking por relevancia
  - Caché de embeddings

### Nivel 3: Componentes - Arquitectura Interna

```python
# Diagrama de componentes para el Sistema RAG
"""
┌─────────────────────────────────────────────────────────────┐
│                    Sistema RAG                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Text Chunker  │  │ Embedding Gen.  │  │ Vector Store │ │
│  │                 │  │                 │  │              │ │
│  │ - Smart split   │  │ - Model mgmt    │  │ - Similarity │ │
│  │ - Overlap ctrl  │  │ - Batch proc.   │  │ - Indexing   │ │
│  │ - Context pres. │  │ - Caching       │  │ - Retrieval  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│           │                     │                    │      │
│           └─────────────────────┼────────────────────┘      │
│                                 │                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Semantic Search Engine                     │ │
│  │                                                         │ │
│  │ - Query processing                                      │ │
│  │ - Similarity calculation                                │ │
│  │ - Result ranking                                        │ │
│  │ - Context assembly                                      │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
"""
```

### Nivel 4: Código - Implementación Detallada

El nivel de código se representa a través de los ejemplos de implementación mostrados anteriormente, destacando:

#### Patrones de Diseño Implementados:
1. **Factory Pattern:** `DocumentProcessorFactory` para crear procesadores específicos
2. **Strategy Pattern:** `AnalysisStrategy` para algoritmos de análisis intercambiables  
3. **Observer Pattern:** `TraceabilityManager` para eventos del sistema
4. **Repository Pattern:** `DocumentRepository` para abstracción de datos
5. **Singleton Pattern:** `AppConfig` para configuración global

#### Elementos Gráficos del Diagrama:

- **Personas (👤):** Representan usuarios del sistema con diferentes roles
- **Sistemas de Software (📦):** Contenedores principales de funcionalidad
- **Sistemas Externos (🔗):** Dependencias externas como Ollama
- **Relaciones (→):** Flujo de datos e interacciones entre componentes
- **Colores:** Codificación visual por tipo de componente
- **Iconos:** Identificación rápida de la función de cada elemento

Este diagrama C4 proporciona una vista completa de la arquitectura desde el contexto general hasta los detalles de implementación, facilitando la comprensión tanto para stakeholders técnicos como no técnicos.

---

## Conclusión

Esta documentación arquitectónica detallada de CogniChat presenta un sistema robusto, escalable y bien estructurado que implementa las mejores prácticas de desarrollo de software. La combinación de patrones de diseño establecidos, arquitectura en capas, integración avanzada de IA y sistemas de caché multinivel resulta en una solución técnicamente sólida para el análisis cualitativo asistido por inteligencia artificial.

La modularidad del sistema permite futuras extensiones y modificaciones sin comprometer la estabilidad, mientras que el sistema de observabilidad garantiza la trazabilidad y el monitoreo continuo del rendimiento.