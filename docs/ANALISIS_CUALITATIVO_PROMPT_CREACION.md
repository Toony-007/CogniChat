# 🤖 Prompt para Crear Módulo de Análisis Cualitativo Paso a Paso

## 📋 Objetivo

Este documento proporciona un **prompt detallado y estructurado** para recrear el módulo de análisis cualitativo de forma **modular, independiente y coherente**. Cada submódulo se puede crear por separado sin afectar a los demás.

---

## 🎯 Principios de Diseño

### Reglas Fundamentales

1. **Separación de Responsabilidades**: Cada clase/módulo tiene una única responsabilidad
2. **Independencia**: Los submódulos no deben depender directamente entre sí
3. **Coherencia**: Todos los submódulos siguen las mismas convenciones
4. **Extensibilidad**: Fácil agregar nuevas funcionalidades
5. **Mantenibilidad**: Código limpio, documentado y organizado

### Arquitectura en UN SOLO ARCHIVO

El módulo completo está contenido en: `modules/qualitative_analysis.py`

**Organización interna del archivo (6,099 líneas)**:

```
qualitative_analysis.py
│
├─ SECCIÓN 1: IMPORTS Y CONFIGURACIÓN (Líneas 1-188)
│  └─ Todas las importaciones y configuración global
│
├─ SECCIÓN 2: ENUMS Y DATACLASSES (Líneas 189-247)
│  ├─ AnalysisType (enum)
│  ├─ VisualizationType (enum)
│  ├─ AnalysisConfig (dataclass)
│  ├─ ConceptData (dataclass)
│  └─ AnalysisResult (dataclass)
│
├─ SECCIÓN 3: CLASES BASE (Líneas 248-271)
│  └─ BaseAnalyzer (abstract class)
│
├─ SECCIÓN 4: GESTIÓN DE CACHE (Líneas 272-324)
│  └─ CacheManager (class)
│
├─ SECCIÓN 5: PREPROCESAMIENTO (Líneas 325-394)
│  └─ TextPreprocessor (class)
│
├─ SECCIÓN 6: EXTRACCIÓN DE CONCEPTOS (Líneas 395-591)
│  └─ ConceptExtractor (class)
│
├─ SECCIÓN 7: ANÁLISIS DE TEMAS (Líneas 592-807)
│  └─ ThemeAnalyzer (class)
│
├─ SECCIÓN 8: ANÁLISIS DE SENTIMIENTOS (Líneas 808-1023)
│  └─ SentimentAnalyzer (class)
│
├─ SECCIÓN 9: CLASE PRINCIPAL (Líneas 1024-6008)
│  ├─ AdvancedQualitativeAnalyzer (class principal)
│  ├─ Métodos de validación y utilidades
│  ├─ Métodos de compatibilidad
│  ├─ Métodos de análisis principales
│  ├─ Resúmenes automáticos
│  ├─ Análisis paralelo
│  ├─ Configuración y optimización
│  ├─ Mapas conceptuales
│  ├─ Mapas mentales
│  ├─ Triangulación
│  ├─ Clustering
│  └─ Nubes de palabras
│
└─ SECCIÓN 10: FUNCIONES DE RENDERIZADO (Líneas 4478-6099)
   ├─ render_advanced_dashboard()
   ├─ render_advanced_themes()
   ├─ render_clustering_analysis()
   ├─ render_sentiment_analysis()
   ├─ render_word_cloud()
   ├─ render_interactive_concept_map()
   ├─ render_interactive_mind_map()
   ├─ render_automatic_summary()
   ├─ render_triangulation_analysis()
   └─ render() - Función principal
```

**Ventajas de esta arquitectura de un solo archivo**:
✅ Todo el código está en un solo lugar
✅ Secciones claramente separadas con comentarios
✅ Fácil de navegar con Ctrl+G + número de línea
✅ Coherencia garantizada en un solo archivo
✅ Sin problemas de imports entre módulos

---

## 📝 Prompts para Crear Cada Sección en el Archivo Único

### PASO 1: Crear Sección de Configuración (Líneas 145-247)

```markdown
# PROMPT PARA IA:

En el archivo `modules/qualitative_analysis.py`, crea la SECCIÓN 2 (líneas 145-247) con las siguientes especificaciones:

## Requisitos:
1. Define enumeraciones para tipos de análisis y visualización
2. Crea dataclasses para configuración, datos de conceptos y resultados
3. Implementa validación de configuración
4. Proporciona valores por defecto sensatos

## Estructura esperada:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

class AnalysisType(Enum):
    """Define todos los tipos de análisis disponibles"""
    # Agregar tipos...

class VisualizationType(Enum):
    """Define tipos de visualización"""
    # Agregar tipos...

@dataclass
class AnalysisConfig:
    """Configuración centralizada para análisis"""
    # Parámetros generales
    min_frequency: int = 2
    max_concepts: int = 50
    similarity_threshold: float = 0.6
    
    # Optimización
    enable_cache: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Validación
    def validate(self) -> bool:
        """Validar que la configuración sea correcta"""
        if self.min_frequency < 1:
            raise ValueError("min_frequency debe ser >= 1")
        if self.max_concepts < 1:
            raise ValueError("max_concepts debe ser >= 1")
        return True

@dataclass
class ConceptData:
    """Estructura para almacenar un concepto extraído"""
    concept: str
    score: float
    frequency: int
    context: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    sentiment: Optional[float] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'concept': self.concept,
            'score': self.score,
            'frequency': self.frequency,
            'context': self.context,
            'related_concepts': self.related_concepts,
            'sentiment': self.sentiment,
            'category': self.category
        }

@dataclass
class AnalysisResult:
    """Resultado estandarizado de cualquier análisis"""
    analysis_type: AnalysisType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'analysis_type': self.analysis_type.value,
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time
        }
```

## Criterios de aceptación:
- [ ] Todas las enumeraciones están completas
- [ ] Dataclasses tienen métodos to_dict()
- [ ] Validación implementada y probada
- [ ] Sin dependencias externas más allá de stdlib
```

---

### PASO 2: Crear Sección de Cache (Líneas 272-324)

```markdown
# PROMPT PARA IA:

En el archivo `modules/qualitative_analysis.py`, crea la SECCIÓN 4 (líneas 272-324) con un sistema de cache thread-safe y eficiente.

## Requisitos:
1. Implementar cache LRU (Least Recently Used)
2. Thread-safe con threading.Lock
3. Métricas de rendimiento (hit ratio, tamaño)
4. Auto-eviction cuando alcanza tamaño máximo
5. Serialización opcional para persistencia

## Estructura esperada:

```python
import threading
from typing import Any, Optional, Dict
from datetime import datetime
import json
from pathlib import Path

class CacheManager:
    """Gestor de cache LRU thread-safe"""
    
    def __init__(self, max_size: int = 1000, persistent: bool = False, 
                 cache_file: Optional[Path] = None):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}
        self.lock = threading.Lock()
        self.persistent = persistent
        self.cache_file = cache_file
        
        # Estadísticas
        self._hits = 0
        self._misses = 0
        
        # Cargar cache persistente si existe
        if persistent and cache_file and cache_file.exists():
            self._load_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener elemento del cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = datetime.now()
                self._hits += 1
                return self.cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Guardar elemento en cache"""
        with self.lock:
            # Eviction si es necesario
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = value
            self.access_times[key] = datetime.now()
            
            # Persistir si está habilitado
            if self.persistent:
                self._save_cache()
    
    def _evict_oldest(self) -> None:
        """Eliminar el elemento menos usado recientemente"""
        if not self.access_times:
            return
        
        oldest_key = min(
            self.access_times.keys(),
            key=lambda k: self.access_times[k]
        )
        
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self) -> None:
        """Limpiar todo el cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        with self.lock:
            hit_ratio = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_ratio': hit_ratio,
                'utilization': len(self.cache) / self.max_size
            }
    
    def _save_cache(self) -> None:
        """Guardar cache en disco"""
        if not self.cache_file:
            return
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error guardando cache: {e}")
    
    def _load_cache(self) -> None:
        """Cargar cache desde disco"""
        if not self.cache_file:
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        except Exception as e:
            print(f"Error cargando cache: {e}")

# Factory para diferentes tipos de cache
class CacheFactory:
    """Factory para crear diferentes tipos de cache"""
    
    @staticmethod
    def create_cache(cache_type: str = 'memory', **kwargs) -> CacheManager:
        """Crear cache según tipo"""
        if cache_type == 'memory':
            return CacheManager(**kwargs)
        elif cache_type == 'persistent':
            return CacheManager(persistent=True, **kwargs)
        elif cache_type == 'redis':
            # Implementar RedisCacheManager
            pass
        else:
            raise ValueError(f"Tipo de cache no soportado: {cache_type}")
```

## Criterios de aceptación:
- [ ] Thread-safe verificado
- [ ] LRU eviction funciona correctamente
- [ ] Estadísticas precisas
- [ ] Persistencia opcional funciona
- [ ] Tests unitarios incluidos
```

---

### PASO 3: Crear Sección de Clases Base (Líneas 248-271)

```markdown
# PROMPT PARA IA:

En el archivo `modules/qualitative_analysis.py`, crea la SECCIÓN 3 (líneas 248-271) con clases base y interfaces abstractas.

## Requisitos:
1. Definir clase abstracta BaseAnalyzer
2. Implementar métodos comunes de validación
3. Proveer interfaz consistente para todos los analizadores
4. Incluir logging y manejo de errores

## Estructura esperada:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
from .config import AnalysisConfig, AnalysisResult

logger = logging.getLogger(__name__)

class BaseAnalyzer(ABC):
    """Clase base abstracta para todos los analizadores"""
    
    def __init__(self, config: AnalysisConfig):
        """
        Args:
            config: Configuración del análisis
        """
        self.config = config
        self.cache = {}
        self.logger = logger
        
        # Validar configuración
        if not self.config.validate():
            raise ValueError("Configuración inválida")
    
    @abstractmethod
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """
        Método abstracto que debe implementar cada analizador
        
        Args:
            chunks: Lista de chunks de documentos
            
        Returns:
            AnalysisResult con los datos del análisis
        """
        pass
    
    def _validate_input(self, chunks: List[Dict]) -> bool:
        """
        Validar que la entrada sea válida
        
        Args:
            chunks: Lista de chunks a validar
            
        Returns:
            True si la entrada es válida
        """
        if not chunks or not isinstance(chunks, list):
            self.logger.warning("Chunks inválidos: lista vacía o tipo incorrecto")
            return False
        
        valid_chunks = sum(
            1 for chunk in chunks 
            if isinstance(chunk, dict) and chunk.get('content')
        )
        
        if valid_chunks == 0:
            self.logger.warning("No hay chunks válidos con contenido")
            return False
        
        self.logger.info(f"Validados {valid_chunks} chunks de {len(chunks)} totales")
        return True
    
    def _generate_cache_key(self, chunks: List[Dict], suffix: str = "") -> str:
        """
        Generar clave única para cache
        
        Args:
            chunks: Lista de chunks
            suffix: Sufijo opcional para la clave
            
        Returns:
            Clave de cache única
        """
        import hashlib
        
        content_hash = hash(
            str(sorted([chunk.get('content', '')[:100] for chunk in chunks]))
        )
        
        return f"{self.__class__.__name__}_{content_hash}{suffix}"
    
    def _measure_processing_time(self, func):
        """Decorador para medir tiempo de procesamiento"""
        from functools import wraps
        from datetime import datetime
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                f"{func.__name__} completado en {processing_time:.2f}s"
            )
            
            return result
        return wrapper

class AnalyzerFactory:
    """Factory para crear analizadores según tipo"""
    
    _analyzers = {}
    
    @classmethod
    def register(cls, analysis_type: str, analyzer_class):
        """Registrar un analizador"""
        cls._analyzers[analysis_type] = analyzer_class
    
    @classmethod
    def create(cls, analysis_type: str, config: AnalysisConfig) -> BaseAnalyzer:
        """Crear analizador según tipo"""
        if analysis_type not in cls._analyzers:
            raise ValueError(f"Analizador no registrado: {analysis_type}")
        
        return cls._analyzers[analysis_type](config)
    
    @classmethod
    def get_available_analyzers(cls) -> List[str]:
        """Obtener lista de analizadores disponibles"""
        return list(cls._analyzers.keys())
```

## Criterios de aceptación:
- [ ] BaseAnalyzer es completamente abstracto
- [ ] Métodos de validación robustos
- [ ] Factory pattern implementado
- [ ] Logging en todas las operaciones
- [ ] Sin dependencias entre analizadores
```

---

### PASO 4: Crear Sección de Preprocesamiento (Líneas 325-394)

```markdown
# PROMPT PARA IA:

En el archivo `modules/qualitative_analysis.py`, crea la SECCIÓN 5 (líneas 325-394) que maneje todo el preprocesamiento de texto.

## Requisitos:
1. Preprocesamiento de texto en español
2. Gestión de stopwords con cache
3. Normalización y limpieza
4. Independiente de otros módulos

## Estructura esperada:

```python
import re
from typing import List, Set, Optional
import logging

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocesador de texto especializado para español"""
    
    def __init__(self, custom_stopwords: Optional[Set[str]] = None):
        """
        Args:
            custom_stopwords: Stopwords adicionales específicas del dominio
        """
        self._stopwords_cache: Optional[List[str]] = None
        self.custom_stopwords = custom_stopwords or set()
        self.logger = logger
        
        # Inicializar NLTK si está disponible
        self._initialize_nltk()
    
    def _initialize_nltk(self) -> None:
        """Inicializar recursos de NLTK"""
        if NLTK_AVAILABLE:
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            except Exception as e:
                self.logger.warning(f"Error descargando recursos NLTK: {e}")
    
    def get_spanish_stopwords(self) -> List[str]:
        """
        Obtener lista de stopwords en español con cache
        
        Returns:
            Lista de stopwords
        """
        if self._stopwords_cache is not None:
            return self._stopwords_cache
        
        stopwords_set = set()
        
        # Intentar cargar de NLTK
        if NLTK_AVAILABLE:
            try:
                stopwords_set.update(stopwords.words('spanish'))
            except Exception as e:
                self.logger.warning(f"Error cargando stopwords de NLTK: {e}")
        
        # Stopwords básicas como fallback
        if not stopwords_set:
            stopwords_set = {
                'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se',
                'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con',
                'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más'
                # ... agregar más
            }
        
        # Agregar stopwords personalizadas
        stopwords_set.update(self.custom_stopwords)
        
        # Agregar stopwords específicas del dominio
        domain_stopwords = {
            'también', 'puede', 'ser', 'está', 'están', 'hacer',
            'tiene', 'tienen', 'muy', 'más', 'menos', 'bien'
        }
        stopwords_set.update(domain_stopwords)
        
        # Cachear y retornar
        self._stopwords_cache = list(stopwords_set)
        return self._stopwords_cache
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = True,
                       min_word_length: int = 2) -> str:
        """
        Preprocesar texto para análisis
        
        Args:
            text: Texto a procesar
            remove_stopwords: Si remover stopwords
            min_word_length: Longitud mínima de palabras
            
        Returns:
            Texto procesado
        """
        if not text:
            return ""
        
        # 1. Normalizar a minúsculas
        text = text.lower()
        
        # 2. Remover caracteres especiales (mantener espacios)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 3. Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. Tokenizar
        words = text.split()
        
        # 5. Filtrar palabras
        if remove_stopwords:
            stopwords = set(self.get_spanish_stopwords())
            words = [
                word for word in words 
                if word not in stopwords and len(word) >= min_word_length
            ]
        else:
            words = [word for word in words if len(word) >= min_word_length]
        
        return ' '.join(words)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Dividir texto en oraciones
        
        Args:
            text: Texto a dividir
            
        Returns:
            Lista de oraciones
        """
        if NLTK_AVAILABLE:
            try:
                from nltk.tokenize import sent_tokenize
                return sent_tokenize(text, language='spanish')
            except Exception as e:
                self.logger.warning(f"Error en tokenización NLTK: {e}")
        
        # Fallback simple
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extraer n-gramas del texto
        
        Args:
            text: Texto a procesar
            n: Tamaño de los n-gramas
            
        Returns:
            Lista de n-gramas
        """
        words = self.preprocess_text(text).split()
        
        if len(words) < n:
            return []
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def add_custom_stopword(self, word: str) -> None:
        """Agregar stopword personalizada"""
        self.custom_stopwords.add(word.lower())
        self._stopwords_cache = None  # Invalidar cache
    
    def remove_custom_stopword(self, word: str) -> None:
        """Remover stopword personalizada"""
        self.custom_stopwords.discard(word.lower())
        self._stopwords_cache = None  # Invalidar cache
```

## Criterios de aceptación:
- [ ] Funciona sin NLTK (fallback)
- [ ] Cache de stopwords implementado
- [ ] Métodos de tokenización robustos
- [ ] Personalización de stopwords
- [ ] Sin dependencias de otros módulos del proyecto
```

---

### PASO 5: Crear Sección de Extracción de Conceptos (Líneas 395-591)

```markdown
# PROMPT PARA IA:

En el archivo `modules/qualitative_analysis.py`, crea la SECCIÓN 6 (líneas 395-591) que extraiga conceptos clave del contenido.

## Requisitos:
1. Implementar extracción con TF-IDF
2. Fallback con análisis de frecuencia
3. Enriquecer conceptos con contexto
4. Encontrar conceptos relacionados
5. Independiente de otros analizadores

## Estructura esperada:

```python
from typing import List, Dict
from datetime import datetime
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# NOTA: No se necesitan imports porque todo está en el mismo archivo
# Las clases ya están definidas en secciones anteriores:
# - BaseAnalyzer (Línea 252)
# - AnalysisConfig, AnalysisResult, AnalysisType, ConceptData (Líneas 218-246)
# - CacheManager (Línea 276)
# - TextPreprocessor (Línea 329)

class ConceptExtractor(BaseAnalyzer):
    """Extractor de conceptos clave usando TF-IDF y análisis de frecuencia"""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.preprocessor = TextPreprocessor()
        self.cache_manager = CacheManager(max_size=500)
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """
        Extraer conceptos clave del contenido
        
        Args:
            chunks: Lista de chunks de documentos
            
        Returns:
            AnalysisResult con conceptos extraídos
        """
        start_time = datetime.now()
        
        # Validar entrada
        if not self._validate_input(chunks):
            return AnalysisResult(
                analysis_type=AnalysisType.CONCEPT_EXTRACTION,
                data={'concepts': []},
                metadata={'error': 'Invalid input data'}
            )
        
        # Verificar cache
        cache_key = self._generate_cache_key(chunks, f"_{self.config.min_frequency}")
        if self.config.enable_cache:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.logger.info("Usando conceptos desde cache")
                return cached_result
        
        # Procesar chunks
        concepts = self._extract_concepts(chunks)
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Crear resultado
        result = AnalysisResult(
            analysis_type=AnalysisType.CONCEPT_EXTRACTION,
            data={'concepts': concepts},
            metadata={
                'total_chunks': len(chunks),
                'concepts_found': len(concepts),
                'processing_method': 'tfidf' if SKLEARN_AVAILABLE else 'frequency',
                'cache_stats': self.cache_manager.get_stats()
            },
            processing_time=processing_time
        )
        
        # Guardar en cache
        if self.config.enable_cache:
            self.cache_manager.set(cache_key, result)
        
        return result
    
    def _extract_concepts(self, chunks: List[Dict]) -> List[ConceptData]:
        """Extraer conceptos usando el mejor método disponible"""
        # Preparar textos
        texts = self._prepare_texts(chunks)
        
        if not texts:
            return []
        
        # Estrategia principal: TF-IDF
        if SKLEARN_AVAILABLE and len(texts) >= 2:
            concepts = self._extract_with_tfidf(texts)
            if concepts:
                return self._enrich_concepts(concepts, chunks)
        
        # Fallback: Frecuencia
        concepts = self._extract_with_frequency(texts)
        return self._enrich_concepts(concepts, chunks)
    
    def _prepare_texts(self, chunks: List[Dict]) -> List[str]:
        """Preparar textos para análisis"""
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            if content:
                processed = self.preprocessor.preprocess_text(content)
                if processed and len(processed.split()) > 5:
                    texts.append(processed)
        return texts
    
    def _extract_with_tfidf(self, texts: List[str]) -> List[ConceptData]:
        """Extraer conceptos usando TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=self.config.max_concepts * 2,
                stop_words=self.preprocessor.get_spanish_stopwords(),
                ngram_range=(1, 3),
                min_df=max(1, len(texts) // 10),
                max_df=0.9
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            concepts = []
            for i, score in enumerate(mean_scores):
                if score > 0:
                    concept = ConceptData(
                        concept=feature_names[i],
                        score=float(score),
                        frequency=int(score * len(texts) * 10)
                    )
                    concepts.append(concept)
            
            return sorted(concepts, key=lambda x: x.score, reverse=True)[:self.config.max_concepts]
            
        except Exception as e:
            self.logger.warning(f"Error en TF-IDF: {e}")
            return []
    
    def _extract_with_frequency(self, texts: List[str]) -> List[ConceptData]:
        """Extraer conceptos usando análisis de frecuencia"""
        from collections import Counter
        
        all_text = " ".join(texts)
        words = all_text.split()
        word_freq = Counter(words)
        
        total_words = len(words)
        concepts = []
        
        for word, freq in word_freq.most_common(self.config.max_concepts * 2):
            if (len(word) > 3 and 
                freq >= self.config.min_frequency and
                word.isalpha()):
                
                concept = ConceptData(
                    concept=word,
                    score=freq / total_words,
                    frequency=freq
                )
                concepts.append(concept)
        
        return concepts[:self.config.max_concepts]
    
    def _enrich_concepts(self, concepts: List[ConceptData], chunks: List[Dict]) -> List[ConceptData]:
        """Enriquecer conceptos con contexto y relaciones"""
        for concept in concepts:
            # Agregar contexto
            concept.context = self._find_context(concept.concept, chunks)
            
            # Agregar conceptos relacionados
            concept.related_concepts = self._find_related(concept.concept, concepts)
        
        return concepts
    
    def _find_context(self, concept: str, chunks: List[Dict], max_contexts: int = 3) -> List[str]:
        """Encontrar contexto donde aparece el concepto"""
        contexts = []
        concept_lower = concept.lower()
        
        for chunk in chunks:
            content = chunk.get('content', '')
            if concept_lower in content.lower():
                sentences = self.preprocessor.tokenize_sentences(content)
                for sentence in sentences:
                    if concept_lower in sentence.lower() and len(sentence.strip()) > 20:
                        contexts.append(sentence.strip())
                        if len(contexts) >= max_contexts:
                            break
                if len(contexts) >= max_contexts:
                    break
        
        return contexts
    
    def _find_related(self, concept: str, all_concepts: List[ConceptData], max_related: int = 5) -> List[str]:
        """Encontrar conceptos relacionados"""
        related = []
        concept_words = set(concept.lower().split())
        
        for other_concept in all_concepts:
            if other_concept.concept == concept:
                continue
            
            other_words = set(other_concept.concept.lower().split())
            
            # Similitud por palabras compartidas
            if concept_words & other_words:
                related.append(other_concept.concept)
                if len(related) >= max_related:
                    break
        
        return related

# NOTA: En el archivo único, las clases se usan directamente
# La clase AdvancedQualitativeAnalyzer (Línea 1024) las instancia en __init__:
# self.concept_extractor = ConceptExtractor(self.config)
```

## Criterios de aceptación:
- [ ] Funciona con y sin scikit-learn
- [ ] Cache implementado y funcional
- [ ] Enriquecimiento de conceptos completo
- [ ] No depende de otros analizadores
- [ ] Registrado en AnalyzerFactory
```

---

### PASO 6: Crear Sección de Análisis de Temas (Líneas 592-807)

```markdown
# PROMPT PARA IA:

En el archivo `modules/qualitative_analysis.py`, crea la SECCIÓN 7 (líneas 592-807) que analice temas del contenido.

## Requisitos:
1. Implementar LDA para modelado de temas
2. Fallback con clustering de palabras clave
3. Calcular coherencia de temas
4. Independiente de otros analizadores

## Estructura esperada:

```python
from typing import List, Dict
from datetime import datetime
import numpy as np
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.base import BaseAnalyzer
from ..core.config import AnalysisConfig, AnalysisResult, AnalysisType
from ..core.cache import CacheManager
from ..preprocessing.text_processor import TextPreprocessor

class ThemeAnalyzer(BaseAnalyzer):
    """Analizador de temas usando LDA y clustering"""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.preprocessor = TextPreprocessor()
        self.cache_manager = CacheManager(max_size=200)
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """
        Analizar temas del contenido
        
        Args:
            chunks: Lista de chunks de documentos
            
        Returns:
            AnalysisResult con temas identificados
        """
        start_time = datetime.now()
        
        # Validar entrada
        if not self._validate_input(chunks):
            return AnalysisResult(
                analysis_type=AnalysisType.THEME_ANALYSIS,
                data={'themes': []},
                metadata={'error': 'Invalid input data'}
            )
        
        # Verificar cache
        cache_key = self._generate_cache_key(chunks, f"_{self.config.max_concepts}")
        if self.config.enable_cache:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.logger.info("Usando temas desde cache")
                return cached_result
        
        # Extraer temas
        themes = self._extract_themes(chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Crear resultado
        result = AnalysisResult(
            analysis_type=AnalysisType.THEME_ANALYSIS,
            data={'themes': themes},
            metadata={
                'total_chunks': len(chunks),
                'themes_found': len(themes),
                'processing_method': 'lda' if SKLEARN_AVAILABLE else 'clustering'
            },
            processing_time=processing_time
        )
        
        # Guardar en cache
        if self.config.enable_cache:
            self.cache_manager.set(cache_key, result)
        
        return result
    
    def _extract_themes(self, chunks: List[Dict]) -> List[Dict]:
        """Extraer temas usando el mejor método disponible"""
        # Preparar textos
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '').strip()
            if content:
                processed = self.preprocessor.preprocess_text(content)
                if processed and len(processed.split()) > 10:
                    texts.append(processed)
        
        if not texts:
            return []
        
        # Estrategia principal: LDA
        if SKLEARN_AVAILABLE and len(texts) >= 3:
            themes = self._extract_with_lda(texts)
            if themes:
                return themes
        
        # Fallback: Clustering
        return self._extract_with_clustering(texts)
    
    def _extract_with_lda(self, texts: List[str]) -> List[Dict]:
        """Extraer temas usando LDA"""
        try:
            n_topics = min(
                self.config.max_concepts,
                max(3, len(texts) // 3)
            )
            
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=self.preprocessor.get_spanish_stopwords(),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100,
                learning_method='batch'
            )
            
            lda.fit(tfidf_matrix)
            
            # Extraer temas
            themes = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                weights = [topic[i] for i in top_indices]
                
                theme = {
                    'id': topic_idx,
                    'name': f"Tema {topic_idx + 1}",
                    'keywords': keywords,
                    'weights': weights,
                    'coherence': self._calculate_coherence(keywords, texts),
                    'description': self._describe_theme(keywords)
                }
                themes.append(theme)
            
            return sorted(themes, key=lambda x: x['coherence'], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error en LDA: {e}")
            return []
    
    def _extract_with_clustering(self, texts: List[str]) -> List[Dict]:
        """Extraer temas usando clustering"""
        from collections import Counter
        
        # Extraer palabras clave
        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend([w for w in words if len(w) > 3])
        
        word_freq = Counter(all_words)
        
        # Top palabras
        keywords = [
            word for word, freq in word_freq.most_common(50)
            if freq >= 2
        ]
        
        if len(keywords) < 5:
            return []
        
        # Agrupar en temas
        n_themes = min(5, max(2, len(keywords) // 10))
        theme_size = len(keywords) // n_themes
        
        themes = []
        for i in range(n_themes):
            start = i * theme_size
            end = start + theme_size if i < n_themes - 1 else len(keywords)
            theme_keywords = keywords[start:end]
            
            theme = {
                'id': i,
                'name': f"Tema {i + 1}",
                'keywords': theme_keywords,
                'weights': [1.0] * len(theme_keywords),
                'coherence': len(theme_keywords) / len(keywords),
                'description': self._describe_theme(theme_keywords)
            }
            themes.append(theme)
        
        return themes
    
    def _calculate_coherence(self, keywords: List[str], texts: List[str]) -> float:
        """Calcular coherencia del tema"""
        if len(keywords) < 2:
            return 0.0
        
        co_occurrences = 0
        total_pairs = 0
        
        for text in texts:
            text_words = set(text.split())
            for i, word1 in enumerate(keywords):
                for word2 in keywords[i+1:]:
                    total_pairs += 1
                    if word1 in text_words and word2 in text_words:
                        co_occurrences += 1
        
        return co_occurrences / total_pairs if total_pairs > 0 else 0.0
    
    def _describe_theme(self, keywords: List[str]) -> str:
        """Generar descripción del tema"""
        if not keywords:
            return "Tema sin descripción"
        
        main_keywords = keywords[:3]
        return f"Tema relacionado con: {', '.join(main_keywords)}"

# Registrar en factory
from ..core.base import AnalyzerFactory
AnalyzerFactory.register('theme_analysis', ThemeAnalyzer)
```

## Criterios de aceptación:
- [ ] LDA funciona correctamente
- [ ] Fallback a clustering robusto
- [ ] Coherencia calculada correctamente
- [ ] Completamente independiente
- [ ] Registrado en factory
```

---

## 🎯 Orden de Creación Recomendado (Todo en UN SOLO ARCHIVO)

### Fase 1: Estructura Base (Día 1)
```
En modules/qualitative_analysis.py, crear de arriba hacia abajo:

1. Líneas 1-188:    Imports y configuración global
2. Líneas 189-247:  Enums, dataclasses y estructuras (AnalysisType, AnalysisConfig, etc.)
3. Líneas 248-271:  Clases base e interfaces (BaseAnalyzer)
4. Líneas 272-324:  Gestión de cache (CacheManager)
5. Líneas 325-394:  Preprocesamiento (TextPreprocessor)
```

### Fase 2: Analizadores Especializados (Día 2-3)
```
6. Líneas 395-591:  Extracción de conceptos (ConceptExtractor)
7. Líneas 592-807:  Análisis de temas (ThemeAnalyzer)
8. Líneas 808-1023: Análisis de sentimientos (SentimentAnalyzer)
```

### Fase 3: Clase Principal - Parte 1 (Día 4-5)
```
9. Líneas 1024-1176:  Inicialización y carga de datos
10. Líneas 1177-1424: Resúmenes automáticos
11. Líneas 1425-1657: Métodos de análisis principales
12. Líneas 1658-1883: Análisis paralelo y optimización
```

### Fase 4: Clase Principal - Parte 2 (Día 6-7)
```
13. Líneas 1977-2292:  Mapas conceptuales interactivos
14. Líneas 2293-2398:  Mapas mentales
15. Líneas 2399-2466:  Triangulación
16. Líneas 3752-3838:  Clustering y agrupación
17. Líneas 3934-4026:  Nubes de palabras
```

### Fase 5: Interfaz de Usuario (Día 8)
```
18. Líneas 4478-6009:  Funciones de renderizado (dashboard, gráficos, etc.)
19. Líneas 6010-6099:  Función principal render()
```

**VENTAJA**: Al estar todo en un solo archivo, puedes:
- ✅ Desarrollar sección por sección de arriba hacia abajo
- ✅ Probar cada sección sin crear archivos separados
- ✅ Mantener coherencia fácilmente
- ✅ No preocuparte por imports entre módulos

---

## 📜 Prompt Maestro Completo

```markdown
# PROMPT MAESTRO PARA CREAR MÓDULO DE ANÁLISIS CUALITATIVO (ARCHIVO ÚNICO)

Eres un experto en arquitectura de software y análisis de texto con NLP. Tu tarea es crear un módulo de análisis cualitativo avanzado en UN SOLO ARCHIVO (`modules/qualitative_analysis.py`) siguiendo estos principios:

## CONTEXTO
El módulo será parte de un sistema RAG (Retrieval-Augmented Generation) llamado CogniChat. Debe analizar contenido de documentos procesados y proporcionar insights mediante NLP y visualizaciones interactivas.

## REQUISITOS FUNCIONALES

### 1. Análisis de Texto
- Extracción de conceptos clave (TF-IDF, BM25)
- Modelado de temas (LDA, NMF, BERTopic)
- Análisis de sentimientos (VADER, TextBlob, BERT)
- Clustering de documentos (K-means, DBSCAN)
- Triangulación multi-fuente

### 2. Visualizaciones
- Mapas conceptuales interactivos (PyVis, Graphviz)
- Mapas mentales jerárquicos (streamlit-agraph)
- Nubes de palabras temáticas
- Dashboards con métricas

### 3. Resúmenes
- Resúmenes con LLM (Ollama)
- Resúmenes extractivos (TextRank, LexRank)
- Resúmenes abstractivos (BART, T5)

## REQUISITOS NO FUNCIONALES

### Arquitectura
- ✅ Modular y extensible
- ✅ Separación de responsabilidades
- ✅ Independencia entre componentes
- ✅ Patrones de diseño apropiados

### Rendimiento
- ✅ Cache inteligente (LRU)
- ✅ Procesamiento paralelo
- ✅ Optimización de memoria
- ✅ Lazy loading cuando sea posible

### Calidad
- ✅ Código limpio y documentado
- ✅ Type hints completos
- ✅ Manejo robusto de errores
- ✅ Logging detallado
- ✅ Tests unitarios

### Compatibilidad
- ✅ Python 3.8+
- ✅ Funciona con/sin dependencias opcionales
- ✅ Fallbacks robustos
- ✅ API consistente

## ESTRUCTURA DEL ARCHIVO ÚNICO

Todo el módulo está en un solo archivo: `modules/qualitative_analysis.py`

**Organización interna (19 secciones en 6,099 líneas)**:

```
modules/qualitative_analysis.py
│
├─ [LÍNEAS 1-188] SECCIÓN 1: Imports y Configuración Global
│  └─ Todas las importaciones y logger setup
│
├─ [LÍNEAS 189-247] SECCIÓN 2: Enums y Dataclasses
│  ├─ AnalysisType
│  ├─ VisualizationType  
│  ├─ AnalysisConfig
│  ├─ ConceptData
│  └─ AnalysisResult
│
├─ [LÍNEAS 248-271] SECCIÓN 3: Clases Base
│  └─ BaseAnalyzer (abstract)
│
├─ [LÍNEAS 272-324] SECCIÓN 4: Cache Manager
│  └─ CacheManager (LRU cache thread-safe)
│
├─ [LÍNEAS 325-394] SECCIÓN 5: Preprocesamiento
│  └─ TextPreprocessor
│
├─ [LÍNEAS 395-591] SECCIÓN 6: Extracción de Conceptos
│  └─ ConceptExtractor
│
├─ [LÍNEAS 592-807] SECCIÓN 7: Análisis de Temas
│  └─ ThemeAnalyzer
│
├─ [LÍNEAS 808-1023] SECCIÓN 8: Análisis de Sentimientos
│  └─ SentimentAnalyzer
│
├─ [LÍNEAS 1024-4477] SECCIÓN 9: Clase Principal
│  └─ AdvancedQualitativeAnalyzer
│     ├─ Validación y utilidades
│     ├─ Compatibilidad con API legacy
│     ├─ Métodos de análisis principales
│     ├─ Sección 10: Mapas conceptuales (Línea 1977)
│     ├─ Sección 11: Mapas mentales (Línea 2293)
│     ├─ Sección 12: Resúmenes (Línea 1177)
│     ├─ Sección 13: Triangulación (Línea 2399)
│     ├─ Sección 14: Nubes de palabras (Línea 3934)
│     ├─ Sección 9: Clustering (Línea 3752)
│     ├─ Sección 16: Análisis paralelo (Línea 1658)
│     └─ Sección 17: Configuración (Línea 1780)
│
└─ [LÍNEAS 4478-6099] SECCIÓN 15-19: Renderizado UI
   ├─ Sección 15: Visualizaciones y gráficos
   ├─ render_advanced_dashboard()
   ├─ render_advanced_themes()
   ├─ render_clustering_analysis()
   ├─ render_sentiment_analysis()
   ├─ render_word_cloud()
   ├─ render_interactive_concept_map()
   ├─ render_interactive_mind_map()
   ├─ render_automatic_summary()
   ├─ render_triangulation_analysis()
   └─ Sección 19: render() - Función principal
```

**Cómo navegar el archivo**:
1. Busca el separador: `# === SECCIÓN X ===`
2. Usa Ctrl+G para ir a la línea específica
3. Toda la funcionalidad relacionada está junta
4. Las secciones están ordenadas lógicamente

## CONVENCIONES DE CÓDIGO

### Naming
- Clases: PascalCase (ConceptExtractor)
- Métodos públicos: snake_case (extract_concepts)
- Métodos privados: _snake_case (_validate_input)
- Constantes: UPPER_SNAKE_CASE (MAX_CONCEPTS)

### Documentación
- Docstrings en Google style
- Type hints obligatorios
- Comentarios inline cuando sea necesario
- README por submódulo

### Logging
- logger.debug() para detalles de depuración
- logger.info() para eventos importantes
- logger.warning() para situaciones recuperables
- logger.error() para errores críticos

## PASOS DE IMPLEMENTACIÓN (TODO EN UN ARCHIVO)

Para cada sección del archivo único, sigue este proceso:

1. **Diseño**
   - Define interfaz pública
   - Identifica dependencias
   - Planifica estructura interna

2. **Implementación**
   - Crea clase/módulo según template
   - Implementa método principal
   - Agrega métodos auxiliares
   - Incluye manejo de errores

3. **Optimización**
   - Implementa cache
   - Optimiza algoritmos
   - Agrega métricas

4. **Documentación**
   - Docstrings completos
   - Ejemplos de uso
   - Notas de implementación

5. **Testing**
   - Tests unitarios
   - Tests de integración
   - Verificación de rendimiento

## EJEMPLO DE SECCIÓN COMPLETA EN EL ARCHIVO ÚNICO

Aquí está un ejemplo completo de cómo debe verse una sección dentro de `modules/qualitative_analysis.py`:

```python
# =============================================================================
# 8. ANÁLISIS DE SENTIMIENTOS
# =============================================================================
# Esta sección proporciona análisis de sentimientos usando múltiples algoritmos:
# - VADER (primario): Análisis léxico optimizado
# - TextBlob (secundario): Análisis de polaridad
# - Básico (fallback): Conteo de palabras positivas/negativas
#
# UBICACIÓN EN EL ARCHIVO: Líneas 808-1023
#
# USO:
#     analyzer = AdvancedQualitativeAnalyzer()
#     result = analyzer.advanced_sentiment_analysis(chunks)
#     print(result['overall_stats'])
#
# MODIFICAR AQUÍ PARA:
# - Cambiar umbrales de sentimiento
# - Agregar nuevos algoritmos (BERT, RoBERTa)
# - Mejorar análisis de emociones
# - Ajustar clasificación de sentimientos
# ============================================================================="

from typing import List, Dict, Optional
from datetime import datetime
import logging

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from ..core.base import BaseAnalyzer
from ..core.config import AnalysisConfig, AnalysisResult, AnalysisType
from ..core.cache import CacheManager

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseAnalyzer):
    """
    Analizador de sentimientos multi-algoritmo
    
    Attributes:
        config: Configuración del análisis
        cache_manager: Gestor de cache
        
    Example:
        >>> analyzer = SentimentAnalyzer(config)
        >>> result = analyzer.analyze(chunks)
        >>> sentiments = result.data['sentiments']
    """
    
    def __init__(self, config: AnalysisConfig):
        """
        Inicializar analizador de sentimientos
        
        Args:
            config: Configuración del análisis
        """
        super().__init__(config)
        self.cache_manager = CacheManager(max_size=300)
        
        # Inicializar analizadores
        self._vader = None
        if VADER_AVAILABLE:
            try:
                self._vader = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"Error inicializando VADER: {e}")
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """
        Analizar sentimientos del contenido
        
        Args:
            chunks: Lista de chunks con contenido
            
        Returns:
            AnalysisResult con análisis de sentimientos
            
        Raises:
            ValueError: Si los chunks son inválidos
        """
        start_time = datetime.now()
        
        # Validar
        if not self._validate_input(chunks):
            raise ValueError("Chunks inválidos")
        
        # Cache
        cache_key = self._generate_cache_key(chunks)
        if self.config.enable_cache:
            cached = self.cache_manager.get(cache_key)
            if cached:
                logger.info("Usando sentimientos desde cache")
                return cached
        
        # Analizar
        sentiments = self._analyze_sentiments(chunks)
        
        # Resultado
        result = AnalysisResult(
            analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
            data={'sentiments': sentiments},
            metadata={
                'method': self._get_method_used(),
                'total_analyzed': len(sentiments['by_chunk'])
            },
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
        # Cache
        if self.config.enable_cache:
            self.cache_manager.set(cache_key, result)
        
        return result
    
    def _analyze_sentiments(self, chunks: List[Dict]) -> Dict:
        """Análisis interno de sentimientos"""
        if self._vader:
            return self._analyze_with_vader(chunks)
        elif TEXTBLOB_AVAILABLE:
            return self._analyze_with_textblob(chunks)
        else:
            return self._analyze_basic(chunks)
    
    def _analyze_with_vader(self, chunks: List[Dict]) -> Dict:
        """Análisis con VADER"""
        # Implementación...
        pass
    
    def _analyze_with_textblob(self, chunks: List[Dict]) -> Dict:
        """Análisis con TextBlob"""
        # Implementación...
        pass
    
    def _analyze_basic(self, chunks: List[Dict]) -> Dict:
        """Análisis básico de fallback"""
        # Implementación...
        pass
    
    def _get_method_used(self) -> str:
        """Retornar método usado"""
        if self._vader:
            return "VADER"
        elif TEXTBLOB_AVAILABLE:
            return "TextBlob"
        else:
            return "Basic"

# Registrar en factory
from ..core.base import AnalyzerFactory
AnalyzerFactory.register('sentiment_analysis', SentimentAnalyzer)
```

## CHECKLIST DE CALIDAD

Para cada submódulo creado, verificar:

- [ ] Sigue la estructura definida
- [ ] Documentación completa
- [ ] Type hints en todos los métodos
- [ ] Manejo de errores robusto
- [ ] Logging apropiado
- [ ] Cache implementado
- [ ] Tests unitarios incluidos
- [ ] Sin dependencias circulares
- [ ] API pública clara
- [ ] Ejemplos de uso incluidos

## INTEGRACIÓN FINAL EN EL ARCHIVO ÚNICO

Todo está integrado en `modules/qualitative_analysis.py`. La clase principal es:

```python
class AdvancedQualitativeAnalyzer:
    """
    Clase principal que orquesta todos los análisis
    
    Esta clase coordina todas las secciones del módulo y proporciona
    una interfaz unificada para el análisis cualitativo.
    
    Ubicación: Líneas 1024-4477
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Inicializar analizador con configuración"""
        self.config = config or AnalysisConfig()
        
        # Componentes especializados (clases definidas arriba en el archivo)
        self.preprocessor = TextPreprocessor()          # Línea 325
        self.cache_manager = CacheManager()             # Línea 272
        self.concept_extractor = ConceptExtractor(self.config)   # Línea 395
        self.theme_analyzer = ThemeAnalyzer(self.config)         # Línea 592
        self.sentiment_analyzer = SentimentAnalyzer(self.config) # Línea 808
    
    # MÉTODOS PÚBLICOS PRINCIPALES:
    
    def extract_key_concepts(self, chunks) → List[Dict]
        """Línea 1429 - Extracción de conceptos"""
    
    def extract_advanced_themes(self, chunks, n_topics) → Dict
        """Línea 1463 - Análisis de temas"""
    
    def advanced_sentiment_analysis(self, chunks) → Dict
        """Línea 1497 - Análisis de sentimientos"""
    
    def perform_clustering(self, chunks, n_clusters) → Dict
        """Línea 3756 - Clustering de documentos"""
    
    def perform_triangulation_analysis(self, chunks) → Dict
        """Línea 2403 - Validación multi-fuente"""
    
    def create_interactive_concept_map(self, chunks, layout_type) → str
        """Línea 2001 - Mapas conceptuales"""
    
    def create_interactive_mind_map(self, chunks, node_spacing) → Dict
        """Línea 2297 - Mapas mentales"""
    
    def generate_intelligent_summary(self, chunks, summary_type) → Dict
        """Línea 1241 - Resumen con LLM"""
    
    def generate_word_cloud(self, chunks, source_filter) → str
        """Línea 3938 - Nube de palabras"""
    
    def perform_parallel_analysis(self, chunks, analysis_types) → Dict
        """Línea 1662 - Análisis paralelo"""
    
    def optimize_performance(self) → Dict
        """Línea 1834 - Optimización automática"""
```

**PUNTO DE ENTRADA**:
La función `render()` en la línea 6013 es el punto de entrada principal que Streamlit llama.

---

## 🧪 Testing de Cada Submódulo

```python
# test_concept_extractor.py
import pytest
from qualitative_analysis.analyzers import ConceptExtractor
from qualitative_analysis.core import AnalysisConfig

def test_concept_extraction_with_valid_input():
    config = AnalysisConfig(max_concepts=10)
    extractor = ConceptExtractor(config)
    
    chunks = [
        {'content': 'Este es un documento sobre inteligencia artificial'},
        {'content': 'La inteligencia artificial es importante'}
    ]
    
    result = extractor.analyze(chunks)
    
    assert result.data['concepts']
    assert len(result.data['concepts']) <= 10
    assert result.processing_time > 0

def test_concept_extraction_with_empty_input():
    config = AnalysisConfig()
    extractor = ConceptExtractor(config)
    
    result = extractor.analyze([])
    
    assert not result.data['concepts']
    assert 'error' in result.metadata
```

---

## 🚀 Beneficios de Esta Arquitectura

### 1. Independencia
- Cada submódulo se puede desarrollar por separado
- Cambios en un módulo no afectan otros
- Fácil testing aislado

### 2. Reutilización
- Componentes reutilizables en otros proyectos
- Factory pattern facilita extender funcionalidad
- Cache compartido entre analizadores

### 3. Mantenibilidad
- Código organizado y limpio
- Fácil localizar bugs
- Simple agregar nuevas funcionalidades

### 4. Escalabilidad
- Procesamiento paralelo built-in
- Cache distribuido posible
- Microservicios ready

### 5. Testing
- Tests aislados por módulo
- Mock dependencies fácilmente
- CI/CD simplificado

---

## 📚 Recursos Adicionales

### Librerías Recomendadas

**Análisis de Texto**:
- scikit-learn: TF-IDF, LDA, clustering
- gensim: Topic modeling, word embeddings
- spaCy: NLP avanzado, NER
- transformers: BERT, GPT, T5

**Visualizaciones**:
- plotly: Gráficos interactivos
- networkx: Grafos y redes
- pyvis: Redes interactivas
- graphviz: Diagramas jerárquicos

**Resúmenes**:
- sumy: Resúmenes extractivos
- transformers: Resúmenes abstractivos
- gensim: TextRank

### Referencias

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Gensim Topic Modeling](https://radimrehurek.com/gensim/)
- [NLTK Book](https://www.nltk.org/book/)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [Streamlit Components](https://streamlit.io/components)

---

## 💡 Tips de Implementación

### 1. Comienza Simple
```python
# Versión 1: Funcionalidad mínima
def analyze(self, chunks):
    return basic_analysis(chunks)

# Versión 2: Agregar optimizaciones
def analyze(self, chunks):
    cached = self.cache.get(key)
    if cached:
        return cached
    
    result = basic_analysis(chunks)
    self.cache.set(key, result)
    return result

# Versión 3: Agregar análisis avanzado
def analyze(self, chunks):
    if advanced_available:
        return advanced_analysis(chunks)
    return basic_analysis(chunks)
```

### 2. Siempre Ten Fallback
```python
def _extract_with_advanced_method(self):
    try:
        # Método avanzado
        return advanced_extraction()
    except Exception as e:
        logger.warning(f"Fallback a método básico: {e}")
        return basic_extraction()
```

### 3. Usa Decoradores para Cross-Cutting Concerns
```python
def cached(func):
    """Decorador para cache automático"""
    def wrapper(self, *args, **kwargs):
        key = f"{func.__name__}_{hash(str(args))}"
        cached = self.cache_manager.get(key)
        if cached:
            return cached
        
        result = func(self, *args, **kwargs)
        self.cache_manager.set(key, result)
        return result
    
    return wrapper

@cached
def extract_concepts(self, chunks):
    # La cache se maneja automáticamente
    return self._do_extraction(chunks)
```

---

## 🎓 Conclusión

Este prompt te permite crear o recrear el módulo de análisis cualitativo en **UN SOLO ARCHIVO**, de forma **incremental, organizada y coherente**. Cada sección es independiente pero todas trabajan juntas armónicamente dentro del mismo archivo.

**Ventajas del enfoque de archivo único**:
- ✅ Todo el código en un solo lugar
- ✅ No hay problemas de imports circulares
- ✅ Secciones claramente separadas con comentarios
- ✅ Navegación fácil con Ctrl+G + número de línea
- ✅ Debugging simplificado (un solo archivo)
- ✅ Coherencia garantizada
- ✅ Mantenimiento más simple

**Estructura del archivo resultante**:
```
modules/qualitative_analysis.py (6,099 líneas)
├─ Secciones 1-8: Clases especializadas (1,023 líneas)
├─ Sección 9: Clase principal orquestadora (3,454 líneas)
└─ Secciones 15-19: Funciones de renderizado (1,622 líneas)
```

Sigue los prompts sección por sección y obtendrás un módulo robusto, escalable y profesional en un solo archivo perfectamente organizado.

