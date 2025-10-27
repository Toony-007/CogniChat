# 📚 Arquitectura del Módulo de Análisis Cualitativo

## 📋 Índice

1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Componentes Principales](#componentes-principales)
4. [Flujo de Datos](#flujo-de-datos)
5. [Análisis Detallado por Sección](#análisis-detallado-por-sección)
6. [Mejoras Potenciales](#mejoras-potenciales)
7. [Guía de Modificación](#guía-de-modificación)

---

## 🎯 Visión General

### Propósito del Módulo

El módulo de **Análisis Cualitativo Avanzado** (`modules/qualitative_analysis.py`) es el componente más complejo del sistema CogniChat. Proporciona análisis profundo de contenido RAG utilizando técnicas de NLP, visualizaciones interactivas y algoritmos de inteligencia artificial.

### Características Principales

- 🔍 **Extracción de Conceptos**: Identificación de conceptos clave usando TF-IDF
- 🎯 **Análisis de Temas**: Modelado de temas con LDA (Latent Dirichlet Allocation)
- 😊 **Análisis de Sentimientos**: Evaluación emocional con VADER y TextBlob
- 🗺️ **Mapas Conceptuales**: Visualizaciones interactivas con PyVis
- 🧠 **Mapas Mentales**: Representaciones jerárquicas con streamlit-agraph
- 📝 **Resúmenes Automáticos**: Generación con LLM y métodos extractivos
- 🔺 **Triangulación**: Validación multi-fuente de conceptos
- ☁️ **Nubes de Palabras**: Visualizaciones de frecuencia
- 🔍 **Clustering**: Agrupación de documentos con K-means y DBSCAN
- ⚡ **Procesamiento Paralelo**: Análisis concurrente optimizado

### Estadísticas del Código

- **Total de líneas**: ~6,100
- **Secciones organizadas**: 19
- **Clases principales**: 6
- **Métodos públicos**: ~25
- **Métodos privados**: ~80
- **Funciones de renderizado**: 12

---

## 🏗️ Arquitectura del Sistema

### Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                  MÓDULO DE ANÁLISIS CUALITATIVO                  │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
    │ DATOS   │         │ ANÁLISIS │         │ VISUAL. │
    │ & CACHE │         │ & PROC.  │         │ & UI    │
    └─────────┘         └─────────┘         └─────────┘
         │                    │                    │
    ┌────▼────────────────────▼────────────────────▼────┐
    │              COMPONENTES ESPECIALIZADOS            │
    ├───────────────────────────────────────────────────┤
    │ • TextPreprocessor    • ConceptExtractor          │
    │ • ThemeAnalyzer       • SentimentAnalyzer         │
    │ • CacheManager        • BaseAnalyzer              │
    └───────────────────────────────────────────────────┘
```

### Patrón de Diseño Utilizado

El módulo implementa varios patrones de diseño:

1. **Strategy Pattern**: Diferentes estrategias de análisis (TF-IDF, LDA, VADER)
2. **Factory Pattern**: Creación de analizadores especializados
3. **Singleton Pattern**: CacheManager para gestión de memoria
4. **Template Method Pattern**: BaseAnalyzer con método `analyze()` abstracto
5. **Observer Pattern**: Sistema de métricas y monitoreo

---

## 🧩 Componentes Principales

### 1. Estructuras de Datos (Líneas 145-260)

#### AnalysisType (Enum)
Define los tipos de análisis disponibles:
```python
class AnalysisType(Enum):
    CONCEPT_EXTRACTION = "concept_extraction"
    THEME_ANALYSIS = "theme_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CLUSTERING = "clustering"
    CONCEPT_MAP = "concept_map"
    MIND_MAP = "mind_map"
    SUMMARY = "summary"
    TRIANGULATION = "triangulation"
```

#### AnalysisConfig (Dataclass)
Configuración centralizada:
```python
@dataclass
class AnalysisConfig:
    min_frequency: int = 2
    max_concepts: int = 50
    similarity_threshold: float = 0.6
    chunk_size: int = 2000
    enable_cache: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
```

#### ConceptData (Dataclass)
Estructura para conceptos extraídos:
```python
@dataclass
class ConceptData:
    concept: str
    score: float
    frequency: int
    context: List[str]
    related_concepts: List[str]
    sentiment: Optional[float]
    category: Optional[str]
```

### 2. Clases Base (Líneas 260-350)

#### BaseAnalyzer (ABC)
Clase abstracta base para todos los analizadores:
```python
class BaseAnalyzer(ABC):
    def __init__(self, config: AnalysisConfig)
    
    @abstractmethod
    def analyze(self, chunks: List[Dict]) -> AnalysisResult
    
    def _validate_input(self, chunks: List[Dict]) -> bool
```

**Propósito**: Garantizar interfaz consistente entre todos los analizadores.

#### CacheManager
Gestor de cache con LRU eviction:
```python
class CacheManager:
    def get(self, key: str) -> Optional[Any]
    def set(self, key: str, value: Any) -> None
    def clear(self) -> None
    def _evict_oldest(self) -> None
    def get_stats(self) -> Dict[str, Any]
```

**Características**:
- Thread-safe con `threading.Lock()`
- Eviction automático de elementos antiguos
- Estadísticas de uso en tiempo real

### 3. Preprocesamiento (Líneas 350-395)

#### TextPreprocessor
Preprocesamiento especializado de texto:
```python
class TextPreprocessor:
    def get_spanish_stopwords(self) -> List[str]
    def preprocess_text(self, text: str) -> str
```

**Funcionalidades**:
- Stopwords en español con NLTK
- Normalización de texto
- Eliminación de caracteres especiales
- Cache de stopwords

### 4. Analizadores Especializados (Líneas 395-900)

#### ConceptExtractor
Extracción de conceptos clave:
```python
class ConceptExtractor(BaseAnalyzer):
    def analyze(self, chunks) → AnalysisResult
    def _extract_with_tfidf(self, texts) → List[ConceptData]
    def _extract_with_frequency(self, texts) → List[ConceptData]
    def _enrich_concepts_with_context(self, concepts, chunks)
```

**Algoritmos Usados**:
- **TF-IDF** (primario): Identifica términos importantes
- **Frecuencia** (fallback): Conteo simple de palabras

#### ThemeAnalyzer
Análisis de temas principal:
```python
class ThemeAnalyzer(BaseAnalyzer):
    def analyze(self, chunks) → AnalysisResult
    def _extract_themes_with_lda(self, texts) → List[Dict]
    def _extract_themes_with_clustering(self, texts) → List[Dict]
```

**Algoritmos Usados**:
- **LDA** (Latent Dirichlet Allocation): Modelado de temas
- **K-means clustering**: Agrupación de palabras clave

#### SentimentAnalyzer
Análisis de sentimientos:
```python
class SentimentAnalyzer(BaseAnalyzer):
    def analyze(self, chunks) → AnalysisResult
    def _analyze_with_textblob_vader(self, chunks) → Dict
    def _analyze_basic_sentiment(self, chunks) → Dict
```

**Algoritmos Usados**:
- **VADER** (primario): Análisis de sentimientos específico para español
- **TextBlob** (secundario): Análisis de polaridad y subjetividad
- **Conteo de palabras** (fallback): Método básico

### 5. Clase Principal (Líneas 900-3700)

#### AdvancedQualitativeAnalyzer
Orquestador principal que coordina todos los análisis:

**Métodos de Análisis**:
```python
# Extracción de conceptos
extract_key_concepts(chunks, min_freq) → List[Dict]

# Análisis de temas
extract_advanced_themes(chunks, n_topics) → Dict

# Análisis de sentimientos
advanced_sentiment_analysis(chunks) → Dict

# Clustering
perform_clustering(chunks, n_clusters) → Dict

# Triangulación
perform_triangulation_analysis(chunks) → Dict

# Mapas conceptuales
create_interactive_concept_map(chunks, layout_type) → Optional[str]

# Mapas mentales
create_interactive_mind_map(chunks, node_spacing) → Optional[Dict]

# Resúmenes
generate_intelligent_summary(chunks, summary_type) → Dict
generate_rag_summary(chunks, max_length) → str
generate_basic_summary(chunks, max_sentences) → str

# Nubes de palabras
generate_word_cloud(chunks, source_filter) → Optional[str]
```

**Métodos de Optimización**:
```python
# Análisis paralelo
perform_parallel_analysis(chunks, analysis_types) → Dict[str, AnalysisResult]

# Optimización de rendimiento
optimize_performance() → Dict[str, Any]
get_performance_metrics() → Dict[str, Any]

# Gestión de cache
clear_cache() → None
get_cache_stats() → Dict
```

---

## 🔄 Flujo de Datos

### Flujo General de Análisis

```
1. ENTRADA DE DATOS
   ├─ Chunks del sistema RAG
   └─ Configuración de análisis

2. VALIDACIÓN
   ├─ Verificar chunks válidos
   └─ Verificar configuración

3. PREPROCESAMIENTO
   ├─ Limpiar texto
   ├─ Remover stopwords
   └─ Normalizar

4. ANÁLISIS
   ├─ Extracción de conceptos (TF-IDF)
   ├─ Análisis de temas (LDA)
   ├─ Análisis de sentimientos (VADER)
   └─ Clustering (K-means)

5. ENRIQUECIMIENTO
   ├─ Agregar contexto
   ├─ Encontrar relaciones
   └─ Calcular métricas

6. VISUALIZACIÓN
   ├─ Generar mapas conceptuales
   ├─ Crear gráficos interactivos
   └─ Preparar dashboards

7. SALIDA
   ├─ AnalysisResult estructurado
   └─ Formato legacy compatible
```

### Flujo de Cache

```
CONSULTA
   ├─ Generar cache_key (hash de contenido)
   ├─ Verificar en CacheManager
   │  ├─ HIT → Retornar resultado cacheado
   │  └─ MISS → Procesar y guardar
   └─ Actualizar access_times
```

---

## 📊 Análisis Detallado por Sección

### Sección 6: EXTRACCIÓN DE CONCEPTOS (Línea 395)

#### Algoritmos Implementados

**1. TF-IDF (Term Frequency-Inverse Document Frequency)**
```python
Ventajas:
✅ Identifica términos importantes específicos del corpus
✅ Reduce peso de palabras comunes
✅ Funciona bien con documentos técnicos

Limitaciones:
❌ No captura relaciones semánticas
❌ Sensible al tamaño del corpus
❌ Requiere múltiples documentos

Parámetros configurables:
- max_features: Número máximo de términos (default: 200)
- ngram_range: Rango de n-gramas (default: 1-3)
- min_df: Frecuencia mínima de documento
- max_df: Frecuencia máxima de documento
```

**2. Análisis de Frecuencia (Fallback)**
```python
Ventajas:
✅ Simple y rápido
✅ No requiere múltiples documentos
✅ Fácil de entender

Limitaciones:
❌ No considera importancia relativa
❌ Puede sobrevalorar palabras comunes
❌ No detecta conceptos complejos

Mejoras sugeridas:
→ Implementar BM25 para mejor ranking
→ Usar Mutual Information para co-ocurrencias
→ Agregar detección de entidades nombradas (NER)
```

#### Mejoras Potenciales

**Algoritmos Tradicionales a Agregar**:

1. **BM25 (Best Matching 25)**
```python
from rank_bm25 import BM25Okapi

def _extract_with_bm25(self, texts: List[str]) -> List[ConceptData]:
    """Extraer conceptos usando BM25"""
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    
    # Obtener palabras únicas
    all_words = set()
    for text in tokenized_texts:
        all_words.update(text)
    
    # Ranking de palabras
    concepts = []
    for word in all_words:
        scores = bm25.get_scores([word])
        avg_score = np.mean(scores)
        if avg_score > threshold:
            concepts.append(ConceptData(
                concept=word,
                score=avg_score,
                frequency=sum(1 for text in tokenized_texts if word in text)
            ))
    
    return sorted(concepts, key=lambda x: x.score, reverse=True)
```

2. **PMI (Pointwise Mutual Information)**
```python
def _extract_collocations_with_pmi(self, texts: List[str]) -> List[ConceptData]:
    """Extraer colocaciones importantes usando PMI"""
    from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
    
    finder = BigramCollocationFinder.from_documents(
        [text.split() for text in texts]
    )
    
    bigram_measures = BigramAssocMeasures()
    scored_bigrams = finder.score_ngrams(bigram_measures.pmi)
    
    concepts = []
    for (word1, word2), score in scored_bigrams[:50]:
        concept = ConceptData(
            concept=f"{word1} {word2}",
            score=score,
            frequency=finder.ngram_fd[(word1, word2)]
        )
        concepts.append(concept)
    
    return concepts
```

**Algoritmos con IA a Agregar**:

1. **BERT para Extracción de Conceptos**
```python
from transformers import pipeline

def _extract_with_bert_ner(self, texts: List[str]) -> List[ConceptData]:
    """Extraer entidades nombradas usando BERT"""
    ner_pipeline = pipeline(
        "ner",
        model="dccuchile/bert-base-spanish-wwm-uncased",
        aggregation_strategy="simple"
    )
    
    concepts = []
    for text in texts:
        entities = ner_pipeline(text)
        for entity in entities:
            concept = ConceptData(
                concept=entity['word'],
                score=entity['score'],
                frequency=1,
                category=entity['entity_group']
            )
            concepts.append(concept)
    
    return concepts
```

2. **Embeddings Semánticos con Sentence-BERT**
```python
from sentence_transformers import SentenceTransformer

def _extract_with_semantic_clustering(self, texts: List[str]) -> List[ConceptData]:
    """Extraer conceptos usando embeddings semánticos"""
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Generar embeddings
    embeddings = model.encode(texts)
    
    # Clustering semántico
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    
    # Extraer conceptos representativos de cada cluster
    concepts = []
    for label in set(labels):
        if label == -1:  # Ruido
            continue
        
        cluster_texts = [texts[i] for i, l in enumerate(labels) if l == label]
        # Extraer concepto representativo
        representative = cluster_texts[0][:50]
        
        concept = ConceptData(
            concept=representative,
            score=len(cluster_texts) / len(texts),
            frequency=len(cluster_texts),
            category=f"cluster_{label}"
        )
        concepts.append(concept)
    
    return concepts
```

---

### Sección 7: ANÁLISIS DE TEMAS (Línea 533)

#### Algoritmos Implementados

**1. LDA (Latent Dirichlet Allocation)**
```python
Ventajas:
✅ Descubre temas latentes automáticamente
✅ Asigna documentos a múltiples temas
✅ Interpretable y bien establecido

Limitaciones:
❌ Requiere especificar número de temas
❌ Sensible a parámetros de inicialización
❌ Computacionalmente costoso

Parámetros clave:
- n_components: Número de temas
- max_iter: Iteraciones máximas (default: 100)
- learning_method: 'batch' o 'online'
- random_state: Semilla para reproducibilidad
```

**2. Clustering de Palabras Clave (Fallback)**
```python
Ventajas:
✅ Rápido y eficiente
✅ No requiere muchos parámetros
✅ Funciona con corpus pequeños

Limitaciones:
❌ Menos sofisticado que LDA
❌ Puede no capturar relaciones sutiles
❌ Depende de frecuencia de palabras
```

#### Mejoras Potenciales

**Algoritmos Tradicionales a Agregar**:

1. **NMF (Non-Negative Matrix Factorization)**
```python
from sklearn.decomposition import NMF

def _extract_themes_with_nmf(self, texts: List[str], n_topics: int) -> List[Dict]:
    """Extraer temas usando NMF (más interpretable que LDA)"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=self.preprocessor.get_spanish_stopwords()
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    nmf = NMF(
        n_components=n_topics,
        random_state=42,
        init='nndsvda',  # Inicialización mejorada
        solver='mu',      # Multiplicative Update
        max_iter=400
    )
    
    W = nmf.fit_transform(tfidf_matrix)  # Matriz documento-tema
    H = nmf.components_                   # Matriz tema-palabra
    
    feature_names = vectorizer.get_feature_names_out()
    
    themes = []
    for topic_idx, topic in enumerate(H):
        top_indices = topic.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        weights = [topic[i] for i in top_indices]
        
        theme = {
            'id': topic_idx,
            'name': f"Tema {topic_idx + 1}",
            'keywords': keywords,
            'weights': weights,
            'coherence': self._calculate_topic_coherence(keywords, texts),
            'description': self._generate_theme_description(keywords)
        }
        themes.append(theme)
    
    return themes
```

2. **LSA (Latent Semantic Analysis)**
```python
from sklearn.decomposition import TruncatedSVD

def _extract_themes_with_lsa(self, texts: List[str], n_topics: int) -> List[Dict]:
    """Extraer temas usando LSA/LSI"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=self.preprocessor.get_spanish_stopwords()
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    svd.fit(tfidf_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    
    themes = []
    for topic_idx, topic in enumerate(svd.components_):
        top_indices = topic.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        
        theme = {
            'id': topic_idx,
            'name': f"Tema {topic_idx + 1}",
            'keywords': keywords,
            'weights': topic[top_indices].tolist(),
            'variance_explained': svd.explained_variance_ratio_[topic_idx]
        }
        themes.append(theme)
    
    return themes
```

**Algoritmos con IA a Agregar**:

1. **BERTopic**
```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def _extract_themes_with_bertopic(self, texts: List[str]) -> List[Dict]:
    """Extraer temas usando BERTopic (estado del arte)"""
    # Modelo de embeddings
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Configurar BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language='spanish',
        calculate_probabilities=True,
        verbose=False
    )
    
    # Extraer temas
    topics, probs = topic_model.fit_transform(texts)
    
    # Convertir a formato esperado
    theme_list = []
    for topic_id in set(topics):
        if topic_id == -1:  # Outliers
            continue
        
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            keywords = [word for word, score in topic_words]
            weights = [score for word, score in topic_words]
            
            theme = {
                'id': topic_id,
                'name': topic_model.get_topic_info()[topic_id]['Name'],
                'keywords': keywords,
                'weights': weights,
                'documents_count': sum(1 for t in topics if t == topic_id),
                'representative_docs': topic_model.get_representative_docs(topic_id)
            }
            theme_list.append(theme)
    
    return theme_list
```

2. **Top2Vec**
```python
from top2vec import Top2Vec

def _extract_themes_with_top2vec(self, texts: List[str]) -> List[Dict]:
    """Extraer temas usando Top2Vec"""
    model = Top2Vec(
        documents=texts,
        embedding_model='universal-sentence-encoder-multilingual',
        speed='learn',
        workers=4
    )
    
    topic_words, word_scores, topic_nums = model.get_topics()
    
    themes = []
    for i in range(len(topic_nums)):
        theme = {
            'id': i,
            'name': f"Tema {i + 1}",
            'keywords': topic_words[i][:10].tolist(),
            'weights': word_scores[i][:10].tolist(),
            'size': model.topic_sizes[i],
            'documents': model.get_documents_topics(topic_nums=[i])
        }
        themes.append(theme)
    
    return themes
```

---

### Sección 8: ANÁLISIS DE SENTIMIENTOS (Línea 749)

#### Algoritmos Implementados

**1. VADER (Valence Aware Dictionary and sEntiment Reasoner)**
```python
Ventajas:
✅ Optimizado para redes sociales y texto informal
✅ Maneja intensificadores y negaciones
✅ Rápido y eficiente

Limitaciones:
❌ Basado en léxico (no contextual)
❌ Puede no capturar ironía
❌ Diseñado originalmente para inglés

Salida:
{
    'neg': 0.0,      # Probabilidad negativa
    'neu': 0.5,      # Probabilidad neutral
    'pos': 0.5,      # Probabilidad positiva
    'compound': 0.5  # Score compuesto [-1, 1]
}
```

**2. TextBlob**
```python
Ventajas:
✅ Simple de usar
✅ Proporciona polaridad y subjetividad
✅ Multiidioma

Limitaciones:
❌ Menos preciso que VADER
❌ No maneja bien contexto
❌ Basado en diccionario

Salida:
sentiment.polarity: float [-1.0, 1.0]
sentiment.subjectivity: float [0.0, 1.0]
```

#### Mejoras Potenciales

**Algoritmos Tradicionales a Agregar**:

1. **Análisis de Aspectos (ABSA)**
```python
def _aspect_based_sentiment_analysis(self, chunks: List[Dict]) -> Dict:
    """Análisis de sentimientos basado en aspectos"""
    results = {
        'aspects': {},
        'overall': {'positive': 0, 'negative': 0, 'neutral': 0}
    }
    
    # Definir aspectos a analizar
    aspects = {
        'calidad': ['calidad', 'excelente', 'bueno', 'malo', 'terrible'],
        'servicio': ['servicio', 'atención', 'ayuda', 'soporte'],
        'precio': ['precio', 'costo', 'barato', 'caro', 'económico']
    }
    
    for chunk in chunks:
        content = chunk.get('content', '').lower()
        
        for aspect_name, keywords in aspects.items():
            if any(keyword in content for keyword in keywords):
                # Analizar sentimiento del aspecto
                blob = TextBlob(content)
                sentiment = blob.sentiment.polarity
                
                if aspect_name not in results['aspects']:
                    results['aspects'][aspect_name] = {
                        'positive': 0, 'negative': 0, 'neutral': 0, 'scores': []
                    }
                
                if sentiment > 0.1:
                    results['aspects'][aspect_name]['positive'] += 1
                elif sentiment < -0.1:
                    results['aspects'][aspect_name]['negative'] += 1
                else:
                    results['aspects'][aspect_name]['neutral'] += 1
                
                results['aspects'][aspect_name]['scores'].append(sentiment)
    
    return results
```

**Algoritmos con IA a Agregar**:

1. **BERT para Análisis de Sentimientos**
```python
from transformers import pipeline

def _analyze_sentiment_with_bert(self, chunks: List[Dict]) -> Dict:
    """Análisis de sentimientos usando BERT en español"""
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=-1  # CPU
    )
    
    results = {
        'by_chunk': [],
        'overall': {'positive': 0, 'negative': 0, 'neutral': 0},
        'confidence_avg': 0.0
    }
    
    confidences = []
    
    for chunk in chunks:
        content = chunk.get('content', '').strip()
        if not content or len(content) < 10:
            continue
        
        # Limitar longitud para BERT
        content = content[:512]
        
        try:
            result = sentiment_pipeline(content)[0]
            label = result['label']
            score = result['score']
            
            # Mapear estrellas a sentimiento
            if label in ['4 stars', '5 stars']:
                sentiment = 'positive'
            elif label in ['1 star', '2 stars']:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            results['by_chunk'].append({
                'content': content[:100],
                'sentiment': sentiment,
                'confidence': score,
                'label': label
            })
            
            results['overall'][sentiment] += 1
            confidences.append(score)
            
        except Exception as e:
            logger.warning(f"Error en BERT sentiment: {e}")
    
    results['confidence_avg'] = np.mean(confidences) if confidences else 0.0
    
    return results
```

2. **Análisis de Emociones con IA**
```python
def _analyze_emotions_with_ai(self, chunks: List[Dict]) -> Dict:
    """Análisis de emociones específicas usando IA"""
    # Usar modelo específico de emociones
    from transformers import pipeline
    
    emotion_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
    
    emotions_result = {
        'joy': 0, 'anger': 0, 'sadness': 0,
        'fear': 0, 'surprise': 0, 'disgust': 0,
        'love': 0, 'by_chunk': []
    }
    
    for chunk in chunks:
        content = chunk.get('content', '')[:512]
        
        if not content:
            continue
        
        try:
            emotions = emotion_pipeline(content)[0]
            
            # Encontrar emoción dominante
            dominant = max(emotions, key=lambda x: x['score'])
            emotions_result[dominant['label']] += 1
            
            chunk_emotions = {
                'content': content[:100],
                'emotions': emotions,
                'dominant': dominant['label']
            }
            emotions_result['by_chunk'].append(chunk_emotions)
            
        except Exception as e:
            logger.warning(f"Error en análisis de emociones: {e}")
    
    return emotions_result
```

---

### Sección 10: MAPAS CONCEPTUALES (Línea 1977)

#### Bibliotecas Utilizadas

**PyVis**: Visualización interactiva basada en vis.js
```python
Ventajas:
✅ Altamente interactivo
✅ Múltiples layouts disponibles
✅ Personalización completa de nodos y aristas

Limitaciones:
❌ Puede ser lento con muchos nodos
❌ Requiere HTML/JavaScript
❌ No se integra nativamente con Streamlit
```

#### Mejoras Potenciales

**Visualizaciones Alternativas**:

1. **Graphviz para Jerarquías Estáticas**
```python
import graphviz

def _create_hierarchical_concept_map(self, chunks: List[Dict]) -> str:
    """Crear mapa conceptual jerárquico con Graphviz"""
    dot = graphviz.Digraph(comment='Mapa Conceptual')
    dot.attr(rankdir='TB')  # Top to Bottom
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Analizar jerarquía
    hierarchy = self._analyze_concept_hierarchy(chunks)
    
    # Nodo principal
    dot.node('main', hierarchy['main_theme']['name'], fillcolor='lightcoral')
    
    # Conceptos principales
    for i, concept in enumerate(hierarchy['main_concepts']):
        node_id = f"concept_{i}"
        dot.node(node_id, concept['name'], fillcolor='lightgreen')
        dot.edge('main', node_id, label='define')
        
        # Sub-conceptos
        for j, sub in enumerate(concept['sub_concepts'][:3]):
            sub_id = f"sub_{i}_{j}"
            dot.node(sub_id, sub['name'], fillcolor='lightyellow')
            dot.edge(node_id, sub_id, label=sub['relation'])
    
    return dot.pipe(format='svg').decode('utf-8')
```

2. **Plotly para Mapas Interactivos 3D**
```python
def _create_3d_concept_network(self, chunks: List[Dict]) -> go.Figure:
    """Crear red de conceptos en 3D con Plotly"""
    # Extraer conceptos
    concepts = self.extract_key_concepts(chunks)
    
    # Crear grafo
    G = nx.Graph()
    for concept in concepts:
        G.add_node(concept['concept'], weight=concept['score'])
    
    # Agregar aristas (relaciones)
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i+1:]:
            similarity = self._calculate_concept_similarity(
                c1['concept'], c2['concept']
            )
            if similarity > 0.3:
                G.add_edge(c1['concept'], c2['concept'], weight=similarity)
    
    # Layout 3D
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Preparar trazas
    edge_trace = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(color='gray', width=2),
                hoverinfo='none'
            )
        )
    
    # Nodos
    node_x, node_y, node_z = [], [], []
    node_text = []
    node_sizes = []
    
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(node)
        node_sizes.append(G.nodes[node]['weight'] * 50)
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=node_sizes,
            color='lightblue',
            line=dict(color='darkblue', width=2)
        )
    )
    
    # Crear figura
    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        title='Mapa Conceptual 3D',
        showlegend=False,
        hovermode='closest',
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False)
        )
    )
    
    return fig
```

---

### Sección 12: RESUMENES AUTOMÁTICOS (Línea 1177)

#### Estrategias Implementadas

**1. Resumen con LLM (IA)**
```python
Tipos disponibles:
- comprehensive: Resumen completo y detallado
- executive: Resumen ejecutivo para decisiones
- analytical: Análisis profundo académico
- thematic: Organizado por temas

Ventajas:
✅ Alta calidad y coherencia
✅ Comprensión contextual profunda
✅ Personalizable con prompts

Limitaciones:
❌ Requiere modelo LLM (Ollama)
❌ Más lento que métodos extractivos
❌ Puede ser costoso computacionalmente
```

**2. Resumen Extractivo Básico**
```python
Algoritmo:
1. Dividir en oraciones
2. Calcular frecuencia de palabras importantes
3. Puntuar oraciones por relevancia
4. Seleccionar top N oraciones
5. Ordenar por aparición original

Ventajas:
✅ Muy rápido
✅ No requiere IA
✅ Preserva texto original

Limitaciones:
❌ No genera texto nuevo
❌ Puede ser incoherente
❌ Limitado por oraciones existentes
```

#### Mejoras Potenciales

**Algoritmos Tradicionales a Agregar**:

1. **TextRank (PageRank para texto)**
```python
def _summarize_with_textrank(self, chunks: List[Dict], num_sentences: int = 5) -> str:
    """Generar resumen usando TextRank"""
    from gensim.summarization import summarize
    from gensim.summarization.textcleaner import split_sentences
    
    # Combinar todo el texto
    all_text = " ".join([chunk.get('content', '') for chunk in chunks])
    
    try:
        # Usar TextRank de gensim
        summary = summarize(
            all_text,
            ratio=0.2,  # 20% del texto original
            word_count=500,  # Máximo 500 palabras
            split=True  # Retornar lista de oraciones
        )
        
        return ". ".join(summary[:num_sentences]) + "."
        
    except Exception as e:
        logger.warning(f"Error en TextRank: {e}")
        return self.generate_basic_summary(chunks, num_sentences)
```

2. **LexRank**
```python
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS

def _summarize_with_lexrank(self, chunks: List[Dict], num_sentences: int = 5) -> str:
    """Generar resumen usando LexRank"""
    # Preparar documentos
    documents = [chunk.get('content', '') for chunk in chunks if chunk.get('content')]
    
    # Configurar LexRank
    lxr = LexRank(documents, stopwords=STOPWORDS['es'])
    
    # Obtener resumen
    summary_sentences = lxr.get_summary(
        documents,
        summary_size=num_sentences,
        threshold=0.1
    )
    
    return ". ".join(summary_sentences) + "."
```

**Algoritmos con IA a Agregar**:

1. **BART para Resumen Abstractivo**
```python
from transformers import pipeline

def _summarize_with_bart(self, chunks: List[Dict], max_length: int = 500) -> str:
    """Resumen abstractivo usando BART"""
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    
    # Combinar contenido
    all_text = " ".join([chunk.get('content', '') for chunk in chunks])
    
    # Dividir en secciones si es muy largo
    max_input_length = 1024
    if len(all_text.split()) > max_input_length:
        # Resumir por secciones y luego combinar
        sections = self._split_into_sections(all_text, max_input_length)
        summaries = []
        
        for section in sections:
            summary = summarizer(
                section,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        
        # Resumir los resúmenes
        combined = " ".join(summaries)
        final_summary = summarizer(
            combined,
            max_length=max_length,
            min_length=100,
            do_sample=False
        )[0]['summary_text']
        
        return final_summary
    else:
        summary = summarizer(
            all_text,
            max_length=max_length,
            min_length=100,
            do_sample=False
        )[0]['summary_text']
        
        return summary
```

2. **T5 para Resumen Multilingüe**
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

def _summarize_with_t5(self, chunks: List[Dict]) -> str:
    """Resumen usando T5 multilingüe"""
    model_name = "google/mt5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Preparar texto
    all_text = " ".join([chunk.get('content', '') for chunk in chunks])
    
    # Tokenizar
    inputs = tokenizer.encode(
        "summarize: " + all_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Generar resumen
    summary_ids = model.generate(
        inputs,
        max_length=200,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary
```

---

## 🔄 Mejoras Sugeridas por Componente

### Cache y Rendimiento

**1. Implementar Redis para Cache Distribuido**
```python
import redis

class RedisCache Manager(CacheManager):
    """Cache distribuido con Redis"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
    
    def get(self, key: str) -> Optional[Any]:
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(value)
        )
```

**2. Procesamiento Asíncrono con AsyncIO**
```python
import asyncio
from typing import List, Dict

async def _async_process_chunks(self, chunks: List[Dict]) -> List[Dict]:
    """Procesar chunks de forma asíncrona"""
    async def process_single_chunk(chunk):
        content = chunk.get('content', '')
        processed = await self._async_preprocess(content)
        return processed
    
    tasks = [process_single_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    return results
```

### Análisis Avanzados

**1. Análisis de Coherencia de Documentos**
```python
def analyze_document_coherence(self, chunks: List[Dict]) -> Dict:
    """Analizar coherencia entre secciones del documento"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Generar embeddings para cada chunk
    embeddings = []
    for chunk in chunks:
        vector = self._get_chunk_embedding(chunk)
        embeddings.append(vector)
    
    embeddings_matrix = np.array(embeddings)
    
    # Calcular similitud entre chunks consecutivos
    coherence_scores = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity(
            [embeddings[i]],
            [embeddings[i + 1]]
        )[0][0]
        coherence_scores.append(similarity)
    
    return {
        'overall_coherence': np.mean(coherence_scores),
        'min_coherence': np.min(coherence_scores),
        'max_coherence': np.max(coherence_scores),
        'coherence_by_section': coherence_scores
    }
```

**2. Análisis de Complejidad Textual**
```python
def analyze_text_complexity(self, chunks: List[Dict]) -> Dict:
    """Analizar complejidad del texto"""
    import textstat
    
    complexity_results = {
        'readability': {},
        'by_chunk': [],
        'overall': {}
    }
    
    all_scores = {
        'flesch_reading_ease': [],
        'gunning_fog': [],
        'automated_readability_index': [],
        'avg_sentence_length': [],
        'avg_word_length': []
    }
    
    for chunk in chunks:
        content = chunk.get('content', '')
        if not content:
            continue
        
        chunk_scores = {
            'flesch_reading_ease': textstat.flesch_reading_ease(content),
            'gunning_fog': textstat.gunning_fog(content),
            'automated_readability_index': textstat.automated_readability_index(content),
            'avg_sentence_length': textstat.avg_sentence_length(content),
            'avg_word_length': textstat.avg_character_per_word(content)
        }
        
        for key, value in chunk_scores.items():
            all_scores[key].append(value)
        
        complexity_results['by_chunk'].append(chunk_scores)
    
    # Calcular promedios
    for key, values in all_scores.items():
        if values:
            complexity_results['overall'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return complexity_results
```

---

## 📖 Guía de Modificación

### Cómo Agregar un Nuevo Tipo de Análisis

**Paso 1: Agregar al Enum** (Línea 145)
```python
class AnalysisType(Enum):
    # ... existentes ...
    COMPLEXITY_ANALYSIS = "complexity_analysis"  # NUEVO
```

**Paso 2: Crear Analizador Especializado**
```python
# Agregar después de SentimentAnalyzer (Línea ~900)
class ComplexityAnalyzer(BaseAnalyzer):
    """Analizador de complejidad textual"""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.preprocessor = TextPreprocessor()
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """Analizar complejidad del texto"""
        # Tu implementación aquí
        pass
```

**Paso 3: Integrar en Clase Principal**
```python
# En __init__ de AdvancedQualitativeAnalyzer
self.complexity_analyzer = ComplexityAnalyzer(self.config)

# Agregar método público
def analyze_complexity(self, chunks: List[Dict]) -> Dict:
    result = self.complexity_analyzer.analyze(chunks)
    return result.data
```

**Paso 4: Agregar Función de Renderizado**
```python
# En sección 15 (Línea 4363)
def render_complexity_analysis(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar análisis de complejidad"""
    st.header("📊 Análisis de Complejidad")
    
    # Realizar análisis
    complexity = analyzer.analyze_complexity(chunks)
    
    # Visualizar resultados
    # ... tu código de visualización
```

**Paso 5: Integrar en render() Principal**
```python
# En función render() (Línea 5890)
tabs = st.tabs([
    # ... existentes ...
    "📊 Complejidad"  # NUEVA
])

with tabs[N]:  # N = posición de la nueva tab
    render_complexity_analysis(analyzer, chunks)
```

### Cómo Mejorar un Algoritmo Existente

**Ejemplo: Mejorar Extracción de Conceptos**

1. **Ve a la Línea 395** (Sección 6: EXTRACCIÓN DE CONCEPTOS)
2. **Localiza** `ConceptExtractor._extract_with_tfidf()`
3. **Modifica** los parámetros o agrega nuevo método:

```python
def _extract_with_tfidf(self, texts: List[str]) -> List[ConceptData]:
    """Extraer conceptos usando TF-IDF MEJORADO"""
    
    # MEJORA 1: Ajustar parámetros
    vectorizer = TfidfVectorizer(
        max_features=300,  # Era 200
        stop_words=self.preprocessor.get_spanish_stopwords(),
        ngram_range=(1, 4),  # Era (1, 3) - ahora detecta frases más largas
        min_df=max(1, len(texts) // 15),  # Era // 10
        max_df=0.85,  # Era 0.9
        sublinear_tf=True,  # NUEVO: Escala logarítmica para TF
        use_idf=True,
        smooth_idf=True  # NUEVO: Suavizado IDF
    )
    
    # MEJORA 2: Agregar análisis semántico
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # MEJORA 3: Calcular similitud semántica entre conceptos
    similarity_matrix = cosine_similarity(tfidf_matrix.T)
    
    # MEJORA 4: Agrupar conceptos relacionados
    related_groups = self._group_related_concepts(
        feature_names, 
        similarity_matrix
    )
    
    # Crear ConceptData enriquecido
    concepts = []
    mean_scores = tfidf_matrix.mean(axis=0).A1
    
    for i, score in enumerate(mean_scores):
        if score > 0:
            concept = ConceptData(
                concept=feature_names[i],
                score=float(score),
                frequency=int(score * len(texts) * 10),
                related_concepts=related_groups.get(feature_names[i], []),  # NUEVO
                category=self._categorize_concept(feature_names[i])  # NUEVO
            )
            concepts.append(concept)
    
    return sorted(concepts, key=lambda x: x.score, reverse=True)
```

---

## 🚀 Roadmap de Mejoras Futuras

### Corto Plazo (1-2 semanas)

1. **Optimización de Rendimiento**
   - [ ] Implementar cache distribuido con Redis
   - [ ] Agregar procesamiento asíncrono
   - [ ] Optimizar uso de memoria con generadores

2. **Mejoras en Algoritmos Tradicionales**
   - [ ] Implementar BM25 para ranking
   - [ ] Agregar TextRank para resúmenes
   - [ ] Incluir PMI para colocaciones

3. **Mejoras en Visualizaciones**
   - [ ] Agregar gráficos 3D con Plotly
   - [ ] Implementar dashboards interactivos
   - [ ] Crear exportación a diferentes formatos

### Mediano Plazo (1 mes)

1. **Integración de IA Avanzada**
   - [ ] Implementar BERT para NER
   - [ ] Agregar BERTopic para temas
   - [ ] Incluir modelos de embeddings semánticos

2. **Análisis Avanzados**
   - [ ] Análisis de coherencia de documentos
   - [ ] Detección de argumentos
   - [ ] Análisis de estructura retórica

3. **Mejoras en UX**
   - [ ] Wizard de configuración
   - [ ] Recomendaciones automáticas
   - [ ] Exportación de reportes profesionales

### Largo Plazo (3 meses)

1. **Machine Learning Personalizado**
   - [ ] Entrenamiento de modelos específicos del dominio
   - [ ] Transfer learning para español
   - [ ] Fine-tuning de modelos LLM

2. **Análisis Multimodal**
   - [ ] Procesamiento de imágenes en documentos
   - [ ] Análisis de tablas y gráficos
   - [ ] OCR integrado

3. **Colaboración y Compartición**
   - [ ] Análisis colaborativo
   - [ ] Compartir configuraciones
   - [ ] Plantillas de análisis reutilizables

---

## 📝 Conclusiones

El módulo de análisis cualitativo es extremadamente completo pero también complejo. Las mejoras propuestas permitirán:

1. **Mejor rendimiento** con cache distribuido y procesamiento asíncrono
2. **Análisis más precisos** con algoritmos de ML avanzados
3. **Mayor usabilidad** con mejor organización y documentación
4. **Escalabilidad** para manejar grandes volúmenes de datos

La arquitectura modular implementada facilita agregar nuevas funcionalidades sin afectar código existente.

