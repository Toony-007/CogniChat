# 🚀 Guía Rápida del Módulo de Análisis Cualitativo

## 📍 Navegación Rápida del Código Actual

### Tabla de Contenidos por Líneas

| 🎯 Funcionalidad | 📍 Línea | 🔧 Qué Modificar |
|------------------|----------|------------------|
| **Configuración y Estructuras** | 145 | Agregar tipos de análisis, cambiar valores por defecto |
| **Cache y Memoria** | 223 | Mejorar gestión de cache, agregar persistencia |
| **Preprocesamiento** | 276 | Cambiar stopwords, mejorar limpieza de texto |
| **Extracción de Conceptos** | 346 | Cambiar TF-IDF a BM25, ajustar parámetros |
| **Análisis de Temas** | 533 | Cambiar LDA a NMF, agregar BERTopic |
| **Análisis de Sentimientos** | 749 | Mejorar VADER, agregar análisis de emociones |
| **Mapas Conceptuales** | 1918 | Cambiar colores, layouts, agregar 3D |
| **Mapas Mentales** | 2214 | Ajustar jerarquía, mejorar visualización |
| **Resúmenes Automáticos** | 1118 | Cambiar prompts LLM, agregar BART |
| **Triangulación** | 2320 | Mejorar validación multi-fuente |
| **Clustering** | 3637 | Cambiar K-means a DBSCAN, agregar jerárquico |
| **Nubes de Palabras** | 3819 | Cambiar colores, tamaños, máscaras |
| **Visualizaciones UI** | 4363 | Modificar dashboards, gráficos |
| **Optimización** | 1589 | Mejorar paralelización, cache distribuido |
| **Configuración** | 1711 | Agregar parámetros, validaciones |
| **Renderizado Principal** | 5890 | Modificar tabs, agregar nuevas vistas |

---

## 🎨 Ejemplos de Modificaciones Comunes

### 1. Cambiar Algoritmo de Extracción de Conceptos

**Ve a:** Línea 346 (Sección 6)

**Cambio:** De TF-IDF a BM25

```python
# ANTES (Línea ~460)
def _extract_with_tfidf(self, texts: List[str]) -> List[ConceptData]:
    vectorizer = TfidfVectorizer(...)
    # ...

# DESPUÉS
def _extract_with_bm25(self, texts: List[str]) -> List[ConceptData]:
    """Extraer conceptos usando BM25"""
    from rank_bm25 import BM25Okapi
    
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    
    # Obtener términos únicos
    all_terms = set()
    for text in tokenized_texts:
        all_terms.update(text)
    
    # Rankear términos
    term_scores = {}
    for term in all_terms:
        scores = bm25.get_scores([term])
        term_scores[term] = np.mean(scores)
    
    # Crear ConceptData
    concepts = []
    for term, score in sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:50]:
        concepts.append(ConceptData(
            concept=term,
            score=score,
            frequency=sum(1 for text in tokenized_texts if term in text)
        ))
    
    return concepts

# Y en el método principal (Línea ~430)
def _extract_concepts_advanced(self, chunks: List[Dict]) -> List[ConceptData]:
    texts = self._prepare_texts(chunks)
    
    # Cambiar a BM25
    if len(texts) >= 2:
        concepts = self._extract_with_bm25(texts)  # ← Cambio aquí
    else:
        concepts = self._extract_with_frequency(texts)
    
    return self._enrich_concepts(concepts, chunks)
```

---

### 2. Agregar Análisis de Emociones

**Ve a:** Línea 749 (Sección 8)

**Agregar después de** `SentimentAnalyzer`:

```python
# Línea ~900 (después de SentimentAnalyzer)
class EmotionAnalyzer(BaseAnalyzer):
    """Analizador de emociones específicas"""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'love']
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """Analizar emociones en el contenido"""
        from transformers import pipeline
        
        emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        
        results = {emotion: 0 for emotion in self.emotions}
        by_chunk = []
        
        for chunk in chunks:
            content = chunk.get('content', '')[:512]
            if not content:
                continue
            
            emotions = emotion_pipeline(content)[0]
            dominant = max(emotions, key=lambda x: x['score'])
            
            results[dominant['label']] += 1
            by_chunk.append({
                'content': content[:100],
                'emotions': emotions,
                'dominant': dominant['label']
            })
        
        return AnalysisResult(
            analysis_type=AnalysisType.SENTIMENT_ANALYSIS,
            data={'emotions': results, 'by_chunk': by_chunk},
            metadata={'model': 'distilroberta-emotion'}
        )

# Registrar en factory
AnalyzerFactory.register('emotion_analysis', EmotionAnalyzer)

# Y en AdvancedQualitativeAnalyzer.__init__ (Línea ~1020)
self.emotion_analyzer = EmotionAnalyzer(self.config)

# Agregar método público (Línea ~1350)
def analyze_emotions(self, chunks: List[Dict]) -> Dict:
    """Analizar emociones del contenido"""
    result = self.emotion_analyzer.analyze(chunks)
    return result.data
```

---

### 3. Mejorar Visualización de Mapas Conceptuales

**Ve a:** Línea 1918 (Sección 10)

**Cambio:** Agregar visualización 3D

```python
# Agregar método nuevo (Línea ~2100)
def create_3d_concept_map(self, chunks: List[Dict]) -> go.Figure:
    """Crear mapa conceptual en 3D"""
    import plotly.graph_objects as go
    import networkx as nx
    
    # Extraer conceptos
    concepts = self.extract_key_concepts(chunks)
    
    # Crear grafo
    G = nx.Graph()
    for concept in concepts[:20]:  # Limitar para claridad
        G.add_node(concept['concept'], weight=concept['score'])
    
    # Agregar aristas (conceptos relacionados)
    for i, c1 in enumerate(concepts[:20]):
        for j, c2 in enumerate(concepts[i+1:20], i+1):
            if c2['concept'] in c1.get('related_concepts', []):
                G.add_edge(c1['concept'], c2['concept'])
    
    # Layout 3D usando spring
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Extraer coordenadas
    edge_traces = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        
        edge_traces.append(
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
    node_text, node_sizes = [], []
    
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(node)
        node_sizes.append(G.nodes[node]['weight'] * 100)
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=node_sizes,
            color='lightblue',
            line=dict(color='darkblue', width=2)
        ),
        hovertext=node_text,
        hoverinfo='text'
    )
    
    # Crear figura
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='Mapa Conceptual 3D Interactivo',
        showlegend=False,
        hovermode='closest',
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title='')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

# Y agregar función de renderizado (Línea ~4800)
def render_3d_concept_map(analyzer, chunks):
    """Renderizar mapa conceptual 3D"""
    st.header("🗺️ Mapa Conceptual 3D")
    
    if st.button("Generar Mapa 3D"):
        with st.spinner("Generando visualización 3D..."):
            fig = analyzer.create_3d_concept_map(chunks)
            st.plotly_chart(fig, use_container_width=True)

# Agregar tab en render() (Línea ~5950)
tabs = st.tabs([
    # ... existentes ...
    "🗺️ Mapa 3D"  # NUEVO
])

with tabs[N]:
    render_3d_concept_map(analyzer, chunks)
```

---

### 4. Implementar Resumen con BART

**Ve a:** Línea 1118 (Sección 12)

**Agregar:**

```python
# Línea ~1350
def generate_abstractive_summary(self, chunks: List[Dict], max_length: int = 500) -> str:
    """Generar resumen abstractivo usando BART"""
    try:
        from transformers import pipeline
        
        # Inicializar pipeline
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        
        # Combinar contenido
        all_text = " ".join([
            chunk.get('content', '') for chunk in chunks 
            if chunk.get('content')
        ])
        
        # Limitar longitud
        words = all_text.split()
        if len(words) > 1024:
            all_text = " ".join(words[:1024])
        
        # Generar resumen
        summary = summarizer(
            all_text,
            max_length=max_length,
            min_length=100,
            do_sample=False
        )[0]['summary_text']
        
        return f"**Resumen Abstractivo (BART)**:\n\n{summary}"
        
    except Exception as e:
        logger.error(f"Error en resumen abstractivo: {e}")
        # Fallback a resumen básico
        return self.generate_rag_summary(chunks, max_length)

# Y en render_automatic_summary (Línea ~4650)
summary_type = st.selectbox(
    "Tipo de Resumen:",
    options=["comprehensive", "executive", "analytical", "thematic", "abstractive"],  # ← Agregar
    format_func=lambda x: {
        "comprehensive": "📋 Comprehensivo",
        "executive": "🎯 Ejecutivo",
        "analytical": "🔍 Analítico",
        "thematic": "🏷️ Temático",
        "abstractive": "🤖 Abstractivo (BART)"  # ← Nuevo
    }[x]
)

# Modificar la generación
if summary_type == "abstractive":
    summary = analyzer.generate_abstractive_summary(chunks, max_length=500)
else:
    result = analyzer.generate_intelligent_summary(chunks, summary_type)
    summary = result['summary']
```

---

## 🔍 Búsqueda Rápida de Funcionalidades

### Por Palabra Clave

| 🔎 Buscar | 📍 Ir a Línea | 🎯 Encontrarás |
|-----------|---------------|----------------|
| `AnalysisConfig` | 145 | Configuración central |
| `CacheManager` | 223 | Sistema de cache |
| `TextPreprocessor` | 276 | Preprocesamiento de texto |
| `ConceptExtractor` | 346 | Extracción de conceptos |
| `ThemeAnalyzer` | 533 | Análisis de temas |
| `SentimentAnalyzer` | 749 | Análisis de sentimientos |
| `create_interactive_concept_map` | 1918 | Mapas conceptuales |
| `create_interactive_mind_map` | 2214 | Mapas mentales |
| `generate_intelligent_summary` | 1200 | Resúmenes con LLM |
| `perform_triangulation` | 2320 | Triangulación |
| `perform_clustering` | 3637 | Clustering |
| `generate_word_cloud` | 3819 | Nubes de palabras |
| `perform_parallel_analysis` | 1589 | Análisis paralelo |
| `optimize_performance` | 1735 | Optimización automática |

---

## 🛠️ Modificaciones Comunes

### Cambiar Número de Conceptos por Defecto

**Línea 145** → `AnalysisConfig`
```python
@dataclass
class AnalysisConfig:
    max_concepts: int = 100  # Era 50
```

### Habilitar/Deshabilitar Cache

**Línea 1020** → `__init__` de `AdvancedQualitativeAnalyzer`
```python
def __init__(self, config: Optional[AnalysisConfig] = None):
    self.config = config or AnalysisConfig(
        enable_cache=False  # Deshabilitar cache
    )
```

### Ajustar Umbrales de Sentimiento

**Línea 800** → `_analyze_with_textblob_vader`
```python
# Cambiar umbrales
if polarity > 0.2:  # Era 0.1
    sentiment = 'positive'
elif polarity < -0.2:  # Era -0.1
    sentiment = 'negative'
```

### Cambiar Colores de Mapas Conceptuales

**Línea 2000** → `create_interactive_concept_map`
```python
professional_palette = {
    'main_theme': {
        'background': '#1a237e',  # Cambiar color principal
        'border': '#0d1642',
        'text': '#ffffff'
    }
}
```

### Agregar Nueva Stopword

**Línea 276** → `TextPreprocessor`
```python
def __init__(self):
    self.stopwords_cache = None
    self.custom_stopwords = {
        'palabra_nueva',  # ← Agregar aquí
        'otra_palabra'
    }
```

---

## 📊 Mejoras Recomendadas por Prioridad

### 🔥 Alta Prioridad (Hacer Primero)

1. **Implementar BM25 para Conceptos** (Línea 346)
   - Mejora: 30% más preciso que TF-IDF
   - Complejidad: Media
   - Tiempo: 2 horas

2. **Agregar Cache Persistente** (Línea 223)
   - Mejora: Reduce 90% tiempo en análisis repetidos
   - Complejidad: Baja
   - Tiempo: 1 hora

3. **Optimizar Procesamiento Paralelo** (Línea 1589)
   - Mejora: 3x más rápido en análisis múltiples
   - Complejidad: Media
   - Tiempo: 3 horas

### ⚡ Media Prioridad (Hacer Después)

4. **Implementar BERTopic** (Línea 533)
   - Mejora: Temas más coherentes y semánticos
   - Complejidad: Alta
   - Tiempo: 5 horas

5. **Agregar Mapas 3D** (Línea 1918)
   - Mejora: Mejor visualización de relaciones
   - Complejidad: Media
   - Tiempo: 3 horas

6. **Resúmenes Abstractivos con BART** (Línea 1118)
   - Mejora: Resúmenes más naturales
   - Complejidad: Alta
   - Tiempo: 4 horas

### 💡 Baja Prioridad (Mejoras Futuras)

7. **Análisis de Complejidad Textual**
   - Mejora: Métricas adicionales
   - Complejidad: Baja
   - Tiempo: 2 horas

8. **Exportación a Formatos Múltiples**
   - Mejora: PDF, DOCX, LaTeX
   - Complejidad: Media
   - Tiempo: 4 horas

9. **Dashboard Personalizable**
   - Mejora: UX mejorada
   - Complejidad: Media
   - Tiempo: 5 horas

---

## 🎯 Checklist de Verificación

Antes de hacer cambios:
- [ ] Leer documentación de la sección
- [ ] Identificar dependencias
- [ ] Hacer backup del código
- [ ] Verificar tests existentes

Durante los cambios:
- [ ] Mantener firma de métodos públicos
- [ ] Agregar logging apropiado
- [ ] Actualizar docstrings
- [ ] Preservar fallbacks

Después de los cambios:
- [ ] Ejecutar tests
- [ ] Verificar linting (sin errores)
- [ ] Probar con datos reales
- [ ] Actualizar documentación

---

## 🐛 Debugging Rápido

### Error: "AttributeError: 'NoneType' object has no attribute..."

**Causa**: Configuración no inicializada correctamente

**Solución**: Verificar en `__init__` (Línea ~1020)
```python
# Asegurar que config no sea None
from config.settings import config as global_config
self.cache_path = Path(global_config.CACHE_DIR) / "rag_cache.json"
```

### Error: "KeyError: 'words'"

**Causa**: Formato inconsistente entre métodos

**Solución**: Usar `.get()` con fallback (Línea ~4526)
```python
keywords = topic.get('words', topic.get('keywords', []))
```

### Error: "TF-IDF no funciona"

**Causa**: Muy pocos documentos

**Solución**: Verificar fallback (Línea ~460)
```python
if SKLEARN_AVAILABLE and len(texts) >= 2:  # Mínimo 2 documentos
    concepts = self._extract_with_tfidf(texts)
else:
    concepts = self._extract_with_frequency(texts)  # Fallback
```

---

## 📝 Plantillas de Código

### Plantilla para Nuevo Analizador

```python
from ..core.base import BaseAnalyzer
from ..core.config import AnalysisConfig, AnalysisResult, AnalysisType

class MiNuevoAnalizador(BaseAnalyzer):
    """Descripción del analizador"""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        # Inicializar componentes
    
    def analyze(self, chunks: List[Dict]) -> AnalysisResult:
        """Método principal de análisis"""
        # 1. Validar
        if not self._validate_input(chunks):
            return AnalysisResult(
                analysis_type=AnalysisType.TU_TIPO,
                data={},
                metadata={'error': 'Invalid input'}
            )
        
        # 2. Cache
        cache_key = self._generate_cache_key(chunks)
        cached = self.cache_manager.get(cache_key)
        if cached:
            return cached
        
        # 3. Procesar
        result_data = self._do_analysis(chunks)
        
        # 4. Crear resultado
        result = AnalysisResult(
            analysis_type=AnalysisType.TU_TIPO,
            data=result_data,
            metadata={'info': 'metadata'}
        )
        
        # 5. Guardar en cache
        self.cache_manager.set(cache_key, result)
        
        return result
    
    def _do_analysis(self, chunks):
        """Lógica de análisis"""
        pass

# Registrar
from ..core.base import AnalyzerFactory
AnalyzerFactory.register('mi_analisis', MiNuevoAnalizador)
```

### Plantilla para Nueva Visualización

```python
def render_mi_visualizacion(analyzer, chunks):
    """Renderizar mi nueva visualización"""
    st.header("📊 Mi Visualización")
    
    # Configuración
    col1, col2 = st.columns([3, 1])
    
    with col1:
        param = st.slider("Parámetro", 1, 10, 5)
    
    with col2:
        if st.button("🔄 Generar"):
            if 'mi_viz_cache' in st.session_state:
                del st.session_state.mi_viz_cache
    
    # Generar
    if 'mi_viz_cache' not in st.session_state:
        with st.spinner("Generando..."):
            st.session_state.mi_viz_cache = analyzer.mi_metodo(chunks, param)
    
    data = st.session_state.mi_viz_cache
    
    # Visualizar
    if data:
        st.plotly_chart(data, use_container_width=True)
    else:
        st.error("No se pudo generar la visualización")
```

---

## 🎓 Conclusión

Esta guía rápida te permite:

✅ Navegar el código actual fácilmente
✅ Hacer modificaciones comunes sin riesgo
✅ Agregar nuevas funcionalidades de forma modular
✅ Debugging rápido de problemas comunes
✅ Seguir mejores prácticas

**Recuerda**: Siempre verifica que tus cambios no rompan funcionalidades existentes ejecutando los tests después de cada modificación.

