# 📊 Módulo de Análisis Cualitativo Avanzado - Documentación Completa

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Arquitectura del Módulo](#arquitectura-del-módulo)
3. [Funcionalidades Principales](#funcionalidades-principales)
4. [Guía de Uso](#guía-de-uso)
5. [API Reference](#api-reference)
6. [Mejoras Implementadas](#mejoras-implementadas)
7. [Ejemplos de Uso](#ejemplos-de-uso)
8. [Troubleshooting](#troubleshooting)
9. [Roadmap](#roadmap)

---

## 🎯 Introducción

El **Módulo de Análisis Cualitativo Avanzado** es un sistema integral de procesamiento y análisis de contenido textual que utiliza técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP) y visualizaciones interactivas. Este módulo ha sido completamente refactorizado y mejorado para ofrecer:

- ✅ **Análisis inteligente** con extracción de conceptos usando n-gramas
- ✅ **Visualizaciones interactivas** (mapas conceptuales y mentales)
- ✅ **Arquitectura modular** con clases especializadas
- ✅ **Análisis de sentimientos** avanzado
- ✅ **Triangulación** para una o múltiples fuentes
- ✅ **Sistema de cache** optimizado
- ✅ **Procesamiento paralelo**

### 🎯 Objetivos del Módulo

1. **Extracción Inteligente**: Identificar conceptos clave usando técnicas avanzadas de NLP
2. **Análisis Estructural**: Comprender la organización y jerarquía del contenido
3. **Visualización**: Crear representaciones gráficas interactivas del conocimiento
4. **Validación**: Triangulación de información para mayor confiabilidad
5. **Optimización**: Rendimiento mejorado con cache y procesamiento paralelo

---

## 🏗️ Arquitectura del Módulo

### 📐 Estructura General

```
modules/qualitative_analysis.py
├── 1. IMPORTS Y CONFIGURACIÓN GLOBAL
├── 2. ENUMS, DATACLASSES Y ESTRUCTURAS DE DATOS
├── 3. CLASES BASE Y INTERFACES
├── 4. GESTIÓN DE CACHE Y MEMORIA
├── 5. PREPROCESAMIENTO DE TEXTO
├── 6. EXTRACCIÓN DE CONCEPTOS
├── 7. ANÁLISIS DE TEMAS
├── 8. ANÁLISIS DE SENTIMIENTOS
├── 9. CLUSTERING Y AGRUPACIÓN
├── 10. MAPAS CONCEPTUALES INTERACTIVOS
├── 11. MAPAS MENTALES
├── 12. RESUMENES AUTOMÁTICOS
├── 13. ANÁLISIS DE TRIANGULACIÓN
├── 14. NUBES DE PALABRAS
├── 15. VISUALIZACIONES Y GRÁFICOS
├── 16. ANÁLISIS PARALELO Y OPTIMIZACIÓN
├── 17. MÉTODOS DE CONFIGURACIÓN
├── 18. FUNCIONES DE RENDERIZADO (STREAMLIT)
└── 19. FUNCIÓN PRINCIPAL DE RENDERIZADO
```

### 🔧 Componentes Principales

#### **1. Clases Especializadas**

```python
class TextPreprocessor:
    """Preprocesador de texto especializado"""
    - get_spanish_stopwords()
    - preprocess_text()

class ConceptExtractor:
    """Extractor de conceptos especializado"""
    - analyze()
    - _extract_concepts_advanced()
    - _extract_with_tfidf()
    - _extract_with_frequency()

class ThemeAnalyzer:
    """Analizador de temas especializado"""
    - analyze()
    - _extract_themes_advanced()
    - _extract_themes_with_lda()
    - _extract_themes_with_clustering()

class SentimentAnalyzer:
    """Analizador de sentimientos especializado"""
    - analyze()
    - _analyze_sentiments_advanced()
    - _analyze_with_textblob_vader()

class CacheManager:
    """Gestor de cache optimizado"""
    - get(), set(), clear()
    - _evict_oldest()
    - get_stats()
```

#### **2. Clase Principal**

```python
class AdvancedQualitativeAnalyzer:
    """Analizador cualitativo avanzado con arquitectura modular mejorada"""
    
    # Métodos principales de análisis
    - extract_key_concepts()
    - extract_advanced_themes()
    - advanced_sentiment_analysis()
    - perform_clustering()
    - perform_triangulation_analysis()
    
    # Visualizaciones
    - create_interactive_concept_map()
    - create_interactive_mind_map()
    - generate_word_cloud()
    
    # Métodos avanzados
    - _extract_concepts_with_ngrams()
    - _identify_intelligent_main_theme()
    - _analyze_concept_hierarchy_with_ai()
    - perform_parallel_analysis()
```

---

## 🚀 Funcionalidades Principales

### 1. 📊 **Extracción de Conceptos**

#### **Método Tradicional**
```python
concepts = analyzer.extract_key_concepts(chunks, min_freq=2)
```

#### **Método Avanzado con N-gramas**
```python
concepts = analyzer._extract_concepts_with_ngrams(chunks, max_concepts=30)
```

**Características:**
- ✅ Detecta frases de 1-3 palabras
- ✅ Prioriza conceptos compuestos
- ✅ Bonus de score para n-gramas
- ✅ Contexto enriquecido

### 2. 🎯 **Análisis de Temas**

#### **LDA (Latent Dirichlet Allocation)**
```python
themes = analyzer.extract_advanced_themes(chunks, n_topics=10)
```

#### **Clustering Jerárquico**
```python
clusters = analyzer.perform_clustering(chunks, n_clusters=5)
```

### 3. 😊 **Análisis de Sentimientos**

#### **TextBlob + VADER**
```python
sentiment = analyzer.advanced_sentiment_analysis(chunks)
```

**Métricas:**
- Polaridad (positivo/negativo)
- Subjetividad
- Tendencias temporales
- Distribución por fuente

### 4. 🗺️ **Mapas Conceptuales**

#### **Generación Normal (Rápida)**
```python
html_file = analyzer.create_interactive_concept_map(chunks, layout_type="spring")
```

#### **Generación con IA**
```python
# Usa análisis semántico profundo con LLM
structure = analyzer._analyze_concept_hierarchy_with_ai(chunks)
```

**Características:**
- ✅ Extracción con n-gramas
- ✅ Tema central inteligente
- ✅ Jerarquía de conceptos
- ✅ Relaciones cruzadas
- ✅ Visualización interactiva con PyVis

### 5. 🧠 **Mapas Mentales**

#### **Generación Mejorada**
```python
mind_map = analyzer.create_interactive_mind_map(chunks, node_spacing=350)
```

**Mejoras Visuales:**
- ✅ Texto blanco/gris claro para legibilidad
- ✅ Contenedor 1400x700px (pantalla completa)
- ✅ Espaciado mejorado (450px por defecto)
- ✅ Física más suave para evitar nodos pegados
- ✅ Tamaños de nodos aumentados

### 6. 🔺 **Triangulación**

#### **Multi-Fuente (Clásica)**
```python
triangulation = analyzer.perform_triangulation_analysis(chunks)
```

#### **Fuente Única (Interna)**
```python
# Automático cuando hay una sola fuente
triangulation = analyzer._perform_single_source_triangulation(chunks, source_name)
```

**Tipos de Validación:**
- ✅ Multi-fuente: Conceptos en múltiples documentos
- ✅ Interna: Conceptos en múltiples secciones del mismo documento
- ✅ Confiabilidad calculada por frecuencia de aparición

---

## 📖 Guía de Uso

### 🔧 **Inicialización**

```python
from modules.qualitative_analysis import AdvancedQualitativeAnalyzer

# Crear instancia del analizador
analyzer = AdvancedQualitativeAnalyzer()

# Opcional: Configuración personalizada
from modules.qualitative_analysis import AnalysisConfig

config = AnalysisConfig(
    min_frequency=3,
    max_concepts=50,
    similarity_threshold=0.7,
    enable_cache=True,
    parallel_processing=True
)

analyzer = AdvancedQualitativeAnalyzer(config)
```

### 📊 **Análisis Básico**

```python
# Cargar datos (chunks de documentos)
chunks = [
    {
        'content': 'Texto del documento...',
        'metadata': {'source_file': 'documento.pdf'}
    }
]

# Extraer conceptos clave
concepts = analyzer.extract_key_concepts(chunks)
print(f"Conceptos encontrados: {len(concepts)}")

# Análisis de temas
themes = analyzer.extract_advanced_themes(chunks, n_topics=8)
print(f"Temas identificados: {len(themes.get('topics', []))}")

# Análisis de sentimientos
sentiment = analyzer.advanced_sentiment_analysis(chunks)
print(f"Sentimiento promedio: {sentiment.get('overall_sentiment', {}).get('polarity', 0)}")
```

### 🗺️ **Visualizaciones**

```python
# Mapa conceptual
html_file = analyzer.create_interactive_concept_map(
    chunks, 
    layout_type="spring"  # "spring", "hierarchical", "circular"
)

# Mapa mental
mind_map_data = analyzer.create_interactive_mind_map(
    chunks,
    node_spacing=350,
    return_data=True
)

# Nube de palabras
wordcloud_file = analyzer.generate_word_cloud(chunks)
```

### 🔺 **Triangulación**

```python
# Triangulación automática (detecta si es 1 o múltiples fuentes)
triangulation = analyzer.perform_triangulation_analysis(chunks)

if triangulation['analysis_mode'] == 'multi-source':
    print("Análisis multi-fuente")
elif triangulation['analysis_mode'] == 'single-source-internal':
    print("Análisis de triangulación interna")

# Mostrar conceptos validados
for concept in triangulation['triangulated_concepts'][:10]:
    print(f"{concept['concept']} - Confiabilidad: {concept['reliability']:.2%}")
```

### ⚡ **Análisis Paralelo**

```python
from modules.qualitative_analysis import AnalysisType

# Ejecutar múltiples análisis en paralelo
analysis_types = [
    AnalysisType.CONCEPT_EXTRACTION,
    AnalysisType.THEME_ANALYSIS,
    AnalysisType.SENTIMENT_ANALYSIS,
    AnalysisType.CLUSTERING
]

results = analyzer.perform_parallel_analysis(chunks, analysis_types)

# Acceder a resultados específicos
concepts_result = results['concept_extraction']
themes_result = results['theme_analysis']
```

---

## 🔌 API Reference

### **Clase Principal: AdvancedQualitativeAnalyzer**

#### **Constructor**
```python
def __init__(self, config: Optional[AnalysisConfig] = None)
```

#### **Métodos de Análisis**

##### `extract_key_concepts(chunks, min_freq=2)`
Extrae conceptos clave de los chunks de texto.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto
- `min_freq` (int): Frecuencia mínima para considerar un concepto

**Retorna:**
- `List[Dict]`: Lista de conceptos con score y contexto

##### `extract_advanced_themes(chunks, n_topics=10)`
Análisis avanzado de temas usando LDA y clustering.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto
- `n_topics` (int): Número de temas a identificar

**Retorna:**
- `Dict`: Diccionario con temas, palabras clave y métricas

##### `advanced_sentiment_analysis(chunks)`
Análisis de sentimientos usando TextBlob y VADER.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto

**Retorna:**
- `Dict`: Análisis de sentimientos con polaridad, subjetividad y tendencias

##### `perform_clustering(chunks, n_clusters=5)`
Agrupación de documentos usando K-means y DBSCAN.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto
- `n_clusters` (int): Número de clusters

**Retorna:**
- `Dict`: Resultados del clustering con clusters y métricas

#### **Métodos de Visualización**

##### `create_interactive_concept_map(chunks, layout_type="spring")`
Crea un mapa conceptual interactivo.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto
- `layout_type` (str): Tipo de layout ("spring", "hierarchical", "circular")

**Retorna:**
- `str`: Ruta al archivo HTML generado

##### `create_interactive_mind_map(chunks, node_spacing=250, return_data=False)`
Crea un mapa mental interactivo.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto
- `node_spacing` (int): Espaciado entre nodos
- `return_data` (bool): Si retornar datos además del HTML

**Retorna:**
- `Optional[Dict]`: Datos del mapa mental o None

##### `generate_word_cloud(chunks, source_filter=None)`
Genera una nube de palabras.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto
- `source_filter` (Optional[str]): Filtrar por fuente específica

**Retorna:**
- `Optional[str]`: Ruta al archivo de imagen o None

#### **Métodos Avanzados**

##### `_extract_concepts_with_ngrams(chunks, max_concepts=30)`
Extracción inteligente de conceptos usando n-gramas.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto
- `max_concepts` (int): Número máximo de conceptos

**Retorna:**
- `List[Dict]`: Lista de conceptos con n-gramas

##### `_identify_intelligent_main_theme(chunks, concepts)`
Identifica el tema central de forma inteligente.

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto
- `concepts` (List[Dict]): Lista de conceptos extraídos

**Retorna:**
- `Dict`: Tema central con nombre y descripción

##### `perform_triangulation_analysis(chunks)`
Realiza análisis de triangulación (1 o múltiples fuentes).

**Parámetros:**
- `chunks` (List[Dict]): Lista de chunks de texto

**Retorna:**
- `Dict`: Resultados de triangulación con conceptos validados

---

## ✨ Mejoras Implementadas

### 🎯 **Mejoras en Mapas Conceptuales**

#### **1. Extracción Inteligente con N-gramas**
```python
# ANTES: Palabras sueltas
concepts = ["inteligencia", "artificial", "machine", "learning"]

# AHORA: Frases completas
concepts = ["inteligencia artificial", "machine learning", "procesamiento natural"]
```

#### **2. Modo Normal por Defecto**
- ✅ Generación 3-5x más rápida
- ✅ Resultados más coherentes
- ✅ Modo IA como opción avanzada

#### **3. Mejor Separación Visual**
- ✅ Distancia entre nodos aumentada
- ✅ Tamaños de nodos optimizados
- ✅ Colores más contrastantes

### 🧠 **Mejoras en Mapas Mentales**

#### **1. Legibilidad Mejorada**
```python
# Texto optimizado para fondo oscuro
'fontColor': '#ffffff'  # Blanco para nodos
'fontColor': '#e0e0e0'  # Gris claro para etiquetas
```

#### **2. Contenedor de Pantalla Completa**
```python
# Tamaño optimizado
width=1400,   # +17% más ancho
height=700,   # +40% más alto
```

#### **3. Espaciado Mejorado**
```python
# Configuración por defecto mejorada
min_value=200,    # +33% más espaciado mínimo
value=450,        # +29% más espaciado por defecto
max_value=800,    # +33% más espaciado máximo
```

### 🔺 **Mejoras en Triangulación**

#### **1. Soporte para Fuente Única**
```python
def _perform_single_source_triangulation(self, chunks, source_name):
    """Triangulación interna dividiendo el documento en secciones"""
    # Divide en secciones de 3-5 chunks
    # Analiza conceptos por sección
    # Identifica conceptos que aparecen en múltiples secciones
```

#### **2. Validación Mejorada**
- ✅ Confiabilidad por frecuencia de aparición
- ✅ Análisis de distribución por secciones
- ✅ Métricas de validación cruzada

### 🏗️ **Mejoras en Arquitectura**

#### **1. Clases Especializadas**
```python
# Separación de responsabilidades
TextPreprocessor    # Preprocesamiento
ConceptExtractor    # Extracción de conceptos
ThemeAnalyzer       # Análisis de temas
SentimentAnalyzer   # Análisis de sentimientos
CacheManager        # Gestión de cache
```

#### **2. Sistema de Cache Optimizado**
```python
class CacheManager:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
```

#### **3. Procesamiento Paralelo**
```python
def perform_parallel_analysis(self, chunks, analysis_types):
    """Ejecuta múltiples análisis en paralelo usando ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
        # Procesamiento concurrente
```

---

## 💡 Ejemplos de Uso

### 📊 **Ejemplo 1: Análisis Completo de Documento**

```python
from modules.qualitative_analysis import AdvancedQualitativeAnalyzer

# Inicializar analizador
analyzer = AdvancedQualitativeAnalyzer()

# Datos de ejemplo
chunks = [
    {
        'content': 'La inteligencia artificial está transformando la educación...',
        'metadata': {'source_file': 'educacion_ia.pdf'}
    },
    {
        'content': 'Machine learning permite personalizar el aprendizaje...',
        'metadata': {'source_file': 'educacion_ia.pdf'}
    }
]

# Análisis completo
print("🔍 Extrayendo conceptos...")
concepts = analyzer.extract_key_concepts(chunks)
print(f"✅ {len(concepts)} conceptos encontrados")

print("🎯 Analizando temas...")
themes = analyzer.extract_advanced_themes(chunks, n_topics=5)
print(f"✅ {len(themes['topics'])} temas identificados")

print("😊 Analizando sentimientos...")
sentiment = analyzer.advanced_sentiment_analysis(chunks)
print(f"✅ Sentimiento: {sentiment['overall_sentiment']['label']}")

print("🔺 Realizando triangulación...")
triangulation = analyzer.perform_triangulation_analysis(chunks)
print(f"✅ {triangulation['validated_concepts']} conceptos validados")

print("🗺️ Generando visualizaciones...")
concept_map = analyzer.create_interactive_concept_map(chunks)
mind_map = analyzer.create_interactive_mind_map(chunks)
wordcloud = analyzer.generate_word_cloud(chunks)

print("🎉 Análisis completo finalizado!")
```

### 🗺️ **Ejemplo 2: Generación de Mapas Interactivos**

```python
# Mapa conceptual con diferentes layouts
layouts = ["spring", "hierarchical", "circular"]

for layout in layouts:
    print(f"Generando mapa conceptual con layout: {layout}")
    html_file = analyzer.create_interactive_concept_map(chunks, layout_type=layout)
    print(f"✅ Mapa guardado en: {html_file}")

# Mapa mental con configuración personalizada
mind_map_config = {
    'node_spacing': 400,
    'return_data': True
}

mind_map_data = analyzer.create_interactive_mind_map(chunks, **mind_map_config)
print(f"✅ Mapa mental generado con {len(mind_map_data['nodes'])} nodos")
```

### 🔺 **Ejemplo 3: Triangulación Avanzada**

```python
# Preparar datos de múltiples fuentes
multi_source_chunks = [
    # Fuente 1: Artículo científico
    {'content': 'La IA en educación...', 'metadata': {'source_file': 'articulo_cientifico.pdf'}},
    {'content': 'Machine learning aplicado...', 'metadata': {'source_file': 'articulo_cientifico.pdf'}},
    
    # Fuente 2: Reporte técnico
    {'content': 'Implementación de IA...', 'metadata': {'source_file': 'reporte_tecnico.pdf'}},
    {'content': 'Algoritmos de aprendizaje...', 'metadata': {'source_file': 'reporte_tecnico.pdf'}},
]

# Triangulación multi-fuente
triangulation = analyzer.perform_triangulation_analysis(multi_source_chunks)

print(f"📊 Modo de análisis: {triangulation['analysis_mode']}")
print(f"📚 Fuentes analizadas: {triangulation['total_sources']}")
print(f"🔍 Conceptos totales: {triangulation['total_concepts']}")
print(f"✅ Conceptos validados: {triangulation['validated_concepts']}")

# Mostrar conceptos más confiables
print("\n🎯 Conceptos más confiables:")
for concept in triangulation['triangulated_concepts'][:5]:
    print(f"• {concept['concept']} - Confiabilidad: {concept['reliability']:.2%}")
```

### ⚡ **Ejemplo 4: Análisis Paralelo**

```python
from modules.qualitative_analysis import AnalysisType

# Configurar tipos de análisis
analysis_types = [
    AnalysisType.CONCEPT_EXTRACTION,
    AnalysisType.THEME_ANALYSIS,
    AnalysisType.SENTIMENT_ANALYSIS,
    AnalysisType.CLUSTERING
]

print("⚡ Ejecutando análisis en paralelo...")
start_time = time.time()

results = analyzer.perform_parallel_analysis(chunks, analysis_types)

end_time = time.time()
print(f"✅ Análisis completado en {end_time - start_time:.2f} segundos")

# Procesar resultados
for analysis_type, result in results.items():
    print(f"\n📊 {analysis_type.value}:")
    if hasattr(result, 'data'):
        print(f"   • Procesamiento exitoso")
        print(f"   • Tiempo: {result.processing_time:.2f}s")
    else:
        print(f"   • Error en procesamiento")
```

---

## 🔧 Troubleshooting

### ❗ **Problemas Comunes**

#### **1. Error: "No hay datos disponibles"**

**Causa:** Los chunks están vacíos o mal formateados.

**Solución:**
```python
# Verificar formato de chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.get('content', 'SIN CONTENIDO')[:50]}...")

# Verificar que tengan contenido
valid_chunks = [chunk for chunk in chunks if chunk.get('content', '').strip()]
print(f"Chunks válidos: {len(valid_chunks)}")
```

#### **2. Error: "streamlit-agraph no está disponible"**

**Causa:** La librería streamlit-agraph no está instalada.

**Solución:**
```bash
pip install streamlit-agraph
```

#### **3. Error: "Error analizando jerarquía de conceptos"**

**Causa:** Problemas con el análisis de conceptos o memoria insuficiente.

**Solución:**
```python
# Reducir el número de conceptos
concepts = analyzer.extract_key_concepts(chunks, min_freq=3)

# Limpiar cache
analyzer.clear_cache()

# Usar modo normal en lugar de IA
concept_map = analyzer.create_interactive_concept_map(chunks, layout_type="spring")
```

#### **4. Error: "Ollama no está disponible"**

**Causa:** El servicio Ollama no está ejecutándose.

**Solución:**
```bash
# Iniciar Ollama
ollama serve

# Verificar modelos disponibles
ollama list
```

#### **5. Problemas de Rendimiento**

**Causa:** Datasets muy grandes o configuración subóptima.

**Solución:**
```python
# Configurar para mejor rendimiento
config = AnalysisConfig(
    max_concepts=30,        # Reducir conceptos
    chunk_size=1500,        # Reducir tamaño de chunks
    enable_cache=True,      # Habilitar cache
    parallel_processing=True, # Procesamiento paralelo
    max_workers=2           # Reducir workers
)

analyzer = AdvancedQualitativeAnalyzer(config)

# Procesar en lotes
batch_size = 10
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    results = analyzer.extract_key_concepts(batch)
```

### 🔍 **Debug y Logging**

```python
import logging

# Habilitar logging detallado
logging.basicConfig(level=logging.DEBUG)

# Verificar estado del cache
cache_stats = analyzer.get_cache_stats()
print(f"Cache hits: {cache_stats['hits']}")
print(f"Cache misses: {cache_stats['misses']}")
print(f"Cache size: {cache_stats['size']}")

# Verificar métricas de rendimiento
metrics = analyzer.get_performance_metrics()
print(f"Métricas: {metrics}")
```

---

## 🗺️ Roadmap

### 📅 **Versión 2.1 (Próxima)**

- ✅ **API REST** para integración externa
- ✅ **Exportación** a múltiples formatos (JSON, XML, CSV)
- ✅ **Métricas avanzadas** de calidad
- ✅ **Optimizaciones** de memoria

### 📅 **Versión 2.2 (Futuro)**

- 🔮 **Integración** con bases de datos vectoriales
- 🔮 **Análisis temporal** de documentos
- 🔮 **Clustering automático** de temas
- 🔮 **Exportación** de reportes automáticos

### 📅 **Versión 3.0 (Largo Plazo)**

- 🚀 **IA generativa** para resúmenes
- 🚀 **Análisis multimodal** (texto + imágenes)
- 🚀 **API GraphQL** completa
- 🚀 **Despliegue en la nube** escalable

---

## 📚 Referencias y Recursos

### 📖 **Documentación Técnica**
- [Guía de Arquitectura](./ANALISIS_CUALITATIVO_ARQUITECTURA.md)
- [Guía Rápida](./ANALISIS_CUALITATIVO_GUIA_RAPIDA.md)
- [Prompt de Creación](./ANALISIS_CUALITATIVO_PROMPT_CREACION.md)

### 🔗 **Enlaces Útiles**
- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Documentación de PyVis](https://pyvis.readthedocs.io/)
- [Documentación de scikit-learn](https://scikit-learn.org/)

### 🛠️ **Herramientas Relacionadas**
- **Ollama**: Modelos LLM locales
- **NLTK**: Procesamiento de lenguaje natural
- **scikit-learn**: Machine learning
- **PyVis**: Visualizaciones de redes

---

## 📄 Licencia

Este módulo es parte del proyecto CogniChat y está licenciado bajo la **Licencia MIT**.

---

**Desarrollado con ❤️ por el equipo de CogniChat**

*"Transformando documentos en conocimiento, una visualización a la vez."* 🧠✨
