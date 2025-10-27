# 📋 Documentación de Requerimientos - CogniChat

Este documento explica el propósito y función de cada dependencia utilizada en el proyecto CogniChat, un sistema RAG (Retrieval-Augmented Generation) avanzado.

## 🖥️ Framework Principal

### `streamlit>=1.29.0`
**Propósito**: Framework principal para crear la interfaz web interactiva de la aplicación.
- Permite crear aplicaciones web de manera rápida y sencilla
- Proporciona componentes UI como pestañas, sidebar, formularios, etc.
- Maneja el estado de la aplicación y la interactividad

### `streamlit-chat>=0.1.1`
**Propósito**: Extensión de Streamlit para crear interfaces de chat más elegantes.
- Proporciona componentes específicos para conversaciones
- Mejora la experiencia visual del chat
- Facilita la implementación de burbujas de mensaje

## 🌐 Comunicación y APIs

### `requests>=2.31.0`
**Propósito**: Biblioteca para realizar peticiones HTTP.
- Comunicación con la API de Ollama
- Verificación del estado del servidor
- Descarga de recursos externos si es necesario

## ⚙️ Configuración y Utilidades

### `python-dotenv>=1.0.0`
**Propósito**: Carga variables de entorno desde archivos `.env`.
- Gestión segura de configuraciones
- Separación de configuración del código
- Facilita el despliegue en diferentes entornos

### `pathlib`
**Propósito**: Manipulación moderna de rutas de archivos.
- Gestión cross-platform de rutas
- Operaciones de archivos y directorios
- Incluido en Python estándar

### `psutil>=5.9.0`
**Propósito**: Información del sistema y monitoreo de procesos.
- Monitoreo de recursos del sistema
- Verificación de procesos en ejecución
- Estadísticas de rendimiento

## 📄 Procesamiento de Documentos

### `PyPDF2>=3.0.0`
**Propósito**: Lectura y manipulación de archivos PDF.
- Extracción de texto de documentos PDF
- Procesamiento de documentos académicos y técnicos
- Soporte para PDFs complejos

### `python-docx>=1.1.0`
**Propósito**: Procesamiento de documentos de Microsoft Word (.docx).
- Extracción de texto y metadatos
- Preservación de formato cuando sea necesario
- Compatibilidad con documentos corporativos

### `openpyxl>=3.1.0`
**Propósito**: Lectura y escritura de archivos Excel (.xlsx).
- Procesamiento de hojas de cálculo
- Extracción de datos tabulares
- Análisis de datos estructurados

### `pandas>=2.0.0`
**Propósito**: Manipulación y análisis de datos estructurados.
- Procesamiento de datos tabulares
- Limpieza y transformación de datos
- Análisis estadístico básico

### `beautifulsoup4>=4.12.0`
**Propósito**: Parsing de HTML y XML.
- Extracción de contenido web
- Limpieza de texto HTML
- Procesamiento de documentos web

## 📊 Logging y Monitoreo

### `loguru>=0.7.2`
**Propósito**: Sistema de logging avanzado y fácil de usar.
- Logging estructurado y colorizado
- Rotación automática de logs
- Mejor experiencia de debugging

## 📈 Visualizaciones y Análisis

### `plotly>=5.17.0`
**Propósito**: Visualizaciones interactivas avanzadas.
- Gráficos interactivos en la interfaz web
- Visualización de métricas del sistema
- Análisis visual de datos

### `networkx>=3.2.1`
**Propósito**: Análisis y visualización de grafos.
- Mapas conceptuales
- Análisis de relaciones entre documentos
- Visualización de conexiones semánticas

### `numpy>=1.24.0`
**Propósito**: Computación numérica fundamental.
- Operaciones matemáticas eficientes
- Soporte para arrays multidimensionales
- Base para otras bibliotecas científicas

## 🤖 Machine Learning y NLP

### `scikit-learn>=1.3.0`
**Propósito**: Algoritmos de machine learning.
- Clustering de documentos
- Clasificación de texto
- Reducción de dimensionalidad
- Métricas de evaluación

### `nltk>=3.8.1`
**Propósito**: Procesamiento de lenguaje natural.
- Tokenización de texto
- Análisis sintáctico
- Preprocesamiento de texto

### `textblob>=0.17.1`
**Propósito**: Análisis de sentimientos y NLP simple.
- Análisis de polaridad
- Extracción de entidades
- Corrección ortográfica

### `wordcloud>=1.9.2`
**Propósito**: Generación de nubes de palabras.
- Visualización de términos frecuentes
- Análisis visual de contenido
- Resúmenes visuales de documentos

## 📊 Análisis Estadístico Avanzado

### `seaborn>=0.12.0` y `matplotlib>=3.7.0`
**Propósito**: Visualizaciones estadísticas.
- Gráficos estadísticos avanzados
- Análisis exploratorio de datos
- Visualizaciones científicas

### `scipy>=1.11.0`
**Propósito**: Computación científica.
- Algoritmos de optimización
- Estadísticas avanzadas
- Procesamiento de señales

### `statsmodels>=0.14.0`
**Propósito**: Modelos estadísticos.
- Análisis de regresión
- Pruebas estadísticas
- Modelado econométrico

## 🔍 Análisis Dimensional

### `umap-learn>=0.5.3`
**Propósito**: Reducción de dimensionalidad no lineal.
- Visualización de embeddings
- Análisis de similitud semántica
- Exploración de espacios vectoriales

### `hdbscan>=0.8.33`
**Propósito**: Clustering basado en densidad.
- Agrupación automática de documentos
- Detección de temas
- Análisis de patrones

## 🧠 Procesamiento de Texto Avanzado

### `spacy>=3.7.0`
**Propósito**: NLP industrial de alto rendimiento.
- Análisis sintáctico avanzado
- Reconocimiento de entidades nombradas
- Procesamiento eficiente de texto

### `transformers>=4.30.0`
**Propósito**: Modelos de transformers pre-entrenados.
- Modelos de lenguaje avanzados
- Embeddings contextuales
- Fine-tuning de modelos

## 💾 Bases de Datos y Almacenamiento

### `sqlite3`
**Propósito**: Base de datos ligera integrada.
- Almacenamiento de metadatos
- Historial de conversaciones
- Configuraciones persistentes

### `chromadb>=0.4.0`
**Propósito**: Base de datos vectorial para embeddings.
- Almacenamiento eficiente de vectores
- Búsqueda de similitud semántica
- Indexación de documentos

## 🛠️ Utilidades de Desarrollo

### `tqdm>=4.65.0`
**Propósito**: Barras de progreso.
- Feedback visual durante procesamiento
- Monitoreo de tareas largas
- Mejor experiencia de usuario

### `joblib>=1.3.0`
**Propósito**: Paralelización y persistencia.
- Procesamiento paralelo
- Caché de resultados
- Serialización eficiente

### `Pillow>=10.0.0`
**Propósito**: Procesamiento de imágenes.
- Manipulación de imágenes
- Conversión de formatos
- Extracción de texto de imágenes (OCR)

## 🕸️ Visualizaciones Interactivas

### `pyvis>=0.3.2`
**Propósito**: Visualización interactiva de redes.
- Mapas conceptuales dinámicos
- Exploración interactiva de grafos
- Visualización de relaciones complejas

### `streamlit-agraph>=0.0.45`
**Propósito**: Componente de grafos para Streamlit.
- Integración de visualizaciones de red
- Interactividad en la interfaz web
- Exploración visual de datos

### `graphviz>=0.20.1`
**Propósito**: Generación de diagramas.
- Diagramas de flujo
- Visualización de estructuras
- Documentación visual

## 🤖 Inteligencia Artificial Local

### `ollama>=0.1.7`
**Propósito**: Cliente para el servidor Ollama.
- Comunicación con modelos locales
- Gestión de modelos de IA
- Inferencia local sin dependencias externas

### `langchain>=0.1.0`
**Propósito**: Framework para aplicaciones con LLMs.
- Cadenas de procesamiento de IA
- Integración con múltiples modelos
- Patrones de diseño para IA

### `sentence-transformers>=2.2.2`
**Propósito**: Embeddings de oraciones y texto.
- Generación de representaciones vectoriales
- Búsqueda semántica
- Comparación de similitud textual

### `torch>=2.0.0` y `torchvision>=0.15.0`
**Propósito**: Framework de deep learning.
- Soporte para modelos de transformers
- Computación en GPU si está disponible
- Base para modelos de IA avanzados

## 🔬 Análisis Semántico

### `gensim>=4.3.0`
**Propósito**: Modelado de temas y análisis semántico.
- Modelos Word2Vec, Doc2Vec
- Análisis de temas (LDA)
- Similitud semántica

### `spacy-transformers>=1.2.5`
**Propósito**: Integración de transformers con spaCy.
- Modelos de lenguaje contextuales
- NLP de última generación
- Análisis semántico avanzado

## 🧪 Desarrollo y Testing

### `pytest>=7.4.0`
**Propósito**: Framework de testing.
- Pruebas unitarias y de integración
- Validación de funcionalidades
- Aseguramiento de calidad

### `black>=23.0.0`
**Propósito**: Formateador de código Python.
- Estilo de código consistente
- Formateo automático
- Mejores prácticas de código

### `flake8>=6.0.0`
**Propósito**: Linter para Python.
- Detección de errores de estilo
- Validación de código
- Mantenimiento de calidad

---

## 📝 Resumen por Categorías

- **🖥️ Interfaz Web**: Streamlit y extensiones
- **📄 Procesamiento de Documentos**: PDF, Word, Excel, HTML
- **🤖 IA y NLP**: Ollama, LangChain, Transformers, spaCy
- **📊 Análisis y Visualización**: Plotly, NetworkX, Seaborn
- **💾 Almacenamiento**: SQLite, ChromaDB
- **🛠️ Desarrollo**: Testing, Linting, Formateo
- **⚙️ Utilidades**: Logging, Configuración, Monitoreo

Cada dependencia ha sido seleccionada cuidadosamente para proporcionar funcionalidades específicas que contribuyen al objetivo general de CogniChat: crear un sistema RAG avanzado, local y fácil de usar.