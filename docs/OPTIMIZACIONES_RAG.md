# Optimizaciones del Sistema RAG - CogniChat

## 📋 Resumen de Optimizaciones Implementadas

Este documento detalla las optimizaciones completas realizadas al sistema RAG (Retrieval-Augmented Generation) de CogniChat para mejorar el rendimiento y permitir respuestas más extensas con mejor uso del contexto.

## 🚀 Mejoras de Configuración

### Parámetros de Generación Optimizados
- **MAX_RESPONSE_TOKENS**: Incrementado a 3000 tokens para respuestas más extensas
- **OLLAMA_TIMEOUT**: Aumentado a 120 segundos para manejar consultas complejas
- **DEFAULT_LLM_MODEL**: Optimizado para `deepseek-m1:7b` con soporte extendido

### Parámetros RAG Optimizados
- **CHUNK_SIZE**: Incrementado a 2000 caracteres para mejor contexto
- **CHUNK_OVERLAP**: Aumentado a 300 caracteres para mayor coherencia
- **MAX_RETRIEVAL_DOCS**: Elevado a 15 documentos para contexto ampliado
- **SIMILARITY_THRESHOLD**: Reducido a 0.6 para incluir contenido más relevante

## 🔍 Sistema de Trazabilidad Completo

### Logging de Chunks
- **Archivo**: `logs/retrieved_chunks.log`
- **Contenido**: Registro detallado de todos los chunks recuperados por consulta
- **Formato**: JSON estructurado con timestamps y metadatos

### Modo Debug Avanzado
- **Activación**: Toggle en la interfaz de usuario
- **Información mostrada**:
  - Query ID único para cada consulta
  - Número de chunks recuperados
  - Caracteres de contexto utilizados
  - Documentos fuente consultados
  - Scores de similitud
  - Contenido de chunks relevantes

### Historial de Consultas
- **Archivo**: `logs/query_history.json`
- **Registro completo**:
  - Consultas realizadas
  - Respuestas generadas
  - Documentos fuente utilizados
  - Chunks asociados
  - Metadatos de procesamiento

## 📁 Nuevos Módulos Implementados

### `utils/traceability.py`
Módulo central para el sistema de trazabilidad que incluye:

#### Clases Principales
- **ChunkTrace**: Estructura para información de chunks
- **QueryTrace**: Estructura para trazas de consultas
- **TraceabilityManager**: Gestor principal del sistema

#### Funcionalidades
- Logging automático de chunks recuperados
- Generación de Query IDs únicos
- Guardado de historial en formato JSON
- Información de debug estructurada

### Actualizaciones en Módulos Existentes

#### `utils/rag_processor.py`
- Integración del sistema de trazabilidad
- Función `get_context_for_query` con soporte para tracing
- Uso de parámetros optimizados de configuración
- Logging automático cuando está habilitado

#### `utils/ollama_client.py`
- Soporte para parámetro `max_tokens`
- Configuraciones optimizadas para DeepSeek
- Mejor manejo de timeouts

#### `modules/chatbot.py`
- Interfaz de usuario mejorada con modo debug
- Integración completa del sistema de trazabilidad
- Configuración de tokens en tiempo real
- Visualización de información de debug

#### `config/settings.py`
- Nuevas variables de configuración
- Soporte para flags de trazabilidad
- Parámetros optimizados por defecto

## 🎛️ Nuevas Funcionalidades de la Interfaz

### Panel de Configuración Expandido
1. **Modelo LLM**: Selección dinámica de modelos disponibles
2. **Toggle RAG**: Activación/desactivación del sistema RAG
3. **Máximo Tokens**: Control deslizante hasta 3000 tokens
4. **Modo Debug**: Activación de información detallada

### Información de Debug en Tiempo Real
- **Estadísticas de Consulta**: Query ID, chunks, caracteres, tokens
- **Documentos Fuente**: Lista de archivos consultados
- **Chunks Recuperados**: Contenido y scores de similitud
- **Metadatos de Procesamiento**: Modelo usado, configuraciones

### Historial Mejorado
- **Timestamps**: Marcas de tiempo en todos los mensajes
- **Contexto Utilizado**: Visualización del contexto RAG usado
- **Métricas del Chat**: Estadísticas de uso en tiempo real

## 📊 Archivos de Configuración

### `.env` - Variables de Entorno
```env
# Configuración Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=120

# Modelos
DEFAULT_LLM_MODEL=deepseek-m1:7b
DEFAULT_EMBEDDING_MODEL=nomic-embed-text

# Parámetros RAG Optimizados
CHUNK_SIZE=2000
CHUNK_OVERLAP=300
MAX_RETRIEVAL_DOCS=15
SIMILARITY_THRESHOLD=0.6

# Configuración de Respuestas
MAX_RESPONSE_TOKENS=3000

# Sistema de Trazabilidad
ENABLE_CHUNK_LOGGING=true
ENABLE_DEBUG_MODE=true
ENABLE_HISTORY_TRACKING=true

# Otros
MAX_FILE_SIZE_MB=50
LOG_LEVEL=INFO
```

## 🔧 Uso del Sistema Optimizado

### 1. Carga de Documentos
- Navegar a "Procesamiento de Documentos RAG"
- Cargar documentos (PDF, DOCX, TXT, etc.)
- Procesar con parámetros optimizados

### 2. Chat con RAG Optimizado
- Activar el toggle "Usar RAG"
- Configurar tokens máximos (hasta 3000)
- Opcional: Activar modo debug para información detallada

### 3. Análisis de Trazabilidad
- Revisar `logs/retrieved_chunks.log` para chunks recuperados
- Consultar `logs/query_history.json` para historial completo
- Usar modo debug para información en tiempo real

## 📈 Beneficios de las Optimizaciones

### Rendimiento Mejorado
- **Respuestas más extensas**: Hasta 3000 tokens
- **Mejor contexto**: Chunks más grandes con mayor solapamiento
- **Recuperación ampliada**: Hasta 15 documentos por consulta

### Trazabilidad Completa
- **Transparencia total**: Visibilidad completa del proceso RAG
- **Debug avanzado**: Información detallada para optimización
- **Historial persistente**: Registro completo de todas las interacciones

### Experiencia de Usuario
- **Interfaz mejorada**: Controles intuitivos y información clara
- **Feedback en tiempo real**: Estadísticas y métricas visibles
- **Flexibilidad**: Configuración adaptable según necesidades

## 🔄 Compatibilidad

Todas las optimizaciones mantienen **compatibilidad completa** con:
- Estructura existente del proyecto
- Módulos `document_processor.py` y `chatbot.py`
- Flujo de trabajo actual
- Configuraciones previas

## 📝 Notas Técnicas

### Archivos de Log
- Los logs se crean automáticamente en el directorio `logs/`
- Formato JSON para fácil procesamiento
- Rotación automática para evitar archivos grandes

### Configuración Dinámica
- Cambios en `.env` requieren reinicio de la aplicación
- Configuraciones de UI se aplican inmediatamente
- Modo debug se puede activar/desactivar en tiempo real

### Optimización para DeepSeek
- Prompts optimizados para razonamiento profundo
- Contexto completo para análisis comprehensivo
- Configuraciones específicas para mejor rendimiento