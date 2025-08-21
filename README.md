# 🧠 CogniChat - Sistema RAG Avanzado

Sistema de Recuperación y Generación Aumentada (RAG) con análisis cualitativo avanzado y capacidades de procesamiento de documentos inteligente.

## 🚀 Características Principales

### 📄 Procesamiento de Documentos
- **Formatos soportados**: PDF, DOCX, TXT, Excel
- **Chunking inteligente** con preservación de contexto
- **Embeddings vectoriales** para búsqueda semántica
- **Cache optimizado** para rendimiento mejorado

### 🤖 Chatbot Inteligente
- **Integración con Ollama** para modelos locales
- **Respuestas contextuales** basadas en documentos
- **Historial de conversaciones**
- **Trazabilidad de fuentes**

### 📊 Análisis Cualitativo Avanzado
- **Extracción de temas** con LDA (Latent Dirichlet Allocation)
- **Clustering de documentos** con K-means y UMAP
- **Análisis de sentimientos** con VADER y TextBlob
- **Mapas conceptuales interactivos** con NetworkX
- **Nubes de palabras** personalizables
- **Dashboard de métricas** en tiempo real

### 🎨 Visualizaciones Interactivas
- **Gráficos Plotly** interactivos
- **Redes de conceptos** con NetworkX
- **Distribuciones estadísticas** con Seaborn
- **Métricas en tiempo real**

## 🛠️ Instalación

### Requisitos Previos
- Python 3.8+
- Ollama instalado y configurado
- Git

### Instalación Rápida

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd CogniChat
```

2. **Verificar dependencias**:
```bash
python scripts/check_dependencies.py
```

3. **Instalar dependencias automáticamente**:
```bash
python scripts/install_requirements.py
```

4. **O instalar manualmente**:
```bash
pip install -r requirements.txt
```

5. **Configurar variables de entorno**:
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

6. **Ejecutar la aplicación**:
```bash
streamlit run app.py
```

## 📦 Dependencias Principales

### Framework y UI
- `streamlit>=1.29.0` - Interfaz web interactiva
- `streamlit-chat>=0.1.1` - Componentes de chat

### Machine Learning y NLP
- `scikit-learn>=1.3.0` - Algoritmos de ML
- `nltk>=3.8.1` - Procesamiento de lenguaje natural
- `textblob>=0.17.1` - Análisis de sentimientos
- `spacy>=3.7.0` - NLP avanzado
- `transformers>=4.30.0` - Modelos de transformers

### Visualizaciones
- `plotly>=5.17.0` - Gráficos interactivos
- `matplotlib>=3.7.0` - Gráficos estáticos
- `seaborn>=0.12.0` - Visualizaciones estadísticas
- `networkx>=3.2.1` - Análisis de redes
- `wordcloud>=1.9.2` - Nubes de palabras

### Análisis Avanzado
- `umap-learn>=0.5.3` - Reducción dimensional
- `hdbscan>=0.8.33` - Clustering jerárquico
- `scipy>=1.11.0` - Computación científica
- `statsmodels>=0.14.0` - Modelos estadísticos

## 🎯 Uso

### Iniciar la aplicación

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

### Modelos disponibles

El sistema detecta automáticamente los modelos instalados en Ollama. Actualmente tienes:

- **llama3.1:8b** - Modelo principal para chat
- **deepseek-r1:7b** - Modelo alternativo con excelente razonamiento
- **nomic-embed-text:latest** - Modelo de embeddings multilingüe
- **all-minilm:latest** - Modelo de embeddings eficiente

### Funcionalidades principales

#### 📄 Gestión de Documentos
- Carga múltiples archivos simultáneamente
- Validación automática de formatos y tamaños
- Visualización de archivos cargados
- Eliminación individual o masiva

#### 💬 Chat Inteligente
- Conversación natural con IA
- Integración automática con documentos cargados
- Selección de modelos en tiempo real
- Historial de conversación persistente

#### 🚨 Centro de Alertas
- Monitoreo en tiempo real del sistema
- Registro detallado de errores y advertencias
- Estado de conectividad con Ollama
- Logs del sistema

#### ⚙️ Configuraciones
- Gestión de modelos LLM y embeddings
- Parámetros de procesamiento de documentos
- Configuración del sistema
- Exportar/importar configuraciones

## 📁 Estructura del Proyecto

```
CogniChat/
├── app.py                      # Aplicación principal
├── requirements.txt            # Dependencias
├── .env.example               # Variables de entorno
├── .gitignore                 # Exclusiones Git
├── README.md                  # Documentación
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuraciones centralizadas
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Sistema de logging
│   ├── error_handler.py       # Manejo de errores
│   └── ollama_client.py       # Cliente Ollama
├── pages/
│   ├── __init__.py
│   ├── document_upload.py     # Pestaña de documentos
│   ├── chatbot.py             # Pestaña de chat
│   ├── alerts.py              # Pestaña de alertas
│   └── settings.py            # Pestaña de configuración
└── data/                      # Datos (creado automáticamente)
    ├── uploads/               # Archivos subidos
    ├── processed/             # Archivos procesados
    └── cache/                 # Cache del sistema
```

## 🔧 Configuración Avanzada

### Modelos de Ollama

Para descargar modelos adicionales:

```bash
# Modelos LLM recomendados
ollama pull llama3.2:3b        # Modelo ligero
ollama pull mistral:7b         # Multilingüe
ollama pull gemma2:9b          # Google Research

# Modelos de embeddings
ollama pull mxbai-embed-large  # Embeddings de alto rendimiento
ollama pull bge-large          # BAAI General Embedding
```

### Variables de entorno

Edita el archivo `.env` para personalizar:

```env
# URL de Ollama (cambiar si usas Docker o servidor remoto)
OLLAMA_BASE_URL=http://localhost:11434

# Modelos por defecto
DEFAULT_LLM_MODEL=llama3.1:8b
DEFAULT_EMBEDDING_MODEL=nomic-embed-text:latest

# Configuración RAG
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_DOCS=5
```

## 🐛 Solución de Problemas

### Ollama no se conecta

1. Verificar que Ollama esté ejecutándose:
   ```bash
   ollama list
   ```

2. Reiniciar el servicio:
   ```bash
   ollama serve
   ```

3. Verificar la URL en configuraciones

### Error al cargar archivos

1. Verificar que el archivo no exceda el tamaño máximo
2. Comprobar que el formato esté soportado
3. Revisar permisos de escritura en la carpeta `data/`

### Problemas de rendimiento

1. Usar modelos más pequeños (3B en lugar de 8B)
2. Reducir el tamaño de chunk en configuraciones
3. Limitar el número de documentos recuperados

## 🚀 Próximas Funcionalidades

- [ ] Sistema RAG completo con vectores
- [ ] Soporte para más formatos de archivo
- [ ] Integración con bases de datos vectoriales
- [ ] API REST para integración externa
- [ ] Modo multi-usuario
- [ ] Análisis de sentimientos
- [ ] Generación de resúmenes automáticos

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Si tienes problemas o preguntas:

1. Revisa la pestaña de **Alertas** en la aplicación
2. Consulta los logs en `data/logs/`
3. Abre un issue en el repositorio

---

**Desarrollado con ❤️ usando Streamlit y Ollama**