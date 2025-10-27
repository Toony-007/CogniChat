# 🐍 Guía de Instalación con Conda - CogniChat

Esta guía te permitirá instalar y configurar CogniChat usando Conda para una gestión óptima de dependencias y entornos.

## 📋 Requisitos Previos

### 1. Instalar Miniconda/Anaconda

**Opción A: Miniconda (Recomendado - más ligero)**
- Descargar desde: https://docs.conda.io/en/latest/miniconda.html
- Instalar siguiendo las instrucciones del instalador

**Opción B: Anaconda (Completo)**
- Descargar desde: https://www.anaconda.com/products/distribution
- Incluye más paquetes preinstalados

### 2. Instalar Ollama

**Windows:**
- Descargar desde: https://ollama.ai/download
- Ejecutar el instalador
- Verificar instalación: `ollama --version`

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 3. Instalar Git
- Descargar desde: https://git-scm.com/downloads
- Configurar usuario y email

## 🚀 Instalación Paso a Paso

### Paso 1: Clonar el Repositorio

```bash
git clone <repository-url>
cd CogniChat
```

### Paso 2: Crear Entorno Conda

```bash
# Crear entorno con Python 3.11 (recomendado)
conda create -n cognichat-py311 python=3.11 -y

# Activar el entorno
conda activate cognichat-py311
```

### Paso 3: Instalar Dependencias Base

```bash
# Instalar paquetes científicos desde conda-forge (más rápido)
conda install -c conda-forge numpy pandas matplotlib seaborn plotly scikit-learn scipy -y

# Instalar paquetes de NLP
conda install -c conda-forge nltk spacy -y

# Instalar Streamlit
conda install -c conda-forge streamlit -y
```

### Paso 4: Instalar Dependencias Específicas con pip

```bash
# Dependencias que no están en conda o son más recientes en pip
pip install streamlit-chat>=0.1.1
pip install textblob>=0.17.1
pip install transformers>=4.30.0
pip install umap-learn>=0.5.3
pip install hdbscan>=0.8.33
pip install networkx>=3.2.1
pip install wordcloud>=1.9.2
pip install statsmodels>=0.14.0

# Dependencias para exportación PDF
pip install reportlab>=4.0.0
pip install pyperclip>=1.8.2

# Otras dependencias específicas
pip install python-docx>=0.8.11
pip install openpyxl>=3.1.0
pip install PyPDF2>=3.0.0
pip install python-dotenv>=1.0.0
pip install requests>=2.31.0
pip install vaderSentiment>=3.3.2
```

### Paso 5: Descargar Modelos de spaCy

```bash
# Modelo en español
python -m spacy download es_core_news_sm

# Modelo en inglés (opcional)
python -m spacy download en_core_web_sm
```

### Paso 6: Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus configuraciones
# En Windows: notepad .env
# En Linux/macOS: nano .env
```

**Contenido sugerido para .env:**
```env
# URL de Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Modelos por defecto
DEFAULT_LLM_MODEL=llama3.1:8b
DEFAULT_EMBEDDING_MODEL=nomic-embed-text:latest

# Configuración RAG
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_DOCS=5

# Configuración de logging
LOG_LEVEL=INFO
LOG_FILE=data/logs/cognichat.log
```

### Paso 7: Descargar Modelos de Ollama

```bash
# Modelos principales (requeridos)
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest

# Modelos adicionales (opcionales)
ollama pull deepseek-r1:7b
ollama pull all-minilm:latest
ollama pull mistral:7b
```

### Paso 8: Verificar Instalación

```bash
# Verificar que el entorno esté activo
conda info --envs

# Verificar dependencias críticas
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import utils.chat_history; print('PDF_AVAILABLE:', utils.chat_history.PDF_AVAILABLE)"

# Verificar conexión con Ollama
python -c "import requests; print('Ollama:', requests.get('http://localhost:11434/api/tags').status_code == 200)"
```

### Paso 9: Ejecutar la Aplicación

```bash
# Asegurarse de que el entorno esté activo
conda activate cognichat-py311

# Ejecutar la aplicación
streamlit run app.py
```

La aplicación estará disponible en: http://localhost:8501

## 📦 Exportar/Importar Entorno

### Exportar Entorno (para migración)

```bash
# Exportar entorno completo
conda env export > environment.yml

# Exportar solo dependencias pip
pip freeze > requirements-conda.txt
```

### Importar Entorno (en nuevo dispositivo)

```bash
# Crear entorno desde archivo
conda env create -f environment.yml

# O crear manualmente y instalar dependencias
conda create -n cognichat-py311 python=3.11 -y
conda activate cognichat-py311
pip install -r requirements-conda.txt
```

## 🔧 Comandos Útiles de Conda

### Gestión de Entornos
```bash
# Listar entornos
conda env list

# Activar entorno
conda activate cognichat-py311

# Desactivar entorno
conda deactivate

# Eliminar entorno
conda env remove -n cognichat-py311
```

### Gestión de Paquetes
```bash
# Listar paquetes instalados
conda list

# Buscar paquete
conda search package_name

# Instalar paquete
conda install package_name

# Actualizar paquete
conda update package_name

# Eliminar paquete
conda remove package_name
```

### Información del Sistema
```bash
# Información de conda
conda info

# Información del entorno activo
conda info --envs

# Limpiar cache
conda clean --all
```

## 🐛 Solución de Problemas Comunes

### Error: "conda no se reconoce como comando"
**Solución:**
1. Reiniciar terminal/PowerShell
2. Verificar que conda esté en PATH
3. Ejecutar: `conda init powershell` (Windows)

### Error: "Solving environment: failed"
**Solución:**
```bash
# Actualizar conda
conda update conda

# Limpiar cache
conda clean --all

# Usar mamba (más rápido)
conda install mamba -c conda-forge
mamba install package_name
```

### Error: "Package not found"
**Solución:**
```bash
# Buscar en diferentes canales
conda search -c conda-forge package_name

# Usar pip como alternativa
pip install package_name
```

### Problemas de Rendimiento
**Solución:**
```bash
# Usar libmamba solver (más rápido)
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## 📋 Lista de Verificación Post-Instalación

- [ ] Entorno conda creado y activado
- [ ] Todas las dependencias instaladas sin errores
- [ ] Modelos de spaCy descargados
- [ ] Ollama funcionando y modelos descargados
- [ ] Variables de entorno configuradas
- [ ] Aplicación ejecutándose en http://localhost:8501
- [ ] Funcionalidad de exportación PDF disponible
- [ ] Conexión con Ollama establecida
- [ ] Carga de documentos funcionando
- [ ] Chat respondiendo correctamente

## 🔄 Migración a Otro Dispositivo

### En el dispositivo origen:
```bash
# 1. Exportar entorno
conda env export > cognichat-environment.yml

# 2. Copiar archivos del proyecto
# Incluir: código fuente, .env, datos importantes
```

### En el dispositivo destino:
```bash
# 1. Instalar conda/miniconda
# 2. Clonar repositorio o copiar archivos
# 3. Crear entorno desde archivo
conda env create -f cognichat-environment.yml

# 4. Activar entorno
conda activate cognichat-py311

# 5. Instalar Ollama y descargar modelos
# 6. Configurar .env
# 7. Ejecutar aplicación
streamlit run app.py
```

## 📞 Soporte Adicional

Si encuentras problemas durante la instalación:

1. **Consulta la documentación oficial:**
   - [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
   - [Ollama Documentation](https://github.com/ollama/ollama)

2. **Revisa los logs:**
   - Logs de conda: `conda info`
   - Logs de la aplicación: `data/logs/cognichat.log`

3. **Verifica el estado del sistema:**
   - Pestaña "Alertas" en la aplicación
   - Centro de alertas para diagnósticos

---

**¡Instalación completada! 🎉**

Ahora puedes disfrutar de todas las funcionalidades de CogniChat con un entorno optimizado y reproducible.