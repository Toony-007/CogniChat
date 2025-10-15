# ================================
# STAGE 1: Builder - Construcción
# ================================
FROM python:3.11-slim as builder

# Variables de entorno para el build
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias de build
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /build

# Copiar archivos de dependencias
COPY requirements.txt pyproject.toml ./

# Crear entorno virtual y instalar dependencias
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Descargar recursos de NLTK en el builder
RUN python -c "import nltk; \
    nltk.download('punkt', download_dir='/opt/nltk_data'); \
    nltk.download('stopwords', download_dir='/opt/nltk_data'); \
    nltk.download('vader_lexicon', download_dir='/opt/nltk_data')"

# ================================
# STAGE 2: Runtime - Producción
# ================================
FROM python:3.11-slim as runtime

# Variables de entorno para Railway
ARG RAILWAY_ENVIRONMENT
ARG RAILWAY_SERVICE_NAME
ARG PORT=8501

# Variables de entorno de producción
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=${PORT}
ENV PATH="/opt/venv/bin:$PATH"

# Configuración específica de Streamlit para producción
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false

# Instalar solo dependencias runtime mínimas
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear usuario no-root para seguridad
RUN groupadd -r cognichat && useradd -r -g cognichat -s /bin/false cognichat

# Crear directorio de aplicación
WORKDIR /app

# Copiar entorno virtual desde builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /app/data/nltk_data

# Copiar código de aplicación
COPY --chown=cognichat:cognichat . .

# Crear directorios necesarios con permisos correctos
RUN mkdir -p data/uploads data/processed data/cache data/chat_history data/temp_exports data/chroma_db && \
    chown -R cognichat:cognichat data/ && \
    chmod -R 755 data/ && \
    chmod +x scripts/*.py

# Crear script de inicio optimizado para producción
COPY --chown=cognichat:cognichat <<'EOF' /app/start.sh
#!/bin/bash
set -e

echo "🚀 Iniciando CogniChat en Producción..."
echo "📝 Puerto: $PORT"
echo "🌐 Entorno: ${RAILWAY_ENVIRONMENT:-production}"
echo "🔧 Servicio: ${RAILWAY_SERVICE_NAME:-cognichat}"
echo "👤 Usuario: $(whoami)"

# Verificar permisos de directorios
echo "🔒 Verificando permisos..."
ls -la data/

# Verificar conexión con Ollama si está configurado
if [ ! -z "$OLLAMA_BASE_URL" ]; then
    echo "🔍 Verificando conexión con Ollama en: $OLLAMA_BASE_URL"
    timeout 10 curl -f "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1 && \
        echo "✅ Ollama disponible" || \
        echo "⚠️  Ollama no disponible, continuando..."
fi

# Verificar que Python y dependencias estén disponibles
echo "🐍 Verificando Python y dependencias..."
python --version
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"

# Iniciar Streamlit con configuración de producción
echo "🎯 Iniciando servidor Streamlit..."
exec streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none \
    --server.runOnSave=false \
    --logger.level=info
EOF

RUN chmod +x /app/start.sh && chown cognichat:cognichat /app/start.sh

# Cambiar a usuario no-root
USER cognichat

# Exponer el puerto
EXPOSE ${PORT}

# Healthcheck mejorado para producción
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Comando de inicio
CMD ["/app/start.sh"]