# ========================================
# DOCKERFILE PARA COGNICHAT - PRODUCCIÓN
# ========================================
# Imagen base optimizada para Python y aplicaciones científicas
FROM python:3.11-slim

# ========================================
# INFORMACIÓN DEL CONTENEDOR
# ========================================
LABEL maintainer="CogniChat Team"
LABEL version="1.0.0"
LABEL description="Sistema de Análisis Cualitativo con IA - Producción"

# ========================================
# CONFIGURACIÓN DE VARIABLES DE ENTORNO
# ========================================
# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Variables de entorno para Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ========================================
# INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA
# ========================================
# Actualizar sistema e instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    # Herramientas de compilación necesarias para algunas librerías Python
    build-essential \
    # Herramientas de red para debugging
    curl \
    wget \
    # Librerías para procesamiento de texto y NLP
    libffi-dev \
    libssl-dev \
    # Limpieza de cache para reducir tamaño de imagen
    && rm -rf /var/lib/apt/lists/*

# ========================================
# CONFIGURACIÓN DE DIRECTORIO DE TRABAJO
# ========================================
# Crear y establecer directorio de trabajo
WORKDIR /app

# ========================================
# INSTALACIÓN DE DEPENDENCIAS PYTHON
# ========================================
# Copiar archivos de dependencias primero (para aprovechar cache de Docker)
COPY requirements.txt pyproject.toml ./

# Actualizar pip e instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ========================================
# COPIA DE CÓDIGO FUENTE
# ========================================
# Copiar todo el código fuente de la aplicación
COPY . .

# ========================================
# CONFIGURACIÓN DE DIRECTORIOS DE DATOS
# ========================================
# Crear directorios necesarios para la aplicación
RUN mkdir -p /app/data/cache \
             /app/data/processed \
             /app/data/uploads \
             /app/data/chat_history \
             /app/data/temp_exports \
             /app/data/nltk_data \
             /app/logs

# ========================================
# CONFIGURACIÓN DE PERMISOS
# ========================================
# Crear usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash cognichat && \
    chown -R cognichat:cognichat /app && \
    chmod -R 775 /app/data && \
    chmod -R 775 /app/logs && \
    chmod 775 /app/data/chat_history && \
    chmod 775 /app/data/processed && \
    chmod 775 /app/data/temp_exports && \
    chmod 775 /app/data/uploads

# Cambiar a usuario no-root
USER cognichat

# ========================================
# DESCARGA DE RECURSOS NLP
# ========================================
# Descargar recursos de NLTK necesarios
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/data/nltk_data'); nltk.download('stopwords', download_dir='/app/data/nltk_data'); nltk.download('vader_lexicon', download_dir='/app/data/nltk_data')"

# ========================================
# CONFIGURACIÓN DE PUERTOS
# ========================================
# Exponer puerto de Streamlit
EXPOSE 8501

# ========================================
# CONFIGURACIÓN DE VOLÚMENES
# ========================================
# Definir volúmenes para persistencia de datos
VOLUME ["/app/data", "/app/logs"]

# ========================================
# COMANDO DE INICIO
# ========================================
# Comando para iniciar la aplicación
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]