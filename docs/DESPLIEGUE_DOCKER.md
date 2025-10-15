# 🐳 Guía de Despliegue con Docker - CogniChat

Esta guía te ayudará a desplegar CogniChat usando Docker tanto en Railway como en desarrollo local.

## 📋 Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Despliegue en Railway](#despliegue-en-railway)
3. [Desarrollo Local con Docker](#desarrollo-local-con-docker)
4. [Variables de Entorno](#variables-de-entorno)
5. [Configuración de Ollama](#configuración-de-ollama)
6. [Solución de Problemas](#solución-de-problemas)

## 🔧 Requisitos Previos

### Para Railway:
- Cuenta en [Railway.app](https://railway.app)
- Repositorio de GitHub con el código de CogniChat
- Servicio Ollama externo (Railway Community o servidor dedicado)

### Para Desarrollo Local:
- Docker Desktop instalado
- Docker Compose
- Al menos 8GB de RAM disponible
- 10GB de espacio en disco

## 🚀 Despliegue en Railway

### Opción 1: Despliegue Directo desde GitHub (Recomendado)

1. **Conectar Repositorio:**
   ```bash
   # Asegúrate de que todos los archivos estén en tu repositorio
   git add .
   git commit -m "feat: configuración Docker para Railway"
   git push origin main
   ```

2. **Crear Proyecto en Railway:**
   - Ve a [Railway.app](https://railway.app)
   - Haz clic en "New Project"
   - Selecciona "Deploy from GitHub repo"
   - Elige tu repositorio de CogniChat

3. **Configurar Variables de Entorno:**
   En el dashboard de Railway, ve a Variables y configura:
   ```env
   # Configuración de Ollama (OBLIGATORIO)
   OLLAMA_BASE_URL=https://tu-ollama-service.railway.app
   OLLAMA_TIMEOUT=120
   
   # Modelos por defecto
   DEFAULT_LLM_MODEL=deepseek-r1:7b
   DEFAULT_EMBEDDING_MODEL=nomic-embed-text:latest
   
   # Configuración RAG
   CHUNK_SIZE=2000
   CHUNK_OVERLAP=300
   MAX_RETRIEVAL_DOCS=15
   SIMILARITY_THRESHOLD=0.6
   
   # Configuración de respuestas
   MAX_RESPONSE_TOKENS=3000
   
   # Configuración de archivos
   MAX_FILE_SIZE_MB=100
   
   # Configuración de logs
   LOG_LEVEL=INFO
   
   # Configuración de análisis
   ENABLE_ADVANCED_ANALYSIS=true
   MAX_TOPICS=10
   MIN_CLUSTER_SIZE=5
   DEFAULT_CLUSTERING_METHOD=kmeans
   
   # Configuración de UI
   DEFAULT_LANGUAGE=es
   THEME=light
   MAX_UPLOAD_SIZE_MB=200
   ```

4. **Desplegar:**
   - Railway detectará automáticamente el `Dockerfile`
   - El despliegue comenzará automáticamente
   - Espera 5-10 minutos para el primer despliegue

### Opción 2: Despliegue con Railway CLI

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Autenticarse
railway login

# Crear proyecto
railway init

# Configurar variables de entorno
railway variables set OLLAMA_BASE_URL=https://tu-ollama-service.railway.app
railway variables set DEFAULT_LLM_MODEL=deepseek-r1:7b
# ... (agregar todas las variables necesarias)

# Desplegar
railway up
```

## 🏠 Desarrollo Local con Docker

### Opción 1: Docker Compose (Recomendado)

```bash
# Clonar el repositorio
git clone <tu-repositorio>
cd CogniChat

# Construir y ejecutar todos los servicios
docker-compose up --build

# Para ejecutar en segundo plano
docker-compose up -d --build

# Ver logs
docker-compose logs -f cognichat
docker-compose logs -f ollama

# Detener servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v
```

### Opción 2: Docker Manual

```bash
# Construir la imagen
docker build -t cognichat:latest .

# Ejecutar el contenedor
docker run -d \
  --name cognichat \
  -p 8501:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e DEFAULT_LLM_MODEL=deepseek-r1:7b \
  -e DEFAULT_EMBEDDING_MODEL=nomic-embed-text:latest \
  -v cognichat_data:/app/data \
  cognichat:latest

# Ver logs
docker logs -f cognichat

# Detener contenedor
docker stop cognichat
docker rm cognichat
```

## 🔧 Variables de Entorno

### Variables Obligatorias

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | URL del servicio Ollama | `http://localhost:11434` |
| `DEFAULT_LLM_MODEL` | Modelo LLM por defecto | `deepseek-r1:7b` |
| `DEFAULT_EMBEDDING_MODEL` | Modelo de embeddings | `nomic-embed-text:latest` |

### Variables Opcionales

Consulta el archivo `.env.example` para ver todas las variables disponibles.

## 🤖 Configuración de Ollama

### Para Railway:

1. **Crear Servicio Ollama Separado:**
   ```bash
   # Crear nuevo servicio en Railway
   railway init ollama-service
   
   # Usar imagen de Ollama
   # En Railway, selecciona "Deploy Docker Image"
   # Imagen: ollama/ollama:latest
   # Puerto: 11434
   ```

2. **Variables de Entorno para Ollama:**
   ```env
   OLLAMA_ORIGINS=*
   OLLAMA_HOST=0.0.0.0:11434
   ```

3. **Inicializar Modelos:**
   ```bash
   # Conectarse al contenedor de Ollama en Railway
   railway shell
   
   # Descargar modelos
   ollama pull deepseek-r1:7b
   ollama pull nomic-embed-text:latest
   ```

### Para Desarrollo Local:

El `docker-compose.yml` incluye:
- Servicio Ollama automático
- Inicialización automática de modelos
- Configuración de red interna

## 🔍 Verificación del Despliegue

### Railway:
1. Ve al dashboard de Railway
2. Verifica que el servicio esté "Running"
3. Haz clic en el dominio generado
4. Deberías ver la interfaz de CogniChat

### Local:
```bash
# Verificar que los contenedores estén ejecutándose
docker-compose ps

# Verificar salud de los servicios
curl http://localhost:8501/_stcore/health
curl http://localhost:11434/api/tags

# Acceder a la aplicación
# Abre http://localhost:8501 en tu navegador
```

## 🐛 Solución de Problemas

### Error: "Ollama no disponible"

```bash
# Verificar conectividad con Ollama
docker-compose exec cognichat curl -f http://ollama:11434/api/tags

# Verificar logs de Ollama
docker-compose logs ollama
```

### Error: "Modelo no encontrado"

```bash
# Conectarse al contenedor de Ollama
docker-compose exec ollama bash

# Listar modelos disponibles
ollama list

# Descargar modelo faltante
ollama pull deepseek-r1:7b
```

### Error de memoria en Railway

```bash
# Reducir configuraciones en variables de entorno
CHUNK_SIZE=1000
MAX_RETRIEVAL_DOCS=10
MAX_RESPONSE_TOKENS=2000
```

### Logs y Debugging

```bash
# Ver logs detallados
docker-compose logs -f --tail=100 cognichat

# Ejecutar en modo debug
docker-compose up --build
# (sin -d para ver logs en tiempo real)

# Acceder al contenedor para debugging
docker-compose exec cognichat bash
```

## 📊 Monitoreo

### Railway:
- Dashboard integrado con métricas
- Logs en tiempo real
- Alertas automáticas

### Local:
```bash
# Monitorear recursos
docker stats

# Ver logs en tiempo real
docker-compose logs -f

# Verificar salud de servicios
docker-compose ps
```

## 🔒 Consideraciones de Seguridad

1. **Variables de Entorno:**
   - Nunca commits archivos `.env` con datos sensibles
   - Usa Railway Variables para datos sensibles

2. **Ollama:**
   - Configura `OLLAMA_ORIGINS` apropiadamente
   - Considera autenticación para producción

3. **Límites:**
   - Configura `MAX_FILE_SIZE_MB` apropiadamente
   - Implementa rate limiting con `RATE_LIMIT_REQUESTS`

## 💰 Costos Estimados

### Railway:
- **CogniChat:** ~$5-10/mes (512MB RAM)
- **Ollama:** ~$20-40/mes (2-4GB RAM)
- **Total:** ~$25-50/mes

### Desarrollo Local:
- Gratis (usa recursos locales)
- Requiere ~8GB RAM y 10GB disco

## 🚀 Próximos Pasos

1. **Optimización:**
   - Implementar cache Redis
   - Configurar CDN para archivos estáticos
   - Optimizar modelos de Ollama

2. **Monitoreo:**
   - Configurar alertas
   - Implementar métricas personalizadas
   - Logs estructurados

3. **Escalabilidad:**
   - Load balancing
   - Base de datos externa
   - Microservicios

---

## 📞 Soporte

Si encuentras problemas:

1. Revisa los logs: `docker-compose logs -f`
2. Verifica las variables de entorno
3. Consulta la documentación de Railway
4. Revisa los issues del repositorio

¡Feliz despliegue! 🎉