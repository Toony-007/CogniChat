# 🐳 Guía de Despliegue con Docker - CogniChat

## 📋 Tabla de Contenidos
- [Requisitos Previos](#requisitos-previos)
- [Configuración Inicial](#configuración-inicial)
- [Despliegue Rápido](#despliegue-rápido)
- [Configuración Avanzada](#configuración-avanzada)
- [Monitoreo y Logs](#monitoreo-y-logs)
- [Solución de Problemas](#solución-de-problemas)
- [Mantenimiento](#mantenimiento)

## 🔧 Requisitos Previos

### Software Necesario
- **Docker Desktop** (versión 4.0 o superior)
- **Windows 10/11** con WSL2 habilitado
- **PowerShell 5.1** o superior
- **Mínimo 8GB RAM** (recomendado 16GB)
- **20GB espacio libre** en disco

### Verificación de Requisitos
```powershell
# Verificar Docker
docker --version

# Verificar Docker Compose
docker compose version

# Verificar PowerShell
$PSVersionTable.PSVersion
```

## ⚙️ Configuración Inicial

### 1. Preparación del Entorno
```powershell
# Clonar o navegar al directorio del proyecto
cd "C:\Users\Antony Salcedo\Documents\Tesis\Desarrollo\CogniChat"

# Verificar archivos necesarios
ls *.yml, *.env*, Dockerfile
```

### 2. Configuración de Variables de Entorno
El archivo `.env.production` ya está configurado con valores optimizados para producción:

```bash
# Configuración principal
OLLAMA_BASE_URL=http://ollama:11434
DEFAULT_LLM_MODEL=deepseek-r1:7b
DEFAULT_EMBEDDING_MODEL=nomic-embed-text:latest

# Configuración de rendimiento
CHUNK_SIZE=1500
MAX_RESPONSE_TOKENS=2500
LOG_LEVEL=WARNING
```

### 3. Estructura de Directorios
El sistema creará automáticamente:
```
docker-data/
├── ollama/          # Modelos de IA descargados
├── cognichat/       # Base de datos y cache
└── logs/            # Logs de la aplicación
uploads/             # Archivos subidos por usuarios
```

## 🚀 Despliegue Rápido

### Opción 1: Script Automatizado (Recomendado)
```powershell
# Configuración completa automática
.\docker-setup.ps1 setup

# Iniciar servicios
.\docker-setup.ps1 start
```

### Opción 2: Comandos Manuales
```powershell
# 1. Crear directorios necesarios
mkdir docker-data\ollama, docker-data\cognichat, docker-data\logs, uploads

# 2. Construir imágenes
docker compose build

# 3. Iniciar servicios
docker compose up -d
```

## 🔗 Acceso a la Aplicación

Una vez iniciado, CogniChat estará disponible en:
- **Aplicación Principal**: http://localhost:8501
- **API Ollama**: http://localhost:11434

### Verificación de Estado
```powershell
# Ver estado de contenedores
docker compose ps

# Ver logs en tiempo real
docker compose logs -f

# Verificar salud de servicios
docker compose exec cognichat curl -f http://localhost:8501/_stcore/health
```

## 🔧 Configuración Avanzada

### Personalización de Recursos
Editar `docker-compose.yml` para ajustar recursos:

```yaml
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 16G  # Aumentar para modelos más grandes
        reservations:
          memory: 8G
```

### Configuración de Red Personalizada
```yaml
networks:
  cognichat-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
```

### Volúmenes Personalizados
```yaml
volumes:
  ollama_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: D:\CogniChat\ollama  # Cambiar ruta si es necesario
```

## 📊 Monitoreo y Logs

### Comandos de Monitoreo
```powershell
# Estado general
.\docker-setup.ps1 status

# Logs específicos por servicio
docker compose logs cognichat
docker compose logs ollama

# Logs con filtro de tiempo
docker compose logs --since="1h" cognichat

# Seguimiento de logs en tiempo real
docker compose logs -f --tail=100
```

### Métricas de Rendimiento
```powershell
# Uso de recursos por contenedor
docker stats

# Información detallada de contenedores
docker compose exec cognichat ps aux

# Espacio usado por volúmenes
docker system df
```

## 🛠️ Comandos de Gestión

### Gestión de Servicios
```powershell
# Iniciar servicios
.\docker-setup.ps1 start

# Detener servicios
.\docker-setup.ps1 stop

# Reiniciar servicios
.\docker-setup.ps1 restart

# Ver logs
.\docker-setup.ps1 logs

# Limpiar recursos
.\docker-setup.ps1 clean
```

### Comandos Docker Directos
```powershell
# Reconstruir sin cache
docker compose build --no-cache

# Iniciar en modo interactivo
docker compose up

# Escalar servicios (si es necesario)
docker compose up --scale cognichat=2

# Actualizar servicios
docker compose pull && docker compose up -d
```

## 🔍 Solución de Problemas

### Problemas Comunes

#### 1. Error de Conexión con Ollama
```powershell
# Verificar que Ollama está ejecutándose
docker compose ps ollama

# Reiniciar Ollama
docker compose restart ollama

# Verificar logs de Ollama
docker compose logs ollama
```

#### 2. Puerto 8501 en Uso
```powershell
# Encontrar proceso usando el puerto
netstat -ano | findstr :8501

# Cambiar puerto en docker-compose.yml
ports:
  - "8502:8501"  # Usar puerto diferente
```

#### 3. Problemas de Memoria
```powershell
# Verificar uso de memoria
docker stats --no-stream

# Limpiar cache de Docker
docker system prune -f

# Reiniciar Docker Desktop
```

#### 4. Errores de Permisos
```powershell
# Ejecutar PowerShell como Administrador
# Verificar que Docker Desktop está ejecutándose
# Reiniciar servicio de Docker
```

### Logs de Diagnóstico
```powershell
# Logs completos del sistema
docker compose logs > cognichat-logs.txt

# Información del sistema Docker
docker system info > docker-info.txt

# Estado de todos los contenedores
docker ps -a > containers-status.txt
```

## 🔄 Mantenimiento

### Actualizaciones
```powershell
# Actualizar imágenes base
docker compose pull

# Reconstruir con nuevos cambios
docker compose build --no-cache

# Aplicar actualizaciones
docker compose up -d
```

### Respaldos
```powershell
# Respaldar volúmenes de datos
docker run --rm -v cognichat_cognichat_data:/data -v ${PWD}:/backup alpine tar czf /backup/cognichat-backup.tar.gz -C /data .

# Respaldar configuración
Copy-Item .env.production, docker-compose.yml backup/
```

### Limpieza Periódica
```powershell
# Limpiar imágenes no utilizadas
docker image prune -f

# Limpiar volúmenes huérfanos
docker volume prune -f

# Limpiar todo el sistema (cuidado!)
.\docker-setup.ps1 clean -Force
```

## 📈 Optimización de Rendimiento

### Configuración de Producción
1. **Aumentar memoria para Ollama** (mínimo 8GB)
2. **Usar SSD** para volúmenes de datos
3. **Configurar límites de recursos** apropiados
4. **Habilitar logging estructurado**

### Monitoreo Continuo
```powershell
# Script de monitoreo automático
while ($true) {
    docker compose ps
    docker stats --no-stream
    Start-Sleep 300  # Cada 5 minutos
}
```

## 🆘 Soporte y Contacto

### Recursos Adicionales
- **Documentación Docker**: https://docs.docker.com/
- **Documentación Ollama**: https://ollama.ai/docs
- **Streamlit Docs**: https://docs.streamlit.io/

### Comandos de Emergencia
```powershell
# Detener todo y limpiar
docker compose down -v
docker system prune -af

# Reinicio completo
.\docker-setup.ps1 clean -Force
.\docker-setup.ps1 setup
.\docker-setup.ps1 start
```

---

**Nota**: Esta guía asume un entorno Windows con Docker Desktop. Para otros sistemas operativos, adaptar los comandos según corresponda.