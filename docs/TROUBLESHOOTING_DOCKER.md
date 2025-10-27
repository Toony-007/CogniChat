# 🔧 GUÍA DE TROUBLESHOOTING - DOCKER

## 🚨 Problemas Comunes y Soluciones

### 1. 📁 Errores de Permisos
```bash
# Síntoma: Permission denied al escribir archivos
# Solución:
docker exec cognichat-app chmod -R 755 /app/data
docker exec cognichat-app chown -R cognichat:cognichat /app/data
```

### 2. 🌐 Problemas de Conectividad
```bash
# Síntoma: Connection refused entre contenedores
# Verificar:
docker network ls
docker exec cognichat-app ping ollama
```

### 3. 💾 Volúmenes No Montados
```bash
# Síntoma: Datos no persisten
# Verificar:
docker inspect cognichat-app | grep -A 10 "Mounts"
```

### 4. 🔧 Variables de Entorno
```bash
# Verificar variables críticas:
docker exec cognichat-app env | grep OLLAMA
docker exec cognichat-app env | grep PYTHONPATH
```

### 5. 📦 Modelos de Ollama Faltantes
```bash
# Verificar modelos descargados:
docker exec cognichat-ollama ollama list
```

## 🛡️ Comandos de Diagnóstico Rápido

### Estado General
```powershell
# Verificar estado de contenedores
docker compose ps

# Verificar logs recientes
docker logs cognichat-app --tail 20
docker logs cognichat-ollama --tail 20
```

### Verificación de Permisos
```bash
# Dentro del contenedor cognichat-app
docker exec cognichat-app ls -la /app/data/
docker exec cognichat-app whoami
docker exec cognichat-app id
```

### Verificación de Conectividad
```bash
# Probar conexión a Ollama
docker exec cognichat-app curl -f http://ollama:11434/api/tags
```

### Verificación de Volúmenes
```powershell
# En el host (Windows)
Test-Path "docker-data\cognichat"
Test-Path "docker-data\ollama"
Get-ChildItem "docker-data" -Recurse | Measure-Object -Property Length -Sum
```

## 🚀 Soluciones Rápidas

### Reinicio Completo
```powershell
docker compose down
docker compose up -d
```

### Reconstrucción Incremental
```powershell
docker compose build cognichat
docker compose up -d cognichat
```

### Reconstrucción Completa (Solo si es necesario)
```powershell
docker compose build --no-cache
docker compose up -d
```

### Limpieza de Recursos
```powershell
# Solo si hay problemas graves
docker system prune -f
docker volume prune -f
```

## 📋 Checklist de Verificación Post-Despliegue

- [ ] Contenedores están "Up" y "healthy"
- [ ] Ollama responde en http://localhost:11434
- [ ] CogniChat accesible en http://localhost:8501
- [ ] Modelos descargados (deepseek-r1:7b, nomic-embed-text)
- [ ] Permisos correctos en /app/data
- [ ] Variables de entorno configuradas
- [ ] Volúmenes montados correctamente
- [ ] Logs sin errores críticos

## 🆘 Comandos de Emergencia

### Si todo falla:
```powershell
# Parar todo
docker compose down -v

# Limpiar (CUIDADO: Borra volúmenes)
docker system prune -a -f

# Reconstruir desde cero
docker compose build --no-cache
docker compose up -d
```

### Backup de Datos Importantes:
```powershell
# Antes de limpiezas drásticas
Copy-Item "docker-data" "docker-data-backup" -Recurse
```