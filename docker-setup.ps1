# ========================================
# COGNICHAT DOCKER SETUP SCRIPT
# ========================================
# Script para gestionar CogniChat en Docker
# Autor: Sistema CogniChat
# Version: 1.0
# ========================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("setup", "start", "stop", "restart", "logs", "status", "clean")]
    [string]$Action
)

# ========================================
# CONFIGURACION
# ========================================
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colores para output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colorMap = @{
        "Red" = "Red"
        "Green" = "Green"
        "Yellow" = "Yellow"
        "Blue" = "Blue"
        "Cyan" = "Cyan"
        "Magenta" = "Magenta"
        "White" = "White"
    }
    
    if ($colorMap.ContainsKey($Color)) {
        Write-Host $Message -ForegroundColor $colorMap[$Color]
    } else {
        Write-Host $Message
    }
}

# ========================================
# FUNCION PARA VERIFICAR DOCKER
# ========================================
function Test-DockerInstallation {
    try {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion) {
            Write-ColorOutput "Docker encontrado: $dockerVersion" "Green"
            return $true
        }
    }
    catch {
        Write-ColorOutput "Docker no esta instalado o no esta en el PATH" "Red"
        Write-ColorOutput "Instala Docker Desktop desde: https://www.docker.com/products/docker-desktop" "Yellow"
        return $false
    }
    return $false
}

# ========================================
# FUNCION PARA VERIFICAR DOCKER COMPOSE
# ========================================
function Test-DockerCompose {
    try {
        $composeVersion = docker compose version 2>$null
        if ($composeVersion) {
            Write-ColorOutput "Docker Compose encontrado: $composeVersion" "Green"
            return $true
        }
    }
    catch {
        Write-ColorOutput "Docker Compose no esta disponible" "Red"
        Write-ColorOutput "Asegurate de tener Docker Desktop actualizado" "Yellow"
        return $false
    }
    return $false
}

# ========================================
# FUNCION PARA CREAR DIRECTORIOS
# ========================================
function New-DockerDirectories {
    $directories = @(
        "data/ollama",
        "data/cognichat",
        "logs"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "Directorio creado: $dir" "Green"
        }
    }
}

# ========================================
# FUNCION PARA CONFIGURAR VARIABLES DE ENTORNO
# ========================================
function Set-EnvironmentVariables {
    Write-ColorOutput "Configurando variables de entorno..." "Yellow"
    
    if (-not (Test-Path ".env.production")) {
        Write-ColorOutput "Archivo .env.production no encontrado" "Red"
        Write-ColorOutput "Creando archivo .env.production basico..." "Yellow"
        return $false
    }
    
    Write-ColorOutput "Archivo .env.production encontrado" "Green"
    return $true
}

# ========================================
# FUNCION PARA CONSTRUIR IMAGENES
# ========================================
function Build-DockerImages {
    Write-ColorOutput "Construyendo imagenes de Docker..." "Yellow"
    
    try {
        Write-ColorOutput "Construyendo imagen de CogniChat..." "Blue"
        docker compose build --no-cache
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Imagenes construidas exitosamente" "Green"
            return $true
        } else {
            Write-ColorOutput "Error al construir las imagenes" "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "Error durante la construccion: $($_.Exception.Message)" "Red"
        return $false
    }
}

# ========================================
# FUNCION PARA INICIAR SERVICIOS
# ========================================
function Start-Services {
    Write-ColorOutput "Iniciando servicios de CogniChat..." "Yellow"
    
    try {
        docker compose up -d
        
        Start-Sleep -Seconds 10
        
        $services = docker compose ps --services
        Write-ColorOutput "Servicios iniciados exitosamente" "Green"
        Write-ColorOutput ""
        Write-ColorOutput "Accede a CogniChat en: http://localhost:8501" "Cyan"
        Write-ColorOutput "Ollama API disponible en: http://localhost:11434" "Cyan"
        return $true
    }
    catch {
        Write-ColorOutput "Error al iniciar servicios: $($_.Exception.Message)" "Red"
        return $false
    }
}

# ========================================
# FUNCION PARA DETENER SERVICIOS
# ========================================
function Stop-Services {
    Write-ColorOutput "Deteniendo servicios de CogniChat..." "Yellow"
    
    try {
        docker compose down
        Write-ColorOutput "Servicios detenidos exitosamente" "Green"
        return $true
    }
    catch {
        Write-ColorOutput "Error al detener servicios: $($_.Exception.Message)" "Red"
        return $false
    }
}

# ========================================
# FUNCION PARA MOSTRAR LOGS
# ========================================
function Show-Logs {
    try {
        Write-ColorOutput "Mostrando logs de los servicios..." "Yellow"
        docker compose logs --tail=50 -f
    }
    catch {
        Write-ColorOutput "Error al mostrar logs: $($_.Exception.Message)" "Red"
    }
}

# ========================================
# FUNCION PARA MOSTRAR ESTADO
# ========================================
function Show-Status {
    try {
        Write-ColorOutput "Estado de los servicios:" "Yellow"
        Write-ColorOutput ""
        docker compose ps
        Write-ColorOutput ""
        Write-ColorOutput "Uso de recursos:" "Yellow"
        docker stats --no-stream
    }
    catch {
        Write-ColorOutput "Error al obtener estado: $($_.Exception.Message)" "Red"
    }
}

# ========================================
# FUNCION PARA LIMPIAR RECURSOS
# ========================================
function Clear-Resources {
    Write-ColorOutput "Limpiando recursos de Docker..." "Yellow"
    
    Write-ColorOutput "Eliminando todos los recursos (incluyendo volumenes)..." "Red"
    docker compose down -v --remove-orphans
    
    Write-ColorOutput "Eliminando contenedores e imagenes..." "Blue"
    docker system prune -f
    docker volume prune -f
    
    Write-ColorOutput "Limpieza completada" "Green"
}

# ========================================
# FUNCION PRINCIPAL
# ========================================
function Main {
    Write-ColorOutput "========================================" "Cyan"
    Write-ColorOutput "    COGNICHAT DOCKER MANAGER" "Cyan"
    Write-ColorOutput "========================================" "Cyan"
    Write-ColorOutput ""
    
    # Verificar Docker
    if (-not (Test-DockerInstallation)) {
        exit 1
    }
    
    if (-not (Test-DockerCompose)) {
        exit 1
    }
    
    # Ejecutar accion
    switch ($Action.ToLower()) {
        "setup" {
            Write-ColorOutput "Configurando CogniChat para Docker..." "Cyan"
            New-DockerDirectories
            if (Set-EnvironmentVariables) {
                if (Build-DockerImages) {
                    Write-ColorOutput ""
                    Write-ColorOutput "Configuracion completada exitosamente" "Green"
                    Write-ColorOutput "Ejecuta './docker-setup.ps1 start' para iniciar los servicios" "Cyan"
                }
            }
        }
        
        "start" {
            Write-ColorOutput "Iniciando CogniChat..." "Cyan"
            New-DockerDirectories
            Start-Services
        }
        
        "stop" {
            Write-ColorOutput "Deteniendo CogniChat..." "Cyan"
            Stop-Services
        }
        
        "restart" {
            Write-ColorOutput "Reiniciando CogniChat..." "Cyan"
            Stop-Services
            Start-Sleep -Seconds 5
            Start-Services
        }
        
        "logs" {
            Show-Logs
        }
        
        "status" {
            Show-Status
        }
        
        "clean" {
            Write-ColorOutput "Limpiando recursos..." "Cyan"
            Clear-Resources
        }
        
        default {
            Write-ColorOutput "Accion no valida: $Action" "Red"
            Write-ColorOutput "Acciones disponibles: setup, start, stop, restart, logs, status, clean" "Yellow"
        }
    }
}

# ========================================
# EJECUCION DEL SCRIPT
# ========================================
try {
    Main
}
catch {
    Write-ColorOutput "Error critico: $($_.Exception.Message)" "Red"
    exit 1
}