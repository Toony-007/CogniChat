# ========================================
# SCRIPT PARA DESCARGAR MODELOS DE OLLAMA
# ========================================
# Este script descarga automáticamente los modelos por defecto
# después de que los contenedores estén ejecutándose

param(
    [switch]$Force = $false
)

# Función para mostrar mensajes con colores
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colorMap = @{
        "Red" = [ConsoleColor]::Red
        "Green" = [ConsoleColor]::Green
        "Yellow" = [ConsoleColor]::Yellow
        "Blue" = [ConsoleColor]::Blue
        "Cyan" = [ConsoleColor]::Cyan
        "Magenta" = [ConsoleColor]::Magenta
        "White" = [ConsoleColor]::White
    }
    
    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

# Verificar que Docker esté ejecutándose
function Test-DockerRunning {
    try {
        docker ps | Out-Null
        return $true
    }
    catch {
        Write-ColorOutput "Docker no está ejecutándose o no está disponible" "Red"
        return $false
    }
}

# Verificar que el contenedor de Ollama esté ejecutándose
function Test-OllamaContainer {
    try {
        $containers = docker ps --format "table {{.Names}}" | Select-String "ollama"
        if ($containers) {
            Write-ColorOutput "Contenedor Ollama encontrado: $containers" "Green"
            return $true
        } else {
            Write-ColorOutput "Contenedor Ollama no está ejecutándose" "Red"
            return $false
        }
    }
    catch {
        Write-ColorOutput "Error al verificar contenedores: $($_.Exception.Message)" "Red"
        return $false
    }
}

# Descargar un modelo específico
function Download-Model {
    param(
        [string]$ModelName,
        [string]$Description
    )
    
    Write-ColorOutput "📥 Descargando $Description ($ModelName)..." "Cyan"
    
    try {
        # Verificar si el modelo ya existe (si no se fuerza la descarga)
        if (-not $Force) {
            $existingModels = docker exec ollama ollama list 2>$null
            if ($existingModels -match $ModelName) {
                Write-ColorOutput "✅ Modelo $ModelName ya está descargado" "Green"
                return $true
            }
        }
        
        # Descargar el modelo
        $result = docker exec ollama ollama pull $ModelName
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ $Description descargado correctamente" "Green"
            return $true
        } else {
            Write-ColorOutput "❌ Error al descargar $Description" "Red"
            Write-ColorOutput "Salida: $result" "Yellow"
            return $false
        }
    }
    catch {
        Write-ColorOutput "❌ Excepción al descargar $Description : $($_.Exception.Message)" "Red"
        return $false
    }
}

# Función principal
function Main {
    Write-ColorOutput "========================================" "Cyan"
    Write-ColorOutput "    DESCARGA DE MODELOS COGNICHAT" "Cyan"
    Write-ColorOutput "========================================" "Cyan"
    Write-ColorOutput ""
    
    # Verificaciones previas
    if (-not (Test-DockerRunning)) {
        Write-ColorOutput "Inicia Docker Desktop y ejecuta 'docker compose up -d' primero" "Yellow"
        exit 1
    }
    
    if (-not (Test-OllamaContainer)) {
        Write-ColorOutput "Inicia los contenedores con 'docker compose up -d' primero" "Yellow"
        exit 1
    }
    
    # Esperar a que Ollama esté completamente listo
    Write-ColorOutput "⏳ Esperando a que Ollama esté listo..." "Yellow"
    Start-Sleep -Seconds 10
    
    # Lista de modelos a descargar
    $models = @(
        @{
            Name = "deepseek-r1:7b"
            Description = "DeepSeek R1 7B (Modelo LLM principal)"
        },
        @{
            Name = "nomic-embed-text:latest"
            Description = "Nomic Embed Text (Modelo de embeddings)"
        }
    )
    
    $successCount = 0
    $totalModels = $models.Count
    
    # Descargar cada modelo
    foreach ($model in $models) {
        if (Download-Model -ModelName $model.Name -Description $model.Description) {
            $successCount++
        }
        Write-ColorOutput "" # Línea en blanco
    }
    
    # Resumen final
    Write-ColorOutput "========================================" "Cyan"
    Write-ColorOutput "RESUMEN DE DESCARGA:" "Cyan"
    Write-ColorOutput "Modelos descargados: $successCount/$totalModels" "$(if($successCount -eq $totalModels){'Green'}else{'Yellow'})"
    
    if ($successCount -eq $totalModels) {
        Write-ColorOutput "🎉 ¡Todos los modelos descargados correctamente!" "Green"
        Write-ColorOutput "CogniChat está listo para usar" "Green"
    } else {
        Write-ColorOutput "⚠️  Algunos modelos no se pudieron descargar" "Yellow"
        Write-ColorOutput "Puedes intentar descargarlos manualmente:" "Yellow"
        Write-ColorOutput "docker exec ollama ollama pull deepseek-r1:7b" "Cyan"
        Write-ColorOutput "docker exec ollama ollama pull nomic-embed-text:latest" "Cyan"
    }
    
    Write-ColorOutput "========================================" "Cyan"
}

# Ejecutar script
try {
    Main
}
catch {
    Write-ColorOutput "Error crítico: $($_.Exception.Message)" "Red"
    exit 1
}