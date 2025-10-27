# Estructura del Proyecto CogniChat

## Descripción General
CogniChat es un sistema RAG (Retrieval-Augmented Generation) avanzado con análisis cualitativo y procesamiento inteligente de documentos.

## 🗂️ Estructura del Proyecto

```
CogniChat/
├── 📄 Archivos de configuración
│   ├── .env                  # Variables de entorno (no incluido en repo)
│   ├── .env.example         # Ejemplo de configuración
│   ├── .gitignore           # Archivos ignorados por Git
│   ├── LICENSE              # Licencia MIT
│   ├── MANIFEST.in          # Configuración de empaquetado
│   ├── PROJECT_STRUCTURE.md # Este archivo
│   ├── README.md            # Documentación principal
│   ├── pyproject.toml       # Configuración moderna del proyecto
│   └── requirements.txt     # Dependencias de Python

├── 📁 app.py                # Aplicación principal Streamlit

├── 📁 config/               # Configuración del sistema
│   ├── __init__.py
│   └── settings.py          # Configuraciones centralizadas

├── 📁 modules/              # Módulos principales
│   ├── __init__.py
│   ├── alerts.py            # Sistema de alertas
│   ├── chatbot.py           # Interfaz de chat
│   ├── document_processor.py # Procesamiento de documentos
│   ├── document_upload.py   # Carga de archivos
│   ├── qualitative_analysis.py # Análisis cualitativo
│   └── settings.py          # Configuraciones de UI

├── 📁 utils/                # Utilidades del sistema
│   ├── __init__.py
│   ├── database.py          # Gestión de base de datos
│   ├── error_handler.py     # Manejo de errores
│   ├── logger.py            # Sistema de logging
│   ├── metrics.py           # Métricas del sistema
│   ├── ollama_client.py     # Cliente Ollama
│   ├── rag_processor.py     # Procesador RAG
│   ├── traceability.py      # Trazabilidad
│   └── validators.py        # Validadores

├── 📁 scripts/              # Scripts de utilidad
│   ├── __init__.py
│   ├── check_dependencies.py # Verificador de dependencias
│   └── install_requirements.py # Instalador automático

├── 📁 docs/                 # Documentación
│   ├── __init__.py
│   ├── CHANGELOG.md         # Historial de cambios
│   └── OPTIMIZACIONES_RAG.md # Documentación técnica

├── 📁 tests/                # Pruebas del sistema
│   ├── __init__.py
│   └── test_validators.py   # Tests de validadores

├── 📁 data/                 # Datos del sistema
│   ├── cache/               # Cache del sistema
│   │   ├── .gitkeep
│   │   └── rag_cache.json
│   ├── processed/           # Documentos procesados
│   │   └── .gitkeep
│   └── uploads/             # Archivos subidos
│       ├── .gitkeep
│       ├── Colombia.docx
│       └── Japon.docx

└── 📁 logs/                 # Archivos de log
    ├── cognichat_*.log      # Logs de aplicación
    ├── errors_*.log         # Logs de errores
    ├── query_history.json   # Historial de consultas
    └── retrieved_chunks.log # Chunks recuperados
```

## Descripción de Componentes

### Aplicación Principal
- **app.py**: Punto de entrada principal con interfaz Streamlit

### Configuración
- **config/settings.py**: Configuraciones centralizadas del sistema
- **pyproject.toml**: Configuración moderna del proyecto (reemplaza setup.py)

### Módulos Principales
- **modules/**: Contiene la lógica de negocio principal
  - Procesamiento de documentos
  - Sistema de chat
  - Análisis cualitativo
  - Alertas y configuraciones

### Utilidades
- **utils/**: Funciones de soporte y utilidades
  - Gestión de base de datos
  - Logging y manejo de errores
  - Métricas y validaciones
  - Cliente Ollama y procesador RAG

### Scripts
- **scripts/**: Scripts de instalación y mantenimiento
  - Verificación de dependencias
  - Instalación automática

### Datos y Logs
- **data/**: Almacenamiento de datos del sistema
- **logs/**: Archivos de registro y trazabilidad

### Testing y Documentación
- **tests/**: Suite de pruebas unitarias
- **docs/**: Documentación técnica y de usuario

## ✅ Mejoras Implementadas

### 🗑️ Eliminación de Duplicación
- ✅ **Archivos duplicados eliminados**: Removidos `CHANGELOG.md`, `OPTIMIZACIONES_RAG.md`, `check_dependencies.py` e `install_requirements.py` de la raíz
- ✅ **Documentación centralizada**: Todos los archivos de documentación movidos a `docs/`
- ✅ **Scripts organizados**: Scripts de utilidad movidos a `scripts/`
- ✅ **Archivos temporales eliminados**: Removidos todos los directorios `__pycache__`

### 📁 Creación de Archivos Faltantes
- ✅ **`utils/database.py`**: Utilidades para manejo de ChromaDB
- ✅ **`utils/metrics.py`**: Sistema de métricas y monitoreo
- ✅ **`utils/validators.py`**: Validadores centralizados
- ✅ **Estructura de tests**: Directorio `tests/` con ejemplos básicos
- ✅ **Documentación organizada**: Directorio `docs/` con archivos técnicos

### 🔧 Mejor Organización
- ✅ **Separación clara**: Cada tipo de archivo en su directorio correspondiente
- ✅ **Configuración actualizada**: `pyproject.toml` y `MANIFEST.in` actualizados
- ✅ **Referencias corregidas**: Todas las rutas actualizadas en documentación
- ✅ **Scripts funcionales**: Rutas corregidas para encontrar `requirements.txt`

## Beneficios de la Nueva Estructura

1. **Separación Clara de Responsabilidades**: Cada directorio tiene un propósito específico
2. **Mantenibilidad**: Código más fácil de mantener y extender
3. **Estándares Modernos**: Uso de `pyproject.toml` en lugar de `setup.py`
4. **Testing**: Estructura clara para pruebas unitarias
5. **Documentación**: Documentación organizada y accesible
6. **Escalabilidad**: Estructura que permite crecimiento del proyecto

## Comandos Útiles

```bash
# Verificar dependencias
python scripts/check_dependencies.py

# Instalar dependencias
python scripts/install_requirements.py

# Ejecutar aplicación
streamlit run app.py

# Ejecutar tests
pytest tests/

# Construir paquete
python -m build
```