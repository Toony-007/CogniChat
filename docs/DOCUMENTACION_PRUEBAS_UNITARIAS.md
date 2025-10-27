# Documentación de Pruebas Unitarias - CogniChat

## Índice
1. [Introducción](#introducción)
2. [Arquitectura de Pruebas Unitarias](#arquitectura-de-pruebas-unitarias)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Configuración y Setup](#configuración-y-setup)
5. [Tipos de Pruebas](#tipos-de-pruebas)
6. [Fixtures y Datos de Prueba](#fixtures-y-datos-de-prueba)
7. [Ejecución de Pruebas](#ejecución-de-pruebas)
8. [Cobertura de Código](#cobertura-de-código)
9. [Mejores Prácticas](#mejores-prácticas)
10. [Troubleshooting](#troubleshooting)

## Introducción

Las pruebas unitarias en CogniChat están diseñadas para validar componentes individuales del sistema de manera aislada. A diferencia de las pruebas de integración existentes en el directorio `tests/`, estas pruebas se enfocan en la lógica de negocio específica de cada módulo sin dependencias externas.

### Objetivos de las Pruebas Unitarias

- **Aislamiento**: Probar cada componente de forma independiente
- **Velocidad**: Ejecución rápida sin dependencias externas
- **Precisión**: Identificar exactamente qué componente falla
- **Mantenibilidad**: Facilitar refactoring y cambios de código
- **Documentación**: Servir como especificación ejecutable

## Arquitectura de Pruebas Unitarias

### Principios de Diseño

1. **Separación de Responsabilidades**: Cada archivo de prueba se enfoca en un módulo específico
2. **Mocking Extensivo**: Uso de mocks para aislar dependencias
3. **Fixtures Reutilizables**: Datos de prueba centralizados y reutilizables
4. **Configuración Centralizada**: Setup común en `conftest.py`

### Diferencias con Pruebas de Integración

| Aspecto | Pruebas Unitarias | Pruebas de Integración |
|---------|-------------------|------------------------|
| **Ubicación** | `unit_tests/` | `tests/` |
| **Enfoque** | Componentes aislados | Interacción entre módulos |
| **Dependencias** | Mockeadas | Reales (cuando es posible) |
| **Velocidad** | Muy rápida | Moderada |
| **Complejidad** | Baja | Media-Alta |

## Estructura del Proyecto

```
unit_tests/
├── __init__.py                     # Configuración del paquete
├── conftest.py                     # Configuración pytest y fixtures globales
├── run_unit_tests.py              # Script de ejecución
├── modules/                       # Pruebas de módulos principales
│   ├── __init__.py
│   ├── test_chatbot_unit.py       # Pruebas unitarias del chatbot
│   ├── test_document_processor_unit.py
│   ├── test_qualitative_analysis_unit.py
│   └── test_settings_unit.py
├── utils/                         # Pruebas de utilidades
│   ├── __init__.py
│   ├── test_rag_processor_unit.py # Pruebas del procesador RAG
│   ├── test_ollama_client_unit.py
│   ├── test_validators_unit.py
│   └── test_logger_unit.py
├── config/                        # Pruebas de configuración
│   ├── __init__.py
│   └── test_settings_unit.py
├── fixtures/                      # Datos de prueba y fixtures
│   ├── __init__.py
│   ├── test_data.py              # Datos de prueba centralizados
│   └── mock_objects.py           # Objetos mock reutilizables
└── reports/                       # Reportes generados
    ├── coverage_html/            # Reporte HTML de cobertura
    ├── coverage.json             # Datos de cobertura en JSON
    ├── junit_report.xml          # Reporte JUnit
    └── test_results.json         # Resultados detallados
```

## Configuración y Setup

### Dependencias Requeridas

```bash
# Instalar dependencias de pruebas
pip install pytest pytest-cov pytest-mock pytest-json-report
```

### Configuración en `conftest.py`

El archivo `conftest.py` contiene:

- **Fixtures globales**: Configuraciones reutilizables
- **Mocks comunes**: Streamlit, OllamaClient, etc.
- **Configuración de pytest**: Marcadores y opciones
- **Setup del entorno**: Paths y configuraciones

### Fixtures Principales

```python
@pytest.fixture(scope="function")
def mock_streamlit():
    """Mock completo de Streamlit para pruebas unitarias."""
    
@pytest.fixture(scope="function") 
def mock_ollama_client():
    """Mock del cliente Ollama con respuestas predefinidas."""
    
@pytest.fixture(scope="function")
def sample_pdf_content():
    """Contenido de PDF de ejemplo para pruebas."""
```

## Tipos de Pruebas

### 1. Pruebas de Módulos (`modules/`)

**Objetivo**: Validar la lógica de negocio de cada módulo principal.

#### `test_chatbot_unit.py`
- Inicialización del chatbot
- Lógica de selección de modelos
- Manejo de mensajes
- Integración con contexto RAG
- Exportación de conversaciones

#### `test_document_processor_unit.py`
- Procesamiento de diferentes tipos de archivo
- Validación de formatos
- Extracción de metadatos
- Manejo de errores

#### `test_qualitative_analysis_unit.py`
- Extracción de conceptos clave
- Análisis temático
- Generación de mapas conceptuales
- Caching de resultados

### 2. Pruebas de Utilidades (`utils/`)

**Objetivo**: Validar funciones de soporte y utilidades.

#### `test_rag_processor_unit.py`
- Procesamiento de documentos
- Creación de chunks
- Operaciones vectoriales
- Funcionalidad de búsqueda
- Manejo de caché

#### `test_ollama_client_unit.py`
- Conexión con Ollama
- Gestión de modelos
- Generación de respuestas
- Manejo de errores de conexión

### 3. Pruebas de Configuración (`config/`)

**Objetivo**: Validar configuraciones y settings.

#### `test_settings_unit.py`
- Carga de configuraciones
- Validación de parámetros
- Variables de entorno
- Configuraciones por defecto

## Fixtures y Datos de Prueba

### Datos de Prueba Centralizados

El archivo `fixtures/test_data.py` contiene:

```python
# Contenido de documentos de ejemplo
SAMPLE_TEXT_CONTENT = "..."
SAMPLE_PDF_CONTENT = "..."

# Modelos de Ollama simulados
SAMPLE_OLLAMA_MODELS = [...]

# Mensajes de chat de ejemplo
SAMPLE_CHAT_MESSAGES = [...]

# Resultados RAG simulados
SAMPLE_RAG_RESULTS = [...]
```

### Funciones Utilitarias

```python
def get_sample_file_content(file_type: str) -> str:
    """Retorna contenido de ejemplo para diferentes tipos de archivo."""

def create_temp_test_file(temp_dir: Path, filename: str, content: str) -> Path:
    """Crea archivos temporales para pruebas."""

def get_mock_streamlit_session():
    """Retorna mock de streamlit session_state."""
```

## Ejecución de Pruebas

### Script de Ejecución

```bash
# Ejecutar todas las pruebas unitarias
python unit_tests/run_unit_tests.py

# Ejecutar con cobertura
python unit_tests/run_unit_tests.py --coverage

# Ejecutar suite específica
python unit_tests/run_unit_tests.py --suite modules

# Salida detallada
python unit_tests/run_unit_tests.py --verbose

# Verificar entorno
python unit_tests/run_unit_tests.py --check-env
```

### Ejecución Directa con Pytest

```bash
# Todas las pruebas unitarias
pytest unit_tests/ -m unit

# Módulo específico
pytest unit_tests/modules/test_chatbot_unit.py -v

# Con cobertura
pytest unit_tests/ --cov=modules --cov=utils --cov-report=html
```

### Opciones de Ejecución

| Opción | Descripción |
|--------|-------------|
| `-v, --verbose` | Salida detallada |
| `-c, --coverage` | Generar reporte de cobertura |
| `-s, --suite` | Ejecutar suite específica |
| `--check-env` | Verificar configuración del entorno |

## Cobertura de Código

### Configuración de Cobertura

```bash
# Generar reporte HTML
pytest unit_tests/ --cov=modules --cov=utils --cov-report=html

# Generar reporte JSON
pytest unit_tests/ --cov=modules --cov=utils --cov-report=json

# Mostrar líneas faltantes
pytest unit_tests/ --cov=modules --cov=utils --cov-report=term-missing
```

### Métricas de Cobertura

- **Objetivo**: >90% de cobertura en módulos críticos
- **Mínimo aceptable**: >80% de cobertura general
- **Reportes**: HTML, JSON, y terminal

### Interpretación de Reportes

```
Name                    Stmts   Miss  Cover   Missing
---------------------------------------------------
modules/chatbot.py        150     15    90%   45-50, 78-82
utils/rag_processor.py    200     20    90%   123-130, 156-163
---------------------------------------------------
TOTAL                     350     35    90%
```

## Mejores Prácticas

### 1. Nomenclatura

```python
# Estructura de nombres de pruebas
def test_[funcionalidad]_[escenario]_[resultado_esperado]():
    """Descripción clara de lo que prueba."""
    
# Ejemplos
def test_chatbot_initialization_with_valid_config_succeeds():
def test_rag_processor_search_with_empty_query_returns_empty_list():
```

### 2. Estructura de Pruebas (AAA Pattern)

```python
def test_example():
    # Arrange - Configurar datos y mocks
    mock_client = Mock()
    mock_client.get_models.return_value = ["model1", "model2"]
    
    # Act - Ejecutar la funcionalidad
    result = function_under_test(mock_client)
    
    # Assert - Verificar resultados
    assert result is not None
    assert len(result) == 2
    mock_client.get_models.assert_called_once()
```

### 3. Uso de Mocks

```python
# Mock de dependencias externas
@patch('modules.chatbot.OllamaClient')
def test_with_mocked_client(mock_ollama):
    # Configurar comportamiento del mock
    mock_client = Mock()
    mock_ollama.return_value = mock_client
    
    # Ejecutar prueba
    result = chatbot.some_function()
    
    # Verificar interacciones
    mock_ollama.assert_called_once()
```

### 4. Manejo de Excepciones

```python
def test_function_raises_exception_on_invalid_input():
    with pytest.raises(ValueError, match="Invalid input"):
        function_under_test("invalid_input")
```

### 5. Parametrización

```python
@pytest.mark.parametrize("input_value,expected", [
    ("valid_input", True),
    ("invalid_input", False),
    ("", False)
])
def test_validation_function(input_value, expected):
    result = validate_input(input_value)
    assert result == expected
```

## Troubleshooting

### Problemas Comunes

#### 1. ImportError en Módulos

**Problema**: `ImportError: cannot import name 'module' from 'package'`

**Solución**:
```python
# En conftest.py o al inicio del archivo de prueba
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

#### 2. Mocks No Funcionan

**Problema**: Los mocks no se aplican correctamente

**Solución**:
```python
# Asegurar que el patch se aplique al lugar correcto
@patch('modules.chatbot.OllamaClient')  # Donde se importa
# No: @patch('utils.ollama_client.OllamaClient')  # Donde se define
```

#### 3. Fixtures No Se Encuentran

**Problema**: `fixture 'fixture_name' not found`

**Solución**:
- Verificar que `conftest.py` esté en el directorio correcto
- Asegurar que el fixture esté definido con `@pytest.fixture`
- Verificar el scope del fixture

#### 4. Problemas de Cobertura

**Problema**: Cobertura no se genera correctamente

**Solución**:
```bash
# Instalar pytest-cov
pip install pytest-cov

# Verificar paths en comando de cobertura
pytest --cov=modules --cov=utils unit_tests/
```

### Debugging de Pruebas

```python
# Usar pytest con debugging
pytest unit_tests/test_file.py::test_function -v -s --pdb

# Logging en pruebas
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Verificación del Entorno

```bash
# Verificar instalación de dependencias
python unit_tests/run_unit_tests.py --check-env

# Verificar estructura de directorios
ls -la unit_tests/
```

## Integración con CI/CD

### GitHub Actions

```yaml
name: Unit Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run unit tests
      run: python unit_tests/run_unit_tests.py --coverage
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Conclusiones

Las pruebas unitarias en CogniChat proporcionan:

1. **Validación Granular**: Cada componente se prueba de forma aislada
2. **Feedback Rápido**: Ejecución rápida para desarrollo iterativo
3. **Documentación Viva**: Las pruebas documentan el comportamiento esperado
4. **Refactoring Seguro**: Confianza para realizar cambios en el código
5. **Calidad Asegurada**: Detección temprana de errores y regresiones

### Próximos Pasos

1. **Expansión de Cobertura**: Agregar más pruebas para módulos específicos
2. **Automatización**: Integrar con pipeline de CI/CD
3. **Métricas Avanzadas**: Implementar métricas de calidad de código
4. **Performance Testing**: Agregar pruebas de rendimiento unitarias

---

**Fecha de Creación**: Octubre 2024  
**Versión**: 1.0.0  
**Autor**: Equipo de Desarrollo CogniChat