# Pruebas Unitarias - CogniChat

## Descripción

Este directorio contiene las pruebas unitarias para CogniChat, diseñadas para validar componentes individuales del sistema de manera aislada y eficiente.

## Estructura

```
unit_tests/
├── README.md                      # Este archivo
├── conftest.py                    # Configuración pytest y fixtures
├── run_unit_tests.py             # Script de ejecución principal
├── modules/                      # Pruebas de módulos principales
├── utils/                        # Pruebas de utilidades
├── config/                       # Pruebas de configuración
├── fixtures/                     # Datos de prueba y mocks
└── reports/                      # Reportes generados
```

## Inicio Rápido

### 1. Instalar Dependencias

```bash
pip install pytest pytest-cov pytest-mock pytest-json-report
```

### 2. Ejecutar Todas las Pruebas

```bash
python unit_tests/run_unit_tests.py
```

### 3. Ejecutar con Cobertura

```bash
python unit_tests/run_unit_tests.py --coverage
```

## Comandos Principales

| Comando | Descripción |
|---------|-------------|
| `python run_unit_tests.py` | Ejecutar todas las pruebas |
| `python run_unit_tests.py --coverage` | Ejecutar con reporte de cobertura |
| `python run_unit_tests.py --suite modules` | Ejecutar solo pruebas de módulos |
| `python run_unit_tests.py --verbose` | Salida detallada |
| `python run_unit_tests.py --check-env` | Verificar configuración |

## Tipos de Pruebas

### Módulos Principales (`modules/`)
- **Chatbot**: Lógica de conversación y manejo de modelos
- **Document Processor**: Procesamiento de documentos
- **Qualitative Analysis**: Análisis cualitativo y extracción de conceptos

### Utilidades (`utils/`)
- **RAG Processor**: Procesamiento y búsqueda vectorial
- **Ollama Client**: Comunicación con modelos Ollama
- **Validators**: Validación de datos y formatos

### Configuración (`config/`)
- **Settings**: Configuraciones del sistema

## Características Principales

- ✅ **Aislamiento Completo**: Cada prueba es independiente
- ✅ **Mocking Extensivo**: Sin dependencias externas
- ✅ **Fixtures Reutilizables**: Datos de prueba centralizados
- ✅ **Reportes Detallados**: HTML, JSON, y JUnit
- ✅ **Cobertura de Código**: Métricas detalladas
- ✅ **Ejecución Rápida**: Optimizado para desarrollo iterativo

## Diferencias con Pruebas de Integración

| Aspecto | Pruebas Unitarias | Pruebas de Integración |
|---------|-------------------|------------------------|
| **Ubicación** | `unit_tests/` | `tests/` |
| **Enfoque** | Componentes aislados | Interacción entre módulos |
| **Velocidad** | Muy rápida | Moderada |
| **Dependencias** | Mockeadas | Reales |

## Documentación Completa

Para documentación detallada, consulta:
- **[Documentación Completa](../docs/DOCUMENTACION_PRUEBAS_UNITARIAS.md)**: Guía exhaustiva
- **[Plan de Desarrollo](../docs/Plan_de_Desarrollo.md)**: Contexto del proyecto

## Contribuir

1. **Crear Pruebas**: Seguir el patrón AAA (Arrange, Act, Assert)
2. **Nomenclatura**: `test_[funcionalidad]_[escenario]_[resultado]`
3. **Cobertura**: Mantener >90% en módulos críticos
4. **Documentación**: Incluir docstrings descriptivos

## Soporte

Para problemas o preguntas:
1. Consultar la sección **Troubleshooting** en la documentación completa
2. Verificar configuración con `python run_unit_tests.py --check-env`
3. Revisar logs en el directorio `reports/`

---

**Versión**: 1.0.0  
**Última Actualización**: Octubre 2024