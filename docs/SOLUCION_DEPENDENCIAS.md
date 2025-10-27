# Solución de Problemas de Dependencias en CogniChat

## Problema Identificado

### Error de Exportación PDF
**Síntoma:** Al intentar exportar conversaciones a PDF, aparece el mensaje "reportlab no está instalado".

**Causa raíz:** Las dependencias `reportlab` y `pyperclip` no estaban instaladas en el entorno conda específico (`cognichat-py311`) que utiliza la aplicación.

### Diagnóstico
```bash
# Verificación del estado de PDF_AVAILABLE
python -c "import utils.chat_history; print('PDF_AVAILABLE:', utils.chat_history.PDF_AVAILABLE)"
# Resultado: PDF_AVAILABLE: False

# Verificación de reportlab
python -c "import reportlab; print(reportlab.Version)"
# Resultado: ModuleNotFoundError: No module named 'reportlab'
```

## Solución Implementada

### 1. Instalación de Dependencias
```bash
# Activar el entorno correcto
conda activate cognichat-py311

# Instalar las dependencias faltantes
pip install reportlab>=4.0.0 pyperclip>=1.8.2
```

### 2. Verificación Post-Instalación
```bash
# Verificar que PDF_AVAILABLE sea True
python -c "import utils.chat_history; print('PDF_AVAILABLE:', utils.chat_history.PDF_AVAILABLE)"
# Resultado esperado: PDF_AVAILABLE: True
```

### 3. Reinicio del Servidor
```bash
# Reiniciar Streamlit para cargar las nuevas dependencias
streamlit run app.py
```

## ¿Por qué se usa Conda en lugar del entorno Python estándar?

### Ventajas de Conda

1. **Gestión de Dependencias Compleja**
   - Conda maneja tanto paquetes Python como dependencias del sistema
   - Resuelve automáticamente conflictos de versiones
   - Incluye bibliotecas compiladas (C/C++) de forma más eficiente

2. **Aislamiento de Entornos**
   - Cada proyecto puede tener su propia versión de Python
   - Evita conflictos entre diferentes versiones de librerías
   - Permite trabajar con múltiples proyectos sin interferencias

3. **Reproducibilidad**
   - Los entornos conda son más fáciles de replicar
   - Mejor control de versiones de dependencias del sistema
   - Facilita el despliegue en diferentes máquinas

4. **Gestión de Paquetes Científicos**
   - Optimizado para librerías de ciencia de datos y ML
   - Mejor manejo de paquetes como NumPy, SciPy, TensorFlow
   - Instalación más rápida de paquetes compilados

### Comparación: Conda vs pip/venv

| Aspecto | Conda | pip/venv |
|---------|-------|----------|
| Gestión de Python | ✅ Múltiples versiones | ❌ Solo la versión instalada |
| Dependencias del sistema | ✅ Incluidas | ❌ Manejo manual |
| Resolución de conflictos | ✅ Automática | ❌ Manual |
| Paquetes científicos | ✅ Optimizado | ⚠️ Puede requerir compilación |
| Velocidad de instalación | ✅ Rápida (binarios) | ⚠️ Puede ser lenta (compilación) |

### Estructura del Entorno en CogniChat

```
cognichat-py311/
├── Python 3.11.x
├── Streamlit
├── Dependencias de ML/NLP
├── reportlab (para PDF)
├── pyperclip (para clipboard)
└── Otras dependencias específicas
```

## Mejores Prácticas

### 1. Verificación de Entorno
```bash
# Siempre verificar el entorno activo
conda info --envs
conda list  # Ver paquetes instalados
```

### 2. Instalación de Dependencias
```bash
# Preferir conda cuando esté disponible
conda install package_name

# Usar pip solo cuando el paquete no esté en conda
pip install package_name
```

### 3. Documentación de Dependencias
- Mantener actualizado `requirements.txt`
- Documentar versiones específicas
- Incluir dependencias del sistema si es necesario

## Prevención de Problemas Futuros

1. **Script de Verificación**
   - Crear script que verifique todas las dependencias críticas
   - Ejecutar antes de iniciar la aplicación

2. **Documentación Clara**
   - Especificar el entorno conda requerido
   - Listar todas las dependencias críticas
   - Incluir pasos de instalación detallados

3. **Testing de Funcionalidades**
   - Probar exportación PDF en cada despliegue
   - Verificar que todas las funcionalidades estén disponibles

## Comandos Útiles para Troubleshooting

```bash
# Verificar entorno activo
echo $CONDA_DEFAULT_ENV

# Listar entornos disponibles
conda env list

# Verificar instalación de paquete específico
python -c "import package_name; print('OK')"

# Ver información detallada del entorno
conda info

# Exportar entorno para reproducibilidad
conda env export > environment.yml
```

Este documento debe consultarse siempre que aparezcan errores relacionados con dependencias faltantes o problemas de entorno en CogniChat.