# 👥 Guía de Usuario - Módulo de Análisis Cualitativo

## 📋 Tabla de Contenidos

1. [Introducción](#introducción)
2. [Primeros Pasos](#primeros-pasos)
3. [Análisis Básico](#análisis-básico)
4. [Visualizaciones Interactivas](#visualizaciones-interactivas)
5. [Análisis Avanzado](#análisis-avanzado)
6. [Casos de Uso](#casos-de-uso)
7. [Consejos y Mejores Prácticas](#consejos-y-mejores-prácticas)
8. [Solución de Problemas](#solución-de-problemas)
9. [FAQ](#faq)

---

## 🎯 Introducción

El **Módulo de Análisis Cualitativo** de CogniChat es una herramienta poderosa que te permite analizar documentos de texto de manera inteligente, extraer conceptos clave, identificar temas y crear visualizaciones interactivas. Esta guía te llevará paso a paso para dominar todas las funcionalidades.

### 🌟 **¿Qué puedes hacer con este módulo?**

- 📊 **Extraer conceptos clave** de tus documentos
- 🎯 **Identificar temas principales** automáticamente
- 😊 **Analizar sentimientos** del contenido
- 🗺️ **Crear mapas conceptuales** interactivos
- 🧠 **Generar mapas mentales** visuales
- 🔺 **Validar información** con triangulación
- ☁️ **Crear nubes de palabras**
- 📈 **Obtener métricas detalladas**

---

## 🚀 Primeros Pasos

### 📝 **Paso 1: Preparar tus Documentos**

Antes de usar el análisis cualitativo, necesitas tener documentos procesados en el sistema RAG:

1. **Ve a la pestaña "Procesamiento de Documentos RAG"**
2. **Carga tus documentos** (PDF, DOCX, TXT, etc.)
3. **Haz clic en "Procesar Documentos"**
4. **Espera a que termine el procesamiento**

✅ **Tip**: Los documentos se procesan en chunks (fragmentos) para análisis más preciso.

### 🎛️ **Paso 2: Acceder al Módulo**

1. **Navega a la pestaña "Análisis Cualitativo"**
2. **Verifica que aparezca el mensaje**: "✅ Datos RAG disponibles"
3. **Si no hay datos**: Vuelve al paso 1 para procesar documentos

### ⚙️ **Paso 3: Configuración Inicial (Opcional)**

En la sección "Configuración", puedes ajustar:

- **Frecuencia mínima**: Mínimo de apariciones para considerar un concepto (recomendado: 2)
- **Máximo de conceptos**: Número máximo de conceptos a extraer (recomendado: 50)
- **Umbral de similitud**: Sensibilidad para agrupar conceptos similares (recomendado: 0.6)
- **Habilitar cache**: Para análisis más rápidos en documentos grandes (recomendado: ✅)

---

## 📊 Análisis Básico

### 🔍 **Extracción de Conceptos**

#### **¿Qué hace?**
Identifica las palabras y frases más importantes de tus documentos.

#### **Cómo usarlo:**
1. **Ve a la sección "Análisis de Conceptos"**
2. **Haz clic en "Extraer Conceptos"**
3. **Espera el procesamiento** (puede tomar unos segundos)

#### **¿Qué verás?**
- **Lista de conceptos** ordenados por relevancia
- **Score de importancia** para cada concepto
- **Contexto** donde aparece cada concepto
- **Frecuencia** de aparición

#### **Ejemplo de resultado:**
```
🎯 Conceptos Principales:
1. "inteligencia artificial" - Score: 0.95 - Frecuencia: 15
2. "machine learning" - Score: 0.89 - Frecuencia: 12
3. "procesamiento de datos" - Score: 0.82 - Frecuencia: 8
```

### 🎯 **Análisis de Temas**

#### **¿Qué hace?**
Agrupa el contenido en temas principales usando técnicas de machine learning.

#### **Cómo usarlo:**
1. **Ve a la sección "Análisis de Temas"**
2. **Selecciona el número de temas** (recomendado: 8-12)
3. **Haz clic en "Analizar Temas"**

#### **¿Qué verás?**
- **Lista de temas** identificados
- **Palabras clave** para cada tema
- **Descripción** automática del tema
- **Distribución** de temas en el documento

#### **Ejemplo de resultado:**
```
🎯 Tema 1: "Tecnología Educativa"
Palabras clave: inteligencia artificial, educación, aprendizaje, tecnología
Descripción: Tecnologías aplicadas al ámbito educativo

🎯 Tema 2: "Análisis de Datos"
Palabras clave: datos, análisis, estadísticas, métricas
Descripción: Procesamiento y análisis de información
```

### 😊 **Análisis de Sentimientos**

#### **¿Qué hace?**
Analiza el tono emocional del contenido (positivo, negativo, neutral).

#### **Cómo usarlo:**
1. **Ve a la sección "Análisis de Sentimientos"**
2. **Haz clic en "Analizar Sentimientos"**

#### **¿Qué verás?**
- **Sentimiento general** del documento
- **Distribución** de sentimientos por sección
- **Tendencias** emocionales
- **Confianza** en el análisis

#### **Ejemplo de resultado:**
```
😊 Sentimiento General: Positivo (75% confianza)
📊 Distribución:
- Positivo: 60%
- Neutral: 30%
- Negativo: 10%
```

---

## 🗺️ Visualizaciones Interactivas

### 🗺️ **Mapas Conceptuales**

Los mapas conceptuales muestran las relaciones entre conceptos de manera visual.

#### **Cómo crearlos:**
1. **Ve a la sección "Mapa Conceptual"**
2. **Selecciona el modo de generación**:
   - **Normal**: Rápido y eficiente (recomendado)
   - **IA**: Más lento pero con análisis semántico profundo
3. **Elige el tipo de layout**:
   - **Spring**: Disposición natural y orgánica
   - **Hierarchical**: Estructura jerárquica
   - **Circular**: Disposición circular
4. **Haz clic en "Generar Mapa"**

#### **Cómo interactuar:**
- **Hacer clic** en nodos para ver detalles
- **Arrastrar** nodos para reorganizar
- **Zoom** con la rueda del mouse
- **Pan** arrastrando el fondo

#### **Leyenda de colores:**
- 🟣 **Morado**: Conceptos principales
- 🔵 **Azul**: Conceptos secundarios
- 🟢 **Verde**: Conceptos relacionados
- 🟡 **Amarillo**: Conexiones importantes

### 🧠 **Mapas Mentales**

Los mapas mentales presentan la información en una estructura radial desde un tema central.

#### **Cómo crearlos:**
1. **Ve a la sección "Mapa Mental"**
2. **Configura los parámetros**:
   - **Espaciado entre nodos**: Controla la separación (recomendado: 450)
   - **Fuerza de física**: Controla la atracción entre nodos (recomendado: 0.8)
3. **Selecciona el modo**:
   - **Normal**: Recomendado para la mayoría de casos
   - **IA**: Para análisis más profundo
4. **Haz clic en "Generar"**

#### **Estructura del mapa:**
- **Centro**: Tema principal del documento
- **Primer nivel**: Conceptos principales
- **Segundo nivel**: Sub-conceptos y detalles

#### **Interacciones:**
- **Hover**: Ver información adicional
- **Clic**: Expandir/contraer nodos
- **Arrastrar**: Reorganizar estructura

### ☁️ **Nubes de Palabras**

Las nubes de palabras muestran visualmente la frecuencia de términos.

#### **Cómo crearlas:**
1. **Ve a la sección "Nube de Palabras"**
2. **Selecciona opciones**:
   - **Filtrar por fuente**: Si tienes múltiples documentos
   - **Número máximo de palabras**: Controla la densidad
3. **Haz clic en "Generar Nube"**

#### **Interpretación:**
- **Tamaño**: Indica frecuencia (más grande = más frecuente)
- **Color**: Diferentes colores para categorización visual
- **Posición**: Organización automática para mejor legibilidad

---

## 🔺 Análisis Avanzado

### 🔺 **Triangulación**

La triangulación valida conceptos comparándolos entre múltiples fuentes o secciones.

#### **¿Cuándo usar triangulación?**
- Cuando tienes **múltiples documentos** sobre el mismo tema
- Cuando quieres **validar** la importancia de conceptos
- Cuando necesitas **confiabilidad** en tus análisis

#### **Tipos de triangulación:**

##### **Multi-Fuente (Múltiples documentos)**
- Compara conceptos entre diferentes documentos
- Identifica conceptos que aparecen en múltiples fuentes
- Mayor confiabilidad para conceptos validados

##### **Fuente Única (Un documento)**
- Divide el documento en secciones
- Identifica conceptos que aparecen en múltiples secciones
- Útil para identificar temas centrales vs. específicos

#### **Cómo interpretar los resultados:**

```
🔺 Conceptos Triangulados:
✅ "inteligencia artificial" - Confiabilidad: 85%
   Aparece en 3 de 4 fuentes
   
⚠️ "machine learning" - Confiabilidad: 60%
   Aparece en 2 de 4 fuentes
   
📍 "algoritmo específico" - Confiabilidad: 25%
   Aparece en 1 de 4 fuentes
```

**Interpretación:**
- **Alta confiabilidad (>70%)**: Concepto central y validado
- **Media confiabilidad (40-70%)**: Concepto importante
- **Baja confiabilidad (<40%)**: Concepto específico o marginal

### ⚡ **Análisis Paralelo**

Para documentos grandes, puedes ejecutar múltiples análisis simultáneamente.

#### **Cómo usarlo:**
1. **Ve a la sección "Análisis Paralelo"**
2. **Selecciona los tipos de análisis**:
   - ✅ Extracción de conceptos
   - ✅ Análisis de temas
   - ✅ Análisis de sentimientos
   - ✅ Clustering
3. **Haz clic en "Ejecutar Análisis Paralelo"**

#### **Ventajas:**
- **Más rápido**: Múltiples análisis simultáneos
- **Eficiente**: Mejor uso de recursos
- **Completo**: Resultados de todos los análisis

---

## 💼 Casos de Uso

### 📚 **Caso 1: Investigación Académica**

**Escenario**: Analizar múltiples papers sobre inteligencia artificial.

**Proceso recomendado:**
1. **Cargar papers** en formato PDF
2. **Procesar documentos** en el sistema RAG
3. **Extraer conceptos** para identificar términos clave
4. **Crear mapa conceptual** para visualizar relaciones
5. **Realizar triangulación** para validar conceptos importantes
6. **Analizar sentimientos** para evaluar tono de los papers

**Resultado esperado**: Comprensión profunda de la literatura, identificación de conceptos centrales y tendencias en el campo.

### 📊 **Caso 2: Análisis de Encuestas**

**Escenario**: Analizar respuestas abiertas de una encuesta de satisfacción.

**Proceso recomendado:**
1. **Cargar respuestas** como documento de texto
2. **Analizar sentimientos** para evaluar satisfacción general
3. **Extraer temas** para identificar categorías de feedback
4. **Crear nube de palabras** para visualizar términos frecuentes
5. **Generar mapa mental** para organizar insights

**Resultado esperado**: Identificación de áreas de mejora, temas de satisfacción y insights accionables.

### 📰 **Caso 3: Análisis de Noticias**

**Escenario**: Analizar artículos de noticias sobre un tema específico.

**Proceso recomendado:**
1. **Cargar artículos** de diferentes fuentes
2. **Realizar triangulación** para identificar información consistente
3. **Analizar sentimientos** para evaluar sesgo mediático
4. **Extraer conceptos** para identificar narrativas principales
5. **Crear visualizaciones** para presentar hallazgos

**Resultado esperado**: Comprensión objetiva del tema, identificación de narrativas dominantes y evaluación de sesgos.

### 📖 **Caso 4: Análisis de Documentos Corporativos**

**Escenario**: Analizar documentos internos para identificar procesos y procedimientos.

**Proceso recomendado:**
1. **Cargar documentos** (manuales, procedimientos, etc.)
2. **Extraer conceptos** para identificar procesos clave
3. **Crear mapa conceptual** para visualizar flujos de trabajo
4. **Analizar temas** para categorizar tipos de documentos
5. **Generar resúmenes** automáticos de cada sección

**Resultado esperado**: Mapeo de procesos, identificación de áreas de mejora y documentación estructurada.

---

## 💡 Consejos y Mejores Prácticas

### 📝 **Preparación de Documentos**

#### ✅ **Buenas Prácticas:**
- **Formato consistente**: Usa el mismo formato para documentos similares
- **Calidad del texto**: Asegúrate de que el OCR funcione correctamente
- **Tamaño apropiado**: Documentos de 5-50 páginas funcionan mejor
- **Contenido relevante**: Elimina páginas irrelevantes antes del procesamiento

#### ❌ **Evitar:**
- **Documentos muy pequeños**: Menos de 1 página pueden dar resultados pobres
- **Documentos muy grandes**: Más de 100 páginas pueden ser lentos
- **Texto con muchos errores**: Corrige errores de OCR antes del análisis
- **Contenido duplicado**: Elimina duplicados para evitar sesgos

### 🎯 **Optimización de Resultados**

#### **Para mejores conceptos:**
- **Ajusta la frecuencia mínima**: Aumenta para conceptos más relevantes
- **Revisa el contexto**: Verifica que los contextos sean apropiados
- **Usa n-gramas**: Habilita la extracción de frases completas

#### **Para mejores temas:**
- **Experimenta con el número de temas**: Prueba 5, 10, 15 temas
- **Revisa las palabras clave**: Asegúrate de que sean representativas
- **Combina con clustering**: Usa ambos métodos para validación cruzada

#### **Para mejores visualizaciones:**
- **Prueba diferentes layouts**: Cada tipo tiene sus ventajas
- **Ajusta el espaciado**: Para mapas mentales, usa espaciado mayor
- **Interactúa con los mapas**: Explora todas las funcionalidades

### 🔧 **Configuración Avanzada**

#### **Para documentos técnicos:**
```python
# Configuración recomendada
config = {
    'min_frequency': 3,           # Conceptos más específicos
    'max_concepts': 30,           # Menos conceptos, más relevantes
    'similarity_threshold': 0.7,  # Mayor precisión
    'n_topics': 8                 # Temas más específicos
}
```

#### **Para documentos generales:**
```python
# Configuración recomendada
config = {
    'min_frequency': 2,           # Más conceptos
    'max_concepts': 50,           # Mayor cobertura
    'similarity_threshold': 0.6,  # Balanceado
    'n_topics': 12                # Más temas
}
```

#### **Para análisis rápido:**
```python
# Configuración recomendada
config = {
    'enable_cache': True,         # Usar cache
    'parallel_processing': True,  # Procesamiento paralelo
    'max_concepts': 20,           # Menos conceptos
    'n_topics': 5                 # Menos temas
}
```

---

## 🚨 Solución de Problemas

### ❗ **Problemas Comunes**

#### **1. "No hay datos disponibles"**

**Causa**: Los documentos no están procesados en el sistema RAG.

**Solución**:
1. Ve a "Procesamiento de Documentos RAG"
2. Verifica que los documentos estén cargados
3. Haz clic en "Procesar Documentos"
4. Espera a que termine el procesamiento
5. Regresa al análisis cualitativo

#### **2. "Error al generar mapa conceptual"**

**Causa**: Problemas con la librería PyVis o memoria insuficiente.

**Solución**:
1. **Usa modo normal** en lugar de IA
2. **Reduce el número de conceptos** en la configuración
3. **Cierra otras aplicaciones** para liberar memoria
4. **Reinicia la aplicación** si el problema persiste

#### **3. "Mapa mental se ve mal"**

**Causa**: Configuración de espaciado o colores inadecuada.

**Solución**:
1. **Aumenta el espaciado** entre nodos (450-600)
2. **Reduce la fuerza de física** (0.5-0.8)
3. **Usa modo normal** para mejor rendimiento
4. **Verifica que el texto sea legible**

#### **4. "Análisis muy lento"**

**Causa**: Documentos muy grandes o configuración subóptima.

**Solución**:
1. **Habilita el cache** en la configuración
2. **Usa procesamiento paralelo**
3. **Reduce el número de conceptos** máximos
4. **Procesa documentos más pequeños**

#### **5. "Conceptos irrelevantes"**

**Causa**: Configuración de frecuencia muy baja o stopwords insuficientes.

**Solución**:
1. **Aumenta la frecuencia mínima** (3-5)
2. **Revisa la lista de stopwords**
3. **Usa n-gramas** para conceptos más coherentes
4. **Filtra manualmente** conceptos irrelevantes

### 🔍 **Debug y Diagnóstico**

#### **Verificar estado del sistema:**
1. **Ve a la configuración**
2. **Revisa las métricas de cache**
3. **Verifica el estado de Ollama** (si usas modo IA)
4. **Comprueba los logs** para errores específicos

#### **Probar con datos de ejemplo:**
1. **Usa documentos pequeños** para probar
2. **Verifica que el análisis básico funcione**
3. **Prueba visualizaciones simples**
4. **Escala gradualmente** a documentos más grandes

---

## ❓ FAQ

### **P: ¿Cuál es la diferencia entre mapa conceptual y mapa mental?**

**R**: 
- **Mapa conceptual**: Muestra relaciones entre conceptos en red
- **Mapa mental**: Estructura radial desde un tema central
- **Usa mapa conceptual** para relaciones complejas
- **Usa mapa mental** para organización jerárquica

### **P: ¿Por qué algunos conceptos aparecen irrelevantes?**

**R**: 
- **Frecuencia muy baja**: Aumenta la frecuencia mínima
- **Stopwords insuficientes**: El sistema está aprendiendo
- **Contexto específico**: Pueden ser relevantes en tu dominio
- **Solución**: Ajusta la configuración y revisa manualmente

### **P: ¿Cómo interpreto el score de los conceptos?**

**R**:
- **Score alto (>0.8)**: Concepto muy relevante
- **Score medio (0.5-0.8)**: Concepto importante
- **Score bajo (<0.5)**: Concepto marginal
- **Considera también**: Frecuencia y contexto

### **P: ¿Qué hacer si el análisis es muy lento?**

**R**:
- **Habilita cache**: Para análisis repetidos
- **Usa procesamiento paralelo**: Para análisis múltiples
- **Reduce conceptos**: Menos conceptos = más rápido
- **Documentos más pequeños**: Divide documentos grandes

### **P: ¿Puedo exportar los resultados?**

**R**:
- **Sí**: Los mapas se guardan como archivos HTML
- **Nubes de palabras**: Se guardan como imágenes
- **Datos**: Se pueden copiar desde la interfaz
- **Reportes**: Usa la función de exportación

### **P: ¿Cómo mejoro la calidad del análisis?**

**R**:
- **Documentos de calidad**: Texto claro y sin errores
- **Configuración apropiada**: Ajusta parámetros según tu caso
- **Múltiples análisis**: Combina diferentes métodos
- **Validación manual**: Revisa y ajusta resultados

### **P: ¿Qué hacer si no hay suficientes conceptos?**

**R**:
- **Reduce la frecuencia mínima**: De 3 a 2 o 1
- **Aumenta el máximo de conceptos**: De 30 a 50
- **Verifica el contenido**: Asegúrate de que sea sustancial
- **Usa n-gramas**: Para capturar frases completas

### **P: ¿Puedo analizar documentos en otros idiomas?**

**R**:
- **Español**: Optimizado por defecto
- **Inglés**: Funciona bien
- **Otros idiomas**: Puede requerir ajustes en stopwords
- **Solución**: Configura stopwords personalizados

---

## 🎉 ¡Felicidades!

Has completado la guía de usuario del módulo de análisis cualitativo. Ahora tienes todas las herramientas necesarias para:

✅ **Extraer insights** valiosos de tus documentos  
✅ **Crear visualizaciones** impactantes  
✅ **Validar información** con triangulación  
✅ **Optimizar** tus análisis  
✅ **Resolver problemas** comunes  

### 🚀 **Próximos Pasos**

1. **Experimenta** con diferentes tipos de documentos
2. **Combina** múltiples análisis para insights más profundos
3. **Comparte** tus visualizaciones con colegas
4. **Itera** y mejora basándote en los resultados

### 📞 **Soporte**

Si necesitas ayuda adicional:
- **Revisa la documentación técnica** en `docs/`
- **Consulta los logs** para errores específicos
- **Experimenta** con configuraciones diferentes
- **Únete a la comunidad** de CogniChat

---

**¡Disfruta explorando el poder del análisis cualitativo!** 🧠✨

*"Transformando documentos en conocimiento, una visualización a la vez."*
