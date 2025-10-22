# 🔬 Módulo de Análisis Cualitativo v2.0

## 📋 Descripción

Sistema modular de análisis cualitativo diseñado específicamente para **asistir a investigadores** en el análisis profundo de contenido documental.

## 🎯 Principios Fundamentales

### 1. 🧭 Asistencia al Investigador
- Explicaciones claras sobre qué hace cada análisis
- Guías de interpretación de resultados
- Información metodológica transparente

### 2. 🧠 Procesamiento Inteligente
- **NO copia y pega** información
- Analiza, sintetiza y contextualiza
- Genera valor añadido mediante procesamiento

### 3. 📚 Fundamentación Completa
- Cada resultado respaldado por citas
- Sistema de trazabilidad completo
- Verificación de fuentes originales

### 4. 🔬 Transparencia Metodológica
- Algoritmos documentados
- Limitaciones explícitas
- Validación humana recomendada

## 📁 Estructura del Módulo

```
qualitative_analysis/
├── __init__.py                      # API pública del módulo
├── README.md                        # Esta documentación
│
├── core/                            # Componentes fundamentales
│   ├── __init__.py
│   ├── config.py                    # Configuración centralizada
│   ├── analyzer.py                  # Clase orquestadora principal
│   └── citation_manager.py          # Sistema de citación y referencias
│
├── extractors/                      # Extractores de información
│   ├── __init__.py
│   └── concept_extractor.py         # ✅ Extractor de conceptos clave (TF-IDF)
│
├── analyzers/                       # Analizadores avanzados (futuro)
│   └── theme_analyzer.py            # 🔜 Análisis de temas (LDA)
│
├── visualizations/                  # Visualizaciones (futuro)
│   └── concept_map.py               # 🔜 Mapas conceptuales
│
└── ui/                              # Interfaz de usuario
    ├── __init__.py
    ├── main_render.py               # Función render() principal
    ├── components/
    │   ├── __init__.py
    │   └── educational.py           # Componentes educativos
    └── tabs/
        ├── __init__.py
        └── concepts_tab.py          # ✅ Tab de conceptos clave
```

## 🚀 Uso Rápido

### Desde la Aplicación (Streamlit)

```python
# En app.py (ya integrado)
from modules import qualitative_analysis

# Renderizar el módulo completo
qualitative_analysis.render()
```

### Uso Programático

```python
from modules.qualitative_analysis import QualitativeAnalyzer, AnalysisConfig

# Crear configuración personalizada
config = AnalysisConfig(
    max_concepts=30,
    use_ngrams=True,
    enable_citations=True
)

# Inicializar analizador
analyzer = QualitativeAnalyzer(config)

# Preparar datos (chunks de documentos)
chunks = [
    {
        'content': 'Texto del documento...',
        'metadata': {
            'source_file': 'documento.pdf',
            'page_number': 1
        }
    },
    # más chunks...
]

# Extraer conceptos
concepts = analyzer.extract_concepts(chunks)

# Ver resultados
for concept in concepts[:10]:
    print(f"{concept.concept}: {concept.relevance_score:.3f}")
    print(f"  Fuentes: {', '.join(concept.sources)}")
    if concept.citations:
        citation = concept.citations[0]
        print(f"  Cita: {citation.format_citation()}")
```

## 📊 Sub-módulos Implementados

### ✅ 1. Extracción de Conceptos Clave

**Estado:** Completamente implementado

**Características:**
- Extracción inteligente usando TF-IDF
- Detección de n-gramas (frases completas)
- Sistema de citación integrado
- Análisis de co-ocurrencia
- Exportación en múltiples formatos

**Algoritmo:** TF-IDF con detección de n-gramas (1-3 palabras)

**Ejemplo de uso:**
```python
from modules.qualitative_analysis.extractors import ConceptExtractor

extractor = ConceptExtractor(config)
concepts = extractor.extract_concepts(chunks)

# Ver concepto con citas
concept = concepts[0]
print(f"Concepto: {concept.concept}")
print(f"Relevancia: {concept.relevance_score:.3f}")
print(f"Frecuencia: {concept.frequency}")
print(f"Fuentes: {len(concept.sources)}")

# Ver primera cita
if concept.citations:
    citation = concept.get_first_citation()
    print(f"Cita: {citation.get_full_context()}")
```

## 🔜 Sub-módulos Planeados

### 2. Análisis de Temas (En desarrollo)
- LDA (Latent Dirichlet Allocation)
- Clustering semántico
- Visualización de temas

### 3. Análisis de Sentimientos (En desarrollo)
- VADER + TextBlob
- Análisis contextual
- Tendencias emocionales

### 4. Triangulación (En desarrollo)
- Validación cruzada entre fuentes
- Análisis de confiabilidad
- Identificación de consensos

### 5. Mapas Conceptuales (En desarrollo)
- Visualizaciones interactivas
- Análisis de relaciones
- Redes semánticas

## 🎓 Guía para Investigadores

### ¿Qué hace este módulo?

Este sistema analiza tus documentos para identificar:
- **Conceptos clave**: Términos y frases más importantes
- **Patrones**: Relaciones y conexiones entre conceptos
- **Fuentes**: De dónde proviene cada información

### ¿Qué NO hace?

- ❌ NO copia y pega texto de documentos
- ❌ NO reemplaza el análisis humano
- ❌ NO interpreta significados automáticamente

### ¿Cómo interpretar resultados?

1. **Score de relevancia (0.0 - 1.0)**:
   - 0.7-1.0: Concepto central
   - 0.4-0.7: Concepto importante
   - 0.0-0.4: Concepto secundario

2. **Frecuencia**: Número de apariciones en corpus

3. **Citas**: Referencias a fuentes originales para verificación

### Mejores Prácticas

1. ✅ **Valida resultados**: Revisa que los conceptos extraídos tengan sentido
2. ✅ **Verifica citas**: Usa las citas para volver a las fuentes originales
3. ✅ **Contextualiza**: Interpreta según tu conocimiento del dominio
4. ✅ **Combina métodos**: Usa junto con análisis manual
5. ✅ **Documenta decisiones**: Registra cómo usas los resultados

## 🔧 Configuración Avanzada

### AnalysisConfig

```python
config = AnalysisConfig(
    # Extracción de conceptos
    min_concept_frequency=2,      # Frecuencia mínima
    max_concepts=30,               # Máximo a extraer
    use_ngrams=True,               # Detectar frases
    ngram_range=(1, 3),            # Rango de n-gramas
    
    # Citación
    enable_citations=True,         # Activar citas
    citation_context_chars=150,    # Contexto en citas
    
    # Interfaz
    show_explanations=True,        # Mostrar explicaciones
    show_methodology=True,         # Mostrar metodología
    show_interpretation_guide=True,# Guías de interpretación
    
    # Rendimiento
    enable_cache=True,             # Activar cache
    parallel_processing=True,      # Procesamiento paralelo
    max_workers=4                  # Workers paralelos
)
```

## 📤 Formatos de Exportación

### JSON Completo
```json
{
  "concept": "inteligencia artificial",
  "frequency": 15,
  "relevance_score": 0.856,
  "sources": ["doc1.pdf", "doc2.pdf"],
  "citations": [
    {
      "citation_id": "abc123",
      "source_file": "doc1.pdf",
      "content": "inteligencia artificial",
      "context_before": "...",
      "context_after": "..."
    }
  ]
}
```

### CSV
```csv
Concepto,Frecuencia,Relevancia,Num_Fuentes,Fuentes
inteligencia artificial,15,0.856,2,doc1.pdf; doc2.pdf
machine learning,12,0.742,2,doc1.pdf; doc2.pdf
```

## 🐛 Troubleshooting

### Error: "scikit-learn es requerido"
```bash
pip install scikit-learn
```

### Error: "No hay documentos disponibles"
1. Ve a "📄 Gestión de Documentos"
2. Sube tus archivos
3. Ve a "🧠 Procesamiento RAG"
4. Procesa los documentos
5. Regresa a "🔬 Análisis Cualitativo"

### Conceptos extraídos no tienen sentido
- Aumenta `min_concept_frequency`
- Verifica que los documentos estén en español
- Revisa que los stopwords estén configurados correctamente

### Performance lento
- Reduce `max_concepts`
- Desactiva `use_ngrams` para búsqueda más rápida
- Activa `parallel_processing`

## 📚 Referencias

### Algoritmos Utilizados

1. **TF-IDF** (Term Frequency-Inverse Document Frequency)
   - Referencia: Salton, G., & McGill, M. J. (1983). Introduction to modern information retrieval.

2. **N-gramas**
   - Detección de frases completas en lugar de palabras individuales

### Dependencias

- `scikit-learn`: Machine learning y TF-IDF
- `nltk`: Procesamiento de lenguaje natural
- `streamlit`: Interfaz de usuario
- `plotly`: Visualizaciones

## 🤝 Contribuir

Este módulo sigue una arquitectura modular que facilita agregar nuevos sub-módulos:

1. Crea nuevo extractor en `extractors/`
2. Agrega interfaz en `ui/tabs/`
3. Actualiza `main_render.py` para incluir nueva tab
4. Documenta metodología y limitaciones

## 📄 Licencia

Parte del proyecto CogniChat - Ver LICENSE en raíz del proyecto.

---

**Desarrollado con enfoque en investigación científica** 🔬✨

