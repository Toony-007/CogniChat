"""
Datos de Prueba y Fixtures - CogniChat

Este módulo contiene datos de prueba, fixtures y configuraciones
reutilizables para las pruebas unitarias.
"""

import json
from datetime import datetime
from pathlib import Path

# Datos de prueba para documentos
SAMPLE_TEXT_CONTENT = """
Este es un documento de ejemplo para pruebas unitarias de CogniChat.

El sistema CogniChat utiliza técnicas de Retrieval-Augmented Generation (RAG)
para proporcionar respuestas contextuales basadas en documentos cargados.

Conceptos clave:
- Procesamiento de lenguaje natural
- Análisis cualitativo de datos
- Inteligencia artificial conversacional
- Búsqueda semántica

El análisis cualitativo permite extraer insights significativos de los datos
textuales, identificando patrones, temas y conceptos relevantes.
"""

SAMPLE_PDF_CONTENT = """
Documento PDF de Ejemplo - CogniChat

Este es el contenido simulado de un archivo PDF para pruebas.
Contiene información sobre metodologías de investigación cualitativa.

Metodologías incluidas:
1. Análisis temático
2. Análisis de contenido
3. Teoría fundamentada
4. Análisis fenomenológico

Cada metodología tiene sus propias características y aplicaciones específicas
en el contexto de la investigación cualitativa.
"""

# Datos de prueba para modelos de Ollama
SAMPLE_OLLAMA_MODELS = [
    {
        "name": "llama2:7b",
        "size": "3.8GB",
        "digest": "sha256:1234567890abcdef",
        "details": {
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        }
    },
    {
        "name": "mistral:7b",
        "size": "4.1GB", 
        "digest": "sha256:abcdef1234567890",
        "details": {
            "format": "gguf",
            "family": "mistral",
            "families": ["mistral"],
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        }
    },
    {
        "name": "codellama:7b",
        "size": "3.9GB",
        "digest": "sha256:567890abcdef1234",
        "details": {
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "7B",
            "quantization_level": "Q4_0"
        }
    }
]

# Datos de prueba para mensajes de chat
SAMPLE_CHAT_MESSAGES = [
    {
        "role": "user",
        "content": "¿Qué es el análisis cualitativo?",
        "timestamp": "2024-01-15T10:30:00Z"
    },
    {
        "role": "assistant", 
        "content": "El análisis cualitativo es un método de investigación que se centra en comprender y interpretar datos no numéricos, como texto, imágenes o audio. Se utiliza para explorar fenómenos complejos y obtener insights profundos sobre comportamientos, motivaciones y experiencias.",
        "timestamp": "2024-01-15T10:30:15Z"
    },
    {
        "role": "user",
        "content": "¿Cuáles son las principales técnicas?",
        "timestamp": "2024-01-15T10:31:00Z"
    },
    {
        "role": "assistant",
        "content": "Las principales técnicas de análisis cualitativo incluyen:\n\n1. **Análisis temático**: Identificación de patrones y temas recurrentes\n2. **Análisis de contenido**: Categorización sistemática del contenido\n3. **Teoría fundamentada**: Desarrollo de teorías basadas en los datos\n4. **Análisis fenomenológico**: Exploración de experiencias vividas\n5. **Análisis narrativo**: Estudio de historias y relatos",
        "timestamp": "2024-01-15T10:31:20Z"
    }
]

# Datos de prueba para resultados RAG
SAMPLE_RAG_RESULTS = [
    {
        "content": "El análisis cualitativo es fundamental para comprender fenómenos complejos en la investigación social.",
        "score": 0.95,
        "source": "metodologia_investigacion.pdf",
        "chunk_id": "chunk_001"
    },
    {
        "content": "Las técnicas de codificación permiten organizar y categorizar los datos cualitativos de manera sistemática.",
        "score": 0.87,
        "source": "tecnicas_analisis.txt", 
        "chunk_id": "chunk_045"
    },
    {
        "content": "La triangulación de datos mejora la validez y confiabilidad de los resultados en investigación cualitativa.",
        "score": 0.82,
        "source": "validez_investigacion.docx",
        "chunk_id": "chunk_023"
    }
]

# Configuración de prueba para análisis cualitativo
QUALITATIVE_ANALYSIS_CONFIG = {
    "max_concepts": 10,
    "min_frequency": 2,
    "stopwords_language": "spanish",
    "similarity_threshold": 0.7,
    "clustering_method": "kmeans",
    "n_clusters": 5
}

# Datos de prueba para conceptos extraídos
SAMPLE_EXTRACTED_CONCEPTS = [
    {
        "concept": "análisis cualitativo",
        "frequency": 15,
        "relevance": 0.92,
        "related_terms": ["investigación", "metodología", "datos"]
    },
    {
        "concept": "procesamiento de lenguaje natural",
        "frequency": 8,
        "relevance": 0.85,
        "related_terms": ["NLP", "texto", "algoritmos"]
    },
    {
        "concept": "inteligencia artificial",
        "frequency": 12,
        "relevance": 0.88,
        "related_terms": ["IA", "machine learning", "automatización"]
    }
]

# Configuración de prueba para validadores
VALIDATOR_TEST_CASES = {
    "valid_emails": [
        "test@example.com",
        "user.name@domain.co.uk",
        "admin+tag@company.org"
    ],
    "invalid_emails": [
        "invalid-email",
        "@domain.com",
        "user@",
        "user name@domain.com"
    ],
    "valid_file_types": [
        "document.pdf",
        "text.txt",
        "report.docx",
        "notes.md"
    ],
    "invalid_file_types": [
        "image.jpg",
        "video.mp4",
        "archive.zip",
        "executable.exe"
    ]
}

# Datos de prueba para métricas del sistema
SAMPLE_SYSTEM_METRICS = {
    "timestamp": datetime.now().isoformat(),
    "memory_usage": {
        "total": "8GB",
        "used": "4.2GB",
        "available": "3.8GB"
    },
    "processing_stats": {
        "documents_processed": 25,
        "total_chunks": 150,
        "average_processing_time": "2.3s"
    },
    "model_performance": {
        "response_time": "1.8s",
        "accuracy_score": 0.89,
        "user_satisfaction": 4.2
    }
}

def get_sample_file_content(file_type: str) -> str:
    """
    Retorna contenido de ejemplo para diferentes tipos de archivo.
    
    Args:
        file_type: Tipo de archivo ('txt', 'pdf', 'docx')
        
    Returns:
        Contenido de ejemplo como string
    """
    content_map = {
        'txt': SAMPLE_TEXT_CONTENT,
        'pdf': SAMPLE_PDF_CONTENT,
        'docx': SAMPLE_TEXT_CONTENT  # Reutilizar contenido de texto
    }
    
    return content_map.get(file_type, SAMPLE_TEXT_CONTENT)

def create_temp_test_file(temp_dir: Path, filename: str, content: str) -> Path:
    """
    Crea un archivo temporal para pruebas.
    
    Args:
        temp_dir: Directorio temporal
        filename: Nombre del archivo
        content: Contenido del archivo
        
    Returns:
        Path del archivo creado
    """
    file_path = temp_dir / filename
    file_path.write_text(content, encoding='utf-8')
    return file_path

def get_mock_streamlit_session():
    """
    Retorna un mock de streamlit session_state para pruebas.
    
    Returns:
        Diccionario simulando session_state
    """
    return {
        'messages': SAMPLE_CHAT_MESSAGES.copy(),
        'selected_model': 'llama2:7b',
        'rag_enabled': True,
        'processed_documents': ['test_doc.pdf'],
        'analysis_cache': {}
    }