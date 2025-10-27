"""
Configuración de Pytest para Pruebas Unitarias - CogniChat

Este archivo contiene la configuración global de pytest y fixtures
compartidos para todas las pruebas unitarias del proyecto.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Fixture que proporciona la ruta raíz del proyecto."""
    return project_root

@pytest.fixture(scope="function")
def mock_streamlit():
    """Fixture que mockea streamlit para pruebas unitarias."""
    with patch('streamlit.session_state', {}):
        with patch('streamlit.sidebar'):
            with patch('streamlit.columns'):
                with patch('streamlit.container'):
                    yield

@pytest.fixture(scope="function")
def mock_ollama_client():
    """Fixture que mockea el cliente Ollama."""
    mock_client = Mock()
    mock_client.get_available_models.return_value = [
        {"name": "llama2:7b"},
        {"name": "mistral:7b"}
    ]
    mock_client.is_connected.return_value = True
    mock_client.get_model_info.return_value = {"status": "available"}
    
    with patch('utils.ollama_client.OllamaClient', return_value=mock_client):
        yield mock_client

@pytest.fixture(scope="function")
def sample_pdf_content():
    """Fixture que proporciona contenido de PDF de ejemplo."""
    return "Este es un contenido de ejemplo de un documento PDF para pruebas unitarias."

@pytest.fixture(scope="function")
def sample_text_content():
    """Fixture que proporciona contenido de texto de ejemplo."""
    return """
    Este es un texto de ejemplo para análisis cualitativo.
    Contiene múltiples párrafos y conceptos clave.
    
    Los conceptos principales incluyen:
    - Análisis de datos
    - Procesamiento de lenguaje natural
    - Inteligencia artificial
    - Aprendizaje automático
    """

@pytest.fixture(scope="function")
def temp_directory(tmp_path):
    """Fixture que proporciona un directorio temporal para pruebas."""
    return tmp_path

@pytest.fixture(scope="function")
def mock_rag_processor():
    """Fixture que mockea el procesador RAG."""
    mock_processor = Mock()
    mock_processor.process_documents.return_value = {
        "processed_count": 1,
        "total_chunks": 5,
        "status": "success"
    }
    mock_processor.search.return_value = [
        {"content": "Resultado de búsqueda 1", "score": 0.95},
        {"content": "Resultado de búsqueda 2", "score": 0.87}
    ]
    return mock_processor

# Configuración de pytest
def pytest_configure(config):
    """Configuración personalizada de pytest."""
    config.addinivalue_line(
        "markers", "unit: marca las pruebas como pruebas unitarias"
    )
    config.addinivalue_line(
        "markers", "slow: marca las pruebas que tardan más tiempo"
    )
    config.addinivalue_line(
        "markers", "integration: marca las pruebas de integración"
    )

def pytest_collection_modifyitems(config, items):
    """Modifica la colección de pruebas para agregar marcadores automáticamente."""
    for item in items:
        # Agregar marcador 'unit' a todas las pruebas en unit_tests/
        if "unit_tests" in str(item.fspath):
            item.add_marker(pytest.mark.unit)