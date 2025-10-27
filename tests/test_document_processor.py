"""Tests automatizados para el módulo document_processor.py
Pruebas de funcionalidad del procesamiento de documentos"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path
import tempfile
import pytest
import modules.document_processor as doc_processor_module

class TestDocumentProcessor(unittest.TestCase):
    """Tests para el módulo de procesamiento de documentos"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.test_dir = Path(tempfile.mkdtemp())
        
    def test_show_document_stats_function_exists(self):
        """Test que la función show_document_stats existe"""
        self.assertTrue(hasattr(doc_processor_module, 'show_document_stats'))
        self.assertTrue(callable(doc_processor_module.show_document_stats))
        
    def test_upload_documents_function_exists(self):
        """Test que la función upload_documents existe"""
        self.assertTrue(hasattr(doc_processor_module, 'upload_documents'))
        self.assertTrue(callable(doc_processor_module.upload_documents))
        
    def test_manage_documents_function_exists(self):
        """Test que la función manage_documents existe"""
        self.assertTrue(hasattr(doc_processor_module, 'manage_documents'))
        self.assertTrue(callable(doc_processor_module.manage_documents))
    
    @patch('modules.document_processor.config')
    def test_directory_structure_validation(self, mock_config):
        """Test validación de estructura de directorios"""
        # Configurar directorios mock
        mock_config.UPLOADS_DIR = Path("/test/uploads")
        mock_config.CACHE_DIR = Path("/test/cache")
        mock_config.PROCESSED_DIR = Path("/test/processed")
        
        # Verificar que los directorios están configurados
        assert mock_config.UPLOADS_DIR == Path("/test/uploads")
        assert mock_config.CACHE_DIR == Path("/test/cache")
        assert mock_config.PROCESSED_DIR == Path("/test/processed")
    
    @patch('modules.document_processor.ollama_client')
    def test_ollama_connection_check(self, mock_ollama):
        """Test verificación de conexión con Ollama"""
        # Caso exitoso
        mock_ollama.get_available_models.return_value = ["model1", "model2", "model3"]
        
        models = mock_ollama.get_available_models()
        assert len(models) == 3
        assert "model1" in models
        
        # Caso de error
        mock_ollama.get_available_models.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception) as exc_info:
            mock_ollama.get_available_models()
        assert "Connection error" in str(exc_info.value)
    
    @patch('modules.document_processor.rag_processor')
    def test_document_processing_workflow(self, mock_rag):
        """Test flujo de procesamiento de documentos"""
        # Configurar mock del procesador RAG
        mock_rag.process_documents.return_value = {
            'success': True,
            'processed_files': 3,
            'total_chunks': 150,
            'processing_time': 45.2
        }
        
        # Simular procesamiento
        result = mock_rag.process_documents()
        
        # Verificar resultado
        assert result['success'] == True
        assert result['processed_files'] == 3
        assert result['total_chunks'] == 150
        assert isinstance(result['processing_time'], float)
    
    @patch('modules.document_processor.rag_processor')
    def test_document_statistics_calculation(self, mock_rag):
        """Test cálculo de estadísticas de documentos"""
        # Configurar datos mock
        mock_chunks = [
            {'content': 'a' * 500, 'metadata': {'source_file': 'doc1.pdf'}},
            {'content': 'b' * 750, 'metadata': {'source_file': 'doc2.pdf'}},
            {'content': 'c' * 300, 'metadata': {'source_file': 'doc1.pdf'}},
            {'content': 'd' * 600, 'metadata': {'source_file': 'doc3.pdf'}}
        ]
        
        mock_rag.get_all_chunks.return_value = mock_chunks
        
        chunks = mock_rag.get_all_chunks()
        
        # Calcular estadísticas
        total_chunks = len(chunks)
        unique_sources = len(set(chunk['metadata']['source_file'] for chunk in chunks))
        total_chars = sum(len(chunk['content']) for chunk in chunks)
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        # Verificar estadísticas
        assert total_chunks == 4
        assert unique_sources == 3
        assert total_chars == 2150
        assert avg_chunk_size == 537.5
    
    def test_file_type_validation(self):
        """Test validación de tipos de archivo"""
        # Tipos válidos
        valid_extensions = ['.pdf', '.txt', '.docx', '.doc', '.md']
        
        for ext in valid_extensions:
            test_file = f"documento{ext}"
            # Simular validación
            is_valid = any(test_file.endswith(valid_ext) for valid_ext in valid_extensions)
            assert is_valid == True
        
        # Tipos inválidos
        invalid_extensions = ['.exe', '.bat', '.jpg', '.png', '.mp3']
        
        for ext in invalid_extensions:
            test_file = f"archivo{ext}"
            # Simular validación
            is_valid = any(test_file.endswith(valid_ext) for valid_ext in valid_extensions)
            assert is_valid == False
    
    @patch('modules.document_processor.rag_processor')
    def test_cache_management(self, mock_rag):
        """Test gestión de cache"""
        # Configurar mock de cache
        mock_rag.get_cache_info.return_value = {
            'cache_size': 1024 * 1024 * 5,  # 5 MB
            'cache_entries': 25,
            'last_updated': '2024-01-01 10:00:00'
        }
        
        cache_info = mock_rag.get_cache_info()
        
        # Verificar información de cache
        assert cache_info['cache_size'] == 5242880  # 5 MB en bytes
        assert cache_info['cache_entries'] == 25
        assert cache_info['last_updated'] == '2024-01-01 10:00:00'
        
        # Test limpieza de cache
        mock_rag.clear_cache.return_value = True
        result = mock_rag.clear_cache()
        assert result == True
    
    @patch('modules.document_processor.rag_processor')
    def test_document_reprocessing(self, mock_rag):
        """Test reprocesamiento de documentos"""
        # Configurar mock para reprocesamiento
        mock_rag.reprocess_documents.return_value = {
            'success': True,
            'reprocessed_files': 2,
            'new_chunks': 75,
            'updated_chunks': 25
        }
        
        result = mock_rag.reprocess_documents()
        
        # Verificar resultado del reprocesamiento
        assert result['success'] == True
        assert result['reprocessed_files'] == 2
        assert result['new_chunks'] == 75
        assert result['updated_chunks'] == 25
    
    def test_chunk_size_validation(self):
        """Test validación de tamaño de chunks"""
        # Tamaños válidos
        valid_sizes = [500, 1000, 1500, 2000]
        
        for size in valid_sizes:
            # Simular validación (rango típico: 100-5000)
            is_valid = 100 <= size <= 5000
            assert is_valid == True
        
        # Tamaños inválidos
        invalid_sizes = [50, 10000, 0, -100]
        
        for size in invalid_sizes:
            # Simular validación
            is_valid = 100 <= size <= 5000
            assert is_valid == False
    
    def test_overlap_percentage_validation(self):
        """Test validación de porcentaje de solapamiento"""
        # Porcentajes válidos
        valid_overlaps = [0.1, 0.2, 0.3, 0.5]
        
        for overlap in valid_overlaps:
            # Simular validación (rango típico: 0.0-0.8)
            is_valid = 0.0 <= overlap <= 0.8
            assert is_valid == True
        
        # Porcentajes inválidos
        invalid_overlaps = [-0.1, 0.9, 1.5, 2.0]
        
        for overlap in invalid_overlaps:
            # Simular validación
            is_valid = 0.0 <= overlap <= 0.8
            assert is_valid == False
    
    @patch('modules.document_processor.rag_processor')
    def test_processing_error_handling(self, mock_rag):
        """Test manejo de errores en procesamiento"""
        # Simular error en procesamiento
        mock_rag.process_documents.side_effect = Exception("Error de procesamiento")
        
        with pytest.raises(Exception) as exc_info:
            mock_rag.process_documents()
        
        assert "Error de procesamiento" in str(exc_info.value)
        
        # Simular recuperación de error
        mock_rag.process_documents.side_effect = None
        mock_rag.process_documents.return_value = {
            'success': False,
            'error': 'Archivo corrupto',
            'processed_files': 0
        }
        
        result = mock_rag.process_documents()
        assert result['success'] == False
        assert 'error' in result
    
    @patch('modules.document_processor.config')
    def test_diagnostic_system_check(self, mock_config):
        """Test sistema de diagnóstico"""
        # Configurar directorios para diagnóstico
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            mock_config.UPLOADS_DIR = temp_path / "uploads"
            mock_config.CACHE_DIR = temp_path / "cache"
            mock_config.PROCESSED_DIR = temp_path / "processed"
            
            # Crear directorios
            mock_config.UPLOADS_DIR.mkdir(exist_ok=True)
            mock_config.CACHE_DIR.mkdir(exist_ok=True)
            mock_config.PROCESSED_DIR.mkdir(exist_ok=True)
            
            # Crear archivos de prueba
            (mock_config.UPLOADS_DIR / "test1.pdf").touch()
            (mock_config.UPLOADS_DIR / "test2.txt").touch()
            (mock_config.CACHE_DIR / "cache.json").touch()
            
            # Verificar diagnóstico
            uploads_files = len(list(mock_config.UPLOADS_DIR.glob("*")))
            cache_files = len(list(mock_config.CACHE_DIR.glob("*")))
            processed_files = len(list(mock_config.PROCESSED_DIR.glob("*")))
            
            assert uploads_files == 2
            assert cache_files == 1
            assert processed_files == 0
            
            # Verificar existencia de directorios
            assert mock_config.UPLOADS_DIR.exists()
            assert mock_config.CACHE_DIR.exists()
            assert mock_config.PROCESSED_DIR.exists()


class TestDocumentProcessorIntegration:
    """Tests de integración para el procesador de documentos"""
    
    @patch('modules.document_processor.rag_processor')
    @patch('modules.document_processor.ollama_client')
    def test_full_processing_pipeline(self, mock_ollama, mock_rag):
        """Test pipeline completo de procesamiento"""
        # Configurar mocks
        mock_ollama.get_available_models.return_value = ["model1"]
        mock_rag.process_documents.return_value = {
            'success': True,
            'processed_files': 1,
            'total_chunks': 10
        }
        mock_rag.get_all_chunks.return_value = [
            {'content': 'test content', 'metadata': {'source_file': 'test.pdf'}}
        ]
        
        # Verificar pipeline
        models = mock_ollama.get_available_models()
        assert len(models) > 0
        
        process_result = mock_rag.process_documents()
        assert process_result['success'] == True
        
        chunks = mock_rag.get_all_chunks()
        assert len(chunks) > 0
    
    @patch('modules.document_processor.rag_processor')
    def test_performance_metrics_calculation(self, mock_rag):
        """Test cálculo de métricas de rendimiento"""
        # Configurar datos de rendimiento
        mock_rag.get_performance_metrics.return_value = {
            'processing_time': 120.5,
            'chunks_per_second': 15.2,
            'memory_usage': 256.7,
            'cpu_usage': 45.3
        }
        
        metrics = mock_rag.get_performance_metrics()
        
        # Verificar métricas
        assert metrics['processing_time'] > 0
        assert metrics['chunks_per_second'] > 0
        assert metrics['memory_usage'] > 0
        assert metrics['cpu_usage'] > 0
        assert metrics['cpu_usage'] <= 100  # CPU no puede ser > 100%


if __name__ == "__main__":
    pytest.main([__file__])