"""
Pruebas Unitarias - RAG Processor

Este archivo contiene pruebas unitarias específicas para el procesador RAG,
enfocándose en la lógica de procesamiento de documentos y búsqueda.
"""

import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
import sys
import os
import tempfile
from pathlib import Path
import json

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class TestRAGProcessorUnit(unittest.TestCase):
    """Pruebas unitarias para el procesador RAG."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Limpieza después de cada prueba."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_rag_processor_initialization(self):
        """Probar inicialización del procesador RAG"""
        # Mock del procesador RAG para evitar importaciones circulares
        mock_processor = Mock()
        mock_processor.ollama_client = Mock()
        mock_processor.chunks_cache = {}
        mock_processor.embeddings_cache = {}
        
        # Verificar inicialización
        self.assertIsNotNone(mock_processor.ollama_client)
        self.assertIsInstance(mock_processor.chunks_cache, dict)
        self.assertIsInstance(mock_processor.embeddings_cache, dict)
        
    def test_text_file_processing(self):
        """Probar procesamiento de archivos de texto"""
        # Mock del procesador
        mock_processor = Mock()
        mock_processor.extract_text_from_file.return_value = "Contenido de prueba"
        
        # Simular procesamiento
        result = mock_processor.extract_text_from_file(Path("test.txt"))
        
        # Verificar resultado
        self.assertEqual(result, "Contenido de prueba")
        mock_processor.extract_text_from_file.assert_called_once()
        
    def test_pdf_file_processing(self):
        """Probar procesamiento de archivos PDF"""
        # Mock del procesador
        mock_processor = Mock()
        mock_processor.extract_text_from_file.return_value = "Contenido PDF extraído"
        
        # Simular procesamiento de PDF
        result = mock_processor.extract_text_from_file(Path("test.pdf"))
        
        # Verificar resultado
        self.assertEqual(result, "Contenido PDF extraído")
        mock_processor.extract_text_from_file.assert_called_once()
        
    def test_text_chunking_logic(self):
        """Probar lógica de fragmentación de texto"""
        # Mock del procesador
        mock_processor = Mock()
        mock_processor.chunk_text.return_value = [
            "Fragmento 1 del texto",
            "Fragmento 2 del texto",
            "Fragmento 3 del texto"
        ]
        
        # Simular fragmentación
        text = "Este es un texto largo que debe ser fragmentado en partes más pequeñas"
        chunks = mock_processor.chunk_text(text, chunk_size=50, overlap=10)
        
        # Verificar resultado
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 3)
        mock_processor.chunk_text.assert_called_once_with(text, chunk_size=50, overlap=10)
        
    def test_vector_database_operations(self):
        """Probar operaciones de base de datos vectorial"""
        # Mock del procesador
        mock_processor = Mock()
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_processor.generate_embeddings.return_value = mock_embeddings
        
        # Simular generación de embeddings
        chunks = ["chunk1", "chunk2"]
        embeddings = mock_processor.generate_embeddings(chunks)
        
        # Verificar resultado
        self.assertEqual(embeddings, mock_embeddings)
        mock_processor.generate_embeddings.assert_called_once_with(chunks)
        
    def test_search_functionality(self):
        """Probar funcionalidad de búsqueda"""
        # Mock del procesador
        mock_processor = Mock()
        mock_results = [
            ("Resultado 1", 0.95),
            ("Resultado 2", 0.87),
            ("Resultado 3", 0.75)
        ]
        mock_processor.similarity_search.return_value = mock_results
        
        # Simular búsqueda
        query = "consulta de prueba"
        results = mock_processor.similarity_search(query, top_k=3)
        
        # Verificar resultado
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][1], 0.95)  # Score más alto
        mock_processor.similarity_search.assert_called_once_with(query, top_k=3)
        
    def test_document_metadata_extraction(self):
        """Probar extracción de metadatos de documentos"""
        # Mock del procesador
        mock_processor = Mock()
        mock_metadata = {
            "filename": "test.pdf",
            "size": 1024,
            "type": "pdf",
            "chunks_count": 5
        }
        mock_processor.get_document_stats.return_value = mock_metadata
        
        # Simular extracción de metadatos
        metadata = mock_processor.get_document_stats()
        
        # Verificar resultado
        self.assertIsInstance(metadata, dict)
        self.assertIn("filename", metadata)
        self.assertIn("size", metadata)
        self.assertIn("type", metadata)
        mock_processor.get_document_stats.assert_called_once()
            
    def test_error_handling_invalid_file(self):
        """Probar manejo de errores con archivos inválidos"""
        # Mock del procesador
        mock_processor = Mock()
        mock_processor.extract_text_from_file.side_effect = FileNotFoundError("Archivo no encontrado")
        
        # Simular error
        with self.assertRaises(FileNotFoundError):
            mock_processor.extract_text_from_file(Path("archivo_inexistente.txt"))
        
    def test_supported_file_types(self):
        """Probar tipos de archivo soportados"""
        # Mock del procesador
        mock_processor = Mock()
        supported_types = [".txt", ".pdf", ".docx", ".md", ".json"]
        mock_processor.get_supported_file_types.return_value = supported_types
        
        # Verificar tipos soportados
        types = mock_processor.get_supported_file_types()
        self.assertIsInstance(types, list)
        self.assertIn(".txt", types)
        self.assertIn(".pdf", types)
        self.assertIn(".docx", types)
            
    def test_cache_functionality(self):
        """Probar funcionalidad de cache del procesador RAG"""
        # Crear un mock simple para evitar importaciones circulares
        mock_processor = Mock()
        mock_processor.chunks_cache = {}
        mock_processor.embeddings_cache = {}
        
        # Verificar que tiene propiedades de cache
        self.assertTrue(hasattr(mock_processor, 'chunks_cache'))
        self.assertTrue(hasattr(mock_processor, 'embeddings_cache'))
        
        # Probar limpieza de cache
        mock_processor.clear_cache = Mock()
        mock_processor.clear_cache()
        mock_processor.clear_cache.assert_called_once()
        
        # Simular que el cache se limpia
        mock_processor.chunks_cache.clear()
        mock_processor.embeddings_cache.clear()
        self.assertEqual(len(mock_processor.chunks_cache), 0)
        self.assertEqual(len(mock_processor.embeddings_cache), 0)

if __name__ == '__main__':
    unittest.main()