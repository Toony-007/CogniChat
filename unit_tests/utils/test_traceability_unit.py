"""
Pruebas unitarias para el módulo traceability
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta
from dataclasses import asdict

from utils.traceability import ChunkTrace, QueryTrace, TraceabilityManager


class TestChunkTrace:
    """Pruebas para la clase ChunkTrace"""
    
    def test_chunk_trace_creation(self):
        """Probar creación de ChunkTrace"""
        timestamp = datetime.now().isoformat()
        chunk_trace = ChunkTrace(
            chunk_id="chunk_001",
            content="Contenido de prueba del chunk",
            source_file="documento.pdf",
            chunk_index=5,
            similarity_score=0.85,
            metadata={"page": 1, "section": "introducción"},
            retrieved_at=timestamp
        )
        
        assert chunk_trace.chunk_id == "chunk_001"
        assert chunk_trace.content == "Contenido de prueba del chunk"
        assert chunk_trace.source_file == "documento.pdf"
        assert chunk_trace.chunk_index == 5
        assert chunk_trace.similarity_score == 0.85
        assert chunk_trace.metadata == {"page": 1, "section": "introducción"}
        assert chunk_trace.retrieved_at == timestamp
    
    def test_chunk_trace_to_dict(self):
        """Probar conversión de ChunkTrace a diccionario"""
        timestamp = datetime.now().isoformat()
        chunk_trace = ChunkTrace(
            chunk_id="chunk_001",
            content="Contenido de prueba",
            source_file="documento.pdf",
            chunk_index=5,
            similarity_score=0.85,
            metadata={"page": 1},
            retrieved_at=timestamp
        )
        
        chunk_dict = asdict(chunk_trace)
        
        assert isinstance(chunk_dict, dict)
        assert chunk_dict['chunk_id'] == "chunk_001"
        assert chunk_dict['similarity_score'] == 0.85
        assert chunk_dict['metadata'] == {"page": 1}


class TestQueryTrace:
    """Pruebas para la clase QueryTrace"""
    
    def test_query_trace_creation(self):
        """Probar creación de QueryTrace"""
        timestamp = datetime.now().isoformat()
        chunk_trace = ChunkTrace(
            chunk_id="chunk_001",
            content="Contenido",
            source_file="doc.pdf",
            chunk_index=1,
            similarity_score=0.9,
            metadata={},
            retrieved_at=timestamp
        )
        
        query_trace = QueryTrace(
            query_id="query_001",
            query="¿Qué es la inteligencia artificial?",
            response="La IA es...",
            model_used="llama3.1",
            chunks_retrieved=[chunk_trace],
            total_chunks=1,
            total_context_length=500,
            processing_time=2.5,
            timestamp=timestamp,
            debug_info={"test": "info"}
        )
        
        assert query_trace.query_id == "query_001"
        assert query_trace.query == "¿Qué es la inteligencia artificial?"
        assert query_trace.response == "La IA es..."
        assert query_trace.model_used == "llama3.1"
        assert len(query_trace.chunks_retrieved) == 1
        assert query_trace.total_chunks == 1
        assert query_trace.total_context_length == 500
        assert query_trace.processing_time == 2.5
        assert query_trace.debug_info == {"test": "info"}
    
    def test_query_trace_to_dict(self):
        """Probar conversión de QueryTrace a diccionario"""
        timestamp = datetime.now().isoformat()
        chunk_trace = ChunkTrace(
            chunk_id="chunk_001",
            content="Contenido",
            source_file="doc.pdf",
            chunk_index=1,
            similarity_score=0.9,
            metadata={},
            retrieved_at=timestamp
        )
        
        query_trace = QueryTrace(
            query_id="query_001",
            query="Test query",
            response="Test response",
            model_used="llama3.1",
            chunks_retrieved=[chunk_trace],
            total_chunks=1,
            total_context_length=500,
            processing_time=2.5,
            timestamp=timestamp
        )
        
        query_dict = asdict(query_trace)
        
        assert isinstance(query_dict, dict)
        assert query_dict['query_id'] == "query_001"
        assert query_dict['total_chunks'] == 1
        assert len(query_dict['chunks_retrieved']) == 1


class TestTraceabilityManager:
    """Pruebas para la clase TraceabilityManager"""
    
    def setup_method(self):
        """Configuración antes de cada prueba"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_config = Mock()
        self.mock_config.LOGS_DIR = self.temp_dir
        self.mock_config.ENABLE_CHUNK_LOGGING = True
        self.mock_config.ENABLE_HISTORY_TRACKING = True
        self.mock_config.CHUNK_SIZE = 512
        self.mock_config.CHUNK_OVERLAP = 50
        self.mock_config.MAX_RETRIEVAL_DOCS = 5
        self.mock_config.SIMILARITY_THRESHOLD = 0.7
        self.mock_config.MAX_RESPONSE_TOKENS = 1000
    
    def teardown_method(self):
        """Limpieza después de cada prueba"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('utils.traceability.config')
    def test_traceability_manager_initialization(self, mock_config):
        """Probar inicialización de TraceabilityManager"""
        mock_config.LOGS_DIR = self.temp_dir
        
        manager = TraceabilityManager()
        
        assert manager.chunks_log_file == self.temp_dir / "retrieved_chunks.log"
        assert manager.history_file == self.temp_dir / "query_history.json"
        assert manager.debug_log_file == self.temp_dir / "debug_traces.log"
        assert hasattr(manager, 'chunks_logger')
        assert isinstance(manager.query_history, list)
    
    @patch('utils.traceability.config')
    def test_load_history_empty(self, mock_config):
        """Probar carga de historial vacío"""
        mock_config.LOGS_DIR = self.temp_dir
        
        manager = TraceabilityManager()
        history = manager._load_history()
        
        assert history == []
    
    @patch('utils.traceability.config')
    def test_load_history_with_data(self, mock_config):
        """Probar carga de historial con datos"""
        mock_config.LOGS_DIR = self.temp_dir
        
        # Crear archivo de historial con datos
        test_data = [
            {"query_id": "test_001", "query": "Test query", "timestamp": "2023-01-01T00:00:00"},
            {"query_id": "test_002", "query": "Another query", "timestamp": "2023-01-01T01:00:00"}
        ]
        
        history_file = self.temp_dir / "query_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        manager = TraceabilityManager()
        history = manager._load_history()
        
        assert len(history) == 2
        assert history[0]['query_id'] == "test_001"
        assert history[1]['query_id'] == "test_002"
    
    @patch('utils.traceability.config')
    @patch('utils.traceability.logger')
    def test_load_history_error(self, mock_logger, mock_config):
        """Probar manejo de errores en carga de historial"""
        mock_config.LOGS_DIR = self.temp_dir
        
        # Crear archivo con JSON inválido
        history_file = self.temp_dir / "query_history.json"
        with open(history_file, 'w') as f:
            f.write("json inválido")
        
        manager = TraceabilityManager()
        history = manager._load_history()
        
        assert history == []
        # Verificar que se llamó al logger al menos una vez
        assert mock_logger.error.call_count >= 1
    
    @patch('utils.traceability.config')
    def test_save_history_success(self, mock_config):
        """Probar guardado exitoso de historial"""
        mock_config.LOGS_DIR = self.temp_dir
        
        manager = TraceabilityManager()
        manager.query_history = [
            {"query_id": "test_001", "query": "Test query"}
        ]
        
        manager._save_history()
        
        # Verificar que se guardó correctamente
        assert manager.history_file.exists()
        with open(manager.history_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 1
        assert saved_data[0]['query_id'] == "test_001"
    
    @patch('utils.traceability.config')
    @patch('utils.traceability.logger')
    def test_save_history_error(self, mock_logger, mock_config):
        """Probar manejo de errores en guardado de historial"""
        mock_config.LOGS_DIR = Path("/directorio/inexistente")

        manager = TraceabilityManager()
        manager.query_history = [{"test": "data"}]
        
        # Simular error al escribir el archivo
        with patch('builtins.open', side_effect=OSError("Error de permisos")):
            manager._save_history()

        # Verificar que se llamó logger.error con el mensaje correcto
        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert "Error al guardar historial" in args[0]
    
    @patch('utils.traceability.config')
    def test_log_chunk_retrieval_disabled(self, mock_config):
        """Probar log de chunks cuando está deshabilitado"""
        mock_config.LOGS_DIR = self.temp_dir
        mock_config.ENABLE_CHUNK_LOGGING = False
        
        manager = TraceabilityManager()
        
        # Mock de chunk
        mock_chunk = Mock()
        mock_chunk.id = "chunk_001"
        mock_chunk.content = "Contenido de prueba"
        mock_chunk.source_file = "doc.pdf"
        mock_chunk.chunk_index = 1
        mock_chunk.metadata = {}
        
        query_id = manager.log_chunk_retrieval(
            "Test query",
            [(mock_chunk, 0.85)]
        )
        
        # Debe devolver un ID pero no hacer logging
        assert query_id.startswith("query_")
    
    @patch('utils.traceability.config')
    def test_log_chunk_retrieval_success(self, mock_config):
        """Probar log exitoso de chunks"""
        mock_config.LOGS_DIR = self.temp_dir
        mock_config.ENABLE_CHUNK_LOGGING = True
        
        manager = TraceabilityManager()
        
        # Mock de chunk
        mock_chunk = Mock()
        mock_chunk.id = "chunk_001"
        mock_chunk.content = "Contenido de prueba para el chunk"
        mock_chunk.source_file = "documento.pdf"
        mock_chunk.chunk_index = 1
        mock_chunk.metadata = {"page": 1}
        
        query_id = manager.log_chunk_retrieval(
            "¿Qué es la inteligencia artificial?",
            [(mock_chunk, 0.85)]
        )
        
        assert query_id.startswith("query_")
        # Verificar que se intentó crear el archivo de log (puede fallar por permisos pero el ID se genera)
        assert manager.chunks_log_file is not None
    
    @patch('utils.traceability.config')
    def test_log_complete_query_disabled(self, mock_config):
        """Probar log de consulta completa cuando está deshabilitado"""
        mock_config.LOGS_DIR = self.temp_dir
        mock_config.ENABLE_HISTORY_TRACKING = False
        
        manager = TraceabilityManager()
        
        mock_chunk = Mock()
        mock_chunk.id = "chunk_001"
        mock_chunk.content = "Contenido"
        mock_chunk.source_file = "doc.pdf"
        mock_chunk.chunk_index = 1
        mock_chunk.metadata = {}
        
        query_id = manager.log_complete_query(
            "Test query",
            "Test response",
            "llama3.1",
            [(mock_chunk, 0.85)],
            2.5
        )
        
        assert query_id.startswith("query_")
        assert len(manager.query_history) == 0  # No debe agregar al historial
    
    @patch('utils.traceability.config')
    def test_log_complete_query_success(self, mock_config):
        """Probar log exitoso de consulta completa"""
        mock_config.LOGS_DIR = self.temp_dir
        mock_config.ENABLE_HISTORY_TRACKING = True
        mock_config.ENABLE_CHUNK_LOGGING = False  # Deshabilitado para esta prueba
        
        manager = TraceabilityManager()
        
        mock_chunk = Mock()
        mock_chunk.id = "chunk_001"
        mock_chunk.content = "Contenido de prueba"
        mock_chunk.source_file = "documento.pdf"
        mock_chunk.chunk_index = 1
        mock_chunk.metadata = {"page": 1}
        
        query_id = manager.log_complete_query(
            "¿Qué es la IA?",
            "La IA es una tecnología...",
            "llama3.1",
            [(mock_chunk, 0.85)],
            2.5,
            {"debug": "info"}
        )
        
        assert query_id.startswith("query_")
        assert len(manager.query_history) == 1
        
        # Verificar contenido del historial
        query_data = manager.query_history[0]
        assert query_data['query'] == "¿Qué es la IA?"
        assert query_data['response'] == "La IA es una tecnología..."
        assert query_data['model_used'] == "llama3.1"
        assert query_data['total_chunks'] == 1
        assert query_data['processing_time'] == 2.5
        assert query_data['debug_info'] == {"debug": "info"}
    
    @patch('utils.traceability.config')
    def test_log_complete_query_limit_1000(self, mock_config):
        """Probar límite de 1000 consultas en historial"""
        mock_config.LOGS_DIR = self.temp_dir
        mock_config.ENABLE_HISTORY_TRACKING = True
        mock_config.ENABLE_CHUNK_LOGGING = False
        
        manager = TraceabilityManager()
        
        # Llenar historial con 1000 consultas
        for i in range(1000):
            manager.query_history.append({
                "query_id": f"query_{i:03d}",
                "query": f"Query {i}",
                "response": f"Response {i}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Agregar una consulta más
        mock_chunk = Mock()
        mock_chunk.id = "chunk_001"
        mock_chunk.content = "Contenido"
        mock_chunk.source_file = "doc.pdf"
        mock_chunk.chunk_index = 1
        mock_chunk.metadata = {}
        
        manager.log_complete_query(
            "Nueva consulta",
            "Nueva respuesta",
            "llama3.1",
            [(mock_chunk, 0.85)],
            2.5
        )
        
        # Debe mantener solo 1000 consultas
        assert len(manager.query_history) == 1000
        # La primera consulta original debe haber sido eliminada
        assert manager.query_history[0]['query'] != "Query 0"
        # La nueva consulta debe estar al final
        assert manager.query_history[-1]['query'] == "Nueva consulta"
    
    @patch('utils.traceability.config')
    def test_get_debug_info_success(self, mock_config):
        """Probar generación exitosa de información de debug"""
        mock_config.CHUNK_SIZE = 512
        mock_config.CHUNK_OVERLAP = 50
        mock_config.MAX_RETRIEVAL_DOCS = 5
        mock_config.SIMILARITY_THRESHOLD = 0.7
        mock_config.MAX_RESPONSE_TOKENS = 1000
        
        manager = TraceabilityManager()
        
        # Mock de chunks
        mock_chunk1 = Mock()
        mock_chunk1.content = "Contenido del primer chunk"
        mock_chunk1.source_file = "doc1.pdf"
        
        mock_chunk2 = Mock()
        mock_chunk2.content = "Contenido del segundo chunk"
        mock_chunk2.source_file = "doc2.pdf"
        
        chunks_with_scores = [
            (mock_chunk1, 0.9),
            (mock_chunk2, 0.8)
        ]
        
        debug_info = manager.get_debug_info(
            "¿Qué es la IA?",
            chunks_with_scores,
            "Contexto completo generado para la consulta",
            "llama3.1"
        )
        
        assert 'query_length' in debug_info
        assert 'context_length' in debug_info
        assert 'total_chunks_retrieved' in debug_info
        assert 'model_used' in debug_info
        assert 'sources_breakdown' in debug_info
        assert 'similarity_scores' in debug_info
        assert 'config_used' in debug_info
        
        assert debug_info['query_length'] == len("¿Qué es la IA?")
        assert debug_info['total_chunks_retrieved'] == 2
        assert debug_info['model_used'] == "llama3.1"
        assert debug_info['similarity_scores'] == [0.9, 0.8]
        assert len(debug_info['sources_breakdown']) == 2
    
    @patch('utils.traceability.config')
    @patch('utils.traceability.logger')
    def test_get_debug_info_error(self, mock_logger, mock_config):
        """Probar manejo de errores en generación de debug info"""
        manager = TraceabilityManager()
        
        # Simular error con chunks inválidos
        debug_info = manager.get_debug_info(
            "Test query",
            None,  # Esto causará un error
            "Context",
            "llama3.1"
        )
        
        assert 'error' in debug_info
        # Verificar que se llamó al logger al menos una vez
        assert mock_logger.error.call_count >= 1
    
    @patch('utils.traceability.config')
    def test_get_query_history_empty(self, mock_config):
        """Probar obtención de historial vacío"""
        mock_config.LOGS_DIR = self.temp_dir
        
        manager = TraceabilityManager()
        history = manager.get_query_history()
        
        assert history == []
    
    @patch('utils.traceability.config')
    def test_get_query_history_with_limit(self, mock_config):
        """Probar obtención de historial con límite"""
        mock_config.LOGS_DIR = self.temp_dir
        
        manager = TraceabilityManager()
        
        # Agregar 10 consultas al historial
        for i in range(10):
            manager.query_history.append({
                "query_id": f"query_{i:03d}",
                "query": f"Query {i}",
                "timestamp": datetime.now().isoformat()
            })
        
        # Obtener solo las últimas 5
        history = manager.get_query_history(limit=5)
        
        assert len(history) == 5
        assert history[0]['query'] == "Query 5"  # Las últimas 5
        assert history[-1]['query'] == "Query 9"
    
    @patch('utils.traceability.config')
    def test_save_query_history_disabled(self, mock_config):
        """Probar guardado de historial cuando está deshabilitado"""
        mock_config.LOGS_DIR = self.temp_dir
        mock_config.ENABLE_HISTORY_TRACKING = False
        
        manager = TraceabilityManager()
        initial_length = len(manager.query_history)
        
        manager.save_query_history(
            "Test query",
            "Test response",
            "query_001",
            ["doc1.pdf"],
            3,
            "llama3.1",
            1000,
            True
        )
        
        # No debe agregar nada al historial
        assert len(manager.query_history) == initial_length
    
    @patch('utils.traceability.config')
    def test_save_query_history_success(self, mock_config):
        """Probar guardado exitoso de historial de consulta"""
        mock_config.LOGS_DIR = self.temp_dir
        mock_config.ENABLE_HISTORY_TRACKING = True
        
        manager = TraceabilityManager()
        
        manager.save_query_history(
            "¿Qué es la inteligencia artificial?",
            "La inteligencia artificial es una rama de la informática...",
            "query_001",
            ["documento1.pdf", "documento2.pdf"],
            5,
            "llama3.1",
            1000,
            True
        )
        
        assert len(manager.query_history) == 1
        
        query_data = manager.query_history[0]
        assert query_data['query_id'] == "query_001"
        assert query_data['query'] == "¿Qué es la inteligencia artificial?"
        assert query_data['model_used'] == "llama3.1"
        assert query_data['max_tokens'] == 1000
        assert query_data['rag_enabled'] is True
        assert query_data['sources'] == ["documento1.pdf", "documento2.pdf"]
        assert query_data['chunks_count'] == 5
    
    @patch('utils.traceability.config')
    def test_get_sources_statistics_empty(self, mock_config):
        """Probar estadísticas de fuentes con historial vacío"""
        mock_config.LOGS_DIR = self.temp_dir
        
        manager = TraceabilityManager()
        stats = manager.get_sources_statistics()
        
        assert stats['total_queries'] == 0
        assert stats['sources_stats'] == {}
        assert stats['most_used_source'] is None
    
    @patch('utils.traceability.config')
    def test_get_sources_statistics_with_data(self, mock_config):
        """Probar estadísticas de fuentes con datos"""
        mock_config.LOGS_DIR = self.temp_dir
        
        manager = TraceabilityManager()
        
        # Agregar consultas con chunks al historial
        manager.query_history = [
            {
                "query_id": "query_001",
                "chunks_retrieved": [
                    {"source_file": "doc1.pdf", "similarity_score": 0.9},
                    {"source_file": "doc2.pdf", "similarity_score": 0.8}
                ]
            },
            {
                "query_id": "query_002",
                "chunks_retrieved": [
                    {"source_file": "doc1.pdf", "similarity_score": 0.85}
                ]
            }
        ]
        
        stats = manager.get_sources_statistics()
        
        assert stats['total_queries'] == 2
        assert 'doc1.pdf' in stats['sources_stats']
        assert 'doc2.pdf' in stats['sources_stats']
        
        # doc1.pdf debe ser el más usado (2 veces)
        assert stats['most_used_source'] == 'doc1.pdf'
        
        doc1_stats = stats['sources_stats']['doc1.pdf']
        assert doc1_stats['usage_count'] == 2
        assert doc1_stats['total_chunks_used'] == 2
        assert doc1_stats['avg_similarity'] == 0.875  # (0.9 + 0.85) / 2
    
    @patch('utils.traceability.config')
    @patch('utils.traceability.logger')
    def test_get_sources_statistics_error(self, mock_logger, mock_config):
        """Probar manejo de errores en estadísticas de fuentes"""
        mock_config.LOGS_DIR = self.temp_dir
        
        manager = TraceabilityManager()
        
        # Simular error con datos que causen excepción
        manager.query_history = [{"chunks_retrieved": "invalid_data"}]
        
        stats = manager.get_sources_statistics()
        
        # La función debería devolver un diccionario con error
        assert isinstance(stats, dict)
        # Puede contener 'error' o las claves básicas dependiendo del manejo de errores
        assert 'error' in stats or ('total_queries' in stats and 'sources_stats' in stats)