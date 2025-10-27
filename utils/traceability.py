"""
Sistema de trazabilidad completa para el RAG
Registra chunks recuperados, documentos fuente y metadatos
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from config.settings import config
from utils.logger import setup_logger

logger = setup_logger("Traceability")

@dataclass
class ChunkTrace:
    """Información de trazabilidad de un chunk"""
    chunk_id: str
    content: str
    source_file: str
    chunk_index: int
    similarity_score: float
    metadata: Dict[str, Any]
    retrieved_at: str

@dataclass
class QueryTrace:
    """Información de trazabilidad de una consulta completa"""
    query_id: str
    query: str
    response: str
    model_used: str
    chunks_retrieved: List[ChunkTrace]
    total_chunks: int
    total_context_length: int
    processing_time: float
    timestamp: str
    debug_info: Optional[Dict[str, Any]] = None

class TraceabilityManager:
    """Gestor de trazabilidad para el sistema RAG"""
    
    def __init__(self):
        self.chunks_log_file = config.LOGS_DIR / "retrieved_chunks.log"
        self.history_file = config.LOGS_DIR / "query_history.json"
        self.debug_log_file = config.LOGS_DIR / "debug_traces.log"
        
        # Configurar logger específico para chunks
        self.chunks_logger = self._setup_chunks_logger()
        
        # Cargar historial existente
        self.query_history = self._load_history()
    
    def _setup_chunks_logger(self) -> logging.Logger:
        """Configurar logger específico para chunks recuperados"""
        chunks_logger = logging.getLogger("ChunksRetrieval")
        chunks_logger.setLevel(logging.INFO)
        
        # Evitar duplicar handlers
        if not chunks_logger.handlers:
            handler = logging.FileHandler(self.chunks_log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - CHUNK_RETRIEVAL - %(message)s'
            )
            handler.setFormatter(formatter)
            chunks_logger.addHandler(handler)
        
        return chunks_logger
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Cargar historial de consultas existente"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error al cargar historial: {e}")
            return []
    
    def _save_history(self):
        """Guardar historial de consultas"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.query_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar historial: {e}")
    
    def log_chunk_retrieval(self, 
                           query: str, 
                           chunks_with_scores: List[Tuple[Any, float]],
                           query_id: str = None) -> str:
        """
        Registrar chunks recuperados para una consulta
        
        Args:
            query: Consulta del usuario
            chunks_with_scores: Lista de tuplas (chunk, similarity_score)
            query_id: ID único de la consulta
        
        Returns:
            ID de la consulta generado
        """
        if not config.ENABLE_CHUNK_LOGGING:
            return query_id or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            query_id = query_id or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            timestamp = datetime.now().isoformat()
            
            # Crear trazas de chunks
            chunk_traces = []
            for chunk, score in chunks_with_scores:
                chunk_trace = ChunkTrace(
                    chunk_id=chunk.id,
                    content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    similarity_score=score,
                    metadata=chunk.metadata,
                    retrieved_at=timestamp
                )
                chunk_traces.append(chunk_trace)
            
            # Log detallado de chunks
            log_entry = {
                "query_id": query_id,
                "query": query,
                "timestamp": timestamp,
                "total_chunks_retrieved": len(chunk_traces),
                "chunks": [asdict(trace) for trace in chunk_traces]
            }
            
            self.chunks_logger.info(json.dumps(log_entry, ensure_ascii=False))
            
            # Log resumido para facilitar lectura
            sources = list(set(trace.source_file for trace in chunk_traces))
            self.chunks_logger.info(
                f"RESUMEN - Query: {query[:100]}... | "
                f"Chunks: {len(chunk_traces)} | "
                f"Fuentes: {', '.join(sources)} | "
                f"Scores: {[f'{t.similarity_score:.3f}' for t in chunk_traces[:3]]}"
            )
            
            return query_id
            
        except Exception as e:
            logger.error(f"Error al registrar chunks: {e}")
            return query_id or f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def log_complete_query(self, 
                          query: str,
                          response: str,
                          model_used: str,
                          chunks_with_scores: List[Tuple[Any, float]],
                          processing_time: float,
                          debug_info: Dict[str, Any] = None) -> str:
        """
        Registrar una consulta completa con toda su información
        
        Args:
            query: Consulta del usuario
            response: Respuesta generada
            model_used: Modelo utilizado
            chunks_with_scores: Chunks recuperados con scores
            processing_time: Tiempo de procesamiento
            debug_info: Información adicional de debug
        
        Returns:
            ID de la consulta
        """
        if not config.ENABLE_HISTORY_TRACKING:
            return f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Crear trazas de chunks
            chunk_traces = []
            total_context_length = 0
            
            for chunk, score in chunks_with_scores:
                chunk_trace = ChunkTrace(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    similarity_score=score,
                    metadata=chunk.metadata,
                    retrieved_at=datetime.now().isoformat()
                )
                chunk_traces.append(chunk_trace)
                total_context_length += len(chunk.content)
            
            # Crear traza completa de la consulta
            query_trace = QueryTrace(
                query_id=query_id,
                query=query,
                response=response,
                model_used=model_used,
                chunks_retrieved=chunk_traces,
                total_chunks=len(chunk_traces),
                total_context_length=total_context_length,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                debug_info=debug_info
            )
            
            # Agregar al historial
            self.query_history.append(asdict(query_trace))
            
            # Mantener solo las últimas 1000 consultas
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]
            
            # Guardar historial
            self._save_history()
            
            # Log de chunks si está habilitado
            if config.ENABLE_CHUNK_LOGGING:
                self.log_chunk_retrieval(query, chunks_with_scores, query_id)
            
            logger.info(f"Consulta completa registrada: {query_id}")
            return query_id
            
        except Exception as e:
            logger.error(f"Error al registrar consulta completa: {e}")
            return f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_debug_info(self, 
                      query: str,
                      chunks_with_scores: List[Tuple[Any, float]],
                      context: str,
                      model_used: str) -> Dict[str, Any]:
        """
        Generar información de debug para una consulta
        
        Args:
            query: Consulta del usuario
            chunks_with_scores: Chunks recuperados con scores
            context: Contexto completo generado
            model_used: Modelo utilizado
        
        Returns:
            Diccionario con información de debug
        """
        try:
            sources_info = {}
            for chunk, score in chunks_with_scores:
                source = chunk.source_file
                if source not in sources_info:
                    sources_info[source] = {
                        'chunks_count': 0,
                        'total_length': 0,
                        'avg_score': 0,
                        'scores': []
                    }
                
                sources_info[source]['chunks_count'] += 1
                sources_info[source]['total_length'] += len(chunk.content)
                sources_info[source]['scores'].append(score)
            
            # Calcular promedios
            for source_info in sources_info.values():
                source_info['avg_score'] = sum(source_info['scores']) / len(source_info['scores'])
                source_info['avg_chunk_length'] = source_info['total_length'] / source_info['chunks_count']
            
            debug_info = {
                'query_length': len(query),
                'context_length': len(context),
                'total_chunks_retrieved': len(chunks_with_scores),
                'model_used': model_used,
                'sources_breakdown': sources_info,
                'similarity_scores': [score for _, score in chunks_with_scores],
                'config_used': {
                    'chunk_size': config.CHUNK_SIZE,
                    'chunk_overlap': config.CHUNK_OVERLAP,
                    'max_retrieval_docs': config.MAX_RETRIEVAL_DOCS,
                    'similarity_threshold': config.SIMILARITY_THRESHOLD,
                    'max_response_tokens': config.MAX_RESPONSE_TOKENS
                }
            }
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error al generar debug info: {e}")
            return {'error': str(e)}
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtener historial de consultas recientes
        
        Args:
            limit: Número máximo de consultas a retornar
        
        Returns:
            Lista de consultas recientes
        """
        return self.query_history[-limit:] if self.query_history else []
    
    def save_query_history(self,
                          query: str,
                          response: str,
                          query_id: str,
                          sources: List[str],
                          chunks_count: int,
                          model_used: str,
                          max_tokens: int,
                          rag_enabled: bool) -> None:
        """
        Guardar una consulta en el historial de manera simplificada
        
        Args:
            query: Consulta del usuario
            response: Respuesta generada
            query_id: ID único de la consulta
            sources: Lista de archivos fuente utilizados
            chunks_count: Número de chunks recuperados
            model_used: Modelo utilizado
            max_tokens: Tokens máximos configurados
            rag_enabled: Si RAG estaba habilitado
        """
        if not config.ENABLE_HISTORY_TRACKING:
            return
        
        try:
            query_entry = {
                "query_id": query_id,
                "query": query,
                "response": response[:500] + "..." if len(response) > 500 else response,  # Limitar tamaño
                "timestamp": datetime.now().isoformat(),
                "model_used": model_used,
                "max_tokens": max_tokens,
                "rag_enabled": rag_enabled,
                "sources": sources,
                "chunks_count": chunks_count,
                "query_length": len(query),
                "response_length": len(response)
            }
            
            # Agregar al historial
            self.query_history.append(query_entry)
            
            # Mantener solo las últimas 1000 consultas
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]
            
            # Guardar historial
            self._save_history()
            
            logger.info(f"Historial de consulta guardado: {query_id}")
            
        except Exception as e:
            logger.error(f"Error al guardar historial de consulta: {e}")

    def get_sources_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de uso de fuentes
        
        Returns:
            Estadísticas de documentos fuente
        """
        try:
            sources_stats = {}
            total_queries = len(self.query_history)
            
            for query_data in self.query_history:
                chunks = query_data.get('chunks_retrieved', [])
                for chunk in chunks:
                    source = chunk.get('source_file', 'unknown')
                    if source not in sources_stats:
                        sources_stats[source] = {
                            'usage_count': 0,
                            'total_chunks_used': 0,
                            'avg_similarity': 0,
                            'similarities': []
                        }
                    
                    sources_stats[source]['usage_count'] += 1
                    sources_stats[source]['total_chunks_used'] += 1
                    similarity = chunk.get('similarity_score', 0)
                    sources_stats[source]['similarities'].append(similarity)
            
            # Calcular promedios
            for source, stats in sources_stats.items():
                if stats['similarities']:
                    stats['avg_similarity'] = sum(stats['similarities']) / len(stats['similarities'])
                stats['usage_percentage'] = (stats['usage_count'] / total_queries * 100) if total_queries > 0 else 0
            
            return {
                'total_queries': total_queries,
                'sources_stats': sources_stats,
                'most_used_source': max(sources_stats.items(), key=lambda x: x[1]['usage_count'])[0] if sources_stats else None
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas de fuentes: {e}")
            return {'error': str(e)}

# Instancia global del gestor de trazabilidad
traceability_manager = TraceabilityManager()