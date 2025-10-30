"""
Sistema RAG completo para procesamiento y análisis de documentos
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import logging

# Importaciones para procesamiento de texto
import re
from dataclasses import dataclass

# Importaciones para diferentes tipos de archivo
import docx
from pypdf import PdfReader
import pandas as pd
from bs4 import BeautifulSoup

from config.settings import config
from utils.logger import setup_logger
from utils.ollama_client import OllamaClient
from utils.error_handler import ErrorHandler
from utils.traceability import traceability_manager
from modules.document_upload import get_valid_uploaded_files

logger = setup_logger()
error_handler = ErrorHandler()

@dataclass
class DocumentChunk:
    """Representa un fragmento de documento procesado"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    source_file: str = ""
    chunk_index: int = 0
    
class RAGProcessor:
    """Procesador RAG completo para documentos"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.chunks_cache = {}
        self.embeddings_cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Cargar cache de embeddings y chunks"""
        try:
            cache_file = config.CACHE_DIR / "rag_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.chunks_cache = cache_data.get('chunks', {})
                    self.embeddings_cache = cache_data.get('embeddings', {})
                logger.info(f"Cache cargado: {len(self.chunks_cache)} chunks, {len(self.embeddings_cache)} embeddings")
        except Exception as e:
            logger.warning(f"No se pudo cargar cache: {e}")
    
    def _save_cache(self):
        """Guardar cache de embeddings y chunks"""
        try:
            cache_file = config.CACHE_DIR / "rag_cache.json"
            cache_data = {
                'chunks': self.chunks_cache,
                'embeddings': self.embeddings_cache,
                'last_updated': datetime.now().isoformat()
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info("Cache guardado exitosamente")
        except Exception as e:
            logger.error(f"Error al guardar cache: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Obtener hash del archivo para cache"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return str(file_path.stat().st_mtime)
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """Extraer texto de diferentes tipos de archivo"""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_ext == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_ext == '.pdf':
                return self._extract_from_pdf(file_path)
            
            elif file_ext in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            
            elif file_ext in ['.xlsx', '.xls']:
                return self._extract_from_excel(file_path)
            
            elif file_ext == '.csv':
                return self._extract_from_csv(file_path)
            
            elif file_ext in ['.html', '.htm']:
                return self._extract_from_html(file_path)
            
            elif file_ext == '.json':
                return self._extract_from_json(file_path)
            
            else:
                logger.warning(f"Tipo de archivo no soportado: {file_ext}")
                return ""
                
        except Exception as e:
            error_handler.handle_error(e, f"Error al extraer texto de {file_path}")
            return ""
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extraer texto de PDF"""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error al leer PDF {file_path}: {e}")
            return ""
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extraer texto de DOCX"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error al leer DOCX {file_path}: {e}")
            return ""
    
    def _extract_from_excel(self, file_path: Path) -> str:
        """Extraer texto de Excel de manera optimizada"""
        try:
            # Leer todas las hojas del Excel
            excel_file = pd.ExcelFile(file_path)
            all_text = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Leer hoja específica
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Filtrar filas y columnas vacías para optimizar
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if df.empty:
                        continue
                    
                    # Convertir a texto de manera más eficiente
                    sheet_text = f"=== HOJA: {sheet_name} ===\n"
                    
                    # Procesar por lotes para archivos grandes
                    batch_size = 1000
                    for i in range(0, len(df), batch_size):
                        batch_df = df.iloc[i:i+batch_size]
                        
                        # Convertir solo las columnas con datos
                        for col in batch_df.columns:
                            if batch_df[col].dtype == 'object':  # Solo columnas de texto
                                non_null_values = batch_df[col].dropna().astype(str)
                                if not non_null_values.empty:
                                    sheet_text += f"{col}: " + " | ".join(non_null_values.tolist()) + "\n"
                    
                    all_text.append(sheet_text)
                    
                except Exception as e:
                    logger.warning(f"Error procesando hoja {sheet_name} en {file_path}: {e}")
                    continue
            
            return "\n".join(all_text)
            
        except Exception as e:
            logger.error(f"Error al leer Excel {file_path}: {e}")
            return ""
    
    def _extract_from_csv(self, file_path: Path) -> str:
        """Extraer texto de CSV de manera optimizada"""
        try:
            # Leer CSV en chunks para archivos grandes
            chunk_size = 10000
            all_text = []
            
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
                # Filtrar filas y columnas vacías
                chunk_df = chunk_df.dropna(how='all').dropna(axis=1, how='all')
                
                if chunk_df.empty:
                    continue
                
                # Convertir solo columnas de texto
                for col in chunk_df.columns:
                    if chunk_df[col].dtype == 'object':  # Solo columnas de texto
                        non_null_values = chunk_df[col].dropna().astype(str)
                        if not non_null_values.empty:
                            all_text.append(f"{col}: " + " | ".join(non_null_values.tolist()))
            
            return "\n".join(all_text)
            
        except Exception as e:
            logger.error(f"Error al leer CSV {file_path}: {e}")
            return ""
    
    def _extract_from_html(self, file_path: Path) -> str:
        """Extraer texto de HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Error al leer HTML {file_path}: {e}")
            return ""
    
    def _extract_from_json(self, file_path: Path) -> str:
        """Extraer texto de JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error al leer JSON {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Dividir texto en chunks con overlap optimizado para archivos grandes"""
        if not text.strip():
            return []
        
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        # Limpiar texto de manera más eficiente
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Para archivos muy grandes, usar chunking más agresivo
        if len(text) > 1000000:  # Más de 1MB de texto
            chunk_size = min(chunk_size, 1500)  # Chunks más pequeños
            overlap = min(overlap, 200)  # Menos overlap
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Si no es el último chunk, buscar un punto de corte natural
            if end < len(text):
                # Buscar el último punto, salto de línea o espacio
                for separator in ['. ', '\n', ' ']:
                    last_sep = text.rfind(separator, start, end)
                    if last_sep > start:
                        end = last_sep + len(separator)
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  # Filtrar chunks muy pequeños
                chunks.append(chunk)
            
            start = max(start + 1, end - overlap)
        
        return chunks
    
    def process_document(self, file_path: Path) -> List[DocumentChunk]:
        """Procesar un documento completo con optimizaciones"""
        try:
            file_hash = self._get_file_hash(file_path)
            cache_key = f"{file_path.name}_{file_hash}"
            
            # Verificar cache
            if cache_key in self.chunks_cache:
                logger.info(f"Usando chunks en cache para {file_path.name}")
                cached_chunks = self.chunks_cache[cache_key]
                return [DocumentChunk(**chunk_data) for chunk_data in cached_chunks]
            
            logger.info(f"Iniciando procesamiento de {file_path.name}...")
            start_time = datetime.now()
            
            # Extraer texto
            text = self.extract_text_from_file(file_path)
            if not text:
                logger.warning(f"No se pudo extraer texto de {file_path.name}")
                return []
            
            logger.info(f"Texto extraído: {len(text)} caracteres")
            
            # Dividir en chunks
            text_chunks = self.chunk_text(text)
            logger.info(f"Texto dividido en {len(text_chunks)} chunks")
            
            # Crear objetos DocumentChunk de manera más eficiente
            document_chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunk_id = f"{file_path.stem}_{i}_{hashlib.md5(chunk_text.encode()).hexdigest()[:8]}"
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_text,
                    metadata={
                        'source_file': file_path.name,
                        'file_type': file_path.suffix,
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'file_size': file_path.stat().st_size,
                        'created_at': datetime.now().isoformat(),
                        'processing_time': (datetime.now() - start_time).total_seconds()
                    },
                    source_file=file_path.name,
                    chunk_index=i
                )
                document_chunks.append(chunk)
            
            # Guardar en cache de manera más eficiente
            cache_data = []
            for chunk in document_chunks:
                cache_data.append({
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'source_file': chunk.source_file,
                    'chunk_index': chunk.chunk_index
                })
            
            self.chunks_cache[cache_key] = cache_data
            self._save_cache()  # Guardar cache inmediatamente
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Documento {file_path.name} procesado: {len(document_chunks)} chunks en {processing_time:.2f}s")
            return document_chunks
            
        except Exception as e:
            error_handler.handle_error(e, f"Error al procesar documento {file_path}")
            return []
    
    def generate_embeddings(self, chunks: List[DocumentChunk], model: str = None) -> List[DocumentChunk]:
        """Generar embeddings para los chunks de manera optimizada"""
        model = model or config.DEFAULT_EMBEDDING_MODEL
        
        # Filtrar chunks que ya tienen embeddings
        chunks_to_process = []
        for chunk in chunks:
            embedding_key = f"{chunk.id}_{model}"
            if embedding_key in self.embeddings_cache:
                chunk.embedding = self.embeddings_cache[embedding_key]
            else:
                chunks_to_process.append(chunk)
        
        if not chunks_to_process:
            logger.info("Todos los chunks ya tienen embeddings en cache")
            return chunks
        
        logger.info(f"Generando embeddings para {len(chunks_to_process)} chunks...")
        
        # Procesar en lotes para mejor rendimiento
        batch_size = 10
        for i in range(0, len(chunks_to_process), batch_size):
            batch = chunks_to_process[i:i+batch_size]
            
            for chunk in batch:
                try:
                    # Generar embedding
                    embedding = self.ollama_client.generate_embeddings(model, chunk.content)
                    if embedding:
                        chunk.embedding = embedding
                        embedding_key = f"{chunk.id}_{model}"
                        self.embeddings_cache[embedding_key] = embedding
                    
                except Exception as e:
                    logger.error(f"Error al generar embedding para chunk {chunk.id}: {e}")
        
        logger.info(f"Embeddings generados para {len(chunks_to_process)} chunks")
        return chunks
    
    def similarity_search(self, query: str, chunks: List[DocumentChunk], 
                         top_k: int = None, threshold: float = None) -> List[Tuple[DocumentChunk, float]]:
        """Búsqueda por similitud usando embeddings"""
        top_k = top_k or config.MAX_RETRIEVAL_DOCS
        threshold = threshold or config.SIMILARITY_THRESHOLD
        
        try:
            # Generar embedding para la query
            query_embedding = self.ollama_client.generate_embeddings(
                config.DEFAULT_EMBEDDING_MODEL, query
            )
            
            if not query_embedding:
                return []
            
            # Calcular similitudes
            similarities = []
            for chunk in chunks:
                if chunk.embedding:
                    similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                    if similarity >= threshold:
                        similarities.append((chunk, similarity))
            
            # Ordenar por similitud y retornar top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            error_handler.handle_error(e, "Error en búsqueda por similitud")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcular similitud coseno entre dos vectores"""
        try:
            import math
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0
    
    def get_context_for_query(self, query: str, enable_tracing: bool = True) -> Tuple[str, List[Tuple[Any, float]], str]:
        """
        Obtener contexto relevante para una consulta con trazabilidad
        
        Args:
            query: Consulta del usuario
            enable_tracing: Si habilitar trazabilidad
        
        Returns:
            Tupla con (contexto, chunks_con_scores, query_id)
        """
        try:
            
            # Obtener todos los chunks procesados
            all_chunks = []
            uploaded_files = get_valid_uploaded_files()
            
            for file_path in uploaded_files:
                chunks = self.process_document(file_path)
                chunks_with_embeddings = self.generate_embeddings(chunks)
                all_chunks.extend(chunks_with_embeddings)
            
            if not all_chunks:
                return "", [], ""
            
            # Buscar chunks relevantes con parámetros optimizados
            relevant_chunks = self.similarity_search(
                query, 
                all_chunks,
                top_k=config.MAX_RETRIEVAL_DOCS,
                threshold=config.SIMILARITY_THRESHOLD
            )
            
            if not relevant_chunks:
                return "", [], ""
            
            # Construir contexto
            context_parts = []
            for chunk, similarity in relevant_chunks:
                context_parts.append(
                    f"[Fuente: {chunk.source_file}] {chunk.content}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Registrar trazabilidad si está habilitado
            query_id = ""
            if enable_tracing and config.ENABLE_CHUNK_LOGGING:
                query_id = traceability_manager.log_chunk_retrieval(
                    query, relevant_chunks
                )
            
            # Guardar cache
            self._save_cache()
            
            logger.info(f"Contexto generado con {len(relevant_chunks)} chunks relevantes")
            return context, relevant_chunks, query_id
            
        except Exception as e:
            error_handler.handle_error(e, "Error al obtener contexto RAG")
            return "", [], ""
    
    def get_full_context_for_comprehensive_analysis(self) -> str:
        """Obtener TODO el contexto de todos los documentos para análisis comprehensivo"""
        try:
            # Obtener todos los chunks del caché existente
            all_chunks = []
            
            # Iterar sobre todos los chunks en el caché
            for cache_key, cached_chunks in self.chunks_cache.items():
                for chunk_data in cached_chunks:
                    # Crear objeto DocumentChunk desde los datos del caché
                    chunk = DocumentChunk(
                        id=chunk_data['id'],
                        content=chunk_data['content'],
                        metadata=chunk_data['metadata'],
                        source_file=chunk_data['source_file'],
                        chunk_index=chunk_data['chunk_index']
                    )
                    all_chunks.append(chunk)
            
            if not all_chunks:
                logger.warning("No hay chunks en el caché para análisis completo")
                return ""
            
            # Organizar por documento
            documents_content = {}
            for chunk in all_chunks:
                if chunk.source_file not in documents_content:
                    documents_content[chunk.source_file] = []
                documents_content[chunk.source_file].append(chunk.content)
            
            # Construir contexto completo
            context_parts = []
            for file_name, chunks_content in documents_content.items():
                full_document_content = "\n".join(chunks_content)
                context_parts.append(
                    f"=== DOCUMENTO: {file_name} ===\n{full_document_content}\n"
                )
            
            full_context = "\n".join(context_parts)
            
            logger.info(f"Contexto completo generado con {len(all_chunks)} chunks de {len(documents_content)} documentos")
            return full_context
            
        except Exception as e:
            error_handler.handle_error(e, "Error al obtener contexto completo")
            return ""
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de documentos procesados"""
        try:
            uploaded_files = get_valid_uploaded_files()
            
            total_chunks = 0
            total_embeddings = 0
            file_types = {}
            cache_size_mb = 0
            
            for file_path in uploaded_files:
                file_hash = self._get_file_hash(file_path)
                cache_key = f"{file_path.name}_{file_hash}"
                
                if cache_key in self.chunks_cache:
                    chunks = self.chunks_cache[cache_key]
                    total_chunks += len(chunks)
                
                file_ext = file_path.suffix.upper()
                file_types[file_ext] = file_types.get(file_ext, 0) + 1
            
            total_embeddings = len(self.embeddings_cache)
            
            # Calcular tamaño del cache
            try:
                cache_file = config.CACHE_DIR / "rag_cache.json"
                if cache_file.exists():
                    cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
            except Exception:
                cache_size_mb = 0
            
            return {
                'total_documents': len(uploaded_files),
                'total_chunks': total_chunks,
                'total_embeddings': total_embeddings,
                'file_types': file_types,
                'cache_size': len(self.chunks_cache),
                'cache_size_mb': cache_size_mb
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {e}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'total_embeddings': 0,
                'file_types': {},
                'cache_size': 0,
                'cache_size_mb': 0
            }
    
    @property
    def cache(self) -> Dict[str, Any]:
        """Propiedad para acceso unificado al cache"""
        # Convertir chunks_cache a formato compatible
        chunks_list = []
        for cache_key, cached_chunks in self.chunks_cache.items():
            for chunk_data in cached_chunks:
                chunks_list.append(DocumentChunk(
                    id=chunk_data['id'],
                    content=chunk_data['content'],
                    metadata=chunk_data['metadata'],
                    source_file=chunk_data['source_file'],
                    chunk_index=chunk_data['chunk_index']
                ))
        
        return {
            "chunks": chunks_list,
            "embeddings": list(self.embeddings_cache.values()),
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "total_documents": len(set(chunk.source_file for chunk in chunks_list)),
                "total_chunks": len(chunks_list),
                "total_embeddings": len(self.embeddings_cache)
            }
        }
    
    @cache.setter
    def cache(self, value: Dict[str, Any]):
        """Setter para la propiedad cache"""
        if isinstance(value, dict):
            # Limpiar caches existentes
            self.chunks_cache.clear()
            self.embeddings_cache.clear()
            
            # Procesar chunks si existen
            chunks = value.get("chunks", [])
            if isinstance(chunks, list):
                for chunk in chunks:
                    if hasattr(chunk, 'source_file') and hasattr(chunk, 'content'):
                        # Crear cache key
                        cache_key = f"{chunk.source_file}_{chunk.chunk_index}"
                        if cache_key not in self.chunks_cache:
                            self.chunks_cache[cache_key] = []
                        
                        self.chunks_cache[cache_key].append({
                            'id': chunk.id if hasattr(chunk, 'id') else f"chunk_{chunk.chunk_index}",
                            'content': chunk.content,
                            'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {},
                            'source_file': chunk.source_file,
                            'chunk_index': chunk.chunk_index if hasattr(chunk, 'chunk_index') else 0
                        })
    
    def save_cache(self):
        """Método público para guardar cache"""
        self._save_cache()
    
    def clear_cache(self):
        """Limpiar cache de chunks y embeddings"""
        self.chunks_cache.clear()
        self.embeddings_cache.clear()
        
        try:
            cache_file = config.CACHE_DIR / "rag_cache.json"
            if cache_file.exists():
                cache_file.unlink()
            logger.info("Cache RAG limpiado")
        except Exception as e:
            logger.error(f"Error al limpiar cache: {e}")
    
    def process_multiple_documents(self, file_paths: List[Path], max_workers: int = 4) -> Dict[str, List[DocumentChunk]]:
        """Procesar múltiples documentos de manera eficiente"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        results = {}
        lock = threading.Lock()
        
        def process_single_doc(file_path: Path) -> Tuple[str, List[DocumentChunk]]:
            try:
                chunks = self.process_document(file_path)
                return str(file_path), chunks
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {e}")
                return str(file_path), []
        
        logger.info(f"Procesando {len(file_paths)} documentos con {max_workers} workers...")
        start_time = datetime.now()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar todas las tareas
            future_to_path = {executor.submit(process_single_doc, path): path for path in file_paths}
            
            # Recoger resultados conforme se completan
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    file_path, chunks = future.result()
                    with lock:
                        results[file_path] = chunks
                        logger.info(f"Completado: {Path(file_path).name} ({len(chunks)} chunks)")
                except Exception as e:
                    logger.error(f"Error en {path}: {e}")
                    with lock:
                        results[str(path)] = []
        
        total_time = (datetime.now() - start_time).total_seconds()
        total_chunks = sum(len(chunks) for chunks in results.values())
        
        logger.info(f"Procesamiento completado: {total_chunks} chunks en {total_time:.2f}s")
        return results

# Instancia global del procesador RAG
rag_processor = RAGProcessor()