"""
Gestor de Cache RAG para Análisis Cualitativo
Maneja la detección automática y actualización del rag_cache.json
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import streamlit as st

from config.settings import config


class RAGCacheManager:
    """
    Gestor del cache RAG para el módulo de análisis cualitativo
    
    Características:
    - Detección automática de rag_cache.json
    - Verificación de cambios en el cache
    - Opciones de actualización manual
    - Conversión de cache a formato de chunks para análisis
    """
    
    def __init__(self):
        self.cache_file = config.CACHE_DIR / "rag_cache.json"
        self.last_check_time = 0
        self.cache_data = None
        self.cache_modified_time = 0
        self.chunks_cache = []
    
    def detect_cache(self) -> Tuple[bool, str]:
        """
        Detectar si existe rag_cache.json y su estado
        
        Returns:
            Tuple[bool, str]: (existe_cache, mensaje_descriptivo)
        """
        try:
            if not self.cache_file.exists():
                return False, "No se encontró rag_cache.json"
            
            # Verificar si el archivo es válido
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache_data = json.load(f)
            except json.JSONDecodeError:
                return False, "rag_cache.json está corrupto"
            except Exception as e:
                return False, f"Error al leer rag_cache.json: {str(e)}"
            
            # Verificar estructura del cache
            if not isinstance(self.cache_data, dict):
                return False, "rag_cache.json no tiene estructura válida"
            
            if 'chunks' not in self.cache_data:
                return False, "rag_cache.json no contiene chunks"
            
            # Obtener información del cache
            chunks_data = self.cache_data.get('chunks', {})
            total_chunks = sum(len(chunk_list) for chunk_list in chunks_data.values())
            last_updated = self.cache_data.get('last_updated', 'Desconocido')
            
            return True, f"Cache detectado: {total_chunks} chunks, actualizado: {last_updated}"
            
        except Exception as e:
            return False, f"Error al detectar cache: {str(e)}"
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Obtener información detallada del cache
        
        Returns:
            Dict con información del cache
        """
        if not self.cache_data:
            return {}
        
        chunks_data = self.cache_data.get('chunks', {})
        total_chunks = sum(len(chunk_list) for chunk_list in chunks_data.values())
        
        # Contar fuentes únicas
        unique_sources = set()
        for chunk_list in chunks_data.values():
            for chunk in chunk_list:
                if isinstance(chunk, dict) and 'metadata' in chunk:
                    source = chunk['metadata'].get('source_file', 'unknown')
                    unique_sources.add(source)
        
        # Calcular tamaño del archivo
        file_size = 0
        if self.cache_file.exists():
            file_size = self.cache_file.stat().st_size
        
        return {
            'total_chunks': total_chunks,
            'unique_sources': len(unique_sources),
            'sources': list(unique_sources),
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'last_updated': self.cache_data.get('last_updated', 'Desconocido'),
            'cache_file_path': str(self.cache_file),
            'cache_exists': True
        }
    
    def convert_cache_to_chunks(self) -> List[Dict[str, Any]]:
        """
        Convertir cache RAG a formato de chunks para análisis cualitativo
        
        Returns:
            Lista de chunks en formato estándar
        """
        if not self.cache_data:
            return []
        
        chunks = []
        chunks_data = self.cache_data.get('chunks', {})
        
        for file_key, chunk_list in chunks_data.items():
            for chunk_data in chunk_list:
                if isinstance(chunk_data, dict):
                    # Convertir formato del cache a formato estándar
                    chunk = {
                        'content': chunk_data.get('content', ''),
                        'metadata': chunk_data.get('metadata', {})
                    }
                    
                    # Asegurar que metadata tenga source_file
                    if 'source_file' not in chunk['metadata']:
                        chunk['metadata']['source_file'] = chunk_data.get('source_file', 'unknown')
                    
                    # Agregar información adicional si está disponible
                    if 'chunk_index' in chunk_data:
                        chunk['metadata']['chunk_index'] = chunk_data['chunk_index']
                    
                    chunks.append(chunk)
        
        return chunks
    
    def check_cache_updated(self) -> bool:
        """
        Verificar si el cache ha sido actualizado desde la última verificación
        
        Returns:
            True si el cache fue actualizado
        """
        if not self.cache_file.exists():
            return False
        
        try:
            current_mtime = self.cache_file.stat().st_mtime
            
            if current_mtime > self.cache_modified_time:
                self.cache_modified_time = current_mtime
                return True
            
            return False
            
        except Exception:
            return False
    
    def force_reload_cache(self) -> bool:
        """
        Forzar recarga del cache desde el archivo
        
        Returns:
            True si se recargó exitosamente
        """
        try:
            if not self.cache_file.exists():
                return False
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache_data = json.load(f)
            
            self.cache_modified_time = self.cache_file.stat().st_mtime
            self.chunks_cache = self.convert_cache_to_chunks()
            
            return True
            
        except Exception as e:
            st.error(f"Error al recargar cache: {str(e)}")
            return False
    
    def get_chunks_for_analysis(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        Obtener chunks para análisis cualitativo
        
        Args:
            force_reload: Si forzar recarga del cache
            
        Returns:
            Lista de chunks para análisis
        """
        # Verificar si necesitamos recargar
        if force_reload or self.check_cache_updated():
            self.force_reload_cache()
        
        # Si no tenemos chunks en memoria, convertir del cache
        if not self.chunks_cache and self.cache_data:
            self.chunks_cache = self.convert_cache_to_chunks()
        
        return self.chunks_cache
    
    def render_cache_status(self):
        """
        Renderizar estado del cache en la interfaz
        """
        cache_exists, message = self.detect_cache()
        
        if cache_exists:
            cache_info = self.get_cache_info()
            
            st.success(f"✅ {message}")
            
            # Mostrar información detallada
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📄 Documentos",
                    cache_info.get('unique_sources', 0),
                    help="Número de documentos procesados"
                )
            
            with col2:
                st.metric(
                    "📝 Chunks",
                    cache_info.get('total_chunks', 0),
                    help="Número total de fragmentos de texto"
                )
            
            with col3:
                st.metric(
                    "💾 Tamaño",
                    f"{cache_info.get('file_size_mb', 0)} MB",
                    help="Tamaño del archivo de cache"
                )
            
            # Mostrar fuentes
            if cache_info.get('sources'):
                with st.expander("📂 Ver fuentes disponibles"):
                    for i, source in enumerate(cache_info['sources'], 1):
                        st.markdown(f"{i}. `{source}`")
            
            # Botones de control
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Actualizar Cache", help="Recargar cache desde archivo"):
                    if self.force_reload_cache():
                        st.success("✅ Cache actualizado")
                        st.rerun()
                    else:
                        st.error("❌ Error al actualizar cache")
            
            with col2:
                if st.button("📊 Ver Estadísticas", help="Ver estadísticas detalladas"):
                    st.session_state.show_cache_stats = True
            
            with col3:
                if st.button("🗑️ Limpiar Cache", help="Limpiar cache de memoria"):
                    self.chunks_cache = []
                    st.success("✅ Cache de memoria limpiado")
                    st.rerun()
            
            # Mostrar estadísticas si se solicita
            if st.session_state.get('show_cache_stats', False):
                self._render_detailed_stats(cache_info)
        
        else:
            st.warning(f"⚠️ {message}")
            
            # Botón para intentar detectar nuevamente
            if st.button("🔍 Buscar Cache Nuevamente"):
                st.rerun()
    
    def _render_detailed_stats(self, cache_info: Dict[str, Any]):
        """
        Renderizar estadísticas detalladas del cache
        """
        st.markdown("### 📊 Estadísticas Detalladas del Cache")
        
        # Información general
        st.markdown("**Información General:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"- **Archivo:** `{cache_info.get('cache_file_path', 'N/A')}`")
            st.markdown(f"- **Última actualización:** {cache_info.get('last_updated', 'N/A')}")
        
        with col2:
            st.markdown(f"- **Tamaño:** {cache_info.get('file_size_mb', 0)} MB")
            st.markdown(f"- **Chunks totales:** {cache_info.get('total_chunks', 0)}")
        
        # Distribución por fuente
        if cache_info.get('sources'):
            st.markdown("**Distribución por Fuente:**")
            
            # Obtener distribución real de chunks por fuente
            chunks_data = self.cache_data.get('chunks', {})
            source_distribution = {}
            
            for file_key, chunk_list in chunks_data.items():
                for chunk in chunk_list:
                    if isinstance(chunk, dict) and 'metadata' in chunk:
                        source = chunk['metadata'].get('source_file', 'unknown')
                        if source not in source_distribution:
                            source_distribution[source] = 0
                        source_distribution[source] += 1
            
            for source, count in source_distribution.items():
                st.markdown(f"- `{source}`: {count} chunks")
        
        # Botón para cerrar estadísticas
        if st.button("❌ Cerrar Estadísticas"):
            st.session_state.show_cache_stats = False
            st.rerun()
    
    def __repr__(self) -> str:
        cache_exists, _ = self.detect_cache()
        return f"RAGCacheManager(cache_exists={cache_exists}, chunks={len(self.chunks_cache)})"
