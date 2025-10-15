"""Página de procesamiento de documentos para el sistema RAG"""

import streamlit as st
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time

# Importaciones locales
from config.settings import AppConfig
from utils.logger import setup_logger
from utils.rag_processor import rag_processor
from utils.ollama_client import ollama_client, get_available_or_default_models, get_default_models
from modules.document_upload import get_valid_uploaded_files

# Inicializar configuración y logger
config = AppConfig()
logger = setup_logger("DocumentProcessor")

def show_document_stats():
    """Mostrar estadísticas de documentos procesados"""
    stats = rag_processor.get_document_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documentos", stats.get('total_documents', 0))
    
    with col2:
        st.metric("Chunks", stats.get('total_chunks', 0))
    
    with col3:
        st.metric("Embeddings", stats.get('total_embeddings', 0))
    
    with col4:
        st.metric("Tamaño Cache", f"{stats.get('cache_size_mb', 0):.1f} MB")

def upload_documents():
    """Interfaz para subir documentos"""
    st.subheader("📤 Subir Documentos")
    
    uploaded_files = st.file_uploader(
        "Selecciona documentos para procesar",
        type=['pdf', 'docx', 'xlsx', 'csv', 'txt', 'md', 'html', 'json'],
        accept_multiple_files=True,
        help="Formatos soportados: PDF, DOCX, XLSX, CSV, TXT, MD, HTML, JSON"
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} archivo(s) seleccionado(s)")
        
        if st.button("🚀 Procesar Documentos", type="primary"):
            process_uploaded_files(uploaded_files)

def process_uploaded_files(uploaded_files):
    """Procesar archivos subidos"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    processed_files = []
    failed_files = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Actualizar progreso
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Procesando: {uploaded_file.name}")
            
            # Guardar archivo temporalmente
            file_path = config.UPLOADS_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Procesar con RAG (pasar Path object, no string)
            chunks = rag_processor.process_document(file_path)
            
            if chunks:  # Si se procesaron chunks exitosamente
                # Generar embeddings
                chunks_with_embeddings = rag_processor.generate_embeddings(chunks)
                processed_files.append(uploaded_file.name)
                logger.info(f"Documento procesado exitosamente: {uploaded_file.name} ({len(chunks)} chunks)")
            else:
                failed_files.append(uploaded_file.name)
                logger.error(f"Error al procesar documento: {uploaded_file.name}")
            
            time.sleep(0.1)  # Pequeña pausa para mostrar progreso
            
        except Exception as e:
            failed_files.append(uploaded_file.name)
            logger.error(f"Error al procesar {uploaded_file.name}: {e}")
    
    # Mostrar resultados
    progress_bar.progress(1.0)
    status_text.text("✅ Procesamiento completado")
    
    if processed_files:
        st.success(f"✅ {len(processed_files)} documento(s) procesado(s) exitosamente:")
        for file_name in processed_files:
            st.write(f"  • {file_name}")
    
    if failed_files:
        st.error(f"❌ {len(failed_files)} documento(s) fallaron:")
        for file_name in failed_files:
            st.write(f"  • {file_name}")
    
    # Actualizar estadísticas
    st.rerun()

def show_uploaded_documents():
    """Mostrar documentos subidos con información detallada"""
    st.subheader("📋 Documentos Procesados")
    
    # Obtener lista de documentos usando función estandarizada
    uploaded_files = get_valid_uploaded_files()
    
    if not uploaded_files:
        st.info("📁 No hay documentos procesados aún.")
        st.markdown("💡 **Sugerencia:** Usa la pestaña 'Gestión de Documentos' para subir archivos.")
        return
    
    # Mostrar estadísticas generales
    stats = rag_processor.get_document_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Total Documentos", len(uploaded_files))
    with col2:
        st.metric("📝 Total Chunks", stats.get('total_chunks', 0))
    with col3:
        st.metric("🧠 Total Embeddings", stats.get('total_embeddings', 0))
    
    st.markdown("---")
    
    # Mostrar documentos en tabla con más información
    documents_data = []
    for file_path in uploaded_files:
        if file_path.is_file():
            stat = file_path.stat()
            
            # Verificar si está en cache
            file_hash = rag_processor._get_file_hash(file_path)
            cache_key = f"{file_path.name}_{file_hash}"
            chunks_count = 0
            if hasattr(rag_processor, 'chunks_cache') and cache_key in rag_processor.chunks_cache:
                chunks_count = len(rag_processor.chunks_cache[cache_key])
            
            documents_data.append({
                "📄 Nombre": file_path.name,
                "📊 Tamaño": f"{stat.st_size / 1024:.1f} KB",
                "🧩 Chunks": chunks_count,
                "📅 Modificado": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                "✅ Estado": "Procesado" if chunks_count > 0 else "Pendiente"
            })
    
    if documents_data:
        # Mostrar tabla
        st.dataframe(documents_data, use_container_width=True, hide_index=True)
        
        # Sección para eliminar documentos individuales
        st.markdown("---")
        st.subheader("🗑️ Eliminar Documentos")
        
        selected_files = st.multiselect(
            "Selecciona documentos para eliminar:",
            options=[f.name for f in uploaded_files],
            help="Selecciona uno o más documentos para eliminar"
        )
        
        if selected_files:
            if st.button("🗑️ Eliminar Seleccionados", type="secondary"):
                delete_selected_documents(uploaded_files, selected_files)

def manage_documents():
    """Gestionar documentos existentes"""
    st.subheader("📋 Documentos Procesados")
    
    # Obtener lista de documentos usando función estandarizada
    uploaded_files = get_valid_uploaded_files()
    
    if not uploaded_files:
        st.info("No hay documentos procesados aún.")
        return
    
    # Mostrar documentos en tabla con más información
    documents_data = []
    for file_path in uploaded_files:
        if file_path.is_file():
            stat = file_path.stat()
            
            # Verificar si está en cache
            file_hash = rag_processor._get_file_hash(file_path)
            cache_key = f"{file_path.name}_{file_hash}"
            chunks_count = 0
            if cache_key in rag_processor.chunks_cache:
                chunks_count = len(rag_processor.chunks_cache[cache_key])
            
            documents_data.append({
                "📄 Nombre": file_path.name,
                "📊 Tamaño": f"{stat.st_size / 1024:.1f} KB",
                "🧩 Chunks": chunks_count,
                "📅 Modificado": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                "🔗 Ruta": str(file_path)
            })
    
    if documents_data:
        # Mostrar tabla
        df = st.dataframe(documents_data, use_container_width=True, hide_index=True)
        
        # Opciones de gestión
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Limpiar Cache", help="Limpiar cache de embeddings"):
                rag_processor.clear_cache()
                st.success("Cache limpiado exitosamente")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reprocesar Todo", help="Reprocesar todos los documentos"):
                reprocess_all_documents(uploaded_files)
        
        with col3:
            if st.button("📊 Actualizar Stats", help="Actualizar estadísticas"):
                st.rerun()
        
        with col4:
            if st.button("💾 Guardar Cache", help="Forzar guardado del cache"):
                rag_processor._save_cache()
                st.success("Cache guardado")
        
        # Sección para eliminar documentos individuales
        st.markdown("---")
        st.subheader("🗑️ Eliminar Documentos")
        
        selected_files = st.multiselect(
            "Selecciona documentos para eliminar:",
            options=[f.name for f in uploaded_files],
            help="Selecciona uno o más documentos para eliminar"
        )
        
        if selected_files:
            if st.button("🗑️ Eliminar Seleccionados", type="secondary"):
                delete_selected_documents(uploaded_files, selected_files)

def delete_selected_documents(uploaded_files, selected_files):
    """Eliminar documentos seleccionados"""
    try:
        deleted_count = 0
        failed_deletions = []
        
        for file_path in uploaded_files:
            if file_path.name in selected_files:
                try:
                    # Eliminar del cache primero
                    file_hash = rag_processor._get_file_hash(file_path)
                    cache_key = f"{file_path.name}_{file_hash}"
                    
                    if cache_key in rag_processor.chunks_cache:
                        del rag_processor.chunks_cache[cache_key]
                    
                    # Eliminar embeddings relacionados
                    embeddings_to_remove = [key for key in rag_processor.embeddings_cache.keys() 
                                          if key.startswith(cache_key.split('_')[0])]
                    for key in embeddings_to_remove:
                        del rag_processor.embeddings_cache[key]
                    
                    # Eliminar archivo físico
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Documento eliminado: {file_path.name}")
                    
                except Exception as e:
                    failed_deletions.append(file_path.name)
                    logger.error(f"Error al eliminar {file_path.name}: {e}")
        
        # Guardar cache actualizado
        rag_processor._save_cache()
        
        # Mostrar resultados
        if deleted_count > 0:
            st.success(f"✅ {deleted_count} documento(s) eliminado(s) exitosamente")
        
        if failed_deletions:
            st.error(f"❌ Error al eliminar: {', '.join(failed_deletions)}")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error general al eliminar documentos: {e}")
        logger.error(f"Error en delete_selected_documents: {e}")

def reprocess_all_documents(file_paths):
    """Reprocesar todos los documentos"""
    if not file_paths:
        return
    
    with st.spinner("Reprocesando documentos..."):
        # Limpiar cache primero
        rag_processor.clear_cache()
        
        success_count = 0
        for file_path in file_paths:
            if file_path.is_file():
                try:
                    # Procesar documento (pasar Path object directamente)
                    chunks = rag_processor.process_document(file_path)
                    if chunks:
                        # Generar embeddings
                        chunks_with_embeddings = rag_processor.generate_embeddings(chunks)
                        success_count += 1
                        logger.info(f"Documento reprocesado: {file_path.name} ({len(chunks)} chunks)")
                except Exception as e:
                    logger.error(f"Error reprocesando {file_path}: {e}")
        
        st.success(f"✅ {success_count}/{len(file_paths)} documentos reprocesados")
        st.rerun()

def test_rag_search():
    """Probar búsqueda RAG"""
    st.subheader("🔍 Probar Búsqueda RAG")
    
    # Configuración de búsqueda
    col1, col2 = st.columns(2)
    
    with col1:
        # Obtener modelos disponibles o usar predefinidos
        try:
            available_models = get_available_or_default_models(ollama_client)
            
            if not available_models:
                st.warning("No hay modelos disponibles")
                return
                
            model = st.selectbox(
                "Modelo para respuesta:",
                available_models,
                index=0 if available_models else None
            )
        except Exception as e:
            st.error(f"Error al obtener modelos: {e}")
            return
    
    with col2:
        similarity_threshold = st.slider(
            "Umbral de similitud:",
            min_value=0.0,
            max_value=1.0,
            value=config.SIMILARITY_THRESHOLD,
            step=0.05,
            help="Mínima similitud requerida para considerar un chunk relevante"
        )
    
    with col2:
        max_results = st.slider(
            "Máximo de resultados:",
            min_value=1,
            max_value=10,
            value=config.MAX_RETRIEVAL_DOCS,
            help="Número máximo de chunks a retornar"
        )
    
    query = st.text_input(
        "Consulta de prueba:",
        placeholder="Escribe una pregunta sobre tus documentos..."
    )
    
    if query and st.button("🔍 Buscar", key="search_documents"):
        with st.spinner("Buscando en documentos..."):
            try:
                # Obtener todos los chunks procesados usando función estandarizada
                all_chunks = []
                uploaded_files = get_valid_uploaded_files()
                
                for file_path in uploaded_files:
                        chunks = rag_processor.process_document(file_path)
                        chunks_with_embeddings = rag_processor.generate_embeddings(chunks)
                        all_chunks.extend(chunks_with_embeddings)
                
                if not all_chunks:
                    st.warning("⚠️ No hay documentos procesados para buscar")
                    return
                
                # Realizar búsqueda con parámetros personalizados
                relevant_chunks = rag_processor.similarity_search(
                    query, all_chunks, top_k=max_results, threshold=similarity_threshold
                )
                
                if relevant_chunks:
                    st.success(f"✅ Se encontraron {len(relevant_chunks)} chunks relevantes:")
                    
                    # Mostrar resultados detallados
                    for i, (chunk, similarity) in enumerate(relevant_chunks, 1):
                        with st.expander(f"📄 Resultado {i} - {chunk.source_file} (Similitud: {similarity:.3f})"):
                            st.write(f"**Archivo:** {chunk.source_file}")
                            st.write(f"**Chunk:** {chunk.chunk_index + 1} de {chunk.metadata.get('total_chunks', 'N/A')}")
                            st.write(f"**Similitud:** {similarity:.3f}")
                            st.text_area("Contenido:", chunk.content, height=150, key=f"chunk_{i}")
                    
                    # Mostrar contexto completo
                    st.markdown("---")
                    st.subheader("📋 Contexto Completo")
                    context_parts = []
                    for chunk, similarity in relevant_chunks:
                        context_parts.append(f"[Fuente: {chunk.source_file}] {chunk.content}")
                    
                    full_context = "\n\n".join(context_parts)
                    st.text_area("Contexto que se enviaría a la IA:", full_context, height=300)
                    
                else:
                    st.warning(f"⚠️ No se encontraron chunks con similitud >= {similarity_threshold}")
                    st.info("💡 Intenta reducir el umbral de similitud o usar términos diferentes")
                
            except Exception as e:
                st.error(f"❌ Error en la búsqueda: {e}")
                logger.error(f"Error en test_rag_search: {e}")
    
    # Información adicional
    if st.checkbox("ℹ️ Mostrar información del sistema"):
        st.markdown("---")
        st.subheader("🔧 Información del Sistema RAG")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Configuración actual:**")
            st.write(f"• Modelo de embeddings: `{config.DEFAULT_EMBEDDING_MODEL}`")
            st.write(f"• Tamaño de chunk: `{config.CHUNK_SIZE}` caracteres")
            st.write(f"• Overlap: `{config.CHUNK_OVERLAP}` caracteres")
        
        with col2:
            st.write("**Estadísticas:**")
            stats = rag_processor.get_document_stats()
            st.write(f"• Documentos: `{stats.get('total_documents', 0)}`")
            st.write(f"• Chunks: `{stats.get('total_chunks', 0)}`")
            st.write(f"• Embeddings: `{stats.get('total_embeddings', 0)}`")

def clear_rag_cache():
    """Limpiar completamente el cache RAG"""
    st.subheader("🗑️ Limpiar Cache RAG")
    
    st.warning("⚠️ Esta acción eliminará todos los chunks y embeddings procesados.")
    st.info("💡 Los archivos originales se mantendrán, pero deberás reprocesarlos.")
    
    if st.button("🗑️ Limpiar Cache Completo", type="secondary"):
        try:
            # Limpiar cache en memoria
            rag_processor.cache = {
                "chunks": [],
                "embeddings": [],
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_documents": 0,
                    "total_chunks": 0,
                    "total_embeddings": 0
                }
            }
            
            # Guardar cache vacío
            rag_processor.save_cache()
            
            st.success("✅ Cache RAG limpiado completamente")
            st.info("💡 Puedes reprocesar tus documentos usando el botón 'Reprocesar Todos'")
            
            # Actualizar estadísticas
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error al limpiar cache: {e}")
            logger.error(f"Error en clear_rag_cache: {e}")

def export_rag_data():
    """Exportar datos RAG para respaldo"""
    st.subheader("📤 Exportar Datos RAG")
    
    try:
        stats = rag_processor.get_document_stats()
        
        if stats.get('total_chunks', 0) == 0:
            st.warning("⚠️ No hay datos RAG para exportar")
            return
        
        # Preparar datos para exportación
        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_documents": stats.get('total_documents', 0),
                "total_chunks": stats.get('total_chunks', 0),
                "total_embeddings": stats.get('total_embeddings', 0),
                "config": {
                    "chunk_size": config.CHUNK_SIZE,
                    "chunk_overlap": config.CHUNK_OVERLAP,
                    "embedding_model": config.DEFAULT_EMBEDDING_MODEL
                }
            },
            "chunks": []
        }
        
        # Agregar chunks (sin embeddings para reducir tamaño)
        for chunk in rag_processor.cache.get("chunks", []):
            chunk_data = {
                "content": chunk.content,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata
            }
            export_data["chunks"].append(chunk_data)
        
        # Convertir a JSON
        json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Botón de descarga
        st.download_button(
            label="📥 Descargar Respaldo RAG",
            data=json_data,
            file_name=f"rag_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success(f"✅ Respaldo preparado ({len(json_data)} caracteres)")
        
    except Exception as e:
        st.error(f"❌ Error al exportar datos: {e}")
        logger.error(f"Error en export_rag_data: {e}")

def advanced_rag_settings():
    """Configuraciones avanzadas del sistema RAG"""
    st.subheader("⚙️ Configuraciones Avanzadas")
    
    with st.expander("🔧 Parámetros de Procesamiento"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Configuración actual:**")
            st.code(f"""
CHUNK_SIZE = {config.CHUNK_SIZE}
CHUNK_OVERLAP = {config.CHUNK_OVERLAP}
SIMILARITY_THRESHOLD = {config.SIMILARITY_THRESHOLD}
MAX_RETRIEVAL_DOCS = {config.MAX_RETRIEVAL_DOCS}
            """)
        
        with col2:
            st.write("**Modelos disponibles:**")
            try:
                if ollama_client.is_available():
                    models = ollama_client.get_available_models()
                    if models:
                        for model in models:
                            st.write(f"• {model['name']}")
                    else:
                        st.warning("No se encontraron modelos instalados")
                        st.write("**Modelos recomendados para instalar:**")
                        default_models = get_default_models()
                        for model in default_models["chat"][:3]:
                            st.write(f"• {model} (Chat)")
                        for model in default_models["embeddings"][:2]:
                            st.write(f"• {model} (Embeddings)")
                else:
                    st.error("Ollama no está disponible")
                    st.write("**Modelos predefinidos recomendados:**")
                    default_models = get_default_models()
                    for model in default_models["chat"][:3]:
                        st.write(f"• {model} (Chat)")
                    for model in default_models["embeddings"][:2]:
                        st.write(f"• {model} (Embeddings)")
            except Exception as e:
                st.error(f"Error al obtener modelos: {e}")
                st.write("**Modelos predefinidos recomendados:**")
                default_models = get_default_models()
                for model in default_models["chat"][:3]:
                    st.write(f"• {model} (Chat)")
                for model in default_models["embeddings"][:2]:
                    st.write(f"• {model} (Embeddings)")
    
    with st.expander("📊 Estadísticas Detalladas"):
        stats = rag_processor.get_document_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documentos", stats.get('total_documents', 0))
            st.metric("Chunks", stats.get('total_chunks', 0))
        
        with col2:
            st.metric("Embeddings", stats.get('total_embeddings', 0))
            st.metric("Cache (MB)", f"{stats.get('cache_size_mb', 0):.2f}")
        
        with col3:
            if stats.get('total_chunks', 0) > 0:
                avg_chunk_size = stats.get('total_characters', 0) / stats.get('total_chunks', 1)
                st.metric("Promedio chars/chunk", f"{avg_chunk_size:.0f}")
            else:
                st.metric("Promedio chars/chunk", "0")
    
    with st.expander("🔍 Diagnóstico del Sistema"):
        if st.button("🔍 Ejecutar Diagnóstico", key="execute_diagnostic"):
            with st.spinner("Ejecutando diagnóstico..."):
                # Verificar conexión Ollama
                try:
                    models = ollama_client.get_available_models()
                    st.success(f"✅ Ollama conectado ({len(models)} modelos)")
                except Exception as e:
                    st.error(f"❌ Error Ollama: {e}")
                
                # Verificar directorios
                for dir_name, dir_path in [
                    ("Uploads", config.UPLOADS_DIR),
                    ("Cache", config.CACHE_DIR),
                    ("Processed", config.PROCESSED_DIR)
                ]:
                    if dir_path.exists():
                        files_count = len(list(dir_path.glob("*")))
                        st.success(f"✅ {dir_name}: {files_count} archivos")
                    else:
                        st.error(f"❌ {dir_name}: Directorio no existe")
                
                # Verificar cache
                cache_file = config.CACHE_DIR / "rag_cache.json"
                if cache_file.exists():
                    cache_size = cache_file.stat().st_size / 1024 / 1024
                    st.success(f"✅ Cache: {cache_size:.2f} MB")
                else:
                    st.warning("⚠️ Cache: Archivo no existe")

def check_ollama_connection():
    """Verificar conexión con Ollama"""
    try:
        return ollama_client.is_available()
    except Exception as e:
        logger.error(f"Error al conectar con Ollama: {e}")
        return False

def main():
    """Función principal de la página de procesamiento de documentos"""
    st.title("📄 Procesamiento de Documentos RAG")
    
    # Verificar conexión con Ollama
    if not check_ollama_connection():
        st.error("❌ No se puede conectar con Ollama. Verifica que esté ejecutándose.")
        st.info("💡 **Para solucionar este problema:**")
        st.code("ollama serve", language="bash")
        st.markdown("Ejecuta este comando en tu terminal y luego recarga la página.")
        st.stop()
    
    # Crear pestañas para organizar funcionalidades
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Gestión de Documentos", 
        "🔍 Búsqueda RAG", 
        "⚙️ Configuración", 
        "📤 Respaldo", 
        "🔧 Avanzado"
    ])
    
    with tab1:
        st.header("📁 Gestión de Documentos")
        
        # Subir documentos
        upload_documents()
        
        st.markdown("---")
        
        # Mostrar documentos existentes
        show_uploaded_documents()
        
        st.markdown("---")
        
        # Botones de acción
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reprocesar Todos", type="primary"):
                # Obtener lista de archivos usando función estandarizada
                uploaded_files = get_valid_uploaded_files()
                reprocess_all_documents(uploaded_files)
        
        with col2:
            if st.button("📊 Actualizar Estadísticas", key="update_stats_processor"):
                st.rerun()
        
        with col3:
            if st.button("💾 Guardar Cache", key="save_cache_processor"):
                try:
                    rag_processor.save_cache()
                    st.success("✅ Cache guardado")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with tab2:
        st.header("🔍 Búsqueda y Pruebas RAG")
        test_rag_search()
    
    with tab3:
        st.header("⚙️ Configuración del Sistema")
        
        # Mostrar estadísticas generales
        stats = rag_processor.get_document_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Documentos", stats.get('total_documents', 0))
        
        with col2:
            st.metric("📝 Chunks", stats.get('total_chunks', 0))
        
        with col3:
            st.metric("🧠 Embeddings", stats.get('total_embeddings', 0))
        
        with col4:
            st.metric("💾 Cache (MB)", f"{stats.get('cache_size_mb', 0):.2f}")
        
        st.markdown("---")
        
        # Limpiar cache
        clear_rag_cache()
    
    with tab4:
        st.header("📤 Respaldo y Exportación")
        export_rag_data()
    
    with tab5:
        st.header("🔧 Configuración Avanzada")
        advanced_rag_settings()

if __name__ == "__main__":
    main()