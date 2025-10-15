"""Pestaña de alertas y monitoreo"""

import streamlit as st
import time
from datetime import datetime, timedelta
from utils.error_handler import ErrorHandler
from utils.logger import setup_logger
from utils.ollama_client import OllamaClient
from config.settings import config
from modules.document_upload import get_valid_uploaded_files

logger = setup_logger()

def render():
    """Renderizar la pestaña de alertas y monitoreo del sistema"""
    st.header("🚨 Alertas y Monitoreo del Sistema")
    
    # Estado general del sistema
    st.subheader("📊 Estado General del Sistema")
    
    # Verificar estado de Ollama
    ollama_client = OllamaClient()
    ollama_available = ollama_client.is_available()
    
    # Cards de estado
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if ollama_available:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2rem;">✅</h2>
                <h4 style="margin: 0.5rem 0 0 0;">Ollama</h4>
                <p style="margin: 0; font-size: 0.9rem;">Conectado</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2rem;">❌</h2>
                <h4 style="margin: 0.5rem 0 0 0;">Ollama</h4>
                <p style="margin: 0; font-size: 0.9rem;">Desconectado</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Verificar documentos usando función estandarizada
        uploaded_files = get_valid_uploaded_files()
        
        if len(uploaded_files) > 0:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2rem;">📚</h2>
                <h4 style="margin: 0.5rem 0 0 0;">Documentos</h4>
                <p style="margin: 0; font-size: 0.9rem;">{len(uploaded_files)} cargados</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2rem;">⚠️</h2>
                <h4 style="margin: 0.5rem 0 0 0;">Documentos</h4>
                <p style="margin: 0; font-size: 0.9rem;">Sin cargar</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Verificar estado RAG
        rag_enabled = st.session_state.get('rag_enabled', False)
        
        # Obtener estadísticas RAG
        try:
            from utils.rag_processor import rag_processor
            rag_stats = rag_processor.get_document_stats()
            documents_available = rag_stats.get('total_documents', 0) > 0
            embeddings_ready = rag_stats.get('total_embeddings', 0) > 0
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas RAG: {e}")
            rag_stats = {}
            documents_available = False
            embeddings_ready = False
        
        if rag_enabled and embeddings_ready:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2rem;">🔍</h2>
                <h4 style="margin: 0.5rem 0 0 0;">RAG</h4>
                <p style="margin: 0; font-size: 0.9rem;">Activado</p>
            </div>
            """, unsafe_allow_html=True)
        elif rag_enabled and not embeddings_ready:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2rem;">⚠️</h2>
                <h4 style="margin: 0.5rem 0 0 0;">RAG</h4>
                <p style="margin: 0; font-size: 0.9rem;">Sin embeddings</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #607D8B 0%, #455A64 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2rem;">⭕</h2>
                <h4 style="margin: 0.5rem 0 0 0;">RAG</h4>
                <p style="margin: 0; font-size: 0.9rem;">Desactivado</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Verificar espacio en disco
        try:
            import shutil
            total, used, free = shutil.disk_usage(config.DATA_DIR)
            free_gb = free // (1024**3)
            
            if free_gb > 5:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="margin: 0; font-size: 2rem;">💾</h2>
                    <h4 style="margin: 0.5rem 0 0 0;">Espacio</h4>
                    <p style="margin: 0; font-size: 0.9rem;">{free_gb} GB libre</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FF5722 0%, #D84315 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="margin: 0; font-size: 2rem;">⚠️</h2>
                    <h4 style="margin: 0.5rem 0 0 0;">Espacio</h4>
                    <p style="margin: 0; font-size: 0.9rem;">{free_gb} GB libre</p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #9E9E9E 0%, #616161 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <h2 style="margin: 0; font-size: 2rem;">❓</h2>
                <h4 style="margin: 0.5rem 0 0 0;">Espacio</h4>
                <p style="margin: 0; font-size: 0.9rem;">No disponible</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Alertas activas
    st.subheader("🔔 Alertas Activas")
    
    alerts = []
    
    # Verificar alertas del sistema
    if not ollama_available:
        alerts.append({
            'type': 'error',
            'title': 'Ollama Desconectado',
            'message': f'No se puede conectar a Ollama. Verifica que el servicio esté ejecutándose.',
            'action': 'Revisar configuración de Ollama'
        })
    
    if len(uploaded_files) == 0:
        alerts.append({
            'type': 'warning',
            'title': 'Sin Documentos',
            'message': 'No hay documentos cargados. El sistema RAG no funcionará correctamente.',
            'action': 'Cargar documentos en la pestaña de Gestión de Documentos'
        })
    
    if not rag_enabled and len(uploaded_files) > 0:
        alerts.append({
            'type': 'info',
            'title': 'RAG Desactivado',
            'message': 'Tienes documentos cargados pero el RAG está desactivado.',
            'action': 'Activar RAG en la pestaña de Chat o Configuración'
        })
    
    # Verificar modelos disponibles
    if ollama_available:
        # Obtener modelos con caché
        if 'available_models_alerts' not in st.session_state or st.session_state.get('models_alerts_last_refresh', 0) < time.time() - 300:
            models = ollama_client.get_available_models()
            st.session_state.available_models_alerts = models
            st.session_state.models_alerts_last_refresh = time.time()
        else:
            models = st.session_state.available_models_alerts
            
        if len(models) == 0:
            alerts.append({
                'type': 'warning',
                'title': 'Sin Modelos',
                'message': 'No hay modelos de IA disponibles en Ollama.',
                'action': 'Instalar modelos usando: ollama pull llama3.1:8b'
            })
    
    # Mostrar alertas
    if alerts:
        for alert in alerts:
            if alert['type'] == 'error':
                st.error(f"🚨 **{alert['title']}**\n\n{alert['message']}\n\n💡 **Acción recomendada:** {alert['action']}")
            elif alert['type'] == 'warning':
                st.warning(f"⚠️ **{alert['title']}**\n\n{alert['message']}\n\n💡 **Acción recomendada:** {alert['action']}")
            else:
                st.info(f"ℹ️ **{alert['title']}**\n\n{alert['message']}\n\n💡 **Acción recomendada:** {alert['action']}")
    else:
        st.success("✅ **Sistema funcionando correctamente** - No hay alertas activas")
    
    st.divider()
    
    # Información detallada del sistema
    st.subheader("🔍 Información Detallada del Sistema")
    
    with st.expander("🤖 Información de Ollama", expanded=False):
        if ollama_available:
            st.success(f"✅ Conectado a Ollama")
            
            # Usar el mismo caché de modelos
            if 'available_models_alerts' in st.session_state:
                models = st.session_state.available_models_alerts
            else:
                models = ollama_client.get_available_models()
                st.session_state.available_models_alerts = models
                st.session_state.models_alerts_last_refresh = time.time()
            
            if models:
                st.write("**Modelos disponibles:**")
                for model in models:
                    # Extraer nombre del modelo correctamente
                    if isinstance(model, dict):
                        model_name = model.get('name', str(model))
                    else:
                        model_name = str(model)
                    st.write(f"• {model_name}")
            else:
                st.warning("No hay modelos instalados")
        else:
            st.error(f"❌ No se puede conectar a Ollama")
            st.write("**Posibles soluciones:**")
            st.write("• Verificar que Ollama esté ejecutándose")
            st.write("• Comprobar la URL en configuración")
            st.write("• Revisar el firewall/antivirus")
    
    with st.expander("📁 Información de Archivos", expanded=False):
        st.write(f"**Directorio de datos:** `{config.DATA_DIR}`")
        st.write(f"**Documentos cargados:** {len(uploaded_files)}")
        
        if uploaded_files:
            total_size = sum(f.stat().st_size for f in uploaded_files) / (1024**2)
            st.write(f"**Tamaño total:** {total_size:.2f} MB")
            
            # Mostrar tipos de archivo
            file_types = {}
            for file in uploaded_files:
                ext = file.suffix.upper()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            st.write("**Tipos de archivo:**")
            for ext, count in file_types.items():
                st.write(f"• {ext}: {count} archivo(s)")
    
    with st.expander("🧠 Información del Sistema RAG", expanded=False):
        st.write(f"**Estado RAG:** {'✅ Activado' if rag_enabled else '❌ Desactivado'}")
        
        if rag_stats:
            st.write(f"**Documentos procesados:** {rag_stats.get('total_documents', 0)}")
            st.write(f"**Chunks generados:** {rag_stats.get('total_chunks', 0)}")
            st.write(f"**Embeddings creados:** {rag_stats.get('total_embeddings', 0)}")
            st.write(f"**Tamaño de cache:** {rag_stats.get('cache_size_mb', 0):.1f} MB")
            
            # Información por documento
            if 'documents' in rag_stats and rag_stats['documents']:
                st.write("**Documentos por tipo:**")
                doc_types = {}
                for doc_info in rag_stats['documents']:
                    doc_type = doc_info.get('type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                for doc_type, count in doc_types.items():
                    st.write(f"• {doc_type.upper()}: {count}")
        else:
            st.write("**No hay estadísticas RAG disponibles**")
    
    with st.expander("⚙️ Configuración Actual", expanded=False):
        # Mostrar configuración actual desde session_state
        st.write("**Configuración de IA:**")
        st.write(f"• Modelo LLM: {st.session_state.get('selected_llm_model', config.DEFAULT_LLM_MODEL)}")
        st.write(f"• Modelo de Embeddings: {st.session_state.get('selected_embedding_model', config.DEFAULT_EMBEDDING_MODEL)}")
        
        st.write("**Configuración RAG:**")
        st.write(f"• Tamaño de chunk: {st.session_state.get('chunk_size', config.CHUNK_SIZE)}")
        st.write(f"• Overlap: {st.session_state.get('chunk_overlap', config.CHUNK_OVERLAP)}")
        st.write(f"• Documentos máximos: {st.session_state.get('max_retrieval_docs', config.MAX_RETRIEVAL_DOCS)}")
        st.write(f"• Umbral de similitud: {st.session_state.get('similarity_threshold', config.SIMILARITY_THRESHOLD)}")
        
        st.write("**Configuración del Sistema:**")
        st.write(f"• URL Ollama: {st.session_state.get('ollama_url', config.OLLAMA_BASE_URL)}")
        st.write(f"• Timeout: {st.session_state.get('ollama_timeout', config.OLLAMA_TIMEOUT)}s")
        st.write(f"• Nivel de log: {st.session_state.get('log_level', config.LOG_LEVEL)}")
    
    st.divider()
    
    # Herramientas de diagnóstico
    st.subheader("🛠️ Herramientas de Diagnóstico")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Verificar Conexión Ollama", key="verify_ollama_connection"):
            with st.spinner("Verificando conexión..."):
                if ollama_client.is_available():
                    st.success("✅ Conexión exitosa")
                else:
                    st.error("❌ Error de conexión")
    
    with col2:
        if st.button("📊 Actualizar Estado", key="update_status_alerts"):
            st.rerun()
    
    with col3:
        if st.button("🧹 Limpiar Cache", key="clear_cache_alerts"):
            try:
                # Limpiar archivos de cache
                cache_files = list(config.CACHE_DIR.glob("*"))
                for file in cache_files:
                    if file.is_file():
                        file.unlink()
                st.success(f"✅ {len(cache_files)} archivos de cache eliminados")
            except Exception as e:
                st.error(f"❌ Error al limpiar cache: {e}")
    
    # Obtener instancia del error handler desde session state
    if 'error_handler' not in st.session_state:
        st.session_state.error_handler = ErrorHandler()
    
    error_handler = st.session_state.error_handler
    
    # Estadísticas de logs
    stats = error_handler.get_error_stats()
    
    # Logs del sistema (si están disponibles)
    if st.checkbox("📋 Mostrar logs del sistema"):
        try:
            # Buscar el archivo de log más reciente
            log_files = list(config.LOGS_DIR.glob("cognichat_*.log"))
            if log_files:
                # Ordenar por fecha de modificación y tomar el más reciente
                log_file = max(log_files, key=lambda f: f.stat().st_mtime)
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_lines = lines[-20:] if len(lines) > 20 else lines
                
                st.text_area(
                    f"Últimas 20 entradas del log ({log_file.name}):",
                    "".join(last_lines),
                    height=200
                )
            else:
                st.info("No hay archivos de log disponibles")
        except Exception as e:
            st.error(f"Error al leer logs: {e}")
    
    st.divider()
    
    # Estadísticas de errores del sistema
    st.subheader("📊 Estadísticas de Errores")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 Errores", stats['total_errors'])
    
    with col2:
        st.metric("🟡 Advertencias", stats['total_warnings'])
    
    with col3:
        st.metric("🔵 Información", stats['total_info'])
    
    # Controles de logs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Actualizar", type="primary"):
            st.rerun()
    
    with col2:
        if st.button("🧹 Limpiar Alertas", type="secondary"):
            error_handler.clear_all()
            st.success("Alertas limpiadas")
            st.rerun()
    
    with col3:
        auto_refresh = st.toggle("🔄 Auto-actualizar", value=False)
    
    # Pestañas para diferentes tipos de alertas
    tab1, tab2, tab3, tab4 = st.tabs(["🔴 Errores", "🟡 Advertencias", "🔵 Información", "📊 Sistema"])
    
    with tab1:
        render_errors_tab(error_handler)
    
    with tab2:
        render_warnings_tab(error_handler)
    
    with tab3:
        render_info_tab(error_handler)
    
    with tab4:
        render_system_tab()
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

def render_errors_tab(error_handler):
    """Renderizar pestaña de errores"""
    st.subheader("🔴 Errores Recientes")
    
    errors = error_handler.get_recent_errors(20)
    
    if not errors:
        st.info("No hay errores registrados")
        return
    
    for error in errors:
        with st.expander(f"❌ {error['type']} - {error['timestamp'].strftime('%H:%M:%S')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Tipo:**", error['type'])
                st.write("**Contexto:**", error['context'])
                st.write("**Timestamp:**", error['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
            
            with col2:
                st.write("**Mensaje:**")
                st.code(error['message'])
            
            if error.get('traceback'):
                st.write("**Traceback:**")
                st.code(error['traceback'])

def render_warnings_tab(error_handler):
    """Renderizar pestaña de advertencias"""
    st.subheader("🟡 Advertencias Recientes")
    
    warnings = error_handler.get_recent_warnings(20)
    
    if not warnings:
        st.info("No hay advertencias registradas")
        return
    
    for warning in warnings:
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.write("⚠️")
            
            with col2:
                st.write(f"**{warning['context']}:** {warning['message']}")
            
            with col3:
                st.caption(warning['timestamp'].strftime('%H:%M:%S'))
            
            st.divider()

def render_info_tab(error_handler):
    """Renderizar pestaña de información"""
    st.subheader("🔵 Mensajes Informativos")
    
    info_messages = error_handler.get_recent_info(20)
    
    if not info_messages:
        st.info("No hay mensajes informativos")
        return
    
    for info in info_messages:
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.write("ℹ️")
            
            with col2:
                st.write(f"**{info['context']}:** {info['message']}")
            
            with col3:
                st.caption(info['timestamp'].strftime('%H:%M:%S'))
            
            st.divider()

def render_system_tab():
    """Renderizar pestaña de información del sistema"""
    st.subheader("📊 Estado del Sistema")
    
    # Estado de Ollama
    ollama_client = OllamaClient()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🤖 Estado de Ollama")
        
        if ollama_client.is_available():
            st.success("✅ Ollama está disponible")
            
            # Obtener modelos disponibles
            models = ollama_client.get_available_models()
            st.write(f"**Modelos disponibles:** {len(models)}")
            
            if models:
                st.write("**Lista de modelos:**")
                for model in models:
                    size_gb = model['size'] / (1024**3) if model['size'] > 0 else 0
                    st.write(f"• {model['name']} ({size_gb:.1f} GB)")
        else:
            st.error("❌ Ollama no está disponible")
            st.info("Ejecuta `ollama serve` para iniciar el servicio")
    
    with col2:
        st.write("### 📁 Estado de Archivos")
        
        # Información de directorios
        try:
            uploads_count = len(list(config.UPLOADS_DIR.glob("*")))
            processed_count = len(list(config.PROCESSED_DIR.glob("*")))
            cache_count = len(list(config.CACHE_DIR.glob("*")))
            
            st.metric("Archivos subidos", uploads_count)
            st.metric("Archivos procesados", processed_count)
            st.metric("Archivos en cache", cache_count)
            
        except Exception as e:
            st.error(f"Error al obtener información de archivos: {e}")
    
    st.divider()
    
    # Información de configuración
    st.write("### ⚙️ Configuración Actual")
    
    config_info = {
        "Modelo LLM por defecto": config.DEFAULT_LLM_MODEL,
        "Modelo de embeddings por defecto": config.DEFAULT_EMBEDDING_MODEL,
        "Tamaño de chunk": config.CHUNK_SIZE,
        "Overlap de chunk": config.CHUNK_OVERLAP,
        "Máximo documentos recuperados": config.MAX_RETRIEVAL_DOCS,
        "Umbral de similitud": config.SIMILARITY_THRESHOLD,
        "Tamaño máximo de archivo (MB)": config.MAX_FILE_SIZE_MB
    }
    
    for key, value in config_info.items():
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{key}:**")
        with col2:
            st.write(str(value))
    
    st.divider()
    
    # Logs recientes
    st.write("### 📝 Logs del Sistema")
    
    try:
        log_file = config.LOGS_DIR / f"cognichat_{datetime.now().strftime('%Y%m%d')}.log"
        
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-10:]  # Últimas 10 líneas
                
                st.code('\n'.join(recent_lines))
        else:
            st.info("No hay logs disponibles para hoy")
            
    except Exception as e:
        st.error(f"Error al leer logs: {e}")