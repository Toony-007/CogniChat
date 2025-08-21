"""
Pestaña de configuraciones
"""

import streamlit as st
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from utils.error_handler import ErrorHandler
from utils.logger import setup_logger
from utils.ollama_client import OllamaClient
from config.settings import config

logger = setup_logger()
error_handler = ErrorHandler()

def render():
    """Renderizar la pestaña de configuraciones"""
    st.header("⚙️ Configuraciones del Sistema")
    
    # Estado de conexión
    ollama_client = OllamaClient()
    ollama_available = ollama_client.is_available()
    
    if ollama_available:
        st.success("✅ Ollama conectado - Configuración disponible")
    else:
        st.error("❌ Ollama desconectado - Funcionalidad limitada")
    
    # Pestañas de configuración con mejor organización
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Modelos IA", 
        "🔍 RAG & Procesamiento", 
        "🔧 Sistema", 
        "💾 Backup & Restore"
    ])
    
    with tab1:
        render_models_config()
    
    with tab2:
        render_processing_config()
    
    with tab3:
        render_system_config()
    
    with tab4:
        render_export_import()

def render_models_config():
    """Configuración de modelos de IA con diseño mejorado"""
    st.subheader("🤖 Configuración de Modelos de IA")
    
    ollama_client = OllamaClient()
    
    if not ollama_client.is_available():
        st.error("❌ Ollama no está disponible. No se pueden configurar modelos.")
        st.markdown("""
        **Para resolver este problema:**
        1. Asegúrate de que Ollama esté instalado
        2. Ejecuta `ollama serve` en tu terminal
        3. Verifica que no haya problemas de firewall
        """)
        return
    
    # Obtener modelos disponibles con caché
    if 'available_models_settings' not in st.session_state or st.session_state.get('models_settings_last_refresh', 0) < time.time() - 300:
        # Actualizar caché cada 5 minutos
        available_models = ollama_client.get_available_models()
        st.session_state.available_models_settings = available_models
        st.session_state.models_settings_last_refresh = time.time()
    else:
        available_models = st.session_state.available_models_settings
    # Extraer nombres de modelos como cadenas
    model_names = []
    for model in available_models:
        if isinstance(model, dict):
            model_names.append(model.get('name', str(model)))
        else:
            model_names.append(str(model))
    
    # Sección de modelos principales
    st.markdown("### 🎯 Modelos Principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">💬 Modelo LLM Principal</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Modelo para generar respuestas</p>
        </div>
        """, unsafe_allow_html=True)
        
        if model_names:
            current_llm = st.session_state.get('selected_llm_model', config.DEFAULT_LLM_MODEL)
            
            selected_llm = st.selectbox(
                "Seleccionar modelo LLM",
                model_names,
                index=model_names.index(current_llm) if current_llm in model_names else 0,
                key="llm_model_selector",
                help="Modelo principal para generar respuestas del chatbot"
            )
            
            # Información del modelo
            if selected_llm:
                st.markdown(f"""
                <div style="background: #2c3e50; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; color: #ecf0f1;">
                    <strong>📊 Información del Modelo</strong><br>
                    <small>• Modelo: {selected_llm}</small><br>
                    <small>• Tipo: Generación de texto</small><br>
                    <small>• Estado: ✅ Disponible</small>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("💾 Guardar Modelo LLM", type="primary", key="save_llm_model"):
                st.session_state.selected_llm_model = selected_llm
                st.success(f"✅ Modelo LLM configurado: {selected_llm}")
                st.rerun()
        else:
            st.warning("⚠️ No hay modelos LLM disponibles")
            st.info("💡 Descarga un modelo usando: `ollama pull llama3.1:8b`")
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">🔤 Modelo de Embeddings</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Modelo para vectorización de texto</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filtrar modelos de embeddings
        embedding_models = []
        for name in model_names:
            if isinstance(name, str) and ('embed' in name.lower() or 'minilm' in name.lower()):
                embedding_models.append(name)
        
        if embedding_models:
            current_embedding = st.session_state.get('selected_embedding_model', config.DEFAULT_EMBEDDING_MODEL)
            
            selected_embedding = st.selectbox(
                "Seleccionar modelo de embeddings",
                embedding_models,
                index=embedding_models.index(current_embedding) if current_embedding in embedding_models else 0,
                key="embedding_model_selector",
                help="Modelo para convertir texto en vectores para búsqueda semántica"
            )
            
            # Información del modelo
            if selected_embedding:
                st.markdown(f"""
                <div style="background: #2c3e50; padding: 1rem; border-radius: 8px; border-left: 4px solid #f093fb; color: #ecf0f1;">
                    <strong>📊 Información del Modelo</strong><br>
                    <small>• Modelo: {selected_embedding}</small><br>
                    <small>• Tipo: Embeddings</small><br>
                    <small>• Estado: ✅ Disponible</small>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("💾 Guardar Modelo Embeddings", type="primary", key="save_embedding_model"):
                st.session_state.selected_embedding_model = selected_embedding
                st.success(f"✅ Modelo de embeddings configurado: {selected_embedding}")
                st.rerun()
        else:
            st.warning("⚠️ No hay modelos de embeddings disponibles")
            st.info("💡 Descarga un modelo usando: `ollama pull nomic-embed-text`")
    
    st.divider()
    
    # Gestión de modelos
    st.markdown("### 📦 Gestión de Modelos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📋 Modelos Instalados:**")
        
        if available_models:
            for i, model in enumerate(available_models):
                # Extraer nombre del modelo correctamente
                if isinstance(model, dict):
                    model_name = model.get('name', str(model))
                else:
                    model_name = str(model)
                
                # Determinar tipo de modelo
                model_type = "🔤 Embeddings" if any(keyword in model_name.lower() for keyword in ['embed', 'minilm']) else "💬 LLM"
                
                st.markdown(f"""
                <div style="background: #34495e; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #00FF99; color: #ecf0f1;">
                    <strong>{model_name}</strong> <span style="color: #bdc3c7;">({model_type})</span><br>
                    <small style="color: #95a5a6;">Estado: ✅ Instalado y disponible</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📦 No hay modelos instalados")
    
    with col2:
        st.markdown("**📥 Descargar Modelo:**")
        
        # Modelos recomendados
        recommended_models = [
            "llama3.1:8b",
            "llama3.2:3b", 
            "nomic-embed-text",
            "all-minilm:latest"
        ]
        
        model_to_download = st.selectbox(
            "Modelo recomendado",
            ["Seleccionar..."] + recommended_models,
            help="Selecciona un modelo recomendado o escribe uno personalizado abajo"
        )
        
        custom_model = st.text_input(
            "O modelo personalizado",
            placeholder="Ej: deepseek-r1:7b",
            help="Nombre exacto del modelo de Ollama"
        )
        
        final_model = custom_model if custom_model else (model_to_download if model_to_download != "Seleccionar..." else "")
        
        if st.button("📥 Descargar Modelo", disabled=not final_model, key="download_model"):
            if final_model:
                with st.spinner(f"Descargando {final_model}..."):
                    success = ollama_client.pull_model(final_model)
                    if success:
                        st.success(f"✅ Modelo {final_model} descargado correctamente")
                        st.rerun()
                    else:
                        st.error(f"❌ Error al descargar {final_model}")

def render_processing_config():
    """Configuración de procesamiento RAG con diseño mejorado"""
    st.subheader("🔍 Configuración RAG y Procesamiento")
    
    # Configuración RAG principal
    st.markdown("### 🎯 Configuración Principal RAG")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">📝 Configuración de Chunks</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">División de documentos en fragmentos</p>
        </div>
        """, unsafe_allow_html=True)
        
        chunk_size = st.slider(
            "Tamaño de chunk (caracteres)",
            min_value=200,
            max_value=2000,
            value=st.session_state.get('chunk_size', config.CHUNK_SIZE),
            step=100,
            help="Tamaño en caracteres de cada fragmento de texto"
        )
        
        chunk_overlap = st.slider(
            "Overlap entre chunks",
            min_value=0,
            max_value=500,
            value=st.session_state.get('chunk_overlap', config.CHUNK_OVERLAP),
            step=50,
            help="Solapamiento entre fragmentos consecutivos"
        )
        
        # Visualización del impacto
        estimated_chunks = 1000 // (chunk_size - chunk_overlap) if chunk_size > chunk_overlap else 1
        st.info(f"📊 Estimación: ~{estimated_chunks} chunks por cada 1000 caracteres")
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; color: #ecf0f1; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #ecf0f1;">🎯 Configuración de Búsqueda</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Parámetros de recuperación de información</p>
        </div>
        """, unsafe_allow_html=True)
        
        max_retrieval_docs = st.slider(
            "Máximo documentos recuperados",
            min_value=1,
            max_value=20,
            value=st.session_state.get('max_retrieval_docs', config.MAX_RETRIEVAL_DOCS),
            help="Número máximo de documentos a recuperar en RAG"
        )
        
        similarity_threshold = st.slider(
            "Umbral de similitud",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('similarity_threshold', config.SIMILARITY_THRESHOLD),
            step=0.1,
            help="Umbral mínimo de similitud para considerar un documento relevante"
        )
        
        # Información sobre el umbral
        if similarity_threshold < 0.3:
            st.warning("⚠️ Umbral muy bajo - puede incluir contenido irrelevante")
        elif similarity_threshold > 0.8:
            st.warning("⚠️ Umbral muy alto - puede excluir contenido relevante")
        else:
            st.success("✅ Umbral en rango óptimo")
    
    # Configuración de archivos
    st.markdown("### 📁 Configuración de Archivos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_file_size = st.slider(
            "Tamaño máximo de archivo (MB)",
            min_value=1,
            max_value=500,
            value=st.session_state.get('max_file_size', config.MAX_FILE_SIZE_MB),
            help="Tamaño máximo permitido para archivos subidos"
        )
    
    with col2:
        st.markdown("**📋 Formatos Soportados:**")
        
        # Mostrar formatos en un diseño más atractivo
        formats_html = ""
        for format_ext in config.SUPPORTED_FORMATS:
            formats_html += f'<span style="background: #34495e; color: #ecf0f1; padding: 0.2rem 0.5rem; border-radius: 4px; margin: 0.1rem; display: inline-block; font-size: 0.8rem;">{format_ext}</span> '
        
        st.markdown(f'<div style="line-height: 2;">{formats_html}</div>', unsafe_allow_html=True)
    
    # Botón para guardar configuración
    if st.button("💾 Guardar Configuración RAG", type="primary", key="save_rag_config"):
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        st.session_state.max_retrieval_docs = max_retrieval_docs
        st.session_state.similarity_threshold = similarity_threshold
        st.session_state.max_file_size = max_file_size
        
        st.success("✅ Configuración RAG guardada correctamente")
        st.rerun()
    
    st.divider()
    
    # Vista previa de configuración actual
    st.markdown("### 👀 Vista Previa de Configuración Actual")
    
    current_config_display = {
        "Tamaño de chunk": f"{chunk_size} caracteres",
        "Overlap": f"{chunk_overlap} caracteres",
        "Docs máximos": f"{max_retrieval_docs} documentos",
        "Umbral similitud": f"{similarity_threshold:.1f}",
        "Tamaño máximo": f"{max_file_size} MB"
    }
    
    cols = st.columns(len(current_config_display))
    for i, (key, value) in enumerate(current_config_display.items()):
        with cols[i]:
            st.metric(key, value)

def render_system_config():
    """Configuración del sistema con diseño mejorado"""
    st.subheader("🔧 Configuración del Sistema")
    
    # Configuración de Ollama
    st.markdown("### 🤖 Configuración de Ollama")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">🌐 Conexión Ollama</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Configuración de servidor Ollama</p>
        </div>
        """, unsafe_allow_html=True)
        
        ollama_url = st.text_input(
            "URL de Ollama",
            value=st.session_state.get('ollama_url', config.OLLAMA_BASE_URL),
            placeholder="http://localhost:11434",
            help="URL del servidor Ollama"
        )
        
        ollama_timeout = st.slider(
            "Timeout (segundos)",
            min_value=10,
            max_value=300,
            value=st.session_state.get('ollama_timeout', config.OLLAMA_TIMEOUT),
            help="Tiempo límite para conexiones con Ollama"
        )
        
        # Test de conexión
        if st.button("🔍 Probar Conexión", key="test_connection"):
            try:
                test_client = OllamaClient()
                if test_client.is_available():
                    st.success("✅ Conexión exitosa con Ollama")
                else:
                    st.error("❌ No se pudo conectar con Ollama")
            except Exception as e:
                st.error(f"❌ Error de conexión: {str(e)}")
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">📊 Configuración de Logs</h4>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Nivel de registro del sistema</p>
        </div>
        """, unsafe_allow_html=True)
        
        log_level = st.selectbox(
            "Nivel de log",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                st.session_state.get('log_level', config.LOG_LEVEL)
            ),
            help="Nivel de detalle en los logs del sistema"
        )
        
        # Información sobre niveles de log
        log_info = {
            "DEBUG": "🔍 Información muy detallada para desarrollo",
            "INFO": "ℹ️ Información general del funcionamiento",
            "WARNING": "⚠️ Solo advertencias y errores",
            "ERROR": "❌ Solo errores críticos"
        }
        
        st.info(log_info[log_level])
    
    # Botón para guardar configuración del sistema
    if st.button("💾 Guardar Configuración del Sistema", type="primary", key="save_system_config_1"):
        st.session_state.ollama_url = ollama_url
        st.session_state.ollama_timeout = ollama_timeout
        st.session_state.log_level = log_level
        
        st.success("✅ Configuración del sistema guardada correctamente")
        st.rerun()
    
    st.divider()
    
    st.divider()
    
    # Configuración de rutas del sistema
    st.markdown("### 📁 Rutas del Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📂 Directorio de Datos**")
        data_dir = st.text_input(
            "Ruta de datos",
            value=st.session_state.get('data_dir', str(config.DATA_DIR)),
            help="Directorio principal para almacenar datos"
        )
        
        # Verificar si existe
        if os.path.exists(data_dir):
            st.success("✅ Directorio existe")
        else:
            st.warning("⚠️ Directorio no existe")
            if st.button("📁 Crear Directorio", key="create_data_dir"):
                try:
                    os.makedirs(data_dir, exist_ok=True)
                    st.success("✅ Directorio creado")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("**📤 Directorio de Subidas**")
        uploads_dir = st.text_input(
            "Ruta de uploads",
            value=st.session_state.get('uploads_dir', str(config.UPLOADS_DIR)),
            help="Directorio para archivos subidos"
        )
        
        if os.path.exists(uploads_dir):
            file_count = len([f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))])
            st.success(f"✅ {file_count} archivos")
        else:
            st.warning("⚠️ Directorio no existe")
    
    with col3:
        st.markdown("**🗄️ Directorio de Cache**")
        cache_dir = st.text_input(
            "Ruta de cache",
            value=st.session_state.get('cache_dir', str(config.CACHE_DIR)),
            help="Directorio para archivos de cache"
        )
        
        if os.path.exists(cache_dir):
            st.success("✅ Directorio existe")
            
            # Mostrar tamaño del cache
            try:
                cache_size = sum(
                    os.path.getsize(os.path.join(cache_dir, f))
                    for f in os.listdir(cache_dir)
                    if os.path.isfile(os.path.join(cache_dir, f))
                )
                cache_size_mb = cache_size / (1024 * 1024)
                st.info(f"📊 Tamaño: {cache_size_mb:.1f} MB")
                
                if st.button("🗑️ Limpiar Cache", key="clear_cache"):
                    try:
                        for f in os.listdir(cache_dir):
                            file_path = os.path.join(cache_dir, f)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        st.success("✅ Cache limpiado")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            except Exception:
                st.warning("⚠️ No se pudo calcular el tamaño")
        else:
            st.warning("⚠️ Directorio no existe")
    
    # Botón para guardar configuración del sistema
    if st.button("💾 Guardar Configuración del Sistema", type="primary", key="save_system_config_2"):
        st.session_state.ollama_url = ollama_url
        st.session_state.ollama_timeout = ollama_timeout
        st.session_state.log_level = log_level
        st.session_state.data_dir = data_dir
        st.session_state.uploads_dir = uploads_dir
        st.session_state.cache_dir = cache_dir
        
        st.success("✅ Configuración del sistema guardada correctamente")
        st.rerun()
    
    st.divider()
    
    # Información del sistema
    st.markdown("### 💻 Información del Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🐍 Python", f"{sys.version.split()[0]}")
        st.metric("💾 Streamlit", st.__version__)
    
    with col2:
        # Información de memoria
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.metric("🧠 RAM Total", f"{memory.total / (1024**3):.1f} GB")
            st.metric("📊 RAM Usada", f"{memory.percent}%")
        except ImportError:
            st.metric("🧠 RAM", "N/A")
            st.metric("📊 Uso", "N/A")
    
    with col3:
        # Información de disco
        try:
            import psutil
            disk = psutil.disk_usage('/')
            st.metric("💽 Disco Total", f"{disk.total / (1024**3):.1f} GB")
            st.metric("📈 Disco Usado", f"{(disk.used / disk.total) * 100:.1f}%")
        except:
            st.metric("💽 Disco", "N/A")
            st.metric("📈 Uso", "N/A")

def render_export_import():
    """Funcionalidad de backup y restore con diseño mejorado"""
    st.subheader("💾 Backup & Restore")
    
    # Información general
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
        <h3 style="margin: 0; color: white;">🔄 Gestión de Configuraciones</h3>
        <p style="margin: 0.5rem 0 0 0;">Exporta, importa y restaura las configuraciones del sistema</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Exportar Configuración")
        
        st.markdown("""
        <div style="background: #27ae60; color: #ecf0f1; padding: 1rem; border-radius: 8px; border-left: 4px solid #2ecc71; margin-bottom: 1rem;">
            <strong>📋 Incluye:</strong><br>
            <small>• Modelos seleccionados</small><br>
            <small>• Configuración RAG</small><br>
            <small>• Configuración del sistema</small><br>
            <small>• Parámetros de procesamiento</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Opciones de exportación
        export_options = st.multiselect(
            "Seleccionar qué exportar",
            [
                "🤖 Configuración de modelos",
                "🔍 Configuración RAG",
                "🔧 Configuración del sistema"
            ],
            default=[
                "🤖 Configuración de modelos",
                "🔍 Configuración RAG",
                "🔧 Configuración del sistema"
            ]
        )
        
        if st.button("📥 Exportar Configuración", type="primary", key="export_config"):
            try:
                # Crear configuración para exportar
                export_config = {}
                
                if "🤖 Configuración de modelos" in export_options:
                    export_config.update({
                        'selected_llm_model': st.session_state.get('selected_llm_model', config.DEFAULT_LLM_MODEL),
                        'selected_embedding_model': st.session_state.get('selected_embedding_model', config.DEFAULT_EMBEDDING_MODEL)
                    })
                
                if "🔍 Configuración RAG" in export_options:
                    export_config.update({
                        'chunk_size': st.session_state.get('chunk_size', config.CHUNK_SIZE),
                        'chunk_overlap': st.session_state.get('chunk_overlap', config.CHUNK_OVERLAP),
                        'max_retrieval_docs': st.session_state.get('max_retrieval_docs', config.MAX_RETRIEVAL_DOCS),
                        'similarity_threshold': st.session_state.get('similarity_threshold', config.SIMILARITY_THRESHOLD),
                        'max_file_size': st.session_state.get('max_file_size', config.MAX_FILE_SIZE_MB)
                    })
                
                if "🔧 Configuración del sistema" in export_options:
                    export_config.update({
                        'ollama_url': st.session_state.get('ollama_url', config.OLLAMA_BASE_URL),
                        'ollama_timeout': st.session_state.get('ollama_timeout', config.OLLAMA_TIMEOUT),
                        'log_level': st.session_state.get('log_level', config.LOG_LEVEL)
                    })
                
                # Agregar metadatos
                from datetime import datetime
                export_config['_metadata'] = {
                    'export_date': datetime.now().isoformat(),
                    'version': '1.0',
                    'app_name': 'CogniChat'
                }
                
                # Convertir a JSON
                config_json = json.dumps(export_config, indent=2, ensure_ascii=False)
                
                # Crear nombre de archivo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cognichat_config_{timestamp}.json"
                
                # Botón de descarga
                st.download_button(
                    label="💾 Descargar Configuración",
                    data=config_json,
                    file_name=filename,
                    mime="application/json"
                )
                
                st.success("✅ Configuración preparada para descarga")
                
            except Exception as e:
                st.error(f"❌ Error al exportar: {str(e)}")
    
    with col2:
        st.markdown("### 📥 Importar Configuración")
        
        st.markdown("""
        <div style="background: #3498db; color: #ecf0f1; padding: 1rem; border-radius: 8px; border-left: 4px solid #2980b9; margin-bottom: 1rem;">
            <strong>⚠️ Importante:</strong><br>
            <small>• Esto sobrescribirá la configuración actual</small><br>
            <small>• Se recomienda hacer backup antes</small><br>
            <small>• Solo archivos JSON válidos</small>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_config = st.file_uploader(
            "Seleccionar archivo de configuración",
            type=['json'],
            help="Archivo JSON exportado previamente"
        )
        
        if uploaded_config is not None:
            try:
                # Leer y parsear el archivo
                config_data = json.load(uploaded_config)
                
                # Mostrar información del archivo
                if '_metadata' in config_data:
                    metadata = config_data['_metadata']
                    st.markdown(f"""
                    <div style="background: #34495e; color: #ecf0f1; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <strong>📄 Información del Archivo:</strong><br>
                        <small>• Fecha: {metadata.get('export_date', 'N/A')}</small><br>
                        <small>• Versión: {metadata.get('version', 'N/A')}</small><br>
                        <small>• App: {metadata.get('app_name', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar vista previa de configuraciones
                st.markdown("**🔍 Vista Previa:**")
                
                preview_items = []
                if 'selected_llm_model' in config_data:
                    preview_items.append(f"• LLM: {config_data['selected_llm_model']}")
                if 'selected_embedding_model' in config_data:
                    preview_items.append(f"• Embeddings: {config_data['selected_embedding_model']}")
                if 'chunk_size' in config_data:
                    preview_items.append(f"• Chunk size: {config_data['chunk_size']}")
                if 'ollama_url' in config_data:
                    preview_items.append(f"• Ollama URL: {config_data['ollama_url']}")
                
                for item in preview_items[:5]:  # Mostrar solo los primeros 5
                    st.text(item)
                
                if len(preview_items) > 5:
                    st.text(f"... y {len(preview_items) - 5} más")
                
                # Botones de acción
                col_import, col_cancel = st.columns(2)
                
                with col_import:
                    if st.button("✅ Importar Configuración", type="primary", key="import_config"):
                        try:
                            # Aplicar configuración (excluyendo metadatos)
                            for key, value in config_data.items():
                                if not key.startswith('_'):
                                    if key == "llm_model":
                                        st.session_state.selected_llm_model = value
                                    elif key == "embedding_model":
                                        st.session_state.selected_embedding_model = value
                                    else:
                                        st.session_state[key] = value
                            
                            st.success("✅ Configuración importada correctamente")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error al importar: {str(e)}")
                
                with col_cancel:
                    if st.button("❌ Cancelar", key="cancel_import"):
                        st.rerun()
                        
            except json.JSONDecodeError:
                st.error("❌ Archivo JSON inválido")
            except Exception as e:
                st.error(f"❌ Error al procesar archivo: {str(e)}")
    
    st.divider()
    
    # Sección de reset
    st.markdown("### 🔄 Restaurar Configuración")
    
    st.markdown("""
    <div style="background: #d68910; color: #ecf0f1; padding: 1rem; border-radius: 8px; border-left: 4px solid #f39c12; margin-bottom: 1rem;">
        <strong>⚠️ Zona de Peligro</strong><br>
        <small>Estas acciones no se pueden deshacer</small>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restaurar a Valores por Defecto", type="secondary", key="restore_defaults"):
            # Confirmar acción
            if 'confirm_reset' not in st.session_state:
                st.session_state.confirm_reset = True
                st.warning("⚠️ ¿Estás seguro? Haz clic nuevamente para confirmar")
            else:
                # Limpiar configuración
                config_keys = [
                    'selected_llm_model', 'selected_embedding_model',
                    'chunk_size', 'chunk_overlap', 'max_retrieval_docs',
                    'similarity_threshold', 'max_file_size',
                    'ollama_url', 'ollama_timeout', 'log_level'
                ]
                
                for key in config_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                
                del st.session_state.confirm_reset
                st.success("✅ Configuración restaurada a valores por defecto")
                st.rerun()
    
    with col2:
        if st.button("🗑️ Limpiar Session State", type="secondary", key="clear_session_state"):
            if 'confirm_clear' not in st.session_state:
                st.session_state.confirm_clear = True
                st.error("🚨 ¿Estás seguro? Esto limpiará toda la sesión. Haz clic nuevamente para confirmar")
            else:
                try:
                    # Limpiar session state
                    for key in list(st.session_state.keys()):
                        if not key.startswith('_'):
                            del st.session_state[key]
                    
                    del st.session_state.confirm_clear
                    st.success("✅ Session state limpiado")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error al limpiar: {str(e)}")
                    if 'confirm_clear' in st.session_state:
                        del st.session_state.confirm_clear