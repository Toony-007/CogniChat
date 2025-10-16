"""
CogniChat - Sistema RAG Avanzado
Aplicación principal con interfaz Streamlit
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from config.settings import config
from utils.logger import setup_logger
from utils.error_handler import ErrorHandler
from modules import document_upload, chatbot, alerts, settings, document_processor, qualitative_analysis
from modules.document_upload import get_valid_uploaded_files

# Configuración de la página
st.set_page_config(
    page_title="CogniChat - Sistema RAG Avanzado",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar logger y manejo de errores
logger = setup_logger()
error_handler = ErrorHandler()

def load_custom_css():
    """Cargar estilos CSS personalizados"""
    st.markdown("""
    <style>
    /* Estilos principales */
    .main-header {
        background: linear-gradient(90deg, #00FF99 0%, #00CC7A 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .tab-container {
        background: #2c3e50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ecf0f1;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #00FF99;
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        background-color: #ff4444;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #34495e 0%, #2c3e50 100%);
    }
    
    .error-container {
        background: #c0392b;
        border: 1px solid #e74c3c;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ecf0f1;
    }
    
    .success-container {
        background: #27ae60;
        border: 1px solid #2ecc71;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ecf0f1;
    }
    
    .warning-container {
        background: #d68910;
        border: 1px solid #f39c12;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ecf0f1;
    }
    </style>
    """, unsafe_allow_html=True)

def check_ollama_status():
    """Verificar el estado de Ollama"""
    try:
        import requests
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error al conectar con Ollama: {e}")
        return False

def render_sidebar_config():
    """Renderizar configuración actual en la barra lateral"""
    st.sidebar.markdown("### 🎛️ Configuración Actual")
    
    # Verificar estado de Ollama
    ollama_status = check_ollama_status()
    status_class = "status-online" if ollama_status else "status-offline"
    status_text = "Conectado" if ollama_status else "Desconectado"
    
    st.sidebar.markdown(f"""
    <div style="padding: 0.8rem; background: #34495e; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {'#00FF99' if ollama_status else '#ff4444'}; color: #ecf0f1;">
        <h4 style="margin: 0; color: #ecf0f1;">🔧 Estado del Sistema</h4>
        <p style="margin: 0.5rem 0 0 0;"><span class="status-indicator {status_class}"></span>Ollama: {status_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if ollama_status:
        # Mostrar configuración de modelos
        st.sidebar.markdown("#### 🤖 Modelos Configurados")
        
        llm_model = st.session_state.get('selected_llm_model', config.DEFAULT_LLM_MODEL)
        embedding_model = st.session_state.get('selected_embedding_model', config.DEFAULT_EMBEDDING_MODEL)
        
        st.sidebar.markdown(f"""
        <div style="background: #2c3e50; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem; color: #ecf0f1;">
            <strong>💬 LLM:</strong><br>
            <code style="color: #00FF99;">{llm_model}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"""
        <div style="background: #2c3e50; padding: 0.5rem; border-radius: 5px; margin-bottom: 1rem; color: #ecf0f1;">
            <strong>🔤 Embeddings:</strong><br>
            <code style="color: #00FF99;">{embedding_model}</code>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar configuración RAG
        st.sidebar.markdown("#### ⚙️ Configuración RAG")
        
        chunk_size = st.session_state.get('chunk_size', config.CHUNK_SIZE)
        chunk_overlap = st.session_state.get('chunk_overlap', config.CHUNK_OVERLAP)
        max_retrieval_docs = st.session_state.get('max_retrieval_docs', config.MAX_RETRIEVAL_DOCS)
        similarity_threshold = st.session_state.get('similarity_threshold', config.SIMILARITY_THRESHOLD)
        
        st.sidebar.markdown(f"""
        <div style="background: #27ae60; padding: 0.8rem; border-radius: 8px; font-size: 0.9rem; color: #ecf0f1;">
            <div style="margin-bottom: 0.3rem;"><strong>📝 Chunk Size:</strong> {chunk_size}</div>
            <div style="margin-bottom: 0.3rem;"><strong>🔄 Overlap:</strong> {chunk_overlap}</div>
            <div style="margin-bottom: 0.3rem;"><strong>📊 Max Docs:</strong> {max_retrieval_docs}</div>
            <div><strong>🎯 Similitud:</strong> {similarity_threshold}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar estadísticas de documentos
        try:
            uploaded_files = get_valid_uploaded_files()
            
            st.sidebar.markdown("#### 📊 Estadísticas")
            st.sidebar.markdown(f"""
            <div style="background: #3498db; padding: 0.8rem; border-radius: 8px; font-size: 0.9rem; color: #ecf0f1;">
                <div><strong>📄 Documentos:</strong> {len(uploaded_files)}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass
    
    else:
        st.sidebar.error("⚠️ Ollama no disponible")
        st.sidebar.markdown("""
        <div style="background: #c0392b; padding: 0.8rem; border-radius: 8px; font-size: 0.9rem; color: #ecf0f1;">
            Para usar CogniChat, inicia Ollama:<br>
            <code style="color: #00FF99;">ollama serve</code>
        </div>
        """, unsafe_allow_html=True)

def initialize_session_state():
    """Inicializar valores por defecto en session_state"""
    if 'selected_llm_model' not in st.session_state:
        st.session_state.selected_llm_model = config.DEFAULT_LLM_MODEL
    
    if 'selected_embedding_model' not in st.session_state:
        st.session_state.selected_embedding_model = config.DEFAULT_EMBEDDING_MODEL
    
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = config.CHUNK_SIZE
    
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = config.CHUNK_OVERLAP
    
    if 'max_retrieval_docs' not in st.session_state:
        st.session_state.max_retrieval_docs = config.MAX_RETRIEVAL_DOCS
    
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = config.SIMILARITY_THRESHOLD

def main():
    """Función principal de la aplicación"""
    try:
        # Inicializar session_state con valores por defecto
        initialize_session_state()
        
        # Cargar estilos CSS
        load_custom_css()
        
        # Header principal mejorado
        st.markdown("""
        <div class="main-header">
            <h1>🧠 CogniChat</h1>
            <p style="color: white; margin: 0; font-size: 1.2rem; opacity: 0.9;">Sistema RAG Avanzado con IA Local</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Renderizar barra lateral con configuración
        render_sidebar_config()
        
        # Navegación por pestañas mejorada
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📄 Gestión de Documentos", 
            "🧠 Procesamiento RAG",
            "💬 Chat Inteligente", 
            "🔬 Análisis Cualitativo",
            "🚨 Monitoreo y Alertas", 
            "⚙️ Configuraciones"
        ])
        
        with tab1:
            document_upload.render()
        
        with tab2:
            document_processor.main()
        
        with tab3:
            chatbot.render()
        
        with tab4:
            qualitative_analysis.render()
        
        with tab5:
            alerts.render()
        
        with tab6:
            settings.render()
            
    except Exception as e:
        error_handler.handle_error(e, "Error en la aplicación principal")
        st.error("Ha ocurrido un error inesperado. Consulta la pestaña de Alertas para más detalles.")

if __name__ == "__main__":
    main()