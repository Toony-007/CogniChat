"""
Función Principal de Renderizado
Integración con la aplicación principal (app.py)
"""

import streamlit as st
from typing import List, Dict, Any

from ..core.config import AnalysisConfig
from ..core.analyzer import QualitativeAnalyzer
from ..core.rag_cache_manager import RAGCacheManager
from .tabs.concepts_tab import render_concepts_tab
from .tabs.topics_tab import render_topics_tab
from .tabs.relations_tab import render_relations_tab
from .components.cache_management import render_cache_management_panel


def render():
    """
    Función principal de renderizado del módulo de análisis cualitativo
    
    Esta función se llama desde app.py y mantiene compatibilidad con la estructura actual.
    
    Renderiza:
    1. Introducción al sistema
    2. Detección automática de rag_cache.json
    3. Verificación de documentos disponibles
    4. Tabs para cada sub-módulo de análisis
    """
    
    # Header principal
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🔬 Análisis Cualitativo Avanzado</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.95;">
            Sistema Inteligente de Análisis de Contenido para Investigación
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Introducción y principios
    with st.expander("ℹ️ Acerca de este Módulo", expanded=False):
        st.markdown("""
        ### 🎯 Principios Fundamentales
        
        Este sistema ha sido diseñado específicamente para **asistir a investigadores**
        en el análisis profundo de contenido, siguiendo estos principios:
        
        #### 1. 🧭 Asistencia al Investigador
        - Explicaciones claras sobre qué hace cada análisis
        - Guías de interpretación de resultados
        - Información sobre metodologías utilizadas
        
        #### 2. 🧠 Procesamiento Inteligente
        - **NO copia y pega** información de las fuentes
        - Analiza, sintetiza y contextualiza el contenido
        - Genera valor añadido mediante procesamiento inteligente
        
        #### 3. 📚 Fundamentación Completa
        - Cada resultado está respaldado por citas a fuentes originales
        - Sistema de citación que permite verificar de dónde viene cada información
        - Trazabilidad completa del análisis
        
        #### 4. 🔬 Transparencia Metodológica
        - Algoritmos y técnicas claramente documentados
        - Limitaciones y consideraciones explícitas
        - Validación humana siempre recomendada
        
        ### 📋 Sub-módulos Disponibles
        
        **Actualmente implementado:**
        - ✅ **Extracción de Conceptos Clave**: Identifica términos y frases más relevantes con TF-IDF
        
        **En desarrollo futuro:**
        - 🔜 **Análisis de Temas**: Identificación de temas principales con LDA
        - 🔜 **Triangulación**: Validación cruzada entre múltiples fuentes
        - 🔜 **Mapas Conceptuales**: Visualización de relaciones entre conceptos
        - 🔜 **Análisis de Relaciones**: Identificación de conexiones y patrones
        
        ### 🎓 Cómo Usar Este Sistema
        
        1. **Prepara tus documentos**: Sube y procesa tus archivos en las pestañas anteriores
        2. **Selecciona un análisis**: Elige el tipo de análisis que necesitas
        3. **Lee la metodología**: Entiende cómo funciona el análisis antes de ejecutarlo
        4. **Ejecuta el análisis**: Configura parámetros y ejecuta
        5. **Interpreta resultados**: Usa las guías de interpretación proporcionadas
        6. **Verifica fuentes**: Revisa las citas para validar los resultados
        7. **Exporta datos**: Guarda los resultados para tu investigación
        
        ### ⚠️ Importante
        
        Este sistema es una **herramienta de asistencia**, no un reemplazo del análisis humano.
        Los resultados deben ser:
        - ✅ Revisados por el investigador
        - ✅ Validados con el conocimiento experto del dominio
        - ✅ Contextualizados según los objetivos de la investigación
        - ✅ Complementados con análisis cualitativo manual cuando sea necesario
        """)
    
    # Inicializar gestor de cache RAG
    cache_manager = RAGCacheManager()
    
    # Detectar cache automáticamente
    st.markdown("### 🔍 Detección Automática de Cache RAG")
    cache_manager.render_cache_status()
    
    st.divider()
    
    # Obtener chunks de documentos procesados
    chunks = _get_processed_chunks_with_cache(cache_manager)
    
    # Verificar si hay documentos disponibles
    if not chunks:
        st.warning("""
        ### ⚠️ No hay documentos disponibles para análisis
        
        Para usar este módulo, primero debes:
        
        1. **📄 Gestión de Documentos**: Sube tus archivos (PDF, DOCX, TXT, etc.)
        2. **🧠 Procesamiento RAG**: Procesa los documentos para crear chunks
        3. **🔬 Análisis Cualitativo**: Regresa aquí para realizar análisis
        
        Una vez que hayas procesado tus documentos, podrás:
        - Extraer conceptos clave
        - Analizar temas y patrones
        - Generar visualizaciones
        - Obtener citas y referencias
        """)
        
        # Mostrar botón para ir a gestión de documentos
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("📄 Ir a Gestión de Documentos", type="primary", use_container_width=True):
                st.info("Por favor, usa las pestañas en la parte superior para navegar")
        
        return
    
    # Información sobre documentos disponibles
    st.success(f"""
    ✅ **{len(chunks)} fragmentos de texto disponibles** de {len(_get_unique_sources(chunks))} documento(s)
    """)
    
    # Inicializar configuración en session_state si no existe
    if 'qualitative_analysis_config' not in st.session_state:
        st.session_state.qualitative_analysis_config = AnalysisConfig()
    
    config = st.session_state.qualitative_analysis_config
    
    # Configuración global en sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuración Global")
        
        with st.expander("🎛️ Opciones de Interfaz", expanded=False):
            config.show_explanations = st.checkbox(
                "Mostrar explicaciones",
                value=config.show_explanations,
                help="Mostrar cajas informativas sobre cómo funciona cada análisis"
            )
            
            config.show_methodology = st.checkbox(
                "Mostrar metodología",
                value=config.show_methodology,
                help="Mostrar detalles técnicos sobre algoritmos utilizados"
            )
            
            config.show_interpretation_guide = st.checkbox(
                "Mostrar guías de interpretación",
                value=config.show_interpretation_guide,
                help="Mostrar guías sobre cómo interpretar resultados"
            )
            
            config.enable_citations = st.checkbox(
                "Habilitar citación",
                value=config.enable_citations,
                help="Generar citas a fuentes originales"
            )
        
        # Información de ayuda
        st.markdown("---")
        st.markdown("### 💡 Ayuda Rápida")
        
        st.markdown("""
        **Funcionalidades Implementadas:**
        - 🔍 **Conceptos Clave**: Identifica términos importantes con TF-IDF
        - 🎯 **Análisis de Temas**: Extrae temas principales usando LDA
        - 🔗 **Análisis de Relaciones**: Descubre conexiones entre conceptos
        - 📚 **Sistema de Citación**: Referencias a fuentes originales
        - 🔧 **Gestión de Cache**: Optimización de rendimiento
        """)
    
    # Tabs para diferentes sub-módulos (orden lógico de análisis)
    tabs = st.tabs([
        "🔧 Gestión de Cache",
        "🔍 Conceptos Clave",
        "🎯 Análisis de Temas", 
        "🔗 Análisis de Relaciones"
    ])
    
    # Tab 1: Gestión de Cache (IMPLEMENTADO)
    with tabs[0]:
        render_cache_management_panel()
    
    # Tab 2: Extracción de Conceptos (IMPLEMENTADO)
    with tabs[1]:
        render_concepts_tab(chunks, config)
    
    # Tab 3: Análisis de Temas (IMPLEMENTADO)
    with tabs[2]:
        render_topics_tab(chunks, config)
    
    # Tab 4: Análisis de Relaciones (IMPLEMENTADO)
    with tabs[3]:
        render_relations_tab(chunks, config)


def _get_processed_chunks_with_cache(cache_manager: RAGCacheManager) -> List[Dict[str, Any]]:
    """
    Obtener chunks procesados de documentos con detección automática de cache
    
    Args:
        cache_manager: Gestor de cache RAG
        
    Returns:
        Lista de chunks con contenido y metadatos
    """
    try:
        # Prioridad 1: Intentar obtener del cache RAG
        chunks = cache_manager.get_chunks_for_analysis()
        if chunks:
            st.info(f"✅ Chunks obtenidos del cache RAG: {len(chunks)} fragmentos")
            return chunks
        
        # Prioridad 2: Intentar obtener del procesador RAG
        from utils.rag_processor import rag_processor
        
        if hasattr(rag_processor, 'vector_store') and rag_processor.vector_store:
            all_chunks = []
            
            try:
                # Hacer una búsqueda amplia para obtener todos los chunks
                results = rag_processor.search("", top_k=1000)
                
                for result in results:
                    chunk = {
                        'content': result.get('content', ''),
                        'metadata': result.get('metadata', {})
                    }
                    
                    # Asegurar que metadata tenga source_file
                    if 'source_file' not in chunk['metadata']:
                        chunk['metadata']['source_file'] = 'unknown'
                    
                    all_chunks.append(chunk)
                
                if all_chunks:
                    st.info(f"✅ Chunks obtenidos del procesador RAG: {len(all_chunks)} fragmentos")
                    return all_chunks
                
            except Exception as e:
                st.warning(f"⚠️ Error al obtener chunks del procesador RAG: {str(e)}")
        
        # Prioridad 3: Buscar en session_state
        if 'processed_chunks' in st.session_state:
            chunks = st.session_state.processed_chunks
            if chunks:
                st.info(f"✅ Chunks obtenidos de session_state: {len(chunks)} fragmentos")
                return chunks
        
        # Si no hay chunks, retornar lista vacía
        return []
    
    except Exception as e:
        st.error(f"❌ Error al obtener chunks: {str(e)}")
        return []


def _get_processed_chunks() -> List[Dict[str, Any]]:
    """
    Obtener chunks procesados de documentos (método legacy)
    
    Returns:
        Lista de chunks con contenido y metadatos
    """
    try:
        # Importar el procesador RAG
        from utils.rag_processor import rag_processor
        
        # Obtener todos los chunks del procesador
        if hasattr(rag_processor, 'vector_store') and rag_processor.vector_store:
            # Si hay vector store, obtener chunks desde ahí
            all_chunks = []
            
            # Intentar obtener chunks de la base de datos vectorial
            try:
                # Hacer una búsqueda amplia para obtener todos los chunks
                results = rag_processor.search("", top_k=1000)
                
                for result in results:
                    chunk = {
                        'content': result.get('content', ''),
                        'metadata': result.get('metadata', {})
                    }
                    
                    # Asegurar que metadata tenga source_file
                    if 'source_file' not in chunk['metadata']:
                        chunk['metadata']['source_file'] = 'unknown'
                    
                    all_chunks.append(chunk)
                
                return all_chunks
            
            except:
                pass
        
        # Fallback: Buscar en session_state
        if 'processed_chunks' in st.session_state:
            return st.session_state.processed_chunks
        
        # Si no hay chunks, retornar lista vacía
        return []
    
    except Exception as e:
        st.error(f"Error al obtener chunks: {str(e)}")
        return []


def _get_unique_sources(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Obtener lista de fuentes únicas
    
    Args:
        chunks: Lista de chunks
        
    Returns:
        Lista de nombres de archivos fuente únicos
    """
    sources = set()
    for chunk in chunks:
        source = chunk.get('metadata', {}).get('source_file', 'unknown')
        sources.add(source)
    
    return list(sources)

