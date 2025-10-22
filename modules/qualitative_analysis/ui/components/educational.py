"""
Componentes Educativos para Asistencia al Investigador
Estos componentes ayudan al investigador a comprender qué hace cada análisis y cómo interpretar los resultados
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from ...extractors.concept_extractor import ExtractedConcept
from ...core.citation_manager import Citation


def show_methodology_box(title: str, description: str, steps: List[str], algorithm_info: Optional[str] = None):
    """
    Mostrar caja explicativa sobre la metodología utilizada
    
    Args:
        title: Título de la metodología
        description: Descripción general
        steps: Lista de pasos del proceso
        algorithm_info: Información adicional sobre el algoritmo (opcional)
    """
    with st.expander("📖 ¿Cómo funciona este análisis?", expanded=False):
        st.markdown(f"### {title}")
        st.markdown(description)
        
        st.markdown("**📋 Proceso paso a paso:**")
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")
        
        if algorithm_info:
            st.info(f"💡 **Algoritmo utilizado:** {algorithm_info}")
        
        st.markdown("---")
        st.markdown("*Este análisis NO copia contenido, sino que identifica patrones y sintetiza información relevante.*")


def show_interpretation_guide(
    what_it_means: str,
    how_to_use: List[str],
    limitations: List[str]
):
    """
    Mostrar guía de interpretación de resultados
    
    Args:
        what_it_means: Qué significan los resultados
        how_to_use: Cómo usar los resultados en investigación
        limitations: Limitaciones a considerar
    """
    with st.expander("🎯 ¿Cómo interpretar estos resultados?", expanded=False):
        st.markdown("### Interpretación de Resultados")
        
        st.markdown("**📊 ¿Qué significa esto?**")
        st.markdown(what_it_means)
        
        st.markdown("**✅ ¿Cómo usar estos resultados?**")
        for item in how_to_use:
            st.markdown(f"- {item}")
        
        st.markdown("**⚠️ Limitaciones y consideraciones:**")
        for item in limitations:
            st.markdown(f"- {item}")


def show_concept_card(
    concept: ExtractedConcept,
    rank: int,
    show_citations: bool = True,
    show_related: bool = True
):
    """
    Mostrar tarjeta con información de un concepto
    
    Args:
        concept: Concepto extraído
        rank: Posición en el ranking
        show_citations: Si mostrar citas
        show_related: Si mostrar conceptos relacionados
    """
    # Color según relevancia
    if concept.relevance_score > 0.7:
        border_color = "#00FF99"  # Verde brillante
        bg_color = "#1a4d2e"
    elif concept.relevance_score > 0.4:
        border_color = "#00CCFF"  # Azul
        bg_color = "#1a3a4d"
    else:
        border_color = "#FFA500"  # Naranja
        bg_color = "#4d3a1a"
    
    # Renderizar tarjeta
    st.markdown(f"""
    <div style="
        background: {bg_color};
        border-left: 4px solid {border_color};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ecf0f1;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
                <h3 style="margin: 0; color: #ecf0f1;">#{rank}. {concept.concept}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #bdc3c7; font-size: 0.9rem;">
                    📊 Frecuencia: {concept.frequency} | 
                    ⭐ Relevancia: {concept.relevance_score:.3f} | 
                    📚 Fuentes: {len(concept.sources)}
                </p>
            </div>
            <div style="
                background: {border_color};
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
            ">
                Score: {concept.relevance_score:.2f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Información adicional en columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Contexto
        if concept.context_examples:
            st.markdown("**📝 Ejemplos de contexto:**")
            for i, example in enumerate(concept.context_examples[:2], 1):
                # Limpiar y truncar
                clean_example = example.replace('[', '**[').replace(']', ']**')
                if len(clean_example) > 200:
                    clean_example = clean_example[:200] + "..."
                st.markdown(f"*{clean_example}*")
        
        # Citas
        if show_citations and concept.citations:
            with st.expander("📚 Ver citas y fuentes", expanded=False):
                show_citation_box(concept.citations[:3])  # Mostrar máximo 3
    
    with col2:
        # Fuentes
        st.markdown("**📂 Fuentes:**")
        source_dist = concept.get_source_distribution()
        for source, count in list(source_dist.items())[:3]:  # Máximo 3 fuentes
            source_name = source.split('/')[-1]  # Solo el nombre del archivo
            if len(source_name) > 30:
                source_name = source_name[:27] + "..."
            st.markdown(f"• `{source_name}` ({count})")
        
        if len(source_dist) > 3:
            st.markdown(f"*... y {len(source_dist) - 3} más*")
        
        # Conceptos relacionados
        if show_related and concept.related_concepts:
            st.markdown("**🔗 Relacionado con:**")
            for related in concept.related_concepts[:3]:
                st.markdown(f"• {related}")


def show_citation_box(citations: List[Citation]):
    """
    Mostrar caja con citas y referencias
    
    Args:
        citations: Lista de citas a mostrar
    """
    if not citations:
        st.info("No hay citas disponibles para este elemento")
        return
    
    st.markdown("**📖 Citas y Referencias:**")
    
    for i, citation in enumerate(citations, 1):
        # Información de la cita
        st.markdown(f"""
        <div style="
            background: #2c3e50;
            border-left: 3px solid #3498db;
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            color: #ecf0f1;
        ">
            <strong>Cita #{i}</strong><br>
            <small>📄 Fuente: {citation.source_file}</small><br>
            <small>🔖 ID: {citation.citation_id}</small>
            {f'<br><small>📃 Página: {citation.page_number}</small>' if citation.page_number else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Contexto completo
        context = citation.get_full_context(max_chars=250)
        st.markdown(f"*{context}*")
        
        # Formato académico
        academic_citation = citation.format_citation(style="academic")
        st.code(academic_citation, language="text")
        
        st.markdown("---")


def show_statistics_panel(stats: Dict[str, Any]):
    """
    Mostrar panel de estadísticas generales
    
    Args:
        stats: Diccionario con estadísticas
    """
    st.markdown("### 📊 Estadísticas del Análisis")
    
    # Métricas en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔍 Conceptos Extraídos",
            value=stats.get('total_concepts', 0),
            help="Número total de conceptos clave identificados"
        )
    
    with col2:
        st.metric(
            label="📚 Fuentes Únicas",
            value=stats.get('unique_sources', 0),
            help="Número de documentos fuente analizados"
        )
    
    with col3:
        st.metric(
            label="📖 Citas Generadas",
            value=stats.get('total_citations', 0),
            help="Número total de citas a fuentes originales"
        )
    
    with col4:
        avg_rel = stats.get('avg_relevance', 0.0)
        st.metric(
            label="⭐ Relevancia Promedio",
            value=f"{avg_rel:.3f}",
            help="Score promedio de relevancia de conceptos"
        )
    
    # Información adicional
    if stats.get('top_concept'):
        st.success(f"🏆 **Concepto más relevante:** {stats['top_concept']}")
    
    # Estadísticas de citación
    if 'citation_stats' in stats and stats['citation_stats']:
        cit_stats = stats['citation_stats']
        
        with st.expander("📖 Detalles de Citación", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Resumen de Citas:**")
                st.markdown(f"- Total de citas: {cit_stats.get('total_citations', 0)}")
                st.markdown(f"- Fuentes citadas: {cit_stats.get('unique_sources', 0)}")
                st.markdown(f"- Relevancia promedio: {cit_stats.get('avg_relevance', 0):.3f}")
            
            with col2:
                if cit_stats.get('most_cited_source'):
                    st.markdown("**Fuente Más Citada:**")
                    st.code(cit_stats['most_cited_source'])
                
                if 'citations_by_source' in cit_stats:
                    st.markdown("**Distribución:**")
                    for source, count in list(cit_stats['citations_by_source'].items())[:3]:
                        source_name = source.split('/')[-1][:30]
                        st.markdown(f"• {source_name}: {count}")


def show_help_sidebar():
    """Mostrar barra lateral de ayuda"""
    with st.sidebar:
        st.markdown("### 💡 Ayuda")
        
        with st.expander("¿Qué son los conceptos clave?"):
            st.markdown("""
            Los **conceptos clave** son los términos y frases más importantes
            que definen el contenido de tus documentos.
            
            Este sistema los identifica usando análisis estadístico (TF-IDF)
            que considera:
            - Frecuencia en el documento
            - Rareza en el corpus general
            - Importancia contextual
            """)
        
        with st.expander("¿Cómo leer el score de relevancia?"):
            st.markdown("""
            El **score de relevancia** (0.0 a 1.0) indica qué tan importante
            es un concepto:
            
            - **0.7 - 1.0**: Muy relevante (concepto central)
            - **0.4 - 0.7**: Relevante (concepto importante)
            - **0.0 - 0.4**: Moderado (concepto secundario)
            
            Este score se calcula con TF-IDF, considerando tanto la
            frecuencia como la especificidad del término.
            """)
        
        with st.expander("¿Para qué sirven las citas?"):
            st.markdown("""
            Las **citas** te permiten:
            
            1. ✅ Verificar de dónde viene cada concepto
            2. ✅ Volver a la fuente original
            3. ✅ Validar la interpretación del sistema
            4. ✅ Fundamentar tu investigación
            
            Cada concepto incluye citas exactas con contexto para
            que puedas rastrear su origen.
            """)

