"""
Tab de Análisis de Temas
Interfaz de usuario para identificación y análisis de temas
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...extractors.topic_extractor import TopicExtractor, TopicAnalysisResult
from ...core.config import AnalysisConfig
from ..components.educational import (
    show_methodology_box,
    show_interpretation_guide,
    show_citation_box,
    show_statistics_panel
)


def render_topics_tab(chunks: List[Dict[str, Any]], config: AnalysisConfig):
    """
    Renderizar tab de análisis de temas
    
    Args:
        chunks: Lista de chunks de texto para análisis
        config: Configuración del análisis
    """
    st.markdown("### 🎯 Análisis de Temas")
    
    # Mostrar metodología
    show_methodology_box(
        title="Metodología de Análisis de Temas",
        description="""
        **¿Qué hace este análisis?**
        
        Este módulo identifica automáticamente los temas principales presentes en tus documentos 
        utilizando técnicas avanzadas de procesamiento de lenguaje natural. No copia contenido, 
        sino que analiza patrones y sintetiza información para revelar temas latentes.
        """,
        steps=[
            "Preprocesamiento: Limpieza y normalización del texto",
            "Vectorización: Conversión a representación numérica (TF-IDF)",
            "Algoritmo de Análisis: LDA o Clustering Semántico",
            "Extracción de Temas: Identificación de patrones temáticos",
            "Validación: Cálculo de coherencia y calidad",
            "Citación: Vinculación con fuentes originales"
        ],
        algorithm_info="LDA (Latent Dirichlet Allocation) o Clustering Semántico con K-Means"
    )
    
    # Mostrar guía de interpretación
    show_interpretation_guide(
        what_it_means="""
        **¿Qué significan estos resultados?**
        
        Los temas identificados representan patrones conceptuales latentes en tus documentos. 
        Cada tema está caracterizado por palabras clave que aparecen frecuentemente juntas, 
        indicando conceptos relacionados que emergen del contenido analizado.
        """,
        how_to_use=[
            "Utiliza los temas para identificar categorías principales en tu investigación",
            "Analiza la coherencia para validar la calidad de los temas identificados",
            "Revisa las citaciones para verificar la relevancia de cada tema",
            "Compara la distribución de temas entre diferentes documentos",
            "Exporta los resultados para análisis posterior o reportes"
        ],
        limitations=[
            "El número óptimo de temas depende del contenido y puede requerir ajuste",
            "Temas con baja coherencia pueden necesitar refinamiento",
            "La calidad depende de la cantidad y diversidad del contenido",
            "Los algoritmos pueden interpretar de manera diferente el mismo contenido"
        ]
    )
    
    # Verificar si hay chunks disponibles
    if not chunks:
        st.warning("⚠️ No hay documentos disponibles para análisis de temas")
        st.info("""
        **Para realizar análisis de temas:**
        1. Asegúrate de que hay documentos procesados en el cache RAG
        2. Los documentos deben contener texto suficiente para análisis
        3. Mínimo recomendado: 10 documentos o 1000 palabras
        """)
        return
    
    # Mostrar estadísticas de entrada
    st.markdown("#### 📊 Estadísticas de Entrada")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📄 Documentos",
            len(set(chunk.get('metadata', {}).get('source_file', 'unknown') 
                   for chunk in chunks)),
            help="Número de documentos únicos"
        )
    
    with col2:
        st.metric(
            "📝 Chunks",
            len(chunks),
            help="Número total de fragmentos de texto"
        )
    
    with col3:
        st.metric(
            "📖 Palabras Estimadas",
            sum(len(chunk.get('content', '').split()) for chunk in chunks),
            help="Número aproximado de palabras"
        )
    
    with col4:
        st.metric(
            "📂 Fuentes Únicas",
            len(set(chunk.get('metadata', {}).get('source_file', 'unknown') 
                   for chunk in chunks)),
            help="Número de fuentes diferentes"
        )
    
    # Configuración del análisis
    st.markdown("#### ⚙️ Configuración del Análisis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        algorithm = st.selectbox(
            "Algoritmo de Análisis",
            ["LDA (Latent Dirichlet Allocation)", "Clustering Semántico"],
            help="LDA: Identifica temas latentes. Clustering: Agrupa documentos por similitud."
        )
    
    with col2:
        n_topics = st.slider(
            "Número de Temas",
            min_value=3,
            max_value=20,
            value=8,
            help="Número de temas a identificar (3-20)"
        )
    
    with col3:
        min_frequency = st.slider(
            "Frecuencia Mínima",
            min_value=1,
            max_value=5,
            value=2,
            help="Frecuencia mínima de palabras para incluir en análisis"
        )
    
    # Opciones avanzadas
    with st.expander("🔧 Opciones Avanzadas"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_features = st.number_input(
                "Máximo de Características",
                min_value=500,
                max_value=2000,
                value=1000,
                help="Número máximo de características para vectorización"
            )
            
            ngram_range = st.selectbox(
                "Rango de N-gramas",
                ["(1,1) - Palabras simples", "(1,2) - Palabras y bigramas", "(1,3) - Hasta trigramas"],
                index=1
            )
        
        with col2:
            min_df = st.slider(
                "Documentos Mínimos",
                min_value=1,
                max_value=5,
                value=2,
                help="Frecuencia mínima de documentos para incluir palabra"
            )
            
            max_df = st.slider(
                "Documentos Máximos (%)",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                help="Frecuencia máxima de documentos para incluir palabra"
            )
    
    # Botón para ejecutar análisis
    if st.button("🚀 Ejecutar Análisis de Temas", type="primary"):
        with st.spinner("Analizando temas..."):
            try:
                # Crear extractor de temas
                extractor = TopicExtractor(config)
                
                # Ejecutar análisis según algoritmo seleccionado
                if "LDA" in algorithm:
                    result = extractor.extract_topics_lda(chunks, n_topics)
                else:
                    result = extractor.extract_topics_clustering(chunks, n_topics)
                
                # Guardar resultado en session_state
                st.session_state['topic_analysis_result'] = result
                st.session_state['topic_analysis_config'] = {
                    'algorithm': algorithm,
                    'n_topics': n_topics,
                    'min_frequency': min_frequency
                }
                
                st.success(f"✅ Análisis completado: {result.total_topics} temas identificados")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("💡 Intenta con menos temas o verifica que hay suficiente contenido")
    
    # Mostrar resultados si existen
    if 'topic_analysis_result' in st.session_state:
        result = st.session_state['topic_analysis_result']
        config_used = st.session_state.get('topic_analysis_config', {})
        
        render_topic_results(result, config_used, chunks)


def render_topic_results(result: TopicAnalysisResult, config: Dict[str, Any], chunks: List[Dict[str, Any]]):
    """
    Renderizar resultados del análisis de temas
    
    Args:
        result: Resultado del análisis de temas
        config: Configuración utilizada
        chunks: Chunks originales para referencias
    """
    st.markdown("#### 📊 Resultados del Análisis de Temas")
    
    # Mostrar resumen del análisis
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Temas Identificados", result.total_topics)
    
    with col2:
        st.metric("Coherencia Promedio", f"{result.coherence_avg:.3f}")
    
    with col3:
        st.metric("Algoritmo", result.algorithm_used)
    
    with col4:
        st.metric("Tiempo de Procesamiento", f"{result.processing_time:.2f}s")
    
    # Tabs para diferentes vistas de resultados
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Lista de Temas",
        "📊 Visualizaciones", 
        "🔗 Relaciones",
        "💾 Exportar"
    ])
    
    with tab1:
        render_topics_list(result)
    
    with tab2:
        render_topic_visualizations(result)
    
    with tab3:
        render_topic_relationships(result)
    
    with tab4:
        render_topic_export(result)


def render_topics_list(result: TopicAnalysisResult):
    """Renderizar lista de temas identificados"""
    st.markdown("#### 📋 Temas Identificados")
    
    # Crear DataFrame para mostrar temas
    topics_data = []
    for topic in result.topics:
        topics_data.append({
            'ID': topic.topic_id + 1,
            'Nombre': topic.topic_name,
            'Palabras Clave': ', '.join(topic.keywords[:5]),
            'Frecuencia': topic.frequency,
            'Coherencia': f"{topic.coherence_score:.3f}",
            'Confianza': f"{topic.confidence:.3f}",
            'Documentos': len(topic.documents)
        })
    
    df = pd.DataFrame(topics_data)
    
    # Mostrar tabla con temas
    st.dataframe(df, use_container_width=True)
    
    # Mostrar detalles de cada tema
    st.markdown("#### 🔍 Detalles de Temas")
    
    for topic in result.topics:
        with st.expander(f"**{topic.topic_name}**"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Palabras Clave:**")
                for i, keyword in enumerate(topic.keywords[:10], 1):
                    st.markdown(f"{i}. {keyword}")
                
                st.markdown("**Descripción:**")
                st.info(topic.description)
            
            with col2:
                st.markdown("**Métricas:**")
                st.metric("Frecuencia", topic.frequency)
                st.metric("Coherencia", f"{topic.coherence_score:.3f}")
                st.metric("Confianza", f"{topic.confidence:.3f}")
                
                if topic.documents:
                    st.markdown("**Documentos Relacionados:**")
                    for doc in topic.documents[:5]:
                        st.markdown(f"• {doc}")
            
            # Mostrar citaciones si existen
            if topic.citations:
                with st.expander("📚 Ver Citaciones"):
                    for i, citation in enumerate(topic.citations[:3], 1):  # Mostrar solo las primeras 3
                        st.markdown(f"**Cita {i}:**")
                        st.markdown(f"📄 **Fuente:** {citation.get('source_file', 'unknown')}")
                        st.markdown(f"📝 **Contenido:** {citation.get('content', '')[:200]}...")
                        st.markdown(f"🔗 **Chunk:** {citation.get('chunk_index', 0)}")
                        st.markdown("---")


def render_topic_visualizations(result: TopicAnalysisResult):
    """Renderizar visualizaciones de temas"""
    st.markdown("#### 📊 Visualizaciones de Temas")
    
    # Crear extractor para generar visualizaciones
    extractor = TopicExtractor(AnalysisConfig())
    
    try:
        # Generar visualizaciones
        viz_data = extractor.generate_topic_visualization(result)
        
        if viz_data:
            # Gráfico de distribución de frecuencia
            st.plotly_chart(viz_data['distribution_chart'], use_container_width=True)
            
            # Gráfico de coherencia
            st.plotly_chart(viz_data['coherence_chart'], use_container_width=True)
            
            # Tabla de datos
            st.markdown("#### 📈 Datos de Visualización")
            df_viz = pd.DataFrame(viz_data['topics_data'])
            st.dataframe(df_viz, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generando visualizaciones: {str(e)}")
        st.info("💡 Intenta regenerar el análisis")


def render_topic_relationships(result: TopicAnalysisResult):
    """Renderizar relaciones entre temas"""
    st.markdown("#### 🔗 Relaciones entre Temas")
    
    # Crear matriz de similitud entre temas
    topics = result.topics
    n_topics = len(topics)
    
    if n_topics < 2:
        st.info("Se necesitan al menos 2 temas para mostrar relaciones")
        return
    
    # Calcular similitud entre temas
    similarity_matrix = []
    for i, topic1 in enumerate(topics):
        row = []
        for j, topic2 in enumerate(topics):
            if i == j:
                row.append(1.0)
            else:
                # Calcular similitud basada en palabras clave compartidas
                keywords1 = set(topic1.keywords[:5])
                keywords2 = set(topic2.keywords[:5])
                intersection = len(keywords1.intersection(keywords2))
                union = len(keywords1.union(keywords2))
                similarity = intersection / union if union > 0 else 0.0
                row.append(similarity)
        similarity_matrix.append(row)
    
    # Crear heatmap de similitud
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f"Tema {i+1}" for i in range(n_topics)],
        y=[f"Tema {i+1}" for i in range(n_topics)],
        colorscale='Blues',
        text=[[f"{val:.2f}" for val in row] for row in similarity_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Matriz de Similitud entre Temas",
        xaxis_title="Temas",
        yaxis_title="Temas"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar temas más similares
    st.markdown("#### 🔍 Temas Más Similares")
    
    similarities = []
    for i in range(n_topics):
        for j in range(i+1, n_topics):
            similarities.append({
                'Tema 1': f"Tema {i+1}",
                'Tema 2': f"Tema {j+1}",
                'Similitud': similarity_matrix[i][j],
                'Palabras Compartidas': len(set(topics[i].keywords[:5]).intersection(set(topics[j].keywords[:5])))
            })
    
    if similarities:
        df_sim = pd.DataFrame(similarities)
        df_sim = df_sim.sort_values('Similitud', ascending=False)
        st.dataframe(df_sim, use_container_width=True)


def render_topic_export(result: TopicAnalysisResult):
    """Renderizar opciones de exportación"""
    st.markdown("#### 💾 Exportar Análisis de Temas")
    
    # Crear extractor para exportación
    extractor = TopicExtractor(AnalysisConfig())
    
    # Exportar datos
    export_data = extractor.export_topic_analysis(result, 'json')
    
    if export_data:
        # Convertir a JSON
        import json
        json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Botón de descarga
        st.download_button(
            label="📥 Descargar Análisis (JSON)",
            data=json_data,
            file_name=f"topic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Mostrar resumen de exportación
        st.markdown("#### 📋 Resumen de Exportación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Metadatos:**")
            st.json(export_data['metadata'])
        
        with col2:
            st.markdown("**Estadísticas:**")
            st.metric("Total de Temas", len(export_data['topics']))
            st.metric("Coherencia Promedio", f"{export_data['metadata']['avg_coherence']:.3f}")
            st.metric("Tiempo de Procesamiento", f"{export_data['metadata']['processing_time']:.2f}s")
        
        # Mostrar información adicional sobre la exportación
        st.info("""
        **💡 Información sobre la exportación:**
        
        - **JSON**: Incluye todos los datos del análisis para procesamiento posterior
        - **Metadatos**: Información sobre el algoritmo y parámetros utilizados
        - **Citaciones**: Referencias a las fuentes originales incluidas
        - **Estadísticas**: Métricas de calidad y coherencia de los temas
        """)
    
    # Botón para limpiar resultados
    if st.button("🗑️ Limpiar Resultados"):
        if 'topic_analysis_result' in st.session_state:
            del st.session_state['topic_analysis_result']
        if 'topic_analysis_config' in st.session_state:
            del st.session_state['topic_analysis_config']
        st.rerun()
