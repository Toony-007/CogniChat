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
        title="Enfoque Híbrido: Algoritmo + DeepSeek R1",
        description="""
        **¿Qué hace este análisis?**
        
        Este módulo identifica automáticamente los temas principales presentes en tus documentos 
        utilizando un **enfoque híbrido** que combina algoritmos avanzados de procesamiento de lenguaje natural 
        con inteligencia artificial para generar temas académicos profundos y significativos.
        """,
        steps=[
            "Preprocesamiento: Limpieza y normalización del texto",
            "Vectorización: Conversión a representación numérica (TF-IDF)",
            "Algoritmo de Análisis: LDA o Clustering Semántico para extracción inicial",
            "🤖 Refinamiento IA: DeepSeek R1 analiza y mejora los temas candidatos",
            "Generación de temas profundos: Crea temas académicos significativos",
            "Validación: Cálculo de coherencia mejorada y métricas de calidad",
            "Citación: Vinculación con fuentes originales",
            "Visualización: Nubes de palabras y análisis de relaciones"
        ],
        algorithm_info="LDA/Clustering + DeepSeek R1 para temas académicos profundos"
    )
    
    # Mostrar guía de interpretación
    show_interpretation_guide(
        what_it_means="""
        **¿Qué significan estos resultados?**
        
        Los temas identificados son **fenómenos académicos profundos** que emergen de tus documentos. 
        Con el refinamiento IA, cada tema captura procesos, relaciones y conceptos complejos que van 
        más allá de simples palabras clave, proporcionando comprensión matizada de los patrones 
        conceptuales presentes en tu investigación.
        """,
        how_to_use=[
            "Identifica los fenómenos centrales de tu investigación",
            "Analiza las explicaciones generadas por la IA para cada tema",
            "Examina las nubes de palabras para entender la composición de cada tema",
            "Estudia las relaciones entre temas para entender patrones complejos",
            "Usa las métricas de coherencia y confianza para validar la calidad",
            "Aprovecha los temas profundos para desarrollar marcos teóricos",
            "Exporta los resultados para fundamentar tu análisis cualitativo"
        ],
        limitations=[
            "Los temas generados por IA requieren validación del investigador",
            "La calidad depende de la cantidad y diversidad del contenido analizado",
            "El LLM puede generar temas que sintetizan información no explícita",
            "Las explicaciones son interpretaciones que deben ser verificadas",
            "Requiere acceso a un modelo LLM (Ollama) para funcionar completamente"
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
    
    # Configuración del análisis (valores predefinidos en el código)
    st.markdown("#### ⚙️ Configuración del Análisis")
    
    st.info("""
    **ℹ️ Configuración Automática:** Los parámetros del análisis de temas están optimizados automáticamente.
    Si necesitas modificar algún parámetro específico, puedes hacerlo directamente en el código.
    """)
    
    # Botón para ejecutar análisis híbrido
    if st.button("🚀 Analizar Temas", type="primary", use_container_width=True):
        with st.spinner("🎯 Analizando temas con enfoque híbrido..."):
            try:
                # Crear extractor de temas
                extractor = TopicExtractor(config)
                
                # Ejecutar análisis híbrido (algoritmo + IA)
                result = extractor.extract_topics_hybrid(chunks)
                
                # Guardar resultado en session_state
                st.session_state['topic_analysis_result'] = result
                st.session_state['topic_analysis_config'] = {
                    'algorithm': config.topic_algorithm,
                    'max_topics': config.max_topics,
                    'enable_refinement': config.enable_topic_refinement
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
    """Renderizar visualizaciones mejoradas de temas"""
    st.markdown("#### 📊 Visualizaciones de Temas")
    
    # Crear extractor para generar visualizaciones
    extractor = TopicExtractor(AnalysisConfig())
    
    try:
        # Generar visualizaciones mejoradas
        viz_data = extractor.generate_topic_visualization(result)
        
        if viz_data:
            # Gráfico de distribución de frecuencia mejorado
            st.markdown("##### 📈 Distribución de Frecuencia por Tema")
            st.plotly_chart(viz_data['distribution_chart'], use_container_width=True)
            
            # Gráfico de coherencia mejorado
            st.markdown("##### 🎯 Coherencia de Temas")
            st.plotly_chart(viz_data['coherence_chart'], use_container_width=True)
            
            # Nuevo gráfico de confianza
            st.markdown("##### ⭐ Confianza de Temas")
            st.plotly_chart(viz_data['confidence_chart'], use_container_width=True)
            
            # Nubes de palabras por tema
            st.markdown("##### ☁️ Nubes de Palabras por Tema")
            
            wordclouds = viz_data.get('wordclouds', {})
            if wordclouds:
                # Mostrar nubes de palabras en columnas
                cols = st.columns(min(3, len(wordclouds)))
                
                for i, (topic_id, wordcloud_data) in enumerate(wordclouds.items()):
                    if wordcloud_data:
                        with cols[i % len(cols)]:
                            topic_name = result.topics[topic_id].topic_name
                            st.markdown(f"**{topic_name}**")
                            st.image(wordcloud_data, use_container_width=True)
                    else:
                        with cols[i % len(cols)]:
                            st.info("Nube de palabras no disponible")
            else:
                st.info("💡 Las nubes de palabras requieren la librería 'wordcloud'. Instala con: pip install wordcloud")
            
            # Tabla de datos mejorada
            st.markdown("#### 📈 Datos de Visualización")
            df_viz = pd.DataFrame(viz_data['topics_data'])
            st.dataframe(df_viz, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generando visualizaciones: {str(e)}")
        st.info("💡 Intenta regenerar el análisis")


def render_topic_relationships(result: TopicAnalysisResult):
    """Renderizar relaciones mejoradas entre temas"""
    st.markdown("#### 🔗 Relaciones entre Temas")
    
    # Crear extractor para calcular relaciones
    extractor = TopicExtractor(AnalysisConfig())
    
    try:
        # Calcular relaciones usando el método mejorado
        relationships = extractor.calculate_topic_relationships(result)
        
        if not relationships['similarity_matrix']:
            st.info("Se necesitan al menos 2 temas para mostrar relaciones")
            return
        
        # Mostrar matriz de similitud mejorada
        st.markdown("##### 📊 Matriz de Similitud entre Temas")
        
        similarity_matrix = relationships['similarity_matrix']
        topic_names = relationships['topic_names']
        
        # Crear heatmap mejorado
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=topic_names,
            y=topic_names,
            colorscale='Blues',
            text=[[f"{val:.3f}" for val in row] for row in similarity_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Matriz de Similitud entre Temas",
            xaxis_title="Temas",
            yaxis_title="Temas",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar temas más similares con datos precisos
        st.markdown("##### 🔍 Temas Más Similares")
        
        similar_topics = relationships['similar_topics']
        
        if similar_topics:
            # Crear DataFrame con datos mejorados
            similarities_data = []
            for rel in similar_topics:
                similarities_data.append({
                    'Tema 1': rel['topic1_name'],
                    'Tema 2': rel['topic2_name'],
                    'Similitud': f"{rel['similarity']:.3f}",
                    'Palabras Compartidas': len(rel['shared_words']),
                    'Palabras': ', '.join(rel['shared_words'][:5]) if rel['shared_words'] else 'Ninguna'
                })
            
            df_sim = pd.DataFrame(similarities_data)
            st.dataframe(df_sim, use_container_width=True)
            
            # Mostrar detalles de relaciones más fuertes
            if similar_topics:
                st.markdown("##### 📝 Detalles de Relaciones Fuertes")
                
                for i, rel in enumerate(similar_topics[:3]):  # Top 3 relaciones
                    with st.expander(f"🔗 {rel['topic1_name']} ↔ {rel['topic2_name']} (Similitud: {rel['similarity']:.3f})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{rel['topic1_name']}**")
                            topic1 = result.topics[rel['topic1_id']]
                            st.markdown(f"Palabras clave: {', '.join(topic1.keywords[:5])}")
                            st.markdown(f"Frecuencia: {topic1.frequency}")
                        
                        with col2:
                            st.markdown(f"**{rel['topic2_name']}**")
                            topic2 = result.topics[rel['topic2_id']]
                            st.markdown(f"Palabras clave: {', '.join(topic2.keywords[:5])}")
                            st.markdown(f"Frecuencia: {topic2.frequency}")
                        
                        st.markdown("**Palabras Compartidas:**")
                        if rel['shared_words']:
                            for word in rel['shared_words']:
                                st.markdown(f"• {word}")
                        else:
                            st.markdown("Ninguna palabra compartida")
        else:
            st.info("No se encontraron temas con similitud significativa (>0.1)")
    
    except Exception as e:
        st.error(f"Error calculando relaciones: {str(e)}")
        st.info("💡 Intenta regenerar el análisis")


def render_topic_export(result: TopicAnalysisResult):
    """Renderizar opciones de exportación mejoradas"""
    st.markdown("#### 💾 Exportar Análisis de Temas")
    
    st.markdown("""
    <div style="background: #34495e; padding: 1rem; border-radius: 8px; border-left: 4px solid #e67e22; color: #ecf0f1; margin: 1rem 0;">
        <strong>ℹ️ Exportación Completa:</strong> Genera documentos profesionales con todos los temas analizados,
        incluyendo visualizaciones, relaciones y métricas de calidad para uso en tu investigación.
    </div>
    """, unsafe_allow_html=True)
    
    # Crear extractor para exportación
    extractor = TopicExtractor(AnalysisConfig())
    
    # Opciones de exportación
    col1, col2 = st.columns(2)
    
    with col1:
        include_relationships = st.checkbox(
            "Incluir análisis de relaciones",
            value=True,
            help="Incluye la matriz de similitud y temas relacionados"
        )
    
    with col2:
        include_visualizations = st.checkbox(
            "Incluir datos de visualización",
            value=True,
            help="Incluye métricas de coherencia y confianza"
        )
    
    # Información del documento
    st.markdown("#### 📄 Información del Documento")
    
    col1, col2 = st.columns(2)
    with col1:
        document_title = st.text_input(
            "Título del documento",
            value="Análisis de Temas",
            help="Título que aparecerá en el documento"
        )
    
    with col2:
        author_name = st.text_input(
            "Autor",
            value="Investigador",
            help="Nombre del autor del análisis"
        )
    
    # Botones de exportación
    col1, col2 = st.columns(2)
    
    with col1:
        # Exportación a Word
        if st.button("📄 Generar Documento Word", type="primary", use_container_width=True):
            try:
                # Crear documento Word
                doc_content = extractor.generate_word_document(
                    result=result,
                    title=document_title,
                    author=author_name
                )
                
                st.download_button(
                    label="💾 Descargar Documento Word",
                    data=doc_content,
                    file_name=f"{document_title.replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                
                st.success("✅ Documento Word generado correctamente")
                
            except Exception as e:
                st.error(f"❌ Error al generar documento Word: {str(e)}")
                st.exception(e)
    
    with col2:
        # Exportación JSON (mejorada)
        export_data = extractor.export_topic_analysis(result, 'json')
        
        if export_data:
            import json
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Descargar Análisis (JSON)",
                data=json_data,
                file_name=f"topic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Mostrar resumen de exportación
    if export_data:
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
        
        - **Word**: Documento profesional con temas, relaciones y métricas
        - **JSON**: Incluye todos los datos del análisis para procesamiento posterior
        - **Metadatos**: Información sobre el algoritmo y parámetros utilizados
        - **Relaciones**: Matriz de similitud y temas relacionados
        - **Estadísticas**: Métricas de calidad y coherencia de los temas
        """)
    
    # Botón para limpiar resultados
    if st.button("🗑️ Limpiar Resultados", type="secondary"):
        if 'topic_analysis_result' in st.session_state:
            del st.session_state['topic_analysis_result']
        if 'topic_analysis_config' in st.session_state:
            del st.session_state['topic_analysis_config']
        st.rerun()
