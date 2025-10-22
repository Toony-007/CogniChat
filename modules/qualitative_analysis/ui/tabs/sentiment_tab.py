"""
Tab de Análisis de Sentimientos
Interfaz de usuario para análisis de polaridad y subjetividad
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...extractors.sentiment_extractor import SentimentExtractor, SentimentAnalysisResult
from ...core.config import AnalysisConfig
from ..components.educational import (
    show_methodology_box,
    show_interpretation_guide,
    show_citation_box,
    show_statistics_panel
)


def render_sentiment_tab(chunks: List[Dict[str, Any]], config: AnalysisConfig):
    """
    Renderizar tab de análisis de sentimientos
    
    Args:
        chunks: Lista de chunks de texto para análisis
        config: Configuración del análisis
    """
    st.markdown("### 😊 Análisis de Sentimientos")
    
    # Mostrar metodología
    show_methodology_box(
        title="Metodología de Análisis de Sentimientos",
        description="""
        **¿Qué hace este análisis?**
        
        Este módulo analiza el tono emocional y la subjetividad presente en tus documentos 
        utilizando técnicas avanzadas de procesamiento de lenguaje natural. Identifica 
        sentimientos positivos, negativos y neutrales, proporcionando insights sobre 
        el contenido emocional de tu investigación.
        """,
        steps=[
            "Preprocesamiento: Limpieza y normalización del texto",
            "Análisis de Polaridad: Identificación de sentimientos positivos/negativos",
            "Análisis de Subjetividad: Medición del contenido objetivo vs subjetivo",
            "Clasificación: Etiquetado automático de sentimientos",
            "Validación: Cálculo de confianza en las clasificaciones",
            "Citación: Vinculación con fuentes originales"
        ],
        algorithm_info="VADER (Valence Aware Dictionary), TextBlob, o enfoque Híbrido"
    )
    
    # Mostrar guía de interpretación
    show_interpretation_guide(
        what_it_means="""
        **¿Qué significan estos resultados?**
        
        Los análisis de sentimientos revelan el tono emocional y la subjetividad del contenido. 
        La polaridad indica si el texto es positivo, negativo o neutral, mientras que la 
        subjetividad mide qué tan objetivo o subjetivo es el contenido analizado.
        """,
        how_to_use=[
            "Utiliza la polaridad para identificar el tono general de tus documentos",
            "Analiza la subjetividad para distinguir contenido objetivo de opiniones",
            "Revisa las citaciones para validar la relevancia de cada análisis",
            "Compara sentimientos entre diferentes documentos o secciones",
            "Exporta los resultados para análisis posterior o reportes"
        ],
        limitations=[
            "El análisis puede variar según el contexto y dominio específico",
            "Textos irónicos o sarcásticos pueden ser malinterpretados",
            "La precisión depende de la calidad y cantidad del contenido",
            "Diferentes algoritmos pueden dar resultados ligeramente diferentes"
        ]
    )
    
    # Verificar si hay chunks disponibles
    if not chunks:
        st.warning("⚠️ No hay documentos disponibles para análisis de sentimientos")
        st.info("""
        **Para realizar análisis de sentimientos:**
        1. Asegúrate de que hay documentos procesados en el cache RAG
        2. Los documentos deben contener texto suficiente para análisis
        3. Mínimo recomendado: 5 documentos o 500 palabras
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
             ["VADER + TextBlob"],
             help="Algoritmo híbrido que combina VADER (especializado en redes sociales) y TextBlob (análisis general) para máxima precisión."
         )
    
    with col2:
        min_confidence = st.slider(
            "Confianza Mínima",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Confianza mínima para incluir en el análisis"
        )
    
    with col3:
        min_text_length = st.slider(
            "Longitud Mínima de Texto",
            min_value=10,
            max_value=100,
            value=20,
            help="Longitud mínima de texto para análisis"
        )
    
    # Opciones avanzadas
    with st.expander("🔧 Opciones Avanzadas"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_context = st.checkbox(
                "Incluir Análisis Contextual",
                value=True,
                help="Generar contexto adicional para cada sentimiento"
            )
            
            filter_subjectivity = st.checkbox(
                "Filtrar por Subjetividad",
                value=False,
                help="Filtrar resultados basado en nivel de subjetividad"
            )
        
        with col2:
            if filter_subjectivity:
                subjectivity_threshold = st.slider(
                    "Umbral de Subjetividad",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    help="Solo incluir textos con subjetividad mayor al umbral"
                )
            else:
                subjectivity_threshold = 0.0
            
            export_format = st.selectbox(
                "Formato de Exportación",
                ["JSON", "CSV", "PDF"],
                help="Formato para exportar resultados"
            )
    
    # Botón para ejecutar análisis
    if st.button("🚀 Ejecutar Análisis de Sentimientos", type="primary", key="execute_sentiment_analysis"):
        with st.spinner("Analizando sentimientos..."):
            try:
                # Crear extractor de sentimientos
                extractor = SentimentExtractor(config)
                
                # Ejecutar análisis híbrido (VADER + TextBlob)
                result = extractor.analyze_sentiments_hybrid(chunks)
                
                # Filtrar por confianza si es necesario
                if min_confidence > 0:
                    result.sentiments = [
                        s for s in result.sentiments 
                        if s.confidence >= min_confidence
                    ]
                
                # Filtrar por longitud de texto
                result.sentiments = [
                    s for s in result.sentiments 
                    if len(s.text) >= min_text_length
                ]
                
                # Filtrar por subjetividad si está habilitado
                if filter_subjectivity:
                    result.sentiments = [
                        s for s in result.sentiments 
                        if s.subjectivity >= subjectivity_threshold
                    ]
                
                # Guardar resultado en session_state
                st.session_state['sentiment_analysis_result'] = result
                st.session_state['sentiment_analysis_config'] = {
                    'algorithm': algorithm,
                    'min_confidence': min_confidence,
                    'min_text_length': min_text_length,
                    'include_context': include_context
                }
                
                st.success(f"✅ Análisis completado: {len(result.sentiments)} textos analizados")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("💡 Intenta con diferentes parámetros o verifica que hay suficiente contenido")
    
    # Mostrar resultados si existen
    if 'sentiment_analysis_result' in st.session_state:
        result = st.session_state['sentiment_analysis_result']
        config_used = st.session_state.get('sentiment_analysis_config', {})
        
        render_sentiment_results(result, config_used, chunks)


def render_sentiment_results(result: SentimentAnalysisResult, config: Dict[str, Any], chunks: List[Dict[str, Any]]):
    """
    Renderizar resultados del análisis de sentimientos
    
    Args:
        result: Resultado del análisis de sentimientos
        config: Configuración utilizada
        chunks: Chunks originales para referencias
    """
    st.markdown("#### 📊 Resultados del Análisis de Sentimientos")
    
    # Mostrar resumen del análisis
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Textos Analizados", result.total_analyzed)
    
    with col2:
        st.metric("Polaridad Promedio", f"{result.average_polarity:.3f}")
    
    with col3:
        st.metric("Subjetividad Promedio", f"{result.average_subjectivity:.3f}")
    
    with col4:
        st.metric("Algoritmo", result.algorithm_used)
    
    # Tabs para diferentes vistas de resultados
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Lista de Sentimientos",
        "📊 Visualizaciones", 
        "🔗 Análisis por Fuente",
        "💾 Exportar"
    ])
    
    with tab1:
        render_sentiments_list(result)
    
    with tab2:
        render_sentiment_visualizations(result)
    
    with tab3:
        render_sentiment_by_source(result)
    
    with tab4:
        render_sentiment_export(result)


def render_sentiments_list(result: SentimentAnalysisResult):
    """Renderizar lista de sentimientos identificados"""
    st.markdown("#### 📋 Sentimientos Identificados")
    
    # Crear DataFrame para mostrar sentimientos
    sentiments_data = []
    for i, sentiment in enumerate(result.sentiments):
        sentiments_data.append({
            'ID': i + 1,
            'Texto': sentiment.text[:100] + "..." if len(sentiment.text) > 100 else sentiment.text,
            'Sentimiento': sentiment.sentiment_label,
            'Polaridad': f"{sentiment.polarity:.3f}",
            'Subjetividad': f"{sentiment.subjectivity:.3f}",
            'Confianza': f"{sentiment.confidence:.3f}",
            'Contexto': sentiment.context[:50] + "..." if len(sentiment.context) > 50 else sentiment.context
        })
    
    df = pd.DataFrame(sentiments_data)
    
    # Mostrar tabla con sentimientos
    st.dataframe(df, use_container_width=True)
    
    # Mostrar detalles de cada sentimiento
    st.markdown("#### 🔍 Detalles de Sentimientos")
    
    for i, sentiment in enumerate(result.sentiments[:10]):  # Mostrar solo los primeros 10
        with st.expander(f"**Sentimiento {i+1}: {sentiment.sentiment_label.upper()}**"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Texto Completo:**")
                st.text_area("Texto", sentiment.text, height=100, disabled=True, label_visibility="collapsed", key=f"sentiment_text_{i}")
                
                st.markdown("**Contexto:**")
                st.info(sentiment.context)
            
            with col2:
                st.markdown("**Métricas:**")
                st.metric("Polaridad", f"{sentiment.polarity:.3f}")
                st.metric("Subjetividad", f"{sentiment.subjectivity:.3f}")
                st.metric("Confianza", f"{sentiment.confidence:.3f}")
                
                # Barra de progreso para visualizar polaridad
                if sentiment.polarity > 0:
                    st.progress(sentiment.polarity, text=f"Positivo: {sentiment.polarity:.1%}")
                else:
                    st.progress(abs(sentiment.polarity), text=f"Negativo: {abs(sentiment.polarity):.1%}")
            
            # Mostrar citaciones si existen
            if sentiment.citations:
                with st.expander("📚 Ver Citaciones"):
                    for j, citation in enumerate(sentiment.citations[:3], 1):
                        st.markdown(f"**Cita {j}:**")
                        st.markdown(f"📄 **Fuente:** {citation.get('source_file', 'unknown')}")
                        st.markdown(f"📝 **Contenido:** {citation.get('content', '')[:200]}...")
                        st.markdown(f"🔗 **Chunk:** {citation.get('chunk_id', 0)}")
                        st.markdown("---")


def render_sentiment_visualizations(result: SentimentAnalysisResult):
    """Renderizar visualizaciones de sentimientos"""
    st.markdown("#### 📊 Visualizaciones de Sentimientos")
    
    # Crear extractor para generar visualizaciones
    extractor = SentimentExtractor(AnalysisConfig())
    
    try:
        # Generar visualizaciones
        viz_data = extractor.generate_sentiment_visualization(result)
        
        if viz_data:
            # Gráfico de distribución de sentimientos
            st.plotly_chart(viz_data['distribution_chart'], use_container_width=True)
            
            # Gráfico de dispersión polaridad vs subjetividad
            st.plotly_chart(viz_data['scatter_chart'], use_container_width=True)
            
            # Gráfico de confianza
            st.plotly_chart(viz_data['confidence_chart'], use_container_width=True)
            
            # Tabla de datos
            st.markdown("#### 📈 Datos de Visualización")
            df_viz = pd.DataFrame(viz_data['sentiments_data'])
            st.dataframe(df_viz, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generando visualizaciones: {str(e)}")
        st.info("💡 Intenta regenerar el análisis")


def render_sentiment_by_source(result: SentimentAnalysisResult):
    """Renderizar análisis de sentimientos por fuente"""
    st.markdown("#### 🔗 Análisis por Fuente")
    
    # Agrupar sentimientos por fuente
    source_sentiments = {}
    for sentiment in result.sentiments:
        for citation in sentiment.citations:
            source = citation.get('source_file', 'unknown')
            if source not in source_sentiments:
                source_sentiments[source] = []
            source_sentiments[source].append(sentiment)
    
    if not source_sentiments:
        st.info("No hay datos de fuentes disponibles")
        return
    
    # Mostrar análisis por fuente
    for source, sentiments in source_sentiments.items():
        with st.expander(f"📄 {source} ({len(sentiments)} sentimientos)"):
            # Calcular estadísticas por fuente
            polarities = [s.polarity for s in sentiments]
            subjectivities = [s.subjectivity for s in sentiments]
            confidences = [s.confidence for s in sentiments]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Polaridad Promedio", f"{np.mean(polarities):.3f}")
                st.metric("Subjetividad Promedio", f"{np.mean(subjectivities):.3f}")
            
            with col2:
                st.metric("Confianza Promedio", f"{np.mean(confidences):.3f}")
                st.metric("Total Sentimientos", len(sentiments))
            
            with col3:
                # Distribución de sentimientos por fuente
                sentiment_counts = {}
                for s in sentiments:
                    sentiment_counts[s.sentiment_label] = sentiment_counts.get(s.sentiment_label, 0) + 1
                
                st.markdown("**Distribución:**")
                for label, count in sentiment_counts.items():
                    st.markdown(f"• {label}: {count}")


def render_sentiment_export(result: SentimentAnalysisResult):
    """Renderizar opciones de exportación"""
    st.markdown("#### 💾 Exportar Análisis de Sentimientos")
    
    # Crear extractor para exportación
    extractor = SentimentExtractor(AnalysisConfig())
    
    # Exportar datos
    export_data = extractor.export_sentiment_analysis(result, 'json')
    
    if export_data:
        # Convertir a JSON
        import json
        json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Botón de descarga
        st.download_button(
            label="📥 Descargar Análisis (JSON)",
            data=json_data,
            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
            st.metric("Total Analizado", len(export_data['sentiments']))
            st.metric("Polaridad Promedio", f"{export_data['metadata']['avg_polarity']:.3f}")
            st.metric("Subjetividad Promedio", f"{export_data['metadata']['avg_subjectivity']:.3f}")
        
        # Mostrar información adicional sobre la exportación
        st.info("""
        **💡 Información sobre la exportación:**
        
        - **JSON**: Incluye todos los datos del análisis para procesamiento posterior
        - **Metadatos**: Información sobre el algoritmo y parámetros utilizados
        - **Citaciones**: Referencias a las fuentes originales incluidas
        - **Estadísticas**: Métricas de polaridad y subjetividad
        """)
    
    # Botón para limpiar resultados
    if st.button("🗑️ Limpiar Resultados", key="clear_sentiment_results"):
        if 'sentiment_analysis_result' in st.session_state:
            del st.session_state['sentiment_analysis_result']
        if 'sentiment_analysis_config' in st.session_state:
            del st.session_state['sentiment_analysis_config']
        st.rerun()
