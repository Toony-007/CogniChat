"""
Tab de Análisis de Relaciones
Interfaz de usuario para identificar y visualizar relaciones entre conceptos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...extractors.relation_extractor import RelationExtractor, RelationAnalysisResult
from ...core.config import AnalysisConfig
from ..components.educational import (
    show_methodology_box,
    show_interpretation_guide,
    show_citation_box,
    show_statistics_panel
)


def render_relations_tab(chunks: List[Dict[str, Any]], config: AnalysisConfig):
    """
    Renderizar tab de análisis de relaciones
    
    Args:
        chunks: Lista de chunks de texto para análisis
        config: Configuración del análisis
    """
    st.markdown("### 🔗 Análisis de Relaciones entre Conceptos")
    
    # Mostrar metodología
    show_methodology_box(
        title="Metodología de Análisis de Relaciones",
        description="""
        **¿Qué hace este análisis?**
        
        Este módulo identifica y visualiza las relaciones entre conceptos clave en tus documentos.
        Utiliza múltiples técnicas para descubrir conexiones:
        - **Co-ocurrencias**: Conceptos que aparecen juntos frecuentemente
        - **Relaciones Semánticas**: Conceptos con significados relacionados
        - **Relaciones Causales**: Conceptos conectados por causa-efecto
        """,
        steps=[
            "Extracción de Conceptos: Identifica términos clave en los documentos",
            "Análisis de Co-ocurrencias: Detecta conceptos que aparecen juntos",
            "Similitud Semántica: Mide relaciones basadas en significado",
            "Detección de Patrones: Identifica relaciones causales y jerárquicas",
            "Cálculo de Fuerza: Mide la intensidad de cada relación",
            "Citación: Vincula cada relación con sus fuentes originales"
        ],
        algorithm_info="TF-IDF + Similitud Coseno + Análisis de Co-ocurrencias + Detección de Patrones Lingüísticos"
    )
    
    # Mostrar guía de interpretación
    show_interpretation_guide(
        what_it_means="""
        **¿Qué significan estos resultados?**
        
        El análisis de relaciones revela cómo los conceptos están conectados en tus documentos:
        - **Fuerza de Relación**: Indica qué tan fuerte es la conexión (0.0 - 1.0)
        - **Tipo de Relación**: Describe la naturaleza de la conexión
        - **Confianza**: Mide la certeza de que la relación es significativa
        """,
        how_to_use=[
            "Identifica grupos de conceptos fuertemente relacionados",
            "Descubre conceptos centrales con muchas conexiones",
            "Explora relaciones causales para entender dependencias",
            "Usa las visualizaciones de red para ver la estructura completa",
            "Verifica cada relación con las citaciones de fuentes originales"
        ],
        limitations=[
            "Las co-ocurrencias no siempre implican relación semántica",
            "Las relaciones causales detectadas pueden requerir validación manual",
            "La fuerza de relación es relativa al corpus analizado",
            "Documentos cortos pueden no revelar todas las relaciones"
        ]
    )
    
    # Verificar si hay chunks disponibles
    if not chunks:
        st.warning("⚠️ No hay documentos disponibles para análisis de relaciones")
        st.info("""
        **Para realizar análisis de relaciones:**
        1. Asegúrate de que hay documentos procesados en el cache RAG
        2. Los documentos deben contener suficientes conceptos relacionados
        3. Mínimo recomendado: 5 documentos o 1000 palabras
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
        analysis_type = st.selectbox(
            "Tipo de Análisis",
            ["Co-ocurrencias", "Semántico", "Causal", "Híbrido"],
            help="Co-ocurrencias: Conceptos que aparecen juntos. Semántico: Significados relacionados. Causal: Causa-efecto. Híbrido: Combina todos."
        )
    
    with col2:
        if analysis_type == "Co-ocurrencias":
            window_size = st.slider(
                "Tamaño de Ventana",
                min_value=10,
                max_value=100,
                value=50,
                help="Distancia máxima (en palabras) para considerar co-ocurrencia"
            )
            min_cooccurrence = st.slider(
                "Co-ocurrencias Mínimas",
                min_value=1,
                max_value=10,
                value=2,
                help="Número mínimo de veces que deben aparecer juntos"
            )
        elif analysis_type == "Semántico":
            similarity_threshold = st.slider(
                "Umbral de Similitud",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Similitud mínima para considerar relación (0.0 - 1.0)"
            )
    
    with col3:
        max_concepts = st.slider(
            "Máximo de Conceptos",
            min_value=10,
            max_value=50,
            value=30,
            help="Número máximo de conceptos a analizar"
        )
        
        min_strength = st.slider(
            "Fuerza Mínima",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Fuerza mínima para mostrar relación"
        )
    
    # Opciones avanzadas
    with st.expander("🔧 Opciones Avanzadas"):
        col1, col2 = st.columns(2)
        
        with col1:
            show_network_graph = st.checkbox(
                "Mostrar Grafo de Red",
                value=True,
                help="Visualización interactiva de la red de conceptos"
            )
            
            show_matrix = st.checkbox(
                "Mostrar Matriz de Relaciones",
                value=True,
                help="Tabla detallada de todas las relaciones"
            )
        
        with col2:
            export_format = st.selectbox(
                "Formato de Exportación",
                ["JSON", "CSV", "GraphML"],
                help="Formato para exportar resultados"
            )
            
            include_weak_relations = st.checkbox(
                "Incluir Relaciones Débiles",
                value=False,
                help="Incluir relaciones con baja fuerza en la visualización"
            )
    
    # Botón para ejecutar análisis
    if st.button("🚀 Ejecutar Análisis de Relaciones", type="primary", key="execute_relation_analysis"):
        with st.spinner("Analizando relaciones..."):
            try:
                # Crear extractor de relaciones
                extractor = RelationExtractor(config)
                
                # Ejecutar análisis según el tipo seleccionado
                if analysis_type == "Co-ocurrencias":
                    result = extractor.analyze_relations_cooccurrence(
                        chunks,
                        window_size=window_size,
                        min_cooccurrence=min_cooccurrence
                    )
                elif analysis_type == "Semántico":
                    result = extractor.analyze_relations_semantic(
                        chunks,
                        similarity_threshold=similarity_threshold
                    )
                elif analysis_type == "Causal":
                    result = extractor.analyze_relations_causal(chunks)
                else:  # Híbrido
                    # Ejecutar todos los análisis y combinar
                    result_cooc = extractor.analyze_relations_cooccurrence(chunks, window_size=50, min_cooccurrence=2)
                    result_sem = extractor.analyze_relations_semantic(chunks, similarity_threshold=0.3)
                    result_caus = extractor.analyze_relations_causal(chunks)
                    
                    # Combinar resultados
                    all_relations = result_cooc.relations + result_sem.relations + result_caus.relations
                    result = RelationAnalysisResult(
                        relations=all_relations,
                        concepts=result_cooc.concepts,
                        relation_matrix=result_cooc.relation_matrix,
                        total_relations=len(all_relations),
                        analysis_type='hybrid',
                        metadata={
                            'cooccurrence_count': len(result_cooc.relations),
                            'semantic_count': len(result_sem.relations),
                            'causal_count': len(result_caus.relations)
                        }
                    )
                
                # Filtrar por fuerza mínima si es necesario
                if not include_weak_relations:
                    result.relations = [
                        r for r in result.relations 
                        if r.strength >= min_strength
                    ]
                    result.total_relations = len(result.relations)
                
                # Guardar resultado en session_state
                st.session_state['relation_analysis_result'] = result
                st.session_state['relation_analysis_config'] = {
                    'analysis_type': analysis_type,
                    'max_concepts': max_concepts,
                    'min_strength': min_strength,
                    'show_network_graph': show_network_graph,
                    'show_matrix': show_matrix
                }
                
                st.success(f"✅ Análisis completado: {result.total_relations} relaciones identificadas")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("💡 Intenta con diferentes parámetros o verifica que hay suficiente contenido")
    
    # Mostrar resultados si existen
    if 'relation_analysis_result' in st.session_state:
        result = st.session_state['relation_analysis_result']
        config_used = st.session_state.get('relation_analysis_config', {})
        
        render_relation_results(result, config_used, chunks)


def render_relation_results(
    result: RelationAnalysisResult,
    config: Dict[str, Any],
    chunks: List[Dict[str, Any]]
):
    """
    Renderizar resultados del análisis de relaciones
    
    Args:
        result: Resultado del análisis
        config: Configuración utilizada
        chunks: Chunks originales para referencias
    """
    st.markdown("#### 📊 Resultados del Análisis de Relaciones")
    
    # Mostrar resumen del análisis
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Conceptos Analizados", len(result.concepts))
    
    with col2:
        st.metric("Relaciones Identificadas", result.total_relations)
    
    with col3:
        st.metric("Tipo de Análisis", result.analysis_type.capitalize())
    
    with col4:
        if result.relations:
            avg_strength = np.mean([r.strength for r in result.relations])
            st.metric("Fuerza Promedio", f"{avg_strength:.3f}")
        else:
            st.metric("Fuerza Promedio", "N/A")
    
    # Tabs para diferentes vistas de resultados
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Lista de Relaciones",
        "🕸️ Grafo de Red",
        "📊 Matriz de Relaciones",
        "📈 Análisis Estadístico",
        "💾 Exportar"
    ])
    
    with tab1:
        render_relations_list(result)
    
    with tab2:
        if config.get('show_network_graph', True):
            render_network_graph(result)
        else:
            st.info("Visualización de grafo deshabilitada en configuración")
    
    with tab3:
        if config.get('show_matrix', True):
            render_relation_matrix(result)
        else:
            st.info("Matriz de relaciones deshabilitada en configuración")
    
    with tab4:
        render_statistical_analysis(result)
    
    with tab5:
        render_relation_export(result)


def render_relations_list(result: RelationAnalysisResult):
    """Renderizar lista de relaciones identificadas"""
    st.markdown("#### 📋 Relaciones Identificadas")
    
    if not result.relations:
        st.info("No se identificaron relaciones con los parámetros seleccionados")
        return
    
    # Crear DataFrame para mostrar relaciones
    relations_data = []
    for i, relation in enumerate(result.relations):
        relations_data.append({
            'ID': i + 1,
            'Concepto Origen': relation.source_concept,
            'Concepto Destino': relation.target_concept,
            'Tipo': relation.relation_type,
            'Fuerza': f"{relation.strength:.3f}",
            'Confianza': f"{relation.confidence:.3f}",
            'Contexto': relation.context[:80] + "..." if len(relation.context) > 80 else relation.context
        })
    
    df = pd.DataFrame(relations_data)
    
    # Mostrar tabla con relaciones
    st.dataframe(df, use_container_width=True)
    
    # Mostrar detalles de cada relación
    st.markdown("#### 🔍 Detalles de Relaciones")
    
    # Filtros para la lista
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox(
            "Filtrar por Tipo",
            ["Todos"] + list(set(r.relation_type for r in result.relations)),
            key="filter_relation_type"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Ordenar por",
            ["Fuerza", "Confianza", "Tipo"],
            key="sort_relations_by"
        )
    
    with col3:
        show_count = st.slider(
            "Mostrar",
            min_value=1,
            max_value=max(5, min(50, len(result.relations))),
            value=min(10, len(result.relations)),
            key="show_relations_count"
        )
    
    # Aplicar filtros
    filtered_relations = result.relations
    if filter_type != "Todos":
        filtered_relations = [r for r in filtered_relations if r.relation_type == filter_type]
    
    # Ordenar
    if sort_by == "Fuerza":
        filtered_relations = sorted(filtered_relations, key=lambda x: x.strength, reverse=True)
    elif sort_by == "Confianza":
        filtered_relations = sorted(filtered_relations, key=lambda x: x.confidence, reverse=True)
    else:  # Tipo
        filtered_relations = sorted(filtered_relations, key=lambda x: x.relation_type)
    
    # Mostrar relaciones filtradas
    for i, relation in enumerate(filtered_relations[:show_count]):
        with st.expander(f"**Relación {i+1}: {relation.source_concept} ↔ {relation.target_concept}**"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Información de la Relación:**")
                st.markdown(f"• **Tipo:** {relation.relation_type}")
                st.markdown(f"• **Fuerza:** {relation.strength:.3f}")
                st.markdown(f"• **Confianza:** {relation.confidence:.3f}")
                
                st.markdown("**Contexto:**")
                st.info(relation.context)
            
            with col2:
                st.markdown("**Métricas Visuales:**")
                
                # Barra de progreso para fuerza
                st.progress(relation.strength, text=f"Fuerza: {relation.strength:.1%}")
                
                # Barra de progreso para confianza
                st.progress(relation.confidence, text=f"Confianza: {relation.confidence:.1%}")
                
                # Metadata adicional
                if relation.metadata:
                    st.markdown("**Metadata:**")
                    st.json(relation.metadata)
            
            # Mostrar citaciones si existen
            if relation.citations:
                st.markdown("---")
                st.markdown("📚 **Citaciones:**")
                
                for j, citation in enumerate(relation.citations[:3], 1):
                    with st.container():
                        st.markdown(f"**Cita {j}:**")
                        st.markdown(f"📄 **Fuente:** {citation.get('source_file', 'unknown')}")
                        st.markdown(f"📝 **Contenido:** {citation.get('content', '')[:200]}...")
                        if j < len(relation.citations[:3]):
                            st.markdown("---")


def render_network_graph(result: RelationAnalysisResult):
    """Renderizar grafo de red interactivo"""
    st.markdown("#### 🕸️ Grafo de Red de Conceptos")
    
    if not result.relations:
        st.info("No hay relaciones para visualizar")
        return
    
    try:
        # Crear grafo de NetworkX
        G = nx.Graph()
        
        # Agregar nodos
        for concept in result.concepts:
            G.add_node(concept)
        
        # Agregar aristas
        for relation in result.relations:
            G.add_edge(
                relation.source_concept,
                relation.target_concept,
                weight=relation.strength,
                relation_type=relation.relation_type
            )
        
        # Calcular layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Crear visualización con Plotly
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Obtener información de la arista
            edge_info = G[edge[0]][edge[1]]
            weight = edge_info.get('weight', 0.5)
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=weight * 5, color='rgba(125, 125, 125, 0.5)'),
                    hoverinfo='none'
                )
            )
        
        # Crear trazas de nodos
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Calcular grado del nodo (número de conexiones)
            degree = G.degree(node)
            node_size.append(20 + degree * 5)
            node_text.append(f"{node}<br>Conexiones: {degree}")
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            hovertext=node_text,
            hoverinfo='text'
        )
        
        # Crear figura
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title="Red de Relaciones entre Conceptos",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas del grafo
        st.markdown("#### 📊 Estadísticas del Grafo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodos", G.number_of_nodes())
        
        with col2:
            st.metric("Aristas", G.number_of_edges())
        
        with col3:
            density = nx.density(G)
            st.metric("Densidad", f"{density:.3f}")
        
        with col4:
            if G.number_of_nodes() > 0:
                avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                st.metric("Grado Promedio", f"{avg_degree:.2f}")
        
    except Exception as e:
        st.error(f"Error generando grafo: {str(e)}")
        st.info("💡 Intenta con menos conceptos o relaciones más fuertes")


def render_relation_matrix(result: RelationAnalysisResult):
    """Renderizar matriz de relaciones"""
    st.markdown("#### 📊 Matriz de Relaciones")
    
    if not result.relations:
        st.info("No hay relaciones para mostrar en matriz")
        return
    
    try:
        # Crear matriz de relaciones
        matrix_size = len(result.concepts)
        relation_matrix = np.zeros((matrix_size, matrix_size))
        
        # Llenar matriz con fuerzas de relaciones
        for relation in result.relations:
            try:
                idx1 = result.concepts.index(relation.source_concept)
                idx2 = result.concepts.index(relation.target_concept)
                relation_matrix[idx1, idx2] = relation.strength
                relation_matrix[idx2, idx1] = relation.strength
            except ValueError:
                continue
        
        # Crear heatmap con Plotly
        fig = go.Figure(data=go.Heatmap(
            z=relation_matrix,
            x=result.concepts,
            y=result.concepts,
            colorscale='Viridis',
            hovertemplate='%{y} ↔ %{x}<br>Fuerza: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Matriz de Fuerza de Relaciones",
            xaxis_title="Conceptos",
            yaxis_title="Conceptos",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de relaciones más fuertes
        st.markdown("#### 🔝 Top Relaciones Más Fuertes")
        
        # Ordenar relaciones por fuerza
        top_relations = sorted(result.relations, key=lambda x: x.strength, reverse=True)[:10]
        
        top_data = []
        for i, rel in enumerate(top_relations, 1):
            top_data.append({
                'Ranking': i,
                'Concepto 1': rel.source_concept,
                'Concepto 2': rel.target_concept,
                'Tipo': rel.relation_type,
                'Fuerza': f"{rel.strength:.3f}"
            })
        
        df_top = pd.DataFrame(top_data)
        st.dataframe(df_top, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generando matriz: {str(e)}")


def render_statistical_analysis(result: RelationAnalysisResult):
    """Renderizar análisis estadístico de relaciones"""
    st.markdown("#### 📈 Análisis Estadístico")
    
    if not result.relations:
        st.info("No hay datos para análisis estadístico")
        return
    
    # Estadísticas por tipo de relación
    st.markdown("##### 📊 Distribución por Tipo de Relación")
    
    relation_types = [r.relation_type for r in result.relations]
    type_counts = pd.Series(relation_types).value_counts()
    
    fig_types = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Distribución de Tipos de Relación"
    )
    st.plotly_chart(fig_types, use_container_width=True)
    
    # Distribución de fuerza de relaciones
    st.markdown("##### 📊 Distribución de Fuerza de Relaciones")
    
    strengths = [r.strength for r in result.relations]
    
    fig_strength = px.histogram(
        x=strengths,
        nbins=20,
        title="Distribución de Fuerza de Relaciones",
        labels={'x': 'Fuerza', 'y': 'Frecuencia'}
    )
    st.plotly_chart(fig_strength, use_container_width=True)
    
    # Estadísticas descriptivas
    st.markdown("##### 📊 Estadísticas Descriptivas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fuerza Media", f"{np.mean(strengths):.3f}")
        st.metric("Fuerza Mediana", f"{np.median(strengths):.3f}")
    
    with col2:
        st.metric("Fuerza Máxima", f"{np.max(strengths):.3f}")
        st.metric("Fuerza Mínima", f"{np.min(strengths):.3f}")
    
    with col3:
        st.metric("Desviación Estándar", f"{np.std(strengths):.3f}")
        st.metric("Coeficiente de Variación", f"{(np.std(strengths) / np.mean(strengths)):.3f}")
    
    # Conceptos más conectados
    st.markdown("##### 🌟 Conceptos Más Conectados")
    
    concept_connections = defaultdict(int)
    for relation in result.relations:
        concept_connections[relation.source_concept] += 1
        concept_connections[relation.target_concept] += 1
    
    top_concepts = sorted(concept_connections.items(), key=lambda x: x[1], reverse=True)[:10]
    
    top_concepts_data = []
    for i, (concept, connections) in enumerate(top_concepts, 1):
        top_concepts_data.append({
            'Ranking': i,
            'Concepto': concept,
            'Conexiones': connections
        })
    
    df_concepts = pd.DataFrame(top_concepts_data)
    st.dataframe(df_concepts, use_container_width=True)


def render_relation_export(result: RelationAnalysisResult):
    """Renderizar opciones de exportación"""
    st.markdown("#### 💾 Exportar Análisis de Relaciones")
    
    # Crear extractor para exportación
    from ...extractors.relation_extractor import RelationExtractor
    extractor = RelationExtractor(AnalysisConfig())
    
    # Exportar datos
    export_data = extractor.export_relations(result, 'json')
    
    if export_data:
        # Convertir a JSON
        import json
        json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Botón de descarga
        st.download_button(
            label="📥 Descargar Análisis (JSON)",
            data=json_data,
            file_name=f"relation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_relation_json"
        )
        
        # Mostrar resumen de exportación
        st.markdown("#### 📋 Resumen de Exportación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Metadatos:**")
            st.json(export_data['metadata'])
        
        with col2:
            st.markdown("**Estadísticas:**")
            st.metric("Total Relaciones", len(export_data['relations']))
            st.metric("Total Conceptos", len(export_data['concepts']))
            st.metric("Total Citaciones", len(export_data['citations']))
        
        # Mostrar información adicional sobre la exportación
        st.info("""
        **💡 Información sobre la exportación:**
        
        - **JSON**: Incluye todos los datos del análisis para procesamiento posterior
        - **Metadatos**: Información sobre el tipo de análisis y parámetros utilizados
        - **Citaciones**: Referencias completas a las fuentes originales
        - **Conceptos**: Lista de todos los conceptos analizados
        - **Relaciones**: Detalles completos de cada relación identificada
        """)
    
    # Botón para limpiar resultados
    if st.button("🗑️ Limpiar Resultados", key="clear_relation_results"):
        if 'relation_analysis_result' in st.session_state:
            del st.session_state['relation_analysis_result']
        if 'relation_analysis_config' in st.session_state:
            del st.session_state['relation_analysis_config']
        st.rerun()

