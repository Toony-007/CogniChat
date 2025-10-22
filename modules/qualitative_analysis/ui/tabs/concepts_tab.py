"""
Tab de Extracción de Conceptos Clave
Primer sub-módulo del sistema de análisis cualitativo
"""

import streamlit as st
from typing import List, Dict, Any
import json

from ...extractors.concept_extractor import ConceptExtractor, ExtractedConcept
from ...core.config import AnalysisConfig
from ..components.educational import (
    show_methodology_box,
    show_interpretation_guide,
    show_concept_card,
    show_statistics_panel,
    show_help_sidebar
)


def render_concepts_tab(chunks: List[Dict[str, Any]], config: AnalysisConfig):
    """
    Renderizar tab de extracción de conceptos clave
    
    Args:
        chunks: Lista de chunks de documentos
        config: Configuración del análisis
    """
    
    # Título con descripción
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #00FF99 0%, #00CC7A 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style="color: white; margin: 0; font-size: 2rem;">🔍 Extracción de Conceptos Clave</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Identifica los conceptos más importantes de tus documentos con fundamentación completa
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostrar ayuda en sidebar
    show_help_sidebar()
    
    # Mostrar metodología si está habilitado
    if config.show_methodology:
        show_methodology_box(
            title="Extracción Inteligente con TF-IDF",
            description="""
            Este análisis identifica los **conceptos más relevantes** de tus documentos
            usando **TF-IDF** (Term Frequency-Inverse Document Frequency), una técnica
            estándar en análisis de textos que identifica términos que son:
            
            - **Frecuentes** en documentos específicos (TF - Term Frequency)
            - **Raros** en el corpus general (IDF - Inverse Document Frequency)
            
            Esto permite encontrar conceptos que son **importantes y distintivos**.
            """,
            steps=[
                "**Preprocesamiento**: Limpia el texto, elimina palabras vacías (stopwords) y normaliza",
                "**Vectorización**: Convierte el texto en representación matemática usando TF-IDF",
                "**Extracción**: Identifica términos individuales y frases completas (n-gramas)",
                "**Puntuación**: Calcula un score de relevancia para cada concepto (0.0 a 1.0)",
                "**Citación**: Identifica y registra todas las fuentes donde aparece cada concepto",
                "**Análisis de relaciones**: Identifica conceptos que aparecen juntos frecuentemente",
                "**Ordenamiento**: Presenta los conceptos ordenados por relevancia"
            ],
            algorithm_info="TF-IDF con detección de n-gramas (frases de 1-3 palabras)"
        )
    
    # Mostrar guía de interpretación si está habilitado
    if config.show_interpretation_guide:
        show_interpretation_guide(
            what_it_means="""
            Los **conceptos extraídos** representan las ideas centrales y términos técnicos
            más importantes en tus documentos. El **score de relevancia** indica qué tan
            característico es cada concepto del contenido analizado.
            
            Los conceptos con mayor score son aquellos que:
            - Aparecen frecuentemente en tus documentos
            - Son específicos de tu contenido (no son palabras comunes)
            - Están bien distribuidos en las fuentes
            """,
            how_to_use=[
                "Identifica los temas principales de tu corpus documental",
                "Verifica que los conceptos extraídos coinciden con tu comprensión del contenido",
                "Usa las citas para volver a las fuentes originales y profundizar",
                "Examina los conceptos relacionados para entender conexiones temáticas",
                "Compara la distribución de conceptos entre diferentes fuentes",
                "Usa estos conceptos como base para análisis posteriores (temas, relaciones, etc.)"
            ],
            limitations=[
                "El sistema identifica palabras y frases frecuentes, pero no comprende significado",
                "Conceptos muy específicos con baja frecuencia pueden no aparecer",
                "La relevancia es estadística, no semántica (requiere validación del investigador)",
                "Términos polisémicos (múltiples significados) no se distinguen automáticamente",
                "La calidad depende de la calidad y cantidad de documentos analizados"
            ]
        )
    
    # Verificar que hay documentos
    if not chunks:
        st.warning("""
        ⚠️ **No hay documentos para analizar**
        
        Por favor, ve a la pestaña **"📄 Gestión de Documentos"** para:
        1. Subir documentos (PDF, DOCX, TXT, etc.)
        2. Luego ve a **"🧠 Procesamiento RAG"** para procesar los documentos
        3. Vuelve aquí para realizar el análisis cualitativo
        """)
        return
    
    # Información sobre los documentos
    st.markdown("### 📂 Documentos Disponibles")
    
    sources = list(set(chunk['metadata'].get('source_file', 'unknown') for chunk in chunks))
    total_content = sum(len(chunk.get('content', '')) for chunk in chunks)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documentos", len(sources))
    with col2:
        st.metric("📝 Chunks", len(chunks))
    with col3:
        st.metric("📊 Caracteres", f"{total_content:,}")
    
    with st.expander("📋 Ver lista de fuentes"):
        for i, source in enumerate(sources, 1):
            st.markdown(f"{i}. `{source}`")
    
    st.divider()
    
    # Configuración del análisis
    st.markdown("### ⚙️ Configuración del Análisis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_concepts = st.slider(
            "Máximo de conceptos",
            min_value=10,
            max_value=50,
            value=config.max_concepts,
            help="Número máximo de conceptos a extraer"
        )
    
    with col2:
        min_frequency = st.slider(
            "Frecuencia mínima",
            min_value=1,
            max_value=10,
            value=config.min_concept_frequency,
            help="Número mínimo de apariciones para considerar un concepto"
        )
    
    with col3:
        use_ngrams = st.checkbox(
            "Detectar frases completas",
            value=config.use_ngrams,
            help="Además de palabras individuales, detectar frases de 2-3 palabras"
        )
    
    # Actualizar configuración
    config.max_concepts = max_concepts
    config.min_concept_frequency = min_frequency
    config.use_ngrams = use_ngrams
    
    st.divider()
    
    # Botón de análisis
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button(
            "🚀 Extraer Conceptos",
            type="primary",
            use_container_width=True,
            help="Iniciar análisis de extracción de conceptos"
        )
    
    # Realizar análisis
    if analyze_button or st.session_state.get('concepts_extracted'):
        
        # Mostrar spinner durante análisis
        with st.spinner("🔍 Analizando documentos y extrayendo conceptos..."):
            try:
                # Crear extractor
                extractor = ConceptExtractor(config)
                
                # Extraer conceptos
                concepts = extractor.extract_concepts(chunks, method='tfidf')
                
                # Guardar en session state
                st.session_state.concepts_extracted = True
                st.session_state.extracted_concepts = concepts
                st.session_state.concept_extractor = extractor
                
                # Obtener resumen
                summary = extractor.get_concept_summary(concepts)
                st.session_state.concepts_summary = summary
                
            except Exception as e:
                st.error(f"❌ **Error en el análisis:** {str(e)}")
                st.exception(e)
                return
        
        # Obtener datos de session state
        concepts = st.session_state.extracted_concepts
        extractor = st.session_state.concept_extractor
        summary = st.session_state.concepts_summary
        
        # Mostrar resultados
        st.success(f"✅ **Análisis completado:** {len(concepts)} conceptos extraídos")
        
        st.divider()
        
        # Panel de estadísticas
        show_statistics_panel(summary)
        
        st.divider()
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Conceptos Principales",
            "🔗 Relaciones",
            "📚 Bibliografía",
            "💾 Exportar"
        ])
        
        # Tab 1: Conceptos principales
        with tab1:
            st.markdown("### 🎯 Conceptos Clave Identificados")
            
            st.markdown("""
            <div style="background: #34495e; padding: 1rem; border-radius: 8px; border-left: 4px solid #00FF99; color: #ecf0f1; margin: 1rem 0;">
                <strong>ℹ️ Interpretación:</strong> Los conceptos están ordenados por relevancia. 
                Los primeros son los más característicos de tu corpus documental.
                Cada concepto incluye <strong>citas a las fuentes originales</strong> para fundamentación.
            </div>
            """, unsafe_allow_html=True)
            
            # Filtros
            col1, col2 = st.columns([3, 1])
            with col1:
                filter_text = st.text_input(
                    "🔍 Filtrar conceptos",
                    placeholder="Escribe para buscar...",
                    help="Filtra los conceptos por texto"
                )
            with col2:
                sort_by = st.selectbox(
                    "Ordenar por",
                    ["Relevancia", "Frecuencia", "Nombre"],
                    help="Criterio de ordenamiento"
                )
            
            # Aplicar filtros
            filtered_concepts = concepts
            if filter_text:
                filtered_concepts = [
                    c for c in concepts
                    if filter_text.lower() in c.concept.lower()
                ]
            
            # Ordenar
            if sort_by == "Frecuencia":
                filtered_concepts = sorted(filtered_concepts, key=lambda c: c.frequency, reverse=True)
            elif sort_by == "Nombre":
                filtered_concepts = sorted(filtered_concepts, key=lambda c: c.concept)
            
            # Paginación
            concepts_per_page = 10
            total_pages = (len(filtered_concepts) + concepts_per_page - 1) // concepts_per_page
            
            if total_pages > 1:
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    page = st.selectbox(
                        "Página",
                        range(1, total_pages + 1),
                        format_func=lambda x: f"Página {x} de {total_pages}"
                    )
            else:
                page = 1
            
            # Calcular rango
            start_idx = (page - 1) * concepts_per_page
            end_idx = min(start_idx + concepts_per_page, len(filtered_concepts))
            
            # Mostrar conceptos
            st.markdown(f"**Mostrando {start_idx + 1}-{end_idx} de {len(filtered_concepts)} conceptos**")
            
            for i, concept in enumerate(filtered_concepts[start_idx:end_idx], start=start_idx + 1):
                show_concept_card(
                    concept=concept,
                    rank=i,
                    show_citations=config.enable_citations,
                    show_related=True
                )
        
        # Tab 2: Relaciones entre conceptos
        with tab2:
            st.markdown("### 🔗 Relaciones entre Conceptos")
            
            st.markdown("""
            <div style="background: #34495e; padding: 1rem; border-radius: 8px; border-left: 4px solid #3498db; color: #ecf0f1; margin: 1rem 0;">
                <strong>ℹ️ Sobre las relaciones:</strong> Los conceptos relacionados son aquellos que
                aparecen frecuentemente juntos en los mismos fragmentos de texto. Esto puede indicar
                conexiones temáticas importantes.
            </div>
            """, unsafe_allow_html=True)
            
            # Seleccionar concepto para ver relaciones
            concept_names = [c.concept for c in concepts[:30]]  # Top 30
            selected_concept_name = st.selectbox(
                "Selecciona un concepto para ver sus relaciones",
                concept_names,
                help="Muestra conceptos que co-ocurren con el seleccionado"
            )
            
            # Encontrar concepto seleccionado
            selected_concept = next(c for c in concepts if c.concept == selected_concept_name)
            
            # Mostrar información del concepto
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### 📌 {selected_concept.concept}")
                st.markdown(f"""
                - **Frecuencia:** {selected_concept.frequency}
                - **Relevancia:** {selected_concept.relevance_score:.3f}
                - **Fuentes:** {len(selected_concept.sources)}
                """)
            
            with col2:
                st.metric(
                    "🔗 Conceptos Relacionados",
                    len(selected_concept.related_concepts)
                )
            
            # Mostrar conceptos relacionados
            if selected_concept.related_concepts:
                st.markdown("#### Conceptos que aparecen junto a este:")
                
                for i, related_name in enumerate(selected_concept.related_concepts, 1):
                    # Encontrar información del concepto relacionado
                    related_concept = next((c for c in concepts if c.concept == related_name), None)
                    
                    if related_concept:
                        st.markdown(f"""
                        <div style="background: #2c3e50; padding: 0.8rem; border-radius: 5px; margin: 0.5rem 0; border-left: 3px solid #3498db; color: #ecf0f1;">
                            <strong>{i}. {related_concept.concept}</strong><br>
                            <small>Frecuencia: {related_concept.frequency} | Relevancia: {related_concept.relevance_score:.3f}</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Este concepto no tiene relaciones fuertes identificadas")
            
            # Matriz de co-ocurrencia (simplificada)
            st.markdown("---")
            st.markdown("#### 📊 Matriz de Co-ocurrencia (Top 10)")
            
            # Crear matriz simple
            top_10 = concepts[:10]
            matrix_data = []
            
            for concept in top_10:
                row = {
                    'Concepto': concept.concept,
                    'Relaciones': len(concept.related_concepts)
                }
                matrix_data.append(row)
            
            st.dataframe(matrix_data, use_container_width=True)
        
        # Tab 3: Bibliografía
        with tab3:
            st.markdown("### 📚 Bibliografía y Referencias")
            
            st.markdown("""
            <div style="background: #34495e; padding: 1rem; border-radius: 8px; border-left: 4px solid #9b59b6; color: #ecf0f1; margin: 1rem 0;">
                <strong>ℹ️ Fundamentación científica:</strong> Cada concepto está respaldado por citas
                específicas a las fuentes originales. Esta sección te permite ver todas las referencias
                y verificar la fundamentación del análisis.
            </div>
            """, unsafe_allow_html=True)
            
            # Estadísticas de citación
            cit_stats = extractor.citation_manager.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📖 Total de Citas", cit_stats['total_citations'])
            with col2:
                st.metric("📂 Fuentes Citadas", cit_stats['unique_sources'])
            with col3:
                st.metric("⭐ Relevancia Promedio", f"{cit_stats['avg_relevance']:.3f}")
            
            # Bibliografía
            st.markdown("#### 📋 Referencias Bibliográficas")
            
            bibliography = extractor.citation_manager.generate_bibliography()
            
            for i, ref in enumerate(bibliography, 1):
                st.markdown(f"{i}. {ref}")
            
            # Distribución de citas por fuente
            if 'citations_by_source' in cit_stats:
                st.markdown("---")
                st.markdown("#### 📊 Distribución de Citas por Fuente")
                
                import plotly.graph_objects as go
                
                sources_list = list(cit_stats['citations_by_source'].keys())
                counts = list(cit_stats['citations_by_source'].values())
                
                # Truncar nombres largos
                sources_display = [s.split('/')[-1][:30] for s in sources_list]
                
                fig = go.Figure([go.Bar(
                    x=counts,
                    y=sources_display,
                    orientation='h',
                    marker_color='#00FF99'
                )])
                
                fig.update_layout(
                    title="Número de Citas por Fuente",
                    xaxis_title="Número de Citas",
                    yaxis_title="Fuente",
                    height=max(400, len(sources_list) * 30),
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Exportar
        with tab4:
            st.markdown("### 💾 Exportar Resultados")
            
            st.markdown("""
            <div style="background: #34495e; padding: 1rem; border-radius: 8px; border-left: 4px solid #e67e22; color: #ecf0f1; margin: 1rem 0;">
                <strong>ℹ️ Formatos de exportación:</strong> Puedes exportar los resultados del análisis
                en diferentes formatos para uso en tu investigación.
            </div>
            """, unsafe_allow_html=True)
            
            # Opciones de exportación
            export_format = st.radio(
                "Selecciona formato de exportación",
                ["JSON (completo)", "JSON (resumen)", "CSV (conceptos)", "Texto (bibliografia)"],
                help="Formato en el que se exportarán los datos"
            )
            
            include_citations = st.checkbox(
                "Incluir citas completas",
                value=True,
                help="Incluye información detallada de citaciones"
            )
            
            # Generar exportación
            if st.button("📥 Generar Archivo", type="primary"):
                try:
                    if export_format == "JSON (completo)":
                        data = extractor.export_concepts(concepts, include_citations=include_citations)
                        json_str = json.dumps(data, indent=2, ensure_ascii=False)
                        
                        st.download_button(
                            label="💾 Descargar JSON Completo",
                            data=json_str,
                            file_name="conceptos_completo.json",
                            mime="application/json"
                        )
                    
                    elif export_format == "JSON (resumen)":
                        data = {
                            'summary': summary,
                            'concepts': [
                                {
                                    'concept': c.concept,
                                    'frequency': c.frequency,
                                    'relevance': c.relevance_score,
                                    'sources': c.sources
                                }
                                for c in concepts
                            ]
                        }
                        json_str = json.dumps(data, indent=2, ensure_ascii=False)
                        
                        st.download_button(
                            label="💾 Descargar JSON Resumen",
                            data=json_str,
                            file_name="conceptos_resumen.json",
                            mime="application/json"
                        )
                    
                    elif export_format == "CSV (conceptos)":
                        import csv
                        import io
                        
                        output = io.StringIO()
                        writer = csv.writer(output)
                        
                        # Header
                        writer.writerow(['Concepto', 'Frecuencia', 'Relevancia', 'Num_Fuentes', 'Fuentes'])
                        
                        # Data
                        for c in concepts:
                            writer.writerow([
                                c.concept,
                                c.frequency,
                                f"{c.relevance_score:.3f}",
                                len(c.sources),
                                '; '.join(c.sources)
                            ])
                        
                        csv_str = output.getvalue()
                        
                        st.download_button(
                            label="💾 Descargar CSV",
                            data=csv_str,
                            file_name="conceptos.csv",
                            mime="text/csv"
                        )
                    
                    elif export_format == "Texto (bibliografia)":
                        bibliography = extractor.citation_manager.generate_bibliography()
                        text_content = "BIBLIOGRAFÍA - Análisis de Conceptos Clave\n"
                        text_content += "=" * 60 + "\n\n"
                        text_content += f"Total de conceptos: {len(concepts)}\n"
                        text_content += f"Total de citas: {cit_stats['total_citations']}\n\n"
                        text_content += "REFERENCIAS:\n"
                        text_content += "-" * 60 + "\n"
                        
                        for i, ref in enumerate(bibliography, 1):
                            text_content += f"{i}. {ref}\n"
                        
                        st.download_button(
                            label="💾 Descargar Bibliografía",
                            data=text_content,
                            file_name="bibliografia.txt",
                            mime="text/plain"
                        )
                    
                    st.success("✅ Archivo generado correctamente")
                    
                except Exception as e:
                    st.error(f"❌ Error al generar exportación: {str(e)}")

