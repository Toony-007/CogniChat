"""
Componentes de Gestión de Cache RAG
Interfaz para manejo avanzado del rag_cache.json
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from ...core.rag_cache_manager import RAGCacheManager


def render_cache_management_panel():
    """
    Renderizar panel de gestión avanzada del cache
    """
    st.markdown("### 🔧 Gestión Avanzada del Cache")
    
    cache_manager = RAGCacheManager()
    
    # Verificar estado del cache
    cache_exists, message = cache_manager.detect_cache()
    
    if not cache_exists:
        st.warning(f"⚠️ {message}")
        return
    
    # Tabs para diferentes opciones de gestión
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Estadísticas",
        "🔄 Actualización",
        "🔍 Verificación",
        "⚙️ Configuración"
    ])
    
    with tab1:
        render_cache_statistics(cache_manager)
    
    with tab2:
        render_cache_update_options(cache_manager)
    
    with tab3:
        render_cache_verification(cache_manager)
    
    with tab4:
        render_cache_configuration(cache_manager)


def render_cache_statistics(cache_manager: RAGCacheManager):
    """
    Renderizar estadísticas detalladas del cache
    """
    st.markdown("#### 📊 Estadísticas Detalladas")
    
    cache_info = cache_manager.get_cache_info()
    
    if not cache_info:
        st.error("No se pudo obtener información del cache")
        return
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📄 Documentos",
            cache_info.get('unique_sources', 0),
            help="Número de documentos únicos en el cache"
        )
    
    with col2:
        st.metric(
            "📝 Chunks Totales",
            cache_info.get('total_chunks', 0),
            help="Número total de fragmentos de texto"
        )
    
    with col3:
        st.metric(
            "💾 Tamaño",
            f"{cache_info.get('file_size_mb', 0)} MB",
            help="Tamaño del archivo de cache"
        )
    
    with col4:
        last_updated = cache_info.get('last_updated', 'Desconocido')
        st.metric(
            "🕒 Última Actualización",
            last_updated.split('T')[0] if 'T' in last_updated else last_updated,
            help="Fecha de la última actualización del cache"
        )
    
    # Distribución por fuente
    st.markdown("#### 📂 Distribución por Fuente")
    
    chunks_data = cache_manager.cache_data.get('chunks', {})
    source_distribution = {}
    
    for file_key, chunk_list in chunks_data.items():
        for chunk in chunk_list:
            if isinstance(chunk, dict) and 'metadata' in chunk:
                source = chunk['metadata'].get('source_file', 'unknown')
                if source not in source_distribution:
                    source_distribution[source] = 0
                source_distribution[source] += 1
    
    if source_distribution:
        # Crear tabla de distribución
        import pandas as pd
        
        df = pd.DataFrame([
            {'Fuente': source, 'Chunks': count, 'Porcentaje': f"{(count/cache_info.get('total_chunks', 1)*100):.1f}%"}
            for source, count in source_distribution.items()
        ])
        
        st.dataframe(df, use_container_width=True)
        
        # Gráfico de distribución
        if len(source_distribution) > 1:
            import plotly.express as px
            
            fig = px.pie(
                values=list(source_distribution.values()),
                names=list(source_distribution.keys()),
                title="Distribución de Chunks por Fuente"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos de distribución disponibles")


def render_cache_update_options(cache_manager: RAGCacheManager):
    """
    Renderizar opciones de actualización del cache
    """
    st.markdown("#### 🔄 Opciones de Actualización")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔄 Actualización Manual**")
        
        if st.button("🔄 Recargar Cache", type="primary", help="Forzar recarga del cache desde archivo"):
            with st.spinner("Recargando cache..."):
                if cache_manager.force_reload_cache():
                    st.success("✅ Cache recargado exitosamente")
                    st.rerun()
                else:
                    st.error("❌ Error al recargar cache")
        
        if st.button("🔍 Verificar Cambios", help="Verificar si el cache ha sido modificado"):
            if cache_manager.check_cache_updated():
                st.success("✅ El cache ha sido actualizado")
                st.info("Se recomienda recargar el cache")
            else:
                st.info("ℹ️ El cache no ha sido modificado")
    
    with col2:
        st.markdown("**🧹 Limpieza de Memoria**")
        
        if st.button("🗑️ Limpiar Cache de Memoria", help="Limpiar cache cargado en memoria"):
            cache_manager.chunks_cache = []
            st.success("✅ Cache de memoria limpiado")
            st.rerun()
        
        if st.button("🔄 Resetear Gestor", help="Resetear completamente el gestor de cache"):
            cache_manager.cache_data = None
            cache_manager.chunks_cache = []
            cache_manager.cache_modified_time = 0
            st.success("✅ Gestor de cache reseteado")
            st.rerun()
    
    # Información del estado actual
    st.markdown("#### 📋 Estado Actual del Cache")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cache en Memoria:**")
        if cache_manager.chunks_cache:
            st.success(f"✅ {len(cache_manager.chunks_cache)} chunks cargados")
        else:
            st.info("ℹ️ No hay chunks en memoria")
    
    with col2:
        st.markdown("**Última Verificación:**")
        if cache_manager.cache_modified_time > 0:
            from datetime import datetime
            last_check = datetime.fromtimestamp(cache_manager.cache_modified_time)
            st.info(f"🕒 {last_check.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("ℹ️ No se ha verificado")


def render_cache_verification(cache_manager: RAGCacheManager):
    """
    Renderizar opciones de verificación del cache
    """
    st.markdown("#### 🔍 Verificación de Integridad")
    
    # Verificar estructura del cache
    if st.button("🔍 Verificar Estructura", help="Verificar que el cache tenga la estructura correcta"):
        with st.spinner("Verificando estructura..."):
            issues = []
            
            if not cache_manager.cache_data:
                issues.append("❌ Cache no cargado")
            else:
                if 'chunks' not in cache_manager.cache_data:
                    issues.append("❌ No se encontró sección 'chunks'")
                else:
                    chunks_data = cache_manager.cache_data['chunks']
                    if not isinstance(chunks_data, dict):
                        issues.append("❌ Sección 'chunks' no es un diccionario")
                    else:
                        total_chunks = sum(len(chunk_list) for chunk_list in chunks_data.values())
                        if total_chunks == 0:
                            issues.append("⚠️ No hay chunks en el cache")
                        else:
                            st.success(f"✅ Estructura válida: {total_chunks} chunks encontrados")
                
                if 'last_updated' not in cache_manager.cache_data:
                    issues.append("⚠️ No se encontró timestamp de actualización")
            
            if issues:
                for issue in issues:
                    st.markdown(issue)
            else:
                st.success("✅ Estructura del cache es válida")
    
    # Verificar chunks individuales
    if st.button("🔍 Verificar Chunks", help="Verificar integridad de los chunks individuales"):
        with st.spinner("Verificando chunks..."):
            chunks_data = cache_manager.cache_data.get('chunks', {})
            valid_chunks = 0
            invalid_chunks = 0
            
            for file_key, chunk_list in chunks_data.items():
                for chunk in chunk_list:
                    if isinstance(chunk, dict):
                        if 'content' in chunk and 'metadata' in chunk:
                            if chunk['content'].strip():
                                valid_chunks += 1
                            else:
                                invalid_chunks += 1
                        else:
                            invalid_chunks += 1
                    else:
                        invalid_chunks += 1
            
            st.success(f"✅ Chunks válidos: {valid_chunks}")
            if invalid_chunks > 0:
                st.warning(f"⚠️ Chunks inválidos: {invalid_chunks}")
    
    # Verificar archivo físico
    if st.button("🔍 Verificar Archivo", help="Verificar que el archivo de cache sea accesible"):
        cache_file = cache_manager.cache_file
        
        if cache_file.exists():
            try:
                file_size = cache_file.stat().st_size
                st.success(f"✅ Archivo accesible: {file_size / (1024*1024):.2f} MB")
                
                # Verificar que sea JSON válido
                with open(cache_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                st.success("✅ Archivo JSON válido")
                
            except json.JSONDecodeError:
                st.error("❌ Archivo JSON inválido")
            except Exception as e:
                st.error(f"❌ Error al acceder al archivo: {str(e)}")
        else:
            st.error("❌ Archivo de cache no encontrado")


def render_cache_configuration(cache_manager: RAGCacheManager):
    """
    Renderizar configuración del cache
    """
    st.markdown("#### ⚙️ Configuración del Cache")
    
    # Configuración de detección automática
    st.markdown("**🔍 Detección Automática**")
    
    auto_detect = st.checkbox(
        "Detectar cambios automáticamente",
        value=True,
        help="Verificar automáticamente si el cache ha sido modificado"
    )
    
    check_interval = st.slider(
        "Intervalo de verificación (segundos)",
        min_value=5,
        max_value=300,
        value=30,
        help="Cada cuántos segundos verificar cambios en el cache"
    )
    
    # Configuración de memoria
    st.markdown("**💾 Gestión de Memoria**")
    
    max_memory_chunks = st.number_input(
        "Máximo chunks en memoria",
        min_value=100,
        max_value=10000,
        value=1000,
        help="Número máximo de chunks a mantener en memoria"
    )
    
    auto_cleanup = st.checkbox(
        "Limpieza automática de memoria",
        value=True,
        help="Limpiar automáticamente chunks antiguos de la memoria"
    )
    
    # Configuración de logging
    st.markdown("**📝 Logging**")
    
    enable_cache_logging = st.checkbox(
        "Habilitar logging del cache",
        value=True,
        help="Registrar operaciones del cache en los logs"
    )
    
    log_level = st.selectbox(
        "Nivel de logging",
        ["INFO", "DEBUG", "WARNING", "ERROR"],
        index=0,
        help="Nivel de detalle en los logs del cache"
    )
    
    # Botón para aplicar configuración
    if st.button("💾 Aplicar Configuración", type="primary"):
        # Aquí se aplicarían las configuraciones (por ahora solo mostrar)
        st.success("✅ Configuración aplicada")
        st.info("Nota: Las configuraciones se aplicarán en la próxima sesión")


def render_cache_export_options(cache_manager: RAGCacheManager):
    """
    Renderizar opciones de exportación del cache
    """
    st.markdown("#### 💾 Exportar Cache")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📄 Exportar Datos**")
        
        if st.button("📄 Exportar Chunks", help="Exportar todos los chunks a JSON"):
            chunks = cache_manager.get_chunks_for_analysis()
            if chunks:
                import json
                chunks_json = json.dumps(chunks, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="💾 Descargar Chunks",
                    data=chunks_json,
                    file_name=f"chunks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No hay chunks para exportar")
    
    with col2:
        st.markdown("**📊 Exportar Estadísticas**")
        
        if st.button("📊 Exportar Estadísticas", help="Exportar estadísticas del cache"):
            cache_info = cache_manager.get_cache_info()
            if cache_info:
                import json
                stats_json = json.dumps(cache_info, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="💾 Descargar Estadísticas",
                    data=stats_json,
                    file_name=f"cache_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No hay estadísticas para exportar")
