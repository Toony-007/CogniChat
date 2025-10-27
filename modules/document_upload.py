"""
Pestaña de carga de documentos
"""

import streamlit as st
import os
from pathlib import Path
from typing import List
from utils.error_handler import ErrorHandler
from utils.logger import setup_logger
from config.settings import config

logger = setup_logger()
error_handler = ErrorHandler()

def render():
    """Renderizar la pestaña de gestión de documentos"""
    st.header("📄 Gestión de Documentos")
    
    # Información del sistema en cards
    col1, col2, col3 = st.columns(3)
    
    # Contar documentos cargados usando función estandarizada
    uploaded_files = get_valid_uploaded_files()
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h2 style="margin: 0; font-size: 2.5rem;">📚</h2>
            <h3 style="margin: 0.5rem 0 0 0;">Documentos Cargados</h3>
            <h1 style="margin: 0; font-size: 2rem;">{len(uploaded_files)}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h2 style="margin: 0; font-size: 2.5rem;">📋</h2>
            <h3 style="margin: 0.5rem 0 0 0;">Formatos Soportados</h3>
            <h1 style="margin: 0; font-size: 2rem;">{len(config.SUPPORTED_FORMATS)}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h2 style="margin: 0; font-size: 2.5rem;">💾</h2>
            <h3 style="margin: 0.5rem 0 0 0;">Tamaño Máximo</h3>
            <h1 style="margin: 0; font-size: 2rem;">{config.MAX_FILE_SIZE_MB} MB</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Sección de carga de documentos
    st.subheader("📤 Cargar Nuevos Documentos")
    
    # Mostrar formatos soportados de manera más visual
    with st.expander("📋 Ver formatos soportados", expanded=False):
        cols = st.columns(4)
        for i, format_ext in enumerate(config.SUPPORTED_FORMATS):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background: #34495e; color: #ecf0f1; padding: 0.5rem; border-radius: 5px; text-align: center; margin: 0.2rem;">
                    <code>{format_ext}</code>
                </div>
                """, unsafe_allow_html=True)
    
    # Área de carga de archivos mejorada
    uploaded_files_new = st.file_uploader(
        "Selecciona archivos para cargar",
        type=[ext.replace('.', '') for ext in config.SUPPORTED_FORMATS],
        accept_multiple_files=True,
        help=f"Tamaño máximo por archivo: {config.MAX_FILE_SIZE_MB} MB"
    )
    
    if uploaded_files_new:
        st.success(f"✅ {len(uploaded_files_new)} archivo(s) seleccionado(s)")
        
        # Mostrar vista previa de archivos a cargar
        with st.expander("👀 Vista previa de archivos", expanded=True):
            for file in uploaded_files_new:
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 **{file.name}**")
                with col2:
                    st.write(f"{file_size_mb:.2f} MB")
                with col3:
                    if file_size_mb > config.MAX_FILE_SIZE_MB:
                        st.error("❌ Muy grande")
                    else:
                        st.success("✅ OK")
        
        # Botón para procesar archivos
        if st.button("📤 Cargar Documentos", type="primary"):
            success_count = 0
            error_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files_new):
                try:
                    status_text.text(f"Procesando: {file.name}")
                    
                    # Validar tamaño
                    file_size_mb = len(file.getvalue()) / (1024 * 1024)
                    if file_size_mb > config.MAX_FILE_SIZE_MB:
                        st.error(f"❌ {file.name}: Archivo muy grande ({file_size_mb:.1f} MB)")
                        error_count += 1
                        continue
                    
                    # Guardar archivo
                    file_path = config.UPLOADS_DIR / file.name
                    
                    # Verificar si ya existe
                    if file_path.exists():
                        st.warning(f"⚠️ {file.name}: Ya existe, sobrescribiendo...")
                    
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    success_count += 1
                    logger.info(f"Archivo guardado: {file.name}")
                    
                except Exception as e:
                    error_handler.handle_error(e, f"Error al cargar {file.name}")
                    st.error(f"❌ Error al cargar {file.name}")
                    error_count += 1
                
                # Actualizar barra de progreso
                progress_bar.progress((i + 1) / len(uploaded_files_new))
            
            status_text.empty()
            progress_bar.empty()
            
            # Mostrar resumen
            if success_count > 0:
                st.success(f"✅ {success_count} archivo(s) cargado(s) exitosamente")
            if error_count > 0:
                st.error(f"❌ {error_count} archivo(s) fallaron")
            
            if success_count > 0:
                st.rerun()
    
    st.divider()
    
    # Gestión de documentos existentes
    st.subheader("📂 Documentos Existentes")
    
    if uploaded_files:
        # Controles de gestión
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{len(uploaded_files)} documento(s) disponible(s)**")
        
        with col2:
            if st.button("🔄 Actualizar Lista", key="update_document_list"):
                st.rerun()
        
        with col3:
            if st.button("🗑️ Eliminar Todos", type="secondary"):
                if st.session_state.get('confirm_delete_all', False):
                    try:
                        for file_path in uploaded_files:
                            file_path.unlink()
                        st.success("✅ Todos los documentos eliminados")
                        st.session_state.confirm_delete_all = False
                        st.rerun()
                    except Exception as e:
                        error_handler.handle_error(e, "Error al eliminar documentos")
                        st.error("❌ Error al eliminar documentos")
                else:
                    st.session_state.confirm_delete_all = True
                    st.warning("⚠️ Haz clic nuevamente para confirmar")
        
        # Lista de documentos con mejor diseño
        for i, file_path in enumerate(uploaded_files):
            try:
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                file_ext = file_path.suffix.upper()
                
                # Determinar icono según extensión
                icon_map = {
                    '.PDF': '📕', '.DOCX': '📘', '.DOC': '📘', 
                    '.TXT': '📄', '.MD': '📝', '.HTML': '🌐',
                    '.JSON': '📊', '.XML': '📋', '.CSV': '📈',
                    '.XLSX': '📊', '.XLS': '📊'
                }
                icon = icon_map.get(file_ext, '📄')
                
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: #34495e; color: #ecf0f1; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #00FF99;">
                            {icon} <strong>{file_path.name}</strong><br>
                            <small style="color: #bdc3c7;">Tipo: {file_ext} • Tamaño: {file_size:.2f} MB</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if file_ext == '.TXT':
                            if st.button("👁️ Ver", key=f"view_{i}"):
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()[:500]
                                    st.text_area("Contenido (primeros 500 caracteres):", content, height=100)
                                except Exception as e:
                                    st.error(f"Error al leer archivo: {e}")
                    
                    with col3:
                        if st.button("📥 Descargar", key=f"download_{i}"):
                            try:
                                with open(file_path, 'rb') as f:
                                    st.download_button(
                                        label="💾 Descargar",
                                        data=f.read(),
                                        file_name=file_path.name,
                                        key=f"download_btn_{i}"
                                    )
                            except Exception as e:
                                st.error(f"Error al descargar: {e}")
                    
                    with col4:
                        if st.button("🗑️ Eliminar", key=f"delete_{i}", type="secondary"):
                            try:
                                file_path.unlink()
                                st.success(f"✅ {file_path.name} eliminado")
                                st.rerun()
                            except Exception as e:
                                error_handler.handle_error(e, f"Error al eliminar {file_path.name}")
                                st.error(f"❌ Error al eliminar {file_path.name}")
                
                if i < len(uploaded_files) - 1:
                    st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                    
            except Exception as e:
                error_handler.handle_error(e, f"Error al mostrar información de {file_path.name}")
                st.error(f"❌ Error al mostrar {file_path.name}")
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #2c3e50; color: #ecf0f1; border-radius: 10px; border: 2px dashed #34495e;">
            <h2 style="color: #bdc3c7;">📂 No hay documentos cargados</h2>
            <p style="color: #95a5a6;">Sube algunos documentos para comenzar a usar el sistema RAG</p>
        </div>
        """, unsafe_allow_html=True)

def process_uploaded_files(uploaded_files):
    """Procesar archivos subidos"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Procesando {file.name}...")
            
            # Verificar tamaño del archivo
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                error_handler.handle_warning(
                    f"Archivo {file.name} excede el tamaño máximo ({file_size_mb:.2f} MB > {config.MAX_FILE_SIZE_MB} MB)",
                    "Carga de documentos"
                )
                continue
            
            # Guardar archivo
            file_path = config.UPLOADS_DIR / file.name
            
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            logger.info(f"Archivo guardado: {file.name}")
            
            # Actualizar progreso
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
        
        status_text.text("✅ Procesamiento completado")
        st.success(f"Se procesaron {len(uploaded_files)} archivos correctamente")
        
        # Limpiar después de 2 segundos
        import time
        time.sleep(2)
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        error_handler.handle_error(e, "Error al procesar archivos subidos")

def get_uploaded_files_count() -> int:
    """Obtener número de archivos cargados"""
    try:
        return len(list(config.UPLOADS_DIR.glob("*")))
    except Exception as e:
        logger.error(f"Error al contar archivos: {e}")
        return 0

def get_uploaded_files_list() -> List[Path]:
    """Obtener lista de archivos cargados"""
    try:
        return list(config.UPLOADS_DIR.glob("*"))
    except Exception as e:
        logger.error(f"Error al listar archivos: {e}")
        return []

def get_valid_uploaded_files() -> List[Path]:
    """Obtener lista de archivos válidos (excluyendo archivos ocultos y directorios)"""
    try:
        files = get_uploaded_files_list()
        return [f for f in files if f.is_file() and not f.name.startswith('.')]
    except Exception as e:
        logger.error(f"Error al obtener archivos válidos: {e}")
        return []

def delete_file(file_path: Path):
    """Eliminar un archivo"""
    try:
        file_path.unlink()
        logger.info(f"Archivo eliminado: {file_path.name}")
        st.success(f"Archivo {file_path.name} eliminado correctamente")
    except Exception as e:
        error_handler.handle_error(e, f"Error al eliminar archivo {file_path.name}")

def clear_all_files():
    """Limpiar todos los archivos"""
    try:
        files = get_uploaded_files_list()
        for file_path in files:
            file_path.unlink()
        
        logger.info(f"Se eliminaron {len(files)} archivos")
        st.success(f"Se eliminaron {len(files)} archivos correctamente")
        
    except Exception as e:
        error_handler.handle_error(e, "Error al limpiar archivos")