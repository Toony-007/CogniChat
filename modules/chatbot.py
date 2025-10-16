"""
Pestaña del chatbot con historial y exportación
"""

import streamlit as st
import time
from datetime import datetime
from config.settings import config
from utils.logger import setup_logger
from utils.error_handler import ErrorHandler
from utils.ollama_client import OllamaClient
from utils.rag_processor import rag_processor
from utils.chat_history import chat_history_manager
from utils.traceability import TraceabilityManager

# Importar exportadores con manejo de errores
try:
    from utils.chat_history import chat_exporter
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

# Importar pyperclip con manejo de errores
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

logger = setup_logger()
error_handler = ErrorHandler()

def render():
    """Renderizar la pestaña del chatbot"""
    # Inicializar el historial de mensajes si no existe (DEBE SER LO PRIMERO)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.header("💬 Chat Inteligente")
    
    # Inicializar cliente Ollama
    ollama_client = OllamaClient()
    
    # Verificar disponibilidad de Ollama
    if not ollama_client.is_available():
        st.error("❌ Ollama no está disponible. Asegúrate de que esté ejecutándose.")
        st.info("Para iniciar Ollama, ejecuta: `ollama serve` en tu terminal")
        return
    
    # Configuración rápida del chat (compacta)
    with st.expander("🎛️ Configuración Rápida del Chat", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Obtener modelos disponibles con caché en session_state
            if 'available_models' not in st.session_state or st.session_state.get('models_last_refresh', 0) < time.time() - 300:
                # Actualizar caché cada 5 minutos
                available_models = ollama_client.get_available_models()
                st.session_state.available_models = available_models
                st.session_state.models_last_refresh = time.time()
            else:
                available_models = st.session_state.available_models
            
            model_names = [model['name'] for model in available_models]
            
            if not model_names:
                st.error("No hay modelos disponibles")
                # Botón para refrescar modelos
                if st.button("🔄 Refrescar modelos", key="refresh_models_chatbot"):
                    st.session_state.available_models = ollama_client.get_available_models(force_refresh=True)
                    st.rerun()
                return
            
            # Usar modelo configurado o por defecto
            current_model = st.session_state.get('selected_llm_model', config.DEFAULT_LLM_MODEL)
            if current_model not in model_names and model_names:
                current_model = model_names[0]
            
            selected_model = st.selectbox(
                "🤖 Modelo LLM",
                model_names,
                index=model_names.index(current_model) if current_model in model_names else 0,
                help="Modelo de lenguaje para el chat",
                key="chat_model_selector"
            )
            
            # Actualizar el modelo seleccionado en session_state
            if selected_model != st.session_state.get('selected_llm_model'):
                st.session_state.selected_llm_model = selected_model
        
        with col2:
            use_rag = st.toggle(
                "📚 Usar RAG", 
                value=st.session_state.get('rag_enabled', True), 
                help="Usar documentos cargados como contexto",
                key="chat_rag_toggle"
            )
            
            # Actualizar configuración RAG en session_state (usar variable estándar)
            st.session_state.rag_enabled = use_rag
            st.session_state.chat_use_rag = use_rag  # Mantener compatibilidad
            
        with col3:
            max_tokens = st.slider(
                "🎯 Máximo tokens", 
                100, config.MAX_RESPONSE_TOKENS, 
                st.session_state.get('chat_max_tokens', config.MAX_RESPONSE_TOKENS),
                help="Longitud máxima de la respuesta",
                key="chat_tokens_slider"
            )
            
            # Actualizar configuración de tokens en session_state
            st.session_state.chat_max_tokens = max_tokens
        
        with col4:
            debug_mode = st.toggle(
                "🔍 Modo Debug",
                value=st.session_state.get('debug_mode', config.ENABLE_DEBUG_MODE),
                help="Mostrar información detallada de procesamiento",
                key="chat_debug_toggle"
            )
            
            # Actualizar configuración de debug en session_state
            st.session_state.debug_mode = debug_mode
    
    # Área principal del chat
    chat_container = st.container()
    
    # Mostrar historial de chat con mejor diseño
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #2c3e50; border-radius: 10px; margin: 1rem 0; color: #ecf0f1;">
                <h3 style="color: #00FF99;">👋 ¡Hola! Soy CogniChat</h3>
                <p style="color: #bdc3c7;">Pregúntame cualquier cosa. Si tienes documentos cargados, puedo usarlos como contexto.</p>
            </div>
            """, unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"🕒 {message['timestamp']}")
                
                # Botones de exportación para mensajes del asistente
                if message["role"] == "assistant":
                    render_message_export_buttons(message, i)
                
                # Mostrar contexto usado si es un mensaje del asistente
                if message["role"] == "assistant" and "context_used" in message and message["context_used"]:
                    with st.expander("📚 Contexto utilizado", expanded=False):
                        st.markdown(f"```\n{message['context_used'][:500]}...\n```")
    
    # Input del usuario mejorado
    if prompt := st.chat_input("💭 Escribe tu mensaje aquí..."):
        # Agregar mensaje del usuario
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕒 {timestamp}")
        
        # Generar respuesta con interfaz mejorada
        with st.chat_message("assistant"):
            # Crear contenedor para el progreso
            progress_container = st.empty()
            response_container = st.empty()
            
            # Inicializar variables de control
            if 'processing_cancelled' not in st.session_state:
                st.session_state.processing_cancelled = False
            
            try:
                # Paso 1: Inicialización
                with progress_container.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("🚀 Iniciando procesamiento...")
                    with col2:
                        if st.button("❌ Cancelar", key="cancel_processing", type="secondary"):
                            st.session_state.processing_cancelled = True
                            st.warning("⚠️ Procesamiento cancelado por el usuario")
                            st.stop()
                
                progress_bar.progress(10)
                time.sleep(0.5)
                
                # Verificar cancelación
                if st.session_state.processing_cancelled:
                    st.session_state.processing_cancelled = False
                    st.stop()
                
                # Obtener configuración de debug
                debug_mode = st.session_state.get('debug_mode', False)
                
                # Paso 2: Procesamiento RAG
                status_text.text("🔍 Analizando documentos...")
                progress_bar.progress(25)
                
                context = None
                relevant_chunks = []
                query_id = None
                
                if use_rag:
                    if debug_mode:
                        # Modo debug: usar trazabilidad
                        context, relevant_chunks, query_id = get_rag_context(prompt, enable_tracing=True)
                    else:
                        # Modo normal: usar contexto completo
                        context = get_rag_context(prompt, enable_tracing=False)
                    
                    progress_bar.progress(50)
                    
                    if context:
                        if debug_mode:
                            status_text.text(f"✅ Contexto recuperado - {len(relevant_chunks)} chunks encontrados")
                        else:
                            status_text.text(f"✅ Contexto completo cargado - {len(context):,} caracteres")
                    else:
                        status_text.text("⚠️ No hay documentos procesados disponibles")
                else:
                    status_text.text("ℹ️ Modo sin RAG - Respuesta general")
                    progress_bar.progress(50)
                
                # Verificar cancelación
                if st.session_state.processing_cancelled:
                    st.session_state.processing_cancelled = False
                    st.stop()
                
                # Paso 3: Preparación del prompt
                status_text.text("📝 Preparando consulta para la IA...")
                progress_bar.progress(70)
                    
                    
                # Construir el prompt optimizado para DeepSeek con contexto completo
                if use_rag:
                    if context:
                        # Prompt optimizado para DeepSeek con razonamiento profundo
                        full_prompt = f"""<think>
El usuario me ha proporcionado documentos completos y me está haciendo una pregunta. Necesito:

1. Analizar cuidadosamente TODA la información disponible en los documentos
2. Identificar qué partes son relevantes para la pregunta específica
3. Razonar sobre las conexiones y relaciones entre diferentes partes del contenido
4. Proporcionar una respuesta completa y bien fundamentada
5. Citar las fuentes específicas cuando sea apropiado

Voy a revisar todo el contexto disponible y pensar profundamente sobre la mejor respuesta.
</think>

INSTRUCCIONES PARA DEEPSEEK:
- Tienes acceso a TODO el contenido de los documentos del usuario
- Analiza profundamente toda la información disponible
- Razona sobre las conexiones entre diferentes partes del contenido
- Proporciona respuestas completas y bien fundamentadas
- SOLO usa información que esté en los documentos proporcionados
- Si la información no está en los documentos, indícalo claramente
- Cita las fuentes específicas cuando sea relevante

CONTEXTO COMPLETO DE TODOS LOS DOCUMENTOS:
{context}

PREGUNTA DEL USUARIO:
{prompt}

RESPUESTA (basada en análisis profundo de todos los documentos):"""
                    else:
                        # Si no hay contexto, informar claramente
                        full_prompt = f"""<think>
El usuario está preguntando algo pero no hay documentos cargados en el sistema. Necesito explicar esto claramente y guiar al usuario sobre cómo cargar documentos.
</think>

No hay documentos cargados en el sistema RAG.

Para obtener respuestas basadas en tus documentos:
1. Ve a la página "Procesamiento de Documentos RAG"
2. Carga tus documentos (PDF, DOCX, TXT, etc.)
3. Procésalos haciendo clic en "Procesar Documentos"
4. Regresa aquí para hacer preguntas sobre su contenido

Tu pregunta: {prompt}

Respuesta: No puedo responder esta pregunta porque no hay documentos cargados para analizar. Por favor, carga y procesa tus documentos primero."""
                else:
                    # Modo sin RAG - advertencia clara
                    full_prompt = f"""<think>
El sistema RAG está deshabilitado, pero el usuario está haciendo una pregunta. Debo informar que para obtener respuestas basadas en documentos específicos, necesita habilitar el RAG.
</think>

NOTA: El sistema RAG está deshabilitado. Para obtener respuestas basadas en documentos específicos, habilita el RAG en la configuración.

Pregunta: {prompt}

Respuesta: Para responder basándome en tus documentos específicos, necesitas habilitar el sistema RAG y cargar tus documentos."""
                
                # Verificar cancelación antes de generar
                if st.session_state.processing_cancelled:
                    st.session_state.processing_cancelled = False
                    st.stop()
                
                # Paso 4: Generación de respuesta
                status_text.text("🤖 Generando respuesta con IA...")
                progress_bar.progress(85)
                
                # Generar respuesta con configuración de tokens
                response = ollama_client.generate_response(
                    model=selected_model,
                    prompt=full_prompt,
                    context=None,  # El contexto ya está incluido en el prompt
                    max_tokens=max_tokens
                )
                
                # Verificar cancelación después de generar
                if st.session_state.processing_cancelled:
                    st.session_state.processing_cancelled = False
                    st.stop()
                
                # Paso 5: Finalización
                status_text.text("✅ Respuesta generada exitosamente")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Limpiar el contenedor de progreso y mostrar respuesta
                progress_container.empty()
                
                if response:
                    with response_container.container():
                        st.markdown(response)
                        response_timestamp = datetime.now().strftime("%H:%M:%S")
                        st.caption(f"🕒 {response_timestamp}")
                        
                        # Mostrar información de contexto si está disponible
                        if use_rag and context:
                            if debug_mode:
                                st.success(f"🧠 Contexto recuperado con trazabilidad (Query ID: {query_id})")
                                st.info(f"📊 {len(relevant_chunks)} chunks relevantes encontrados ({len(context):,} caracteres)")
                            else:
                                st.success("🧠 Contexto COMPLETO de todos los documentos utilizado")
                                st.info(f"📊 Analizados {len(context):,} caracteres de contenido")
                        
                        # Mostrar información de debug si está habilitado
                        if debug_mode and use_rag and context:
                            with st.expander("🔍 Información de Debug", expanded=False):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("📊 Estadísticas")
                                    st.write(f"**Query ID:** {query_id}")
                                    st.write(f"**Chunks recuperados:** {len(relevant_chunks)}")
                                    st.write(f"**Caracteres de contexto:** {len(context):,}")
                                    st.write(f"**Tokens máximos:** {max_tokens}")
                                    st.write(f"**Modelo usado:** {selected_model}")
                                
                                with col2:
                                    st.subheader("📚 Documentos fuente")
                                    if relevant_chunks:
                                        sources = set()
                                        for chunk_data in relevant_chunks:
                                            # relevant_chunks es una lista de tuplas (DocumentChunk, float)
                                            if isinstance(chunk_data, tuple) and len(chunk_data) >= 2:
                                                chunk, score = chunk_data
                                                if hasattr(chunk, 'source_file'):
                                                    sources.add(chunk.source_file)
                                            elif hasattr(chunk_data, 'source_file'):
                                                sources.add(chunk_data.source_file)
                                        for source in sources:
                                            st.write(f"• {source}")
                                    else:
                                        st.write("No hay chunks específicos (contexto completo)")
                                
                                if relevant_chunks:
                                    st.subheader("🔍 Chunks recuperados")
                                    for i, chunk_data in enumerate(relevant_chunks[:5]):  # Mostrar solo los primeros 5
                                        # Extraer chunk y score de la tupla
                                        if isinstance(chunk_data, tuple) and len(chunk_data) >= 2:
                                            chunk, score = chunk_data
                                            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                                            source = chunk.source_file if hasattr(chunk, 'source_file') else 'N/A'
                                            score_display = f"{score:.3f}" if isinstance(score, float) else str(score)
                                        else:
                                            content = str(chunk_data)
                                            source = 'N/A'
                                            score_display = 'N/A'
                                        
                                        with st.expander(f"Chunk {i+1} (Score: {score_display})"):
                                            st.write(f"**Fuente:** {source}")
                                            st.write(f"**Contenido:** {content[:300]}...")
                        
                        # Registrar en el historial con trazabilidad si está habilitado
                        message_data = {
                            "role": "assistant",
                            "content": response,
                            "timestamp": response_timestamp,
                            "context_used": context if context else None
                        }
                        
                        # Agregar información de debug al historial
                        if debug_mode and query_id:
                            message_data.update({
                                "query_id": query_id,
                                "chunks_count": len(relevant_chunks),
                                "model_used": selected_model,
                                "max_tokens": max_tokens
                            })
                        
                        st.session_state.messages.append(message_data)
                        
                        # Registrar en el sistema de historial si está habilitado
                        if config.ENABLE_HISTORY_TRACKING:
                            try:
                                traceability_manager = TraceabilityManager()
                                
                                # Preparar datos para el historial
                                sources = []
                                if relevant_chunks:
                                    for chunk_data in relevant_chunks:
                                        # relevant_chunks es una lista de tuplas (DocumentChunk, float)
                                        if isinstance(chunk_data, tuple) and len(chunk_data) >= 2:
                                            chunk, score = chunk_data
                                            if hasattr(chunk, 'source_file'):
                                                sources.append(chunk.source_file)
                                        elif hasattr(chunk_data, 'source_file'):
                                            sources.append(chunk_data.source_file)
                                    sources = list(set(sources))  # Eliminar duplicados
                                
                                traceability_manager.save_query_history(
                                    query=prompt,
                                    response=response,
                                    query_id=query_id,
                                    sources=sources,
                                    chunks_count=len(relevant_chunks),
                                    model_used=selected_model,
                                    max_tokens=max_tokens,
                                    rag_enabled=use_rag
                                )
                            except Exception as e:
                                logger.warning(f"Error al guardar historial: {e}")
                else:
                    with response_container.container():
                        st.error("❌ No se pudo generar una respuesta")
                        
            except Exception as e:
                # Limpiar contenedores en caso de error
                progress_container.empty()
                with response_container.container():
                    error_handler.handle_error(e, "Error al generar respuesta del chat")
                    st.error("❌ Ha ocurrido un error al generar la respuesta")
            
            finally:
                # Limpiar estado de cancelación
                if 'processing_cancelled' in st.session_state:
                    st.session_state.processing_cancelled = False
    
    # Panel de control del chat
    if st.session_state.messages:
        st.divider()
        
        # Métricas de la conversación
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.metric("💬 Mensajes enviados", user_messages)
        
        with col2:
            assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            st.metric("🤖 Respuestas recibidas", assistant_messages)
        
        with col3:
            rag_responses = len([m for m in st.session_state.messages if m["role"] == "assistant" and m.get("context_used")])
            st.metric("📚 Respuestas con RAG", rag_responses)
        
        with col4:
            if st.button("🧹 Limpiar Chat", type="secondary"):
                st.session_state.messages = []
                st.rerun()
        
        # Controles de exportación de conversación completa
        st.divider()
        render_conversation_export_controls()
    
    # Renderizar sidebar de historial
    render_history_sidebar()

def render_history_sidebar():
    """Renderizar sidebar para gestión de historial"""
    with st.sidebar:
        st.header("📚 Historial de Conversaciones")
        
        # Guardar conversación actual
        if st.session_state.messages:
            st.subheader("💾 Guardar Conversación")
            
            conversation_name = st.text_input(
                "Nombre de la conversación",
                placeholder="Ej: Análisis de datos 2024",
                key="conversation_name_input"
            )
            
            if st.button("💾 Guardar Conversación Actual", type="primary"):
                try:
                    filename = chat_history_manager.save_conversation(
                        st.session_state.messages,
                        conversation_name if conversation_name else None
                    )
                    st.success(f"✅ Conversación guardada: {filename}")
                    # Usar st.rerun() en lugar de modificar directamente el session_state
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al guardar: {str(e)}")
        
        st.divider()
        
        # Cargar conversaciones guardadas
        st.subheader("📂 Conversaciones Guardadas")
        
        conversations = chat_history_manager.list_conversations()
        
        if conversations:
            for conv in conversations:
                with st.expander(f"📄 {conv['name']}", expanded=False):
                    st.write(f"**Fecha:** {conv.get('created_at', 'N/A')[:19]}")
                    st.write(f"**Mensajes:** {conv.get('total_messages', 0)}")
                    st.write(f"**Tamaño:** {conv.get('file_size', 0)} bytes")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"📥 Cargar", key=f"load_{conv['filename']}"):
                            try:
                                conversation_data = chat_history_manager.load_conversation(conv['filename'])
                                if conversation_data:
                                    st.session_state.messages = conversation_data['messages']
                                    st.success(f"✅ Conversación cargada: {conv['name']}")
                                    st.rerun()
                                else:
                                    st.error("❌ Error al cargar la conversación")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                    
                    with col2:
                        if st.button(f"🗑️ Eliminar", key=f"delete_{conv['filename']}"):
                            if chat_history_manager.delete_conversation(conv['filename']):
                                st.success(f"✅ Conversación eliminada")
                                st.rerun()
                            else:
                                st.error("❌ Error al eliminar")
        else:
            st.info("No hay conversaciones guardadas")

def render_message_export_buttons(message: dict, message_index: int):
    """Renderizar botones de exportación para un mensaje individual"""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📄 DOCX", key=f"docx_{message_index}", help="Exportar a Word"):
            try:
                buffer = chat_exporter.export_message_to_docx(message)
                if buffer:
                    timestamp = message.get('timestamp', 'mensaje').replace(':', '-')
                    filename = f"cognichat_mensaje_{timestamp}.docx"
                    
                    st.download_button(
                        label="⬇️ Descargar DOCX",
                        data=buffer.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"download_docx_{message_index}"
                    )
                else:
                    st.error("❌ Error al generar DOCX")
            except ImportError:
                st.error("❌ python-docx no está instalado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("📑 PDF", key=f"pdf_{message_index}", help="Exportar a PDF"):
            try:
                buffer = chat_exporter.export_message_to_pdf(message)
                if buffer:
                    timestamp = message.get('timestamp', 'mensaje').replace(':', '-')
                    filename = f"cognichat_mensaje_{timestamp}.pdf"
                    
                    st.download_button(
                        label="⬇️ Descargar PDF",
                        data=buffer.getvalue(),
                        file_name=filename,
                        mime="application/pdf",
                        key=f"download_pdf_{message_index}"
                    )
                else:
                    st.error("❌ Error al generar PDF")
            except ImportError:
                st.error("❌ reportlab no está instalado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col3:
        if st.button("📋 Copiar", key=f"copy_{message_index}", help="Copiar al portapapeles"):
            try:
                if CLIPBOARD_AVAILABLE:
                    text = chat_exporter.get_message_text_for_clipboard(message)
                    pyperclip.copy(text)
                    st.success("✅ Copiado al portapapeles")
                else:
                    # Fallback: mostrar texto para copiar manualmente
                    text = chat_exporter.get_message_text_for_clipboard(message)
                    st.text_area(
                        "Copiar manualmente:",
                        value=text,
                        height=100,
                        key=f"manual_copy_{message_index}"
                    )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def render_conversation_export_controls():
    """Renderizar controles de exportación de conversación completa"""
    st.subheader("📤 Exportar Conversación Completa")
    
    if not st.session_state.messages:
        st.info("No hay mensajes para exportar")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Exportar a DOCX", type="secondary"):
            try:
                buffer = chat_exporter.export_conversation_to_docx(st.session_state.messages)
                if buffer:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"cognichat_conversacion_{timestamp}.docx"
                    
                    st.download_button(
                        label="⬇️ Descargar Conversación DOCX",
                        data=buffer.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="download_conversation_docx"
                    )
                else:
                    st.error("❌ Error al generar DOCX")
            except ImportError:
                st.error("❌ python-docx no está instalado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("📋 Copiar Todo", type="secondary"):
            try:
                if CLIPBOARD_AVAILABLE:
                    # Crear texto completo de la conversación
                    conversation_text = []
                    conversation_text.append("=" * 60)
                    conversation_text.append("CONVERSACIÓN COMPLETA DE COGNICHAT")
                    conversation_text.append(f"Exportado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    conversation_text.append("=" * 60)
                    conversation_text.append("")
                    
                    for i, message in enumerate(st.session_state.messages, 1):
                        role_text = "Usuario" if message['role'] == 'user' else "Asistente"
                        timestamp = message.get('timestamp', 'N/A')
                        
                        conversation_text.append(f"--- MENSAJE {i} - {role_text} ({timestamp}) ---")
                        conversation_text.append(message['content'])
                        conversation_text.append("")
                    
                    conversation_text.append("=" * 60)
                    
                    full_text = "\n".join(conversation_text)
                    pyperclip.copy(full_text)
                    st.success("✅ Conversación copiada al portapapeles")
                else:
                    st.error("❌ pyperclip no está disponible")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col3:
        # Estadísticas de la conversación
        total_chars = sum(len(m['content']) for m in st.session_state.messages)
        estimated_tokens = total_chars // 4  # Estimación aproximada
        
        st.metric("📊 Caracteres totales", f"{total_chars:,}")
        st.metric("🎯 Tokens estimados", f"{estimated_tokens:,}")

def get_rag_context(query: str, enable_tracing: bool = False):
    """
    Obtiene el contexto RAG para una consulta específica.
    
    Args:
        query: La consulta del usuario
        enable_tracing: Si habilitar el trazado de chunks
        
    Returns:
        tuple: (contexto, chunks_relevantes, query_id) si enable_tracing es True
               str: contexto si enable_tracing es False
    """
    try:
        
        if enable_tracing:
            # Usar la función optimizada con trazabilidad
            context, relevant_chunks, query_id = rag_processor.get_context_for_query(
                query, enable_tracing=True
            )
            return context, relevant_chunks, query_id
        else:
            # Usar el contexto completo como antes
            context = rag_processor.get_full_context_for_comprehensive_analysis()
            
            if context:
                logger.info(f"Contexto COMPLETO generado para DeepSeek: {query[:50]}... ({len(context)} caracteres)")
                return context
            else:
                logger.info("No se encontraron documentos procesados")
                return None
        
    except Exception as e:
        logger.error(f"Error al obtener contexto completo: {e}")
        return None if not enable_tracing else (None, [], None)