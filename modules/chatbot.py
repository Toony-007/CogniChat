"""
Pestaña del chatbot
"""

import streamlit as st
import time
from datetime import datetime
from utils.error_handler import ErrorHandler
from utils.logger import setup_logger
from utils.ollama_client import OllamaClient
from utils.traceability import TraceabilityManager
from utils.rag_processor import rag_processor
from config.settings import config

logger = setup_logger()
error_handler = ErrorHandler()

def render():
    """Renderizar la pestaña del chatbot"""
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
                if st.button("🔄 Refrescar modelos"):
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
    
    # Inicializar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
        
        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("🤔 Pensando..."):
                try:
                    # Obtener configuración de debug
                    debug_mode = st.session_state.get('debug_mode', False)
                    
                    # Obtener contexto RAG si está habilitado
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
                        
                        if context:
                            if debug_mode:
                                st.success(f"🧠 Contexto recuperado con trazabilidad (Query ID: {query_id})")
                                st.info(f"📊 {len(relevant_chunks)} chunks relevantes encontrados ({len(context):,} caracteres)")
                            else:
                                st.success("🧠 Contexto COMPLETO de todos los documentos cargado para DeepSeek")
                                st.info(f"📊 Analizando {len(context):,} caracteres de contenido")
                        else:
                            st.warning("📄 No hay documentos procesados disponibles")
                    
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
                    
                    # Generar respuesta con configuración de tokens
                    response = ollama_client.generate_response(
                        model=selected_model,
                        prompt=full_prompt,
                        context=None,  # El contexto ya está incluido en el prompt
                        max_tokens=max_tokens
                    )
                    
                    if response:
                        st.markdown(response)
                        response_timestamp = datetime.now().strftime("%H:%M:%S")
                        st.caption(f"🕒 {response_timestamp}")
                        
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
                        st.error("❌ No se pudo generar una respuesta")
                        
                except Exception as e:
                    error_handler.handle_error(e, "Error al generar respuesta del chat")
                    st.error("❌ Ha ocurrido un error al generar la respuesta")
    
    # Panel de control del chat
    if st.session_state.messages:
        st.divider()
        
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