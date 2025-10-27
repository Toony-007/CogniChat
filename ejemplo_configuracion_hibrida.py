#!/usr/bin/env python3
"""
Ejemplo de configuración para el enfoque híbrido
Muestra cómo configurar los nuevos parámetros de la interfaz
"""

from modules.qualitative_analysis.core.config import AnalysisConfig

def ejemplo_configuracion_basica():
    """Configuración básica para usuarios principiantes"""
    print("🔧 CONFIGURACIÓN BÁSICA (Recomendada para principiantes)")
    print("=" * 60)
    
    config = AnalysisConfig(
        # Parámetros TF-IDF (mantienen funcionalidad actual)
        max_concepts=30,                    # Conceptos candidatos iniciales
        min_concept_frequency=2,            # Frecuencia mínima
        use_ngrams=True,                    # Detectar frases completas
        
        # Nuevos parámetros LLM
        enable_llm_refinement=True,         # Activar refinamiento con IA
        llm_model="deepseek-r1:7b",        # Modelo LLM
        llm_temperature=0.3,               # Temperatura (precisión)
        max_final_concepts=15,             # Conceptos finales
        include_concept_explanations=True, # Incluir explicaciones
        llm_max_tokens=2000               # Máximo tokens respuesta
    )
    
    print("✅ Configuración aplicada:")
    print(f"   • Conceptos candidatos: {config.max_concepts}")
    print(f"   • Frecuencia mínima: {config.min_concept_frequency}")
    print(f"   • Detectar frases: {config.use_ngrams}")
    print(f"   • Refinamiento IA: {config.enable_llm_refinement}")
    print(f"   • Modelo LLM: {config.llm_model}")
    print(f"   • Temperatura: {config.llm_temperature}")
    print(f"   • Conceptos finales: {config.max_final_concepts}")
    print(f"   • Explicaciones: {config.include_concept_explanations}")
    
    return config

def ejemplo_configuracion_avanzada():
    """Configuración avanzada para usuarios expertos"""
    print("\n🔧 CONFIGURACIÓN AVANZADA (Para usuarios expertos)")
    print("=" * 60)
    
    config = AnalysisConfig(
        # Parámetros TF-IDF optimizados
        max_concepts=50,                    # Más candidatos para mejor refinamiento
        min_concept_frequency=3,            # Mayor frecuencia mínima
        use_ngrams=True,
        ngram_range=(2, 4),                 # Frases de 2-4 palabras
        
        # Parámetros LLM avanzados
        enable_llm_refinement=True,
        llm_model="deepseek-r1:7b",
        llm_temperature=0.5,                 # Más creatividad
        max_final_concepts=20,              # Más conceptos finales
        include_concept_explanations=True,
        llm_max_tokens=3000,                # Respuestas más largas
        
        # Parámetros de procesamiento
        chunk_size=3000,                    # Chunks más grandes
        chunk_overlap=300,                  # Mayor solapamiento
        enable_citations=True,
        citation_context_chars=200          # Más contexto en citas
    )
    
    print("✅ Configuración avanzada aplicada:")
    print(f"   • Conceptos candidatos: {config.max_concepts}")
    print(f"   • Frecuencia mínima: {config.min_concept_frequency}")
    print(f"   • N-gramas: {config.ngram_range}")
    print(f"   • Modelo LLM: {config.llm_model}")
    print(f"   • Temperatura: {config.llm_temperature}")
    print(f"   • Conceptos finales: {config.max_final_concepts}")
    print(f"   • Tokens máximos: {config.llm_max_tokens}")
    
    return config

def ejemplo_configuracion_solo_tfidf():
    """Configuración solo TF-IDF (sin LLM)"""
    print("\n🔧 CONFIGURACIÓN SOLO TF-IDF (Sin refinamiento LLM)")
    print("=" * 60)
    
    config = AnalysisConfig(
        # Solo TF-IDF
        max_concepts=25,
        min_concept_frequency=2,
        use_ngrams=True,
        
        # LLM desactivado
        enable_llm_refinement=False,
        
        # Otros parámetros
        enable_citations=True,
        show_explanations=True
    )
    
    print("✅ Configuración TF-IDF puro:")
    print(f"   • Conceptos: {config.max_concepts}")
    print(f"   • Frecuencia mínima: {config.min_concept_frequency}")
    print(f"   • Detectar frases: {config.use_ngrams}")
    print(f"   • Refinamiento IA: {config.enable_llm_refinement}")
    
    return config

def mostrar_interfaz_propuesta():
    """Mostrar cómo se vería la interfaz de usuario"""
    print("\n🖥️  INTERFAZ DE USUARIO PROPUESTA")
    print("=" * 60)
    
    print("""
┌─ Configuración del Análisis ─────────────────────────────────┐
│                                                              │
│ 📊 Extracción Inicial (TF-IDF)                              │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ Máximo de conceptos candidatos: [30] ───────────────── │ │
│ │ Frecuencia mínima: [2] ─────────────────────────────── │ │
│ │ ☑ Detectar frases completas                            │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ 🤖 Refinamiento con IA                                      │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ ☑ Activar refinamiento con DeepSeek R1                 │ │
│ │ Modelo: [deepseek-r1:7b] ▼                              │ │
│ │ Temperatura: [0.3] ─────────────────────────────────── │ │
│ │ Máximo conceptos finales: [15] ─────────────────────── │ │
│ │ ☑ Incluir explicaciones de conceptos                   │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ [🔍 Extraer Conceptos]                                      │
│ ✅ Análisis completado: 15 conceptos refinados             │
└──────────────────────────────────────────────────────────────┘
    """)

def main():
    """Función principal"""
    print("🚀 EJEMPLOS DE CONFIGURACIÓN PARA ENFOQUE HÍBRIDO")
    print("TF-IDF + DeepSeek R1")
    print("=" * 60)
    
    # Mostrar ejemplos de configuración
    config_basica = ejemplo_configuracion_basica()
    config_avanzada = ejemplo_configuracion_avanzada()
    config_tfidf = ejemplo_configuracion_solo_tfidf()
    
    # Mostrar interfaz propuesta
    mostrar_interfaz_propuesta()
    
    print("\n✨ VENTAJAS DEL ENFOQUE HÍBRIDO:")
    print("• Elimina conceptos irrelevantes (letras sueltas, palabras genéricas)")
    print("• Consolida conceptos relacionados")
    print("• Genera nombres más precisos y académicos")
    print("• Añade categorización automática")
    print("• Proporciona explicaciones contextuales")
    print("• Mantiene compatibilidad con el sistema actual")
    
    print("\n🎯 CASOS DE USO:")
    print("• Configuración básica: Usuarios nuevos, análisis estándar")
    print("• Configuración avanzada: Investigadores expertos, análisis complejos")
    print("• Solo TF-IDF: Cuando no hay acceso a LLM o se prefiere velocidad")

if __name__ == "__main__":
    main()

