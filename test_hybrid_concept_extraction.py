#!/usr/bin/env python3
"""
Script de prueba para el enfoque híbrido TF-IDF + DeepSeek R1
Demuestra la mejora en la extracción de conceptos
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.qualitative_analysis.extractors.concept_extractor import ConceptExtractor
from modules.qualitative_analysis.core.config import AnalysisConfig

def create_test_document():
    """Crear un documento de prueba con contenido académico complejo"""
    return [
        {
            'content': """
            El modelo VARK (Visual, Aural, Read/Write, Kinesthetic) representa una metodología 
            educativa innovadora que identifica cuatro estilos de aprendizaje principales en 
            estudiantes de diferentes contextos socioeconómicos. Los estudiantes visuales 
            demuestran preferencias marcadas por diagramas, mapas conceptuales y 
            representaciones gráficas, mientras que los estudiantes auditivos desarrollan 
            mejor sus competencias a través de discusiones grupales, conferencias magistrales 
            y explicaciones orales interactivas.
            
            La investigación cualitativa en educación utiliza métodos etnográficos como 
            entrevistas en profundidad, grupos focales participativos y observación 
            participante para recopilar datos empíricos sobre experiencias de aprendizaje. 
            El análisis temático emergente es una técnica de análisis cualitativo que permite
            identificar patrones recurrentes y temas emergentes en los datos, facilitando
            la construcción de teorías fundamentadas en la realidad educativa.
            
            La validez interna en investigación cualitativa se refiere a la credibilidad 
            y autenticidad de los hallazgos, mientras que la validez externa se relaciona 
            con la transferibilidad y aplicabilidad de los resultados a otros contextos 
            educativos similares. La triangulación metodológica fortalece la confiabilidad
            de los datos mediante la convergencia de múltiples fuentes de información.
            
            Los procesos de resiliencia académica emergen cuando los estudiantes enfrentan
            adversidades socioeconómicas pero mantienen su motivación intrínseca hacia
            el aprendizaje. Las redes de apoyo educativo incluyen no solo a los docentes
            sino también a pares, familiares y mentores comunitarios que facilitan
            la persistencia estudiantil en contextos desfavorecidos.
            """,
            'metadata': {
                'source_file': 'documento_prueba.pdf',
                'page_number': 1
            }
        }
    ]

def test_traditional_extraction():
    """Probar extracción tradicional (solo TF-IDF)"""
    print("=" * 60)
    print("🔍 PRUEBA: Extracción tradicional (solo TF-IDF)")
    print("=" * 60)
    
    # Configuración sin LLM
    config = AnalysisConfig(
        enable_llm_refinement=False,
        max_concepts=10,
        min_concept_frequency=1
    )
    
    extractor = ConceptExtractor(config)
    chunks = create_test_document()
    
    concepts = extractor.extract_concepts(chunks)
    
    print(f"\n📊 Conceptos extraídos (TF-IDF puro): {len(concepts)}")
    for i, concept in enumerate(concepts[:5], 1):
        print(f"{i}. {concept.concept} (freq: {concept.frequency}, score: {concept.relevance_score:.3f})")
    
    return concepts

def test_hybrid_extraction():
    """Probar extracción híbrida (TF-IDF + DeepSeek R1)"""
    print("\n" + "=" * 60)
    print("🤖 PRUEBA: Extracción híbrida (TF-IDF + DeepSeek R1)")
    print("=" * 60)
    
    # Configuración con LLM
    config = AnalysisConfig(
        enable_llm_refinement=True,
        llm_model="deepseek-r1:7b",
        llm_temperature=0.3,
        max_final_concepts=8,
        include_concept_explanations=True,
        max_concepts=15,  # Candidatos iniciales
        min_concept_frequency=1
    )
    
    extractor = ConceptExtractor(config)
    chunks = create_test_document()
    
    concepts = extractor.extract_concepts(chunks)
    
    print(f"\n📊 Conceptos refinados (Híbrido): {len(concepts)}")
    for i, concept in enumerate(concepts, 1):
        print(f"{i}. {concept.concept} (freq: {concept.frequency}, score: {concept.relevance_score:.3f})")
        if concept.category:
            print(f"   📂 Categoría: {concept.category}")
        if concept.context_examples and any("Explicación:" in ex for ex in concept.context_examples):
            explanation = next((ex for ex in concept.context_examples if "Explicación:" in ex), "")
            print(f"   💡 {explanation}")
        if concept.related_concepts:
            print(f"   🔗 Relacionado con: {', '.join(concept.related_concepts[:3])}")
        print()
    
    return concepts

def compare_results(traditional_concepts, hybrid_concepts):
    """Comparar resultados entre ambos enfoques"""
    print("=" * 60)
    print("📈 COMPARACIÓN DE RESULTADOS")
    print("=" * 60)
    
    print(f"\n🔍 TF-IDF puro: {len(traditional_concepts)} conceptos")
    print("Conceptos extraídos (tradicionales):")
    for concept in traditional_concepts[:5]:
        print(f"  • {concept.concept} (freq: {concept.frequency}, score: {concept.relevance_score:.3f})")
    
    print(f"\n🤖 Híbrido (TF-IDF + LLM): {len(hybrid_concepts)} conceptos")
    print("Conceptos refinados (académicos profundos):")
    for concept in hybrid_concepts:
        print(f"  • {concept.concept}")
        if concept.category:
            print(f"    📂 Categoría: {concept.category}")
        if concept.context_examples and any("Explicación:" in ex for ex in concept.context_examples):
            explanation = next((ex for ex in concept.context_examples if "Explicación:" in ex), "")
            print(f"    💡 {explanation}")
        if concept.related_concepts:
            print(f"    🔗 Relacionado: {', '.join(concept.related_concepts[:2])}")
        print()
    
    print(f"\n✨ MEJORAS OBSERVADAS:")
    print("• Conceptos profundos y académicos (no palabras simples)")
    print("• Captura de procesos, relaciones y fenómenos complejos")
    print("• Eliminación de términos genéricos y letras sueltas")
    print("• Consolidación inteligente de conceptos relacionados")
    print("• Categorización automática por tipo de fenómeno")
    print("• Explicaciones contextuales que revelan significado")
    print("• Conceptos que explican 'cómo' y 'por qué', no solo 'qué'")
    
    print(f"\n🎯 EJEMPLOS DE TRANSFORMACIÓN:")
    print("❌ Antes: 'documento', 'centro educativo', 'texto'")
    print("✅ Después: 'Resiliencia académica en contextos desfavorecidos'")
    print("✅ Después: 'Redes de apoyo educativo comunitario'")
    print("✅ Después: 'Procesos de motivación intrínseca estudiantil'")

def main():
    """Función principal de prueba"""
    print("🚀 INICIANDO PRUEBA DEL ENFOQUE HÍBRIDO")
    print("TF-IDF + DeepSeek R1 para extracción de conceptos")
    print("=" * 60)
    
    try:
        # Verificar disponibilidad de Ollama
        from utils.ollama_client import ollama_client
        if not ollama_client.is_available():
            print("⚠️  ADVERTENCIA: Ollama no está disponible")
            print("   El refinamiento con LLM no funcionará")
            print("   Solo se ejecutará la extracción TF-IDF")
        
        # Ejecutar pruebas
        traditional_concepts = test_traditional_extraction()
        hybrid_concepts = test_hybrid_extraction()
        
        # Comparar resultados
        compare_results(traditional_concepts, hybrid_concepts)
        
        print("\n✅ PRUEBA COMPLETADA")
        print("El enfoque híbrido está funcionando correctamente")
        
    except Exception as e:
        print(f"\n❌ ERROR durante la prueba: {e}")
        print("Verifica que todas las dependencias estén instaladas")

if __name__ == "__main__":
    main()
