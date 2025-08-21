"""
Módulo de Análisis Cualitativo Avanzado
Análisis profundo de contenido RAG con técnicas de NLP y visualizaciones interactivas
Incluye mapas conceptuales interactivos, mapas mentales, resúmenes automáticos y triangulación
"""

import streamlit as st
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import sys
import tempfile
import base64

# Análisis de texto avanzado
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import LatentDirichletAllocation, PCA
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE
    import nltk
    from textblob import TextBlob
    from wordcloud import WordCloud
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False
    st.warning("⚠️ Algunas funcionalidades avanzadas no están disponibles. Instala las dependencias adicionales.")

# Mapas conceptuales interactivos
try:
    from pyvis.network import Network
    import streamlit.components.v1 as components
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from utils.rag_processor import RAGProcessor
from utils.logger import setup_logger
from config.settings import config

logger = setup_logger()

class AdvancedQualitativeAnalyzer:
    """Analizador cualitativo avanzado con técnicas de NLP y mapas conceptuales interactivos"""
    
    def __init__(self):
        self.rag_processor = RAGProcessor()
        self.cache_path = Path(config.CACHE_DIR) / "rag_cache.json"
        self.analysis_cache_path = Path(config.CACHE_DIR) / "qualitative_analysis_cache.json"
        self._initialize_nltk()
        
    def _initialize_nltk(self):
        """Inicializar recursos de NLTK"""
        if ADVANCED_ANALYSIS_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('wordnet', quiet=True)
            except Exception as e:
                logger.warning(f"Error inicializando NLTK: {e}")
        
    def load_rag_data(self) -> List[Dict]:
        """Cargar datos del cache RAG"""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = []
                    
                    if 'chunks' in data and isinstance(data['chunks'], dict):
                        for filename, chunk_list in data['chunks'].items():
                            if isinstance(chunk_list, list):
                                chunks.extend(chunk_list)
                    
                    return chunks
            return []
        except Exception as e:
            logger.error(f"Error cargando datos RAG: {e}")
            return []
    
    def generate_rag_summary(self, chunks: List[Dict], max_length: int = 500) -> str:
        """Generar resumen automático extenso usando contexto RAG completo"""
        try:
            if not chunks:
                return "No hay contenido disponible para resumir."
            
            # Usar TODOS los chunks disponibles para un resumen más completo
            all_content = []
            sources = set()
            
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                source = chunk.get('source', 'Documento')
                
                if content and len(content) > 30:  # Contenido mínimo más bajo
                    all_content.append(content)
                    sources.add(source)
            
            if not all_content:
                return "No hay contenido suficiente para generar un resumen."
            
            combined_text = " ".join(all_content)
            
            # Análisis más profundo con TextBlob
            blob = TextBlob(combined_text)
            sentences = list(blob.sentences)
            
            if len(sentences) <= 5:
                # Si hay pocas oraciones, devolver todo el contenido
                result = combined_text
                if len(result) > max_length:
                    result = result[:max_length] + "..."
                return f"**Resumen Completo** (basado en {len(sources)} fuente(s)):\n\n{result}"
            
            # Análisis de frecuencia de palabras para identificar temas principales
            words = combined_text.lower().split()
            stop_words = self._get_spanish_stopwords()
            
            # Filtrar palabras significativas
            significant_words = [word for word in words 
                               if len(word) > 3 and 
                               word not in stop_words and 
                               word.isalpha()]
            
            word_freq = Counter(significant_words)
            top_words = dict(word_freq.most_common(20))
            
            # Puntuación de oraciones basada en palabras clave y posición
            sentence_scores = {}
            
            for i, sentence in enumerate(sentences):
                sentence_text = str(sentence).lower()
                sentence_words = sentence_text.split()
                
                # Puntuación base por palabras clave
                keyword_score = sum(top_words.get(word, 0) for word in sentence_words 
                                  if word in top_words)
                
                # Bonus por posición (primeras y últimas oraciones son importantes)
                position_bonus = 0
                if i < 3:  # Primeras 3 oraciones
                    position_bonus = 10
                elif i >= len(sentences) - 3:  # Últimas 3 oraciones
                    position_bonus = 5
                
                # Bonus por longitud (oraciones ni muy cortas ni muy largas)
                length_bonus = 0
                sentence_length = len(sentence_words)
                if 10 <= sentence_length <= 30:
                    length_bonus = 5
                
                total_score = keyword_score + position_bonus + length_bonus
                sentence_scores[str(sentence)] = total_score
            
            # Seleccionar las mejores oraciones (más que antes)
            num_sentences = max(5, min(15, len(sentences) // 3))  # Entre 5 y 15 oraciones
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Mantener orden cronológico de las oraciones seleccionadas
            selected_sentences = [sent[0] for sent in top_sentences[:num_sentences]]
            
            # Reordenar según aparición original
            ordered_sentences = []
            for sentence in sentences:
                if str(sentence) in selected_sentences:
                    ordered_sentences.append(str(sentence))
            
            # Construir resumen estructurado
            summary_parts = []
            
            # Introducción
            summary_parts.append(f"**Resumen Analítico** (basado en {len(chunks)} fragmentos de {len(sources)} fuente(s)):")
            summary_parts.append("")
            
            # Contenido principal
            main_content = " ".join(ordered_sentences)
            
            # Dividir en párrafos lógicos si es muy largo
            if len(main_content) > max_length * 0.8:
                # Dividir en párrafos por puntos o temas
                paragraphs = []
                current_paragraph = []
                
                for sentence in ordered_sentences:
                    current_paragraph.append(sentence)
                    
                    # Crear nuevo párrafo cada 3-4 oraciones
                    if len(current_paragraph) >= 3:
                        paragraphs.append(" ".join(current_paragraph))
                        current_paragraph = []
                
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                
                # Agregar párrafos con separación
                for i, paragraph in enumerate(paragraphs[:5]):  # Máximo 5 párrafos
                    summary_parts.append(f"**{i+1}.** {paragraph}")
                    summary_parts.append("")
            else:
                summary_parts.append(main_content)
                summary_parts.append("")
            
            # Información adicional
            summary_parts.append(f"**Palabras clave principales:** {', '.join(list(top_words.keys())[:10])}")
            
            final_summary = "\n".join(summary_parts)
            
            # Ajustar longitud final
            if len(final_summary) > max_length:
                # Truncar manteniendo estructura
                truncated = final_summary[:max_length]
                last_period = truncated.rfind('.')
                if last_period > max_length * 0.8:
                    final_summary = truncated[:last_period + 1] + "\n\n[Resumen truncado por límite de longitud]"
                else:
                    final_summary = truncated + "..."
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Error generando resumen RAG: {e}")
            return f"Error generando resumen: {str(e)}"
    
    def extract_key_concepts(self, chunks: List[Dict], min_freq: int = 2) -> List[Dict]:
        """Extraer conceptos clave del contenido RAG"""
        try:
            if not chunks:
                return []
            
            # Procesar cada chunk por separado para TF-IDF
            chunk_texts = []
            for chunk in chunks:
                content = chunk.get('content', '').strip()
                if content:
                    processed = self.preprocess_text(content)
                    if processed and len(processed.split()) > 5:  # Mínimo 5 palabras
                        chunk_texts.append(processed)
            
            if not chunk_texts:
                return []
            
            # Extraer conceptos usando TF-IDF si está disponible y hay suficientes documentos
            if ADVANCED_ANALYSIS_AVAILABLE and len(chunk_texts) >= 2:
                try:
                    # Configurar TF-IDF para múltiples documentos
                    num_docs = len(chunk_texts)
                    adjusted_min_df = max(1, min(min_freq, num_docs // 4))
                    adjusted_max_df = min(0.95, max(0.5, (num_docs - 1) / num_docs))
                    
                    vectorizer = TfidfVectorizer(
                        max_features=100,
                        stop_words=self._get_spanish_stopwords(),
                        ngram_range=(1, 2),
                        min_df=adjusted_min_df,
                        max_df=adjusted_max_df
                    )
                    
                    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Calcular puntuaciones promedio para todos los documentos
                    mean_scores = tfidf_matrix.mean(axis=0).A1
                    
                    concepts = []
                    for i, score in enumerate(mean_scores):
                        if score > 0:
                            concepts.append({
                                'concept': feature_names[i],
                                'score': float(score),
                                'type': 'tfidf'
                            })
                    
                    if concepts:
                        return sorted(concepts, key=lambda x: x['score'], reverse=True)[:30]
                    
                except Exception as e:
                    logger.warning(f"Error en TF-IDF: {e}")
            
            # Fallback: análisis básico de frecuencia con todo el texto
            all_text = " ".join(chunk_texts)
            words = all_text.split()
            word_freq = Counter(words)
            stop_words = self._get_spanish_stopwords()
            
            concepts = []
            total_words = len(words)
            
            for word, freq in word_freq.most_common(50):
                if (len(word) > 3 and 
                    word not in stop_words and 
                    freq >= min_freq and
                    word.isalpha()):  # Solo palabras alfabéticas
                    concepts.append({
                        'concept': word,
                        'score': freq / total_words,
                        'type': 'frequency'
                    })
            
            return concepts[:30]
            
        except Exception as e:
            logger.error(f"Error extrayendo conceptos: {e}")
            return []
    
    def create_interactive_concept_map(self, chunks: List[Dict], layout_type: str = "spring") -> Optional[str]:
        """Crear mapa conceptual interactivo estructurado usando PyVis"""
        if not PYVIS_AVAILABLE:
            return None
        
        try:
            # Análisis inteligente del contenido para crear estructura jerárquica
            concept_structure = self._analyze_concept_hierarchy(chunks)
            if not concept_structure:
                return None
            
            # Crear red con configuración mejorada
            net = Network(
                height="700px",
                width="100%",
                bgcolor="#f8f9fa",
                font_color="black",
                directed=True  # Dirigido para mostrar relaciones jerárquicas
            )
            
            # Configurar física para mejor organización jerárquica
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 200},
                "hierarchicalRepulsion": {
                  "centralGravity": 0.2,
                  "springLength": 120,
                  "springConstant": 0.01,
                  "nodeDistance": 150,
                  "damping": 0.09
                }
              },
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "levelSeparation": 150,
                  "nodeSpacing": 100,
                  "treeSpacing": 200,
                  "blockShifting": true,
                  "edgeMinimization": true,
                  "parentCentralization": true,
                  "direction": "UD",
                  "sortMethod": "directed"
                }
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": false,
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
              },
              "edges": {
                "arrows": {
                  "to": {"enabled": true, "scaleFactor": 1, "type": "arrow"}
                },
                "smooth": {
                  "enabled": true,
                  "type": "continuous"
                }
              }
            }
            """)
            
            # Agregar nodo principal (tema central)
            main_theme = concept_structure['main_theme']
            net.add_node(
                "main",
                label=main_theme['name'],
                title=f"Tema Principal: {main_theme['name']}\n{main_theme['description']}",
                size=60,
                color="#e74c3c",
                font={'size': 18, 'color': 'white'},
                level=0,
                shape="box"
            )
            
            # Agregar conceptos principales (nivel 1)
            for i, concept in enumerate(concept_structure['main_concepts']):
                node_id = f"concept_{i}"
                net.add_node(
                    node_id,
                    label=concept['name'],
                    title=f"Concepto: {concept['name']}\nRelevancia: {concept['relevance']:.2f}\nContexto: {concept['context'][:200]}...",
                    size=max(30, min(50, concept['relevance'] * 100)),
                    color="#3498db",
                    font={'size': 14, 'color': 'white'},
                    level=1,
                    shape="ellipse"
                )
                
                # Conectar con el tema principal
                net.add_edge(
                    "main", 
                    node_id,
                    width=3,
                    color={'color': '#2c3e50'},
                    title="representa"
                )
                
                # Agregar sub-conceptos (nivel 2)
                for j, sub_concept in enumerate(concept['sub_concepts'][:4]):  # Máximo 4 sub-conceptos
                    sub_id = f"sub_{i}_{j}"
                    net.add_node(
                        sub_id,
                        label=sub_concept['name'],
                        title=f"Sub-concepto: {sub_concept['name']}\nRelación: {sub_concept['relation']}\nContexto: {sub_concept['context'][:150]}...",
                        size=max(20, sub_concept['relevance'] * 80),
                        color="#f39c12",
                        font={'size': 12, 'color': 'black'},
                        level=2,
                        shape="dot"
                    )
                    
                    # Conectar sub-concepto con concepto principal
                    net.add_edge(
                        node_id,
                        sub_id,
                        width=2,
                        color={'color': '#7f8c8d'},
                        title=sub_concept['relation']
                    )
            
            # Agregar relaciones cruzadas entre conceptos del mismo nivel
            for relation in concept_structure['cross_relations']:
                net.add_edge(
                    relation['from'],
                    relation['to'],
                    width=1,
                    color={'color': '#95a5a6', 'opacity': 0.7},
                    title=relation['type'],
                    dashes=True
                )
            
            # Generar HTML
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8')
            net.save_graph(temp_file.name)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error creando mapa conceptual: {e}")
            return None
    
    def create_interactive_mind_map(self, chunks: List[Dict], node_spacing: int = 250, return_data: bool = False) -> Optional[Dict]:
        """Crear mapa mental interactivo estructurado usando streamlit-agraph con análisis inteligente"""
        if not AGRAPH_AVAILABLE:
            return None
        
        try:
            # Análisis inteligente mejorado para estructura de mapa mental
            mind_structure = self._analyze_intelligent_mind_map_structure(chunks)
            if not mind_structure:
                return None
            
            nodes = []
            edges = []
            
            # Nodo central (tema principal)
            central_theme = mind_structure['central_theme']
            nodes.append({
                'id': "central",
                'label': central_theme['name'],
                'title': f"Tema Central: {central_theme['name']}\n\n{central_theme['description']}",
                'size': 50,
                'color': "#2c3e50",
                'font_size': 18
            })
            
            # Ramas principales (conceptos de primer nivel)
            branch_colors = ["#e74c3c", "#3498db", "#f39c12", "#27ae60", "#9b59b6", "#e67e22"]
            
            for i, branch in enumerate(mind_structure['main_branches']):
                branch_id = f"branch_{i}"
                branch_color = branch_colors[i % len(branch_colors)]
                
                # Nodo de rama principal
                nodes.append({
                    'id': branch_id,
                    'label': branch['name'],
                    'title': f"Rama: {branch['name']}\nImportancia: {branch['importance']:.2f}\n\n{branch['description']}",
                    'size': max(30, int(branch['importance'] * 40)),
                    'color': branch_color,
                    'font_size': 16
                })
                
                # Conectar con el centro
                edges.append({
                    'from': "central",
                    'to': branch_id,
                    'width': max(3, int(branch['importance'] * 5)),
                    'color': branch_color
                })
                
                # Sub-ramas (conceptos de segundo nivel)
                for j, sub_branch in enumerate(branch['sub_branches'][:4]):  # Máximo 4 sub-ramas
                    sub_id = f"sub_{i}_{j}"
                    
                    nodes.append({
                        'id': sub_id,
                        'label': sub_branch['name'],
                        'title': f"Sub-rama: {sub_branch['name']}\nRelación: {sub_branch['relation']}\n\n{sub_branch['context'][:150]}...",
                        'size': max(15, int(sub_branch['relevance'] * 25)),
                        'color': self._lighten_color_hex(branch_color),
                        'font_size': 12
                    })
                    
                    # Conectar con la rama principal
                    edges.append({
                        'from': branch_id,
                        'to': sub_id,
                        'width': max(2, int(sub_branch['relevance'] * 3)),
                        'color': branch_color
                    })
            
            # Estadísticas
            stats = {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'main_branches': len(mind_structure['main_branches']),
                'detailed_concepts': sum(len(branch['sub_branches']) for branch in mind_structure['main_branches'])
            }
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': stats,
                'structure_info': mind_structure.get('info', {})
            }
            
        except Exception as e:
            logger.error(f"Error creando mapa mental: {e}")
            return None
    
    def perform_triangulation_analysis(self, chunks: List[Dict]) -> Dict:
        """Realizar triangulación de información para validar conceptos"""
        try:
            if not chunks:
                return {'error': 'No hay datos disponibles'}
            
            # Agrupar chunks por fuente
            sources = defaultdict(list)
            for chunk in chunks:
                source = chunk.get('metadata', {}).get('source_file', 'Desconocido')
                sources[source].append(chunk)
            
            if len(sources) < 2:
                return {'error': 'Se necesitan al menos 2 fuentes para triangulación'}
            
            # Extraer conceptos por fuente
            concepts_by_source = {}
            for source, source_chunks in sources.items():
                concepts = self.extract_key_concepts(source_chunks)
                concepts_by_source[source] = {c['concept']: c['score'] for c in concepts}
            
            # Encontrar conceptos comunes (triangulación)
            all_concepts = set()
            for concepts in concepts_by_source.values():
                all_concepts.update(concepts.keys())
            
            triangulated_concepts = []
            for concept in all_concepts:
                sources_with_concept = []
                total_score = 0
                
                for source, concepts in concepts_by_source.items():
                    if concept in concepts:
                        sources_with_concept.append(source)
                        total_score += concepts[concept]
                
                if len(sources_with_concept) >= 2:  # Concepto presente en al menos 2 fuentes
                    triangulated_concepts.append({
                        'concept': concept,
                        'sources': sources_with_concept,
                        'source_count': len(sources_with_concept),
                        'avg_score': total_score / len(sources_with_concept),
                        'reliability': len(sources_with_concept) / len(sources)
                    })
            
            # Ordenar por confiabilidad
            triangulated_concepts.sort(key=lambda x: (x['reliability'], x['avg_score']), reverse=True)
            
            return {
                'triangulated_concepts': triangulated_concepts[:20],
                'total_sources': len(sources),
                'total_concepts': len(all_concepts),
                'validated_concepts': len(triangulated_concepts),
                'sources': list(sources.keys())
            }
            
        except Exception as e:
            logger.error(f"Error en triangulación: {e}")
            return {'error': str(e)}
    
    def _get_concept_context(self, concept: str, chunks: List[Dict], max_length: int = 300) -> str:
        """Obtener contexto de un concepto específico"""
        try:
            contexts = []
            concept_lower = concept.lower()
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if concept_lower in content:
                    # Encontrar la posición del concepto
                    pos = content.find(concept_lower)
                    start = max(0, pos - 100)
                    end = min(len(content), pos + len(concept) + 100)
                    context = content[start:end].strip()
                    contexts.append(context)
            
            if contexts:
                # Combinar contextos únicos
                unique_contexts = list(set(contexts))
                combined = " ... ".join(unique_contexts[:3])
                return combined[:max_length]
            
            return f"Contexto no encontrado para: {concept}"
            
        except Exception as e:
            return f"Error obteniendo contexto: {str(e)}"
    
    def _find_related_concepts(self, main_concept: str, chunks: List[Dict]) -> List[Dict]:
        """Encontrar conceptos relacionados a un concepto principal"""
        try:
            related = []
            main_concept_lower = main_concept.lower()
            
            # Buscar en chunks que contengan el concepto principal
            relevant_chunks = []
            for chunk in chunks:
                if main_concept_lower in chunk.get('content', '').lower():
                    relevant_chunks.append(chunk)
            
            if relevant_chunks:
                # Extraer conceptos de chunks relevantes
                related_concepts = self.extract_key_concepts(relevant_chunks)
                
                # Filtrar conceptos relacionados (excluyendo el principal)
                for concept in related_concepts:
                    if (concept['concept'].lower() != main_concept_lower and 
                        len(concept['concept']) > 3 and
                        concept['score'] > 0.01):
                        related.append({
                            'concept': concept['concept'],
                            'score': concept['score'],
                            'relation': 'relacionado con'
                        })
                        
                        if len(related) >= 5:  # Máximo 5 conceptos relacionados
                            break
            
            return related
            
        except Exception as e:
            logger.warning(f"Error encontrando conceptos relacionados: {e}")
            return []
    
    def _analyze_concept_hierarchy(self, chunks: List[Dict]) -> Optional[Dict]:
        """Analizar jerarquía de conceptos para mapa conceptual estructurado"""
        try:
            # Extraer conceptos clave con contexto
            concepts = self.extract_key_concepts(chunks)
            if not concepts:
                return None
            
            # Identificar tema principal
            main_theme = self._identify_main_theme(chunks)
            
            # Clasificar conceptos por niveles jerárquicos
            hierarchy = {
                'main_theme': main_theme,
                'main_concepts': [],
                'sub_concepts': [],
                'cross_relations': [],
                'info': {
                    'total_concepts': len(concepts),
                    'analysis_method': 'hierarchical_semantic'
                }
            }
            
            # Conceptos principales (nivel 1)
            main_concepts = concepts[:6]  # Top 6 conceptos principales
            for concept in main_concepts:
                concept_data = {
                    'name': concept['concept'],
                    'relevance': concept['score'],  # Cambiar 'importance' por 'relevance'
                    'context': self._get_concept_context(concept['concept'], chunks)[:200],
                    'sub_concepts': []
                }
                
                # Buscar sub-conceptos relacionados
                related_concepts = self._find_related_concepts(concept['concept'], chunks)
                for related in related_concepts[:4]:  # Máximo 4 sub-conceptos
                    sub_concept = {
                        'name': related['concept'],
                        'relation': related.get('relation', 'relacionado con'),
                        'relevance': related['score'],
                        'context': self._get_concept_context(related['concept'], chunks)[:150]
                    }
                    concept_data['sub_concepts'].append(sub_concept)
                
                hierarchy['main_concepts'].append(concept_data)
            
            # Identificar relaciones cruzadas
            hierarchy['cross_relations'] = self._identify_cross_relations(hierarchy['main_concepts'])
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error analizando jerarquía de conceptos: {e}")
            return None
    
    def _analyze_intelligent_mind_map_structure(self, chunks: List[Dict]) -> Optional[Dict]:
        """Análisis inteligente mejorado usando modelos de lenguaje para estructura de mapa mental"""
        try:
            # 1. Usar modelo de lenguaje para análisis semántico profundo
            semantic_analysis = self._llm_semantic_analysis(chunks)
            
            # 2. Identificar tema central usando IA
            central_theme = self._llm_identify_central_theme(chunks, semantic_analysis)
            
            # 3. Extraer ramas principales usando clustering inteligente
            main_branches = self._llm_extract_semantic_branches(chunks, semantic_analysis)
            
            # 4. Identificar conexiones usando análisis de relaciones semánticas
            cross_connections = self._llm_identify_semantic_connections(main_branches, chunks)
            
            structure = {
                'central_theme': central_theme,
                'main_branches': main_branches,
                'cross_connections': cross_connections,
                'info': {
                    'total_branches': len(main_branches),
                    'analysis_method': 'llm_intelligent_semantic_analysis',
                    'semantic_depth': len(semantic_analysis.get('concepts', []))
                }
            }
            
            return structure
            
        except Exception as e:
            logger.error(f"Error en análisis inteligente de mapa mental: {e}")
            return None
    
    def _llm_semantic_analysis(self, chunks: List[Dict]) -> Dict:
        """Análisis semántico profundo usando modelo de lenguaje"""
        try:
            from utils.ollama_client import OllamaClient
            
            # Combinar contenido de chunks
            combined_content = "\n\n".join([
                chunk.get('content', '')[:500] for chunk in chunks[:10]  # Limitar para eficiencia
            ])
            
            # Prompt para análisis semántico
            analysis_prompt = f"""
            Analiza el siguiente texto y extrae información estructurada para crear un mapa mental inteligente:

            TEXTO:
            {combined_content}

            INSTRUCCIONES:
            1. Identifica los 5-8 conceptos más importantes y relevantes
            2. Para cada concepto, identifica 2-4 sub-conceptos relacionados
            3. Determina el tema central que conecta todos los conceptos
            4. Identifica relaciones semánticas entre conceptos
            5. Proporciona contexto breve para cada concepto

            FORMATO DE RESPUESTA (JSON):
            {{
                "tema_central": "tema principal del contenido",
                "conceptos_principales": [
                    {{
                        "nombre": "nombre del concepto",
                        "importancia": 0.9,
                        "descripcion": "descripción breve",
                        "sub_conceptos": [
                            {{
                                "nombre": "sub-concepto",
                                "relacion": "tipo de relación",
                                "relevancia": 0.8
                            }}
                        ]
                    }}
                ],
                "relaciones": [
                    {{
                        "concepto_1": "nombre concepto 1",
                        "concepto_2": "nombre concepto 2",
                        "tipo_relacion": "causa-efecto/similitud/oposición/etc",
                        "fuerza": 0.7
                    }}
                ]
            }}
            """
            
            # Usar cliente Ollama para análisis
            ollama_client = OllamaClient()
            response = ollama_client.generate_response(analysis_prompt)
            
            # Intentar parsear respuesta JSON
            import json
            import re
            
            # Extraer JSON de la respuesta
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    semantic_data = json.loads(json_match.group())
                    return semantic_data
                except json.JSONDecodeError:
                    pass
            
            # Fallback: análisis básico si falla el LLM
            return self._fallback_semantic_analysis(chunks)
            
        except Exception as e:
            logger.error(f"Error en análisis semántico LLM: {e}")
            return self._fallback_semantic_analysis(chunks)
    
    def _llm_identify_central_theme(self, chunks: List[Dict], semantic_analysis: Dict) -> Dict:
        """Identificar tema central usando análisis LLM"""
        try:
            tema_central = semantic_analysis.get('tema_central', 'Análisis Cualitativo')
            
            return {
                'name': tema_central,
                'description': f"Tema central identificado mediante análisis semántico: {tema_central}"
            }
            
        except Exception as e:
            logger.error(f"Error identificando tema central LLM: {e}")
            return {'name': 'Tema Principal', 'description': 'Análisis de contenido'}
    
    def _llm_extract_semantic_branches(self, chunks: List[Dict], semantic_analysis: Dict) -> List[Dict]:
        """Extraer ramas principales usando análisis LLM"""
        try:
            branches = []
            conceptos_principales = semantic_analysis.get('conceptos_principales', [])
            
            for i, concepto in enumerate(conceptos_principales[:8]):  # Máximo 8 ramas
                branch = {
                    'name': concepto.get('nombre', f'Concepto {i+1}'),
                    'importance': concepto.get('importancia', 0.5),
                    'description': concepto.get('descripcion', 'Concepto identificado automáticamente'),
                    'sub_branches': []
                }
                
                # Agregar sub-conceptos
                sub_conceptos = concepto.get('sub_conceptos', [])
                for sub_concepto in sub_conceptos[:4]:  # Máximo 4 sub-ramas
                    sub_branch = {
                        'name': sub_concepto.get('nombre', 'Sub-concepto'),
                        'relation': sub_concepto.get('relacion', 'relacionado con'),
                        'relevance': sub_concepto.get('relevancia', 0.5),
                        'context': f"Sub-concepto de {branch['name']}"
                    }
                    branch['sub_branches'].append(sub_branch)
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.error(f"Error extrayendo ramas semánticas LLM: {e}")
            return self._fallback_extract_branches(chunks)
    
    def _llm_identify_semantic_connections(self, branches: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Identificar conexiones semánticas usando análisis LLM"""
        try:
            connections = []
            
            # Crear conexiones basadas en las relaciones identificadas por el LLM
            for i, branch1 in enumerate(branches):
                for j, branch2 in enumerate(branches[i+1:], i+1):
                    # Simular conexión basada en similitud semántica
                    if self._calculate_semantic_similarity(branch1['name'], branch2['name']) > 0.3:
                        connections.append({
                            'from': f"branch_{i}",
                            'to': f"branch_{j}",
                            'relation': 'relacionado semánticamente',
                            'strength': self._calculate_semantic_similarity(branch1['name'], branch2['name'])
                        })
            
            return connections[:5]  # Máximo 5 conexiones
            
        except Exception as e:
            logger.error(f"Error identificando conexiones semánticas LLM: {e}")
            return []
    
    def _calculate_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Calcular similitud semántica básica entre dos conceptos"""
        try:
            # Similitud básica basada en palabras comunes
            words1 = set(concept1.lower().split())
            words2 = set(concept2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def _fallback_semantic_analysis(self, chunks: List[Dict]) -> Dict:
        """Análisis semántico de respaldo si falla el LLM"""
        try:
            # Análisis básico usando técnicas tradicionales
            concept_network = self._build_concept_network(chunks)
            
            # Convertir a formato esperado
            conceptos_principales = []
            for concept, data in list(concept_network.items())[:6]:
                concepto = {
                    'nombre': concept.title(),
                    'importancia': min(1.0, data['frequency'] / 10),
                    'descripcion': f"Concepto extraído automáticamente: {concept}",
                    'sub_conceptos': []
                }
                
                # Agregar sub-conceptos basados en co-ocurrencias
                for related_concept, count in list(data['cooccurrences'].items())[:3]:
                    sub_concepto = {
                        'nombre': related_concept.title(),
                        'relacion': 'co-ocurre con',
                        'relevancia': min(1.0, count / 5)
                    }
                    concepto['sub_conceptos'].append(sub_concepto)
                
                conceptos_principales.append(concepto)
            
            return {
                'tema_central': 'Análisis de Contenido',
                'conceptos_principales': conceptos_principales,
                'relaciones': []
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de respaldo: {e}")
            return {
                'tema_central': 'Análisis Cualitativo',
                'conceptos_principales': [],
                'relaciones': []
            }
    
    def _fallback_extract_branches(self, chunks: List[Dict]) -> List[Dict]:
        """Extracción de ramas de respaldo"""
        try:
            # Usar método tradicional como respaldo
            concept_network = self._build_concept_network(chunks)
            branches = []
            
            for i, (concept, data) in enumerate(list(concept_network.items())[:6]):
                branch = {
                    'name': concept.title(),
                    'importance': min(1.0, data['frequency'] / 10),
                    'description': f"Concepto identificado: {concept}",
                    'sub_branches': []
                }
                
                # Agregar sub-ramas
                for related_concept, count in list(data['cooccurrences'].items())[:3]:
                    sub_branch = {
                        'name': related_concept.title(),
                        'relation': 'relacionado con',
                        'relevance': min(1.0, count / 5),
                        'context': f"Relacionado con {concept}"
                    }
                    branch['sub_branches'].append(sub_branch)
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.error(f"Error en extracción de ramas de respaldo: {e}")
            return []
    
    def _build_concept_network(self, chunks: List[Dict]) -> Dict:
        """Construir red de conceptos basada en co-ocurrencia y frecuencia"""
        try:
            concept_network = defaultdict(lambda: {'frequency': 0, 'cooccurrences': defaultdict(int), 'contexts': []})
            
            # Extraer conceptos de cada chunk
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                
                # Extraer frases importantes (2-4 palabras)
                important_phrases = self._extract_important_phrases(content)
                
                # Registrar frecuencias y co-ocurrencias
                for phrase in important_phrases:
                    concept_network[phrase]['frequency'] += 1
                    concept_network[phrase]['contexts'].append(content[:200])
                    
                    # Co-ocurrencias con otras frases en el mismo chunk
                    for other_phrase in important_phrases:
                        if phrase != other_phrase:
                            concept_network[phrase]['cooccurrences'][other_phrase] += 1
            
            return dict(concept_network)
            
        except Exception as e:
            logger.error(f"Error construyendo red de conceptos: {e}")
            return {}
    
    def _extract_important_phrases(self, text: str) -> List[str]:
        """Extraer frases importantes usando técnicas de NLP"""
        try:
            # Limpiar texto
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            
            # Filtrar palabras vacías
            stop_words = set(self._get_spanish_stopwords())
            filtered_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
            
            phrases = []
            
            # Extraer bigramas y trigramas significativos
            for i in range(len(filtered_words) - 1):
                bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
                phrases.append(bigram.lower())
                
                if i < len(filtered_words) - 2:
                    trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
                    phrases.append(trigram.lower())
            
            # Filtrar frases por relevancia (frecuencia mínima)
            phrase_counts = Counter(phrases)
            relevant_phrases = [phrase for phrase, count in phrase_counts.items() if count >= 2]
            
            return relevant_phrases[:20]  # Top 20 frases más relevantes
            
        except Exception as e:
            logger.error(f"Error extrayendo frases importantes: {e}")
            return []
    
    def _identify_central_theme_advanced(self, chunks: List[Dict]) -> Dict:
        """Identificar tema central usando análisis semántico avanzado"""
        try:
            # Combinar todo el contenido
            all_content = " ".join([chunk.get('content', '') for chunk in chunks])
            
            # Extraer conceptos más frecuentes y significativos
            important_phrases = self._extract_important_phrases(all_content)
            phrase_counts = Counter(important_phrases)
            
            # Seleccionar el tema central basado en frecuencia y centralidad
            if phrase_counts:
                central_concept = phrase_counts.most_common(1)[0][0]
                
                # Generar descripción del tema central
                description = self._generate_theme_description(central_concept, chunks)
                
                return {
                    'name': central_concept.title(),
                    'description': description
                }
            
            return {
                'name': 'Análisis Cualitativo',
                'description': 'Tema central identificado automáticamente'
            }
            
        except Exception as e:
            logger.error(f"Error identificando tema central avanzado: {e}")
            return {'name': 'Tema Principal', 'description': 'Análisis de contenido'}
    
    def _extract_semantic_branches(self, chunks: List[Dict], concept_network: Dict) -> List[Dict]:
        """Extraer ramas principales usando clustering semántico"""
        try:
            branches = []
            
            # Ordenar conceptos por importancia (frecuencia + co-ocurrencias)
            concept_scores = []
            for concept, data in concept_network.items():
                importance_score = data['frequency'] + sum(data['cooccurrences'].values()) * 0.1
                concept_scores.append((concept, importance_score, data))
            
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Crear ramas principales (top 6 conceptos)
            for i, (concept, score, data) in enumerate(concept_scores[:6]):
                branch = {
                    'name': concept.title(),
                    'importance': min(1.0, score / max(concept_scores[0][1], 1)),  # Normalizar
                    'description': self._generate_concept_description(concept, chunks),
                    'sub_branches': []
                }
                
                # Encontrar sub-ramas relacionadas
                related_concepts = sorted(
                    data['cooccurrences'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:4]
                
                for related_concept, cooccurrence_count in related_concepts:
                    if related_concept in concept_network:
                        sub_branch = {
                            'name': related_concept.title(),
                            'relation': 'relacionado con',
                            'relevance': min(1.0, cooccurrence_count / max(data['cooccurrences'].values(), 1)),
                            'context': self._get_concept_context(related_concept, chunks)[:150]
                        }
                        branch['sub_branches'].append(sub_branch)
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.error(f"Error extrayendo ramas semánticas: {e}")
            return []
    
    def _generate_theme_description(self, theme: str, chunks: List[Dict]) -> str:
        """Generar descripción del tema basada en el contexto"""
        try:
            relevant_contexts = []
            theme_lower = theme.lower()
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if theme_lower in content:
                    # Extraer oración que contiene el tema
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if theme_lower in sentence and len(sentence.strip()) > 20:
                            relevant_contexts.append(sentence.strip())
                            break
                
                if len(relevant_contexts) >= 3:
                    break
            
            if relevant_contexts:
                return ". ".join(relevant_contexts[:2])[:200] + "..."
            
            return f"Tema central relacionado con {theme}"
            
        except Exception as e:
            logger.error(f"Error generando descripción del tema: {e}")
            return f"Análisis de {theme}"
    
    def _generate_concept_description(self, concept: str, chunks: List[Dict]) -> str:
        """Generar descripción del concepto basada en el contexto"""
        try:
            concept_lower = concept.lower()
            contexts = []
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if concept_lower in content:
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if concept_lower in sentence and len(sentence.strip()) > 15:
                            contexts.append(sentence.strip())
                            break
                
                if len(contexts) >= 2:
                    break
            
            if contexts:
                return ". ".join(contexts)[:200] + "..."
            
            return f"Concepto relacionado con {concept}"
            
        except Exception as e:
            logger.error(f"Error generando descripción del concepto: {e}")
            return f"Análisis de {concept}"
    
    def _identify_semantic_connections(self, branches: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Identificar conexiones semánticas entre ramas"""
        try:
            connections = []
            
            # Buscar conexiones entre ramas principales
            for i, branch1 in enumerate(branches):
                for j, branch2 in enumerate(branches[i+1:], i+1):
                    # Verificar si los conceptos aparecen juntos en el contenido
                    concept1 = branch1['name'].lower()
                    concept2 = branch2['name'].lower()
                    
                    connection_strength = 0
                    for chunk in chunks:
                        content = chunk.get('content', '').lower()
                        if concept1 in content and concept2 in content:
                            connection_strength += 1
                    
                    if connection_strength >= 2:  # Aparecen juntos en al menos 2 chunks
                        connections.append({
                            'from': f"branch_{i}",
                            'to': f"branch_{j}",
                            'relation': 'co-ocurre con',
                            'strength': connection_strength
                        })
            
            return connections[:5]  # Máximo 5 conexiones
            
        except Exception as e:
            logger.error(f"Error identificando conexiones semánticas: {e}")
            return []
    
    def _lighten_color_hex(self, hex_color: str, factor: float = 0.3) -> str:
        """Aclarar un color hexadecimal"""
        try:
            # Remover el # si está presente
            hex_color = hex_color.lstrip('#')
            
            # Convertir a RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Aclarar cada componente
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            
            # Convertir de vuelta a hex
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception:
            return "#cccccc"  # Color por defecto si hay error
    
    def _get_spanish_stopwords(self) -> List[str]:
        """Obtener lista de palabras vacías en español"""
        return [
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son',
            'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'le', 'ya', 'o', 'fue', 'este',
            'ha', 'si', 'porque', 'esta', 'son', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene', 'también',
            'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante',
            'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'fueron', 'ese', 'eso', 'había', 'ante', 'ellos', 'e',
            'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa',
            'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'sea', 'poco', 'ella', 'estar', 'haber', 'estas',
            'estaba', 'estamos', 'pueden', 'hacen', 'cada', 'fin', 'incluso', 'primero', 'además', 'mientras', 'sin',
            'nueva', 'las', 'suelen', 'cómo', 'después', 'gran', 'tiempo', 'años', 'sobre', 'otras', 'manera', 'bien',
            'trabajo', 'vida', 'ejemplo', 'llevar', 'crear', 'gustar', 'hablar', 'hacer', 'ver', 'dar', 'ir', 'venir',
            'decir', 'deber', 'poder', 'saber', 'querer', 'llegar', 'poner', 'parecer', 'seguir', 'encontrar', 'llamar',
            'venir', 'pensar', 'salir', 'volver', 'tomar', 'conocer', 'vivir', 'sentir', 'tratar', 'mirar', 'contar',
            'empezar', 'esperar', 'buscar', 'existir', 'entrar', 'trabajar', 'escribir', 'perder', 'producir', 'ocurrir'
        ]
    
    def _get_concept_context(self, concept: str, chunks: List[Dict]) -> str:
        """Obtener contexto de un concepto en los chunks"""
        try:
            concept_lower = concept.lower()
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if concept_lower in content:
                    # Encontrar la oración que contiene el concepto
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if concept_lower in sentence and len(sentence.strip()) > 10:
                            return sentence.strip()
            
            return f"Contexto de {concept}"
            
        except Exception as e:
            logger.error(f"Error obteniendo contexto del concepto: {e}")
            return f"Análisis de {concept}"
    
    def _identify_main_theme(self, chunks: List[Dict]) -> Dict:
        """Identificar el tema principal del contenido"""
        try:
            # Generar resumen principal
            main_summary = self.generate_rag_summary(chunks, 150)
            
            # Extraer conceptos más frecuentes
            all_text = " ".join([chunk.get('content', '') for chunk in chunks])
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = Counter(words)
            
            # Filtrar palabras vacías
            stop_words = set(self._get_spanish_stopwords())
            relevant_words = [word for word, count in word_freq.most_common(20) 
                            if len(word) > 3 and word not in stop_words]
            
            # Crear nombre del tema principal
            theme_name = " ".join(relevant_words[:3]).title() if relevant_words else "Tema Principal"
            
            return {
                'name': theme_name,
                'description': main_summary
            }
            
        except Exception as e:
            logger.error(f"Error identificando tema principal: {e}")
            return {'name': 'Tema Principal', 'description': 'Análisis de contenido'}
    
    def _group_concepts_into_branches(self, concepts: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Agrupar conceptos en ramas temáticas para mapa mental"""
        try:
            branches = []
            
            # Tomar los conceptos principales como ramas
            main_concepts = concepts[:6]  # Máximo 6 ramas principales
            
            for concept in main_concepts:
                branch = {
                    'name': concept['concept'],
                    'importance': concept['score'],
                    'description': self._get_concept_context(concept['concept'], chunks)[:200],
                    'sub_branches': [],
                    'details': []
                }
                
                # Buscar sub-ramas relacionadas
                related_concepts = self._find_related_concepts(concept['concept'], chunks)
                for related in related_concepts[:4]:  # Máximo 4 sub-ramas
                    sub_branch = {
                        'name': related['concept'],
                        'relation': related.get('relation', 'relacionado'),
                        'relevance': related['score'],
                        'context': self._get_concept_context(related['concept'], chunks)[:150],
                        'details': []
                    }
                    
                    # Agregar detalles específicos
                    details = self._extract_concept_details(related['concept'], chunks)
                    sub_branch['details'] = details[:3]  # Máximo 3 detalles
                    
                    branch['sub_branches'].append(sub_branch)
                
                branches.append(branch)
            
            return branches
            
        except Exception as e:
            logger.error(f"Error agrupando conceptos en ramas: {e}")
            return []
    
    def _extract_concept_details(self, concept: str, chunks: List[Dict]) -> List[Dict]:
        """Extraer detalles específicos de un concepto"""
        try:
            details = []
            concept_lower = concept.lower()
            
            for chunk in chunks:
                content = chunk.get('content', '').lower()
                if concept_lower in content:
                    # Extraer oraciones que contienen el concepto
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if concept_lower in sentence and len(sentence.strip()) > 20:
                            # Extraer palabras clave de la oración
                            words = re.findall(r'\b\w+\b', sentence)
                            key_words = [w for w in words if len(w) > 4 and w != concept_lower]
                            
                            if key_words:
                                detail_name = key_words[0].title()
                                details.append({
                                    'name': detail_name,
                                    'description': sentence.strip()[:100]
                                })
                            
                            if len(details) >= 5:  # Máximo 5 detalles por concepto
                                break
                
                if len(details) >= 5:
                    break
            
            return details
            
        except Exception as e:
            logger.error(f"Error extrayendo detalles del concepto: {e}")
            return []
    
    def _identify_cross_relations(self, main_concepts: List[Dict]) -> List[Dict]:
        """Identificar relaciones cruzadas entre conceptos principales"""
        try:
            relations = []
            
            for i, concept1 in enumerate(main_concepts):
                for j, concept2 in enumerate(main_concepts[i+1:], i+1):
                    # Buscar relaciones semánticas entre conceptos
                    relation_strength = self._calculate_concept_similarity(
                        concept1['name'], concept2['name']
                    )
                    
                    if relation_strength > 0.3:  # Umbral de relación
                        relations.append({
                            'from': f"concept_{i}",
                            'to': f"concept_{j}",
                            'relation': 'relacionado',
                            'strength': relation_strength
                        })
            
            return relations
            
        except Exception as e:
            logger.error(f"Error identificando relaciones cruzadas: {e}")
            return []
    
    def _identify_branch_connections(self, branches: List[Dict]) -> List[Dict]:
        """Identificar conexiones entre ramas del mapa mental"""
        try:
            connections = []
            
            for i, branch1 in enumerate(branches):
                for j, branch2 in enumerate(branches[i+1:], i+1):
                    # Calcular similitud entre ramas
                    similarity = self._calculate_concept_similarity(
                        branch1['name'], branch2['name']
                    )
                    
                    if similarity > 0.25:  # Umbral para conexiones cruzadas
                        connections.append({
                            'from': f"branch_{i}",
                            'to': f"branch_{j}",
                            'relation': 'conectado',
                            'strength': similarity
                        })
            
            return connections
            
        except Exception as e:
            logger.error(f"Error identificando conexiones entre ramas: {e}")
            return []
    
    def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calcular similitud básica entre dos conceptos"""
        try:
            # Similitud básica basada en palabras comunes
            words1 = set(concept1.lower().split())
            words2 = set(concept2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.error(f"Error calculando similitud de conceptos: {e}")
            return 0.0
    
    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Aclarar un color hexadecimal"""
        try:
            # Remover el # si está presente
            hex_color = hex_color.lstrip('#')
            
            # Convertir a RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Aclarar cada componente
            r = min(255, int(r + (255 - r) * factor))
            g = min(255, int(g + (255 - g) * factor))
            b = min(255, int(b + (255 - b) * factor))
            
            # Convertir de vuelta a hexadecimal
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception as e:
            logger.error(f"Error aclarando color: {e}")
            return "#95a5a6"  # Color por defecto
    
    def _get_concept_color(self, score: float) -> str:
        """Obtener color basado en la puntuación del concepto"""
        if score > 0.1:
            return "#e74c3c"  # Rojo para alta relevancia
        elif score > 0.05:
            return "#f39c12"  # Naranja para relevancia media
        elif score > 0.02:
            return "#3498db"  # Azul para relevancia baja
        else:
            return "#95a5a6"  # Gris para muy baja relevancia
    
    def preprocess_text(self, text: str) -> str:
        """Preprocesar texto para análisis"""
        if not text:
            return ""
        
        # Limpiar texto
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_advanced_themes(self, chunks: List[Dict], n_topics: int = 10) -> Dict:
        """Extracción avanzada de temas usando LDA"""
        if not chunks or not ADVANCED_ANALYSIS_AVAILABLE:
            return self._basic_theme_extraction(chunks)
        
        try:
            # Preparar textos
            texts = [self.preprocess_text(chunk.get('content', '')) for chunk in chunks]
            texts = [text for text in texts if len(text) > 50]  # Filtrar textos muy cortos
            
            if len(texts) < 2:
                return self._basic_theme_extraction(chunks)
            
            # Vectorización TF-IDF con manejo de errores
            # Ajustar parámetros según el número de documentos
            num_docs = len(texts)
            adjusted_min_df = max(1, min(2, num_docs // 5))
            adjusted_max_df = min(0.95, max(0.5, 1.0 - (2.0 / num_docs)))
            
            try:
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words=self._get_spanish_stopwords(),
                    ngram_range=(1, 2),
                    min_df=adjusted_min_df,
                    max_df=adjusted_max_df
                )
                
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # LDA para extracción de temas
                lda = LatentDirichletAllocation(
                    n_components=min(n_topics, len(texts)),
                    random_state=42,
                    max_iter=10
                )
                
                lda.fit(tfidf_matrix)
                
                # Extraer temas
                topics = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topic_weight = topic[top_words_idx].sum()
                    
                    topics.append({
                        'id': topic_idx,
                        'words': top_words,
                        'weight': float(topic_weight),
                        'coherence': self._calculate_topic_coherence(top_words, texts)
                    })
                
                # Asignar documentos a temas
                doc_topic_probs = lda.transform(tfidf_matrix)
                
                return {
                    'topics': topics,
                    'document_topics': doc_topic_probs.tolist(),
                    'feature_names': feature_names.tolist(),
                    'method': 'LDA',
                    'n_topics': len(topics)
                }
                
            except Exception as tfidf_error:
                logger.warning(f"Error en TF-IDF para temas avanzados: {tfidf_error}")
                return self._basic_theme_extraction(chunks)
            
        except Exception as e:
            logger.error(f"Error en análisis avanzado de temas: {e}")
            return self._basic_theme_extraction(chunks)
    
    def _basic_theme_extraction(self, chunks: List[Dict]) -> Dict:
        """Extracción básica de temas por frecuencia"""
        all_text = " ".join([chunk.get('content', '') for chunk in chunks])
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)
        
        stop_words = self._get_spanish_stopwords()
        themes = {word: count for word, count in word_freq.most_common(20) 
                 if len(word) > 3 and word not in stop_words}
        
        return {
            'themes': themes,
            'method': 'frequency',
            'total_words': len(words)
        }
    
    def _get_spanish_stopwords(self) -> list:
        """Obtener palabras vacías en español"""
        spanish_stopwords = [
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 
            'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'más', 'pero', 'sus', 'me', 'ya', 
            'muy', 'mi', 'sin', 'sobre', 'este', 'ser', 'tiene', 'todo', 'esta', 'era', 'cuando', 'él', 'uno', 
            'donde', 'bien', 'tiempo', 'cada', 'aquí', 'hacer', 'cómo', 'sólo', 'durante', 'todos', 'lugar', 
            'vida', 'gran', 'hasta', 'desde', 'había', 'sido', 'años', 'año', 'puede', 'tienen', 'país', 'parte', 
            'entre', 'ciudad', 'mundo', 'forma', 'estado', 'mayor', 'día', 'grupo', 'agua', 'casa', 'caso', 
            'gobierno', 'sistema', 'trabajo', 'desarrollo', 'proceso', 'nivel', 'área', 'población', 'región', 
            'zona', 'tipo', 'número', 'millones', 'miles', 'importante', 'nacional', 'social', 'general', 'local', 
            'público', 'principal', 'internacional', 'natural', 'cultural', 'económico', 'político', 'histórico'
        ]
        
        if ADVANCED_ANALYSIS_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                nltk_stopwords = stopwords.words('spanish')
                spanish_stopwords.extend(nltk_stopwords)
                # Eliminar duplicados manteniendo el orden
                spanish_stopwords = list(dict.fromkeys(spanish_stopwords))
            except:
                pass
        
        return spanish_stopwords
    
    def _calculate_topic_coherence(self, words: List[str], texts: List[str]) -> float:
        """Calcular coherencia de un tema"""
        try:
            # Implementación básica de coherencia
            word_counts = Counter()
            for text in texts:
                text_words = set(text.lower().split())
                for word in words:
                    if word in text_words:
                        word_counts[word] += 1
            
            if not word_counts:
                return 0.0
            
            # Coherencia basada en co-ocurrencia
            coherence = sum(word_counts.values()) / len(words)
            return min(coherence / len(texts), 1.0)
        except:
            return 0.0
    
    def perform_clustering(self, chunks: List[Dict], n_clusters: int = 5) -> Dict:
        """Realizar clustering de documentos"""
        if not chunks or not ADVANCED_ANALYSIS_AVAILABLE:
            return {'error': 'Clustering no disponible'}
        
        try:
            # Preparar textos
            texts = [self.preprocess_text(chunk.get('content', '')) for chunk in chunks]
            texts = [text for text in texts if len(text) > 50]
            
            if len(texts) < 2:
                return {'error': 'Insuficientes documentos para clustering'}
            
            # Vectorización con manejo de errores
            # Ajustar parámetros según el número de documentos
            num_docs = len(texts)
            adjusted_min_df = max(1, min(2, num_docs // 4))
            adjusted_max_df = min(0.95, max(0.5, 1.0 - (1.0 / num_docs)))
            
            try:
                vectorizer = TfidfVectorizer(
                    max_features=500,
                    stop_words=self._get_spanish_stopwords(),
                    ngram_range=(1, 2),
                    min_df=adjusted_min_df,
                    max_df=adjusted_max_df
                )
                
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                # K-means clustering
                n_clusters = min(n_clusters, len(texts))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                
                # Reducción de dimensionalidad para visualización
                if len(texts) > 2:
                    try:
                        from umap import UMAP
                        reducer = UMAP(n_components=2, random_state=42)
                        embedding = reducer.fit_transform(tfidf_matrix.toarray())
                    except ImportError:
                        # Fallback a PCA
                        pca = PCA(n_components=2, random_state=42)
                        embedding = pca.fit_transform(tfidf_matrix.toarray())
                else:
                    embedding = np.random.rand(len(texts), 2)
                
                # Preparar resultados
                clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    
                    source_file = chunks[i].get('metadata', {}).get('source_file', 'Desconocido')
                    clusters[label].append({
                        'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                        'source': source_file,
                        'embedding': embedding[i].tolist()
                    })
                
                return {
                    'clusters': clusters,
                    'embeddings': embedding.tolist(),
                    'labels': cluster_labels.tolist(),
                    'n_clusters': n_clusters,
                    'method': 'K-means + UMAP'
                }
                
            except Exception as clustering_error:
                logger.warning(f"Error en clustering TF-IDF: {clustering_error}")
                return {'error': f'Error en clustering: {str(clustering_error)}'}
            
        except Exception as e:
            logger.error(f"Error en clustering: {e}")
            return {'error': str(e)}
    
    def advanced_sentiment_analysis(self, chunks: List[Dict]) -> Dict:
        """Análisis avanzado de sentimientos"""
        if not chunks:
            return {'error': 'No hay datos disponibles'}
        
        try:
            if ADVANCED_ANALYSIS_AVAILABLE:
                from nltk.sentiment import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                method = 'VADER (NLTK)'
            else:
                # Fallback a TextBlob
                method = 'TextBlob'
            
            results = {
                'by_source': defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0, 'scores': []}),
                'overall_stats': {'distribution': {}, 'mean_score': 0},
                'method': method
            }
            
            all_scores = []
            
            for chunk in chunks:
                content = chunk.get('content', '')
                source = chunk.get('metadata', {}).get('source_file', 'Desconocido')
                
                if not content:
                    continue
                
                # Análisis de sentimiento
                if ADVANCED_ANALYSIS_AVAILABLE and 'sia' in locals():
                    scores = sia.polarity_scores(content)
                    compound_score = scores['compound']
                else:
                    # Fallback con TextBlob
                    blob = TextBlob(content)
                    compound_score = blob.sentiment.polarity
                
                all_scores.append(compound_score)
                results['by_source'][source]['scores'].append(compound_score)
                
                # Clasificar sentimiento
                if compound_score >= 0.05:
                    results['by_source'][source]['positive'] += 1
                elif compound_score <= -0.05:
                    results['by_source'][source]['negative'] += 1
                else:
                    results['by_source'][source]['neutral'] += 1
            
            # Estadísticas generales
            if all_scores:
                total_positive = sum(1 for score in all_scores if score >= 0.05)
                total_negative = sum(1 for score in all_scores if score <= -0.05)
                total_neutral = len(all_scores) - total_positive - total_negative
                
                results['overall_stats'] = {
                    'distribution': {
                        'positive': total_positive,
                        'neutral': total_neutral,
                        'negative': total_negative
                    },
                    'mean_score': np.mean(all_scores) if all_scores else 0
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis de sentimientos: {e}")
            return {'error': str(e)}
    
    def generate_word_cloud(self, chunks: List[Dict], source_filter: Optional[str] = None) -> Optional[str]:
        """Generar nube de palabras"""
        if not chunks or not ADVANCED_ANALYSIS_AVAILABLE:
            return None
        
        try:
            # Filtrar por fuente si se especifica
            if source_filter:
                filtered_chunks = [
                    chunk for chunk in chunks 
                    if chunk.get('metadata', {}).get('source_file', 'Desconocido') == source_filter
                ]
            else:
                filtered_chunks = chunks
            
            if not filtered_chunks:
                return None
            
            # Combinar todo el texto
            all_text = " ".join([chunk.get('content', '') for chunk in filtered_chunks])
            processed_text = self.preprocess_text(all_text)
            
            if len(processed_text) < 100:
                return None
            
            # Generar nube de palabras
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=self._get_spanish_stopwords(),
                max_words=100,
                colormap='viridis',
                relative_scaling=0.5,
                random_state=42
            ).generate(processed_text)
            
            # Guardar imagen temporal
            temp_path = Path(config.CACHE_DIR) / f"wordcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            temp_path.parent.mkdir(exist_ok=True)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(temp_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Error generando nube de palabras: {e}")
            return None
    
    def generate_advanced_concept_map(self, chunks: List[Dict]) -> Dict:
        """Generar mapa conceptual avanzado"""
        if not chunks:
            return {'error': 'No hay datos disponibles'}
        
        try:
            # Preparar textos
            texts = [self.preprocess_text(chunk.get('content', '')) for chunk in chunks]
            texts = [text for text in texts if len(text) > 50]
            
            if len(texts) < 2:
                return {'error': 'Insuficientes documentos para generar mapa conceptual'}
            
            # Extraer términos importantes
            if ADVANCED_ANALYSIS_AVAILABLE:
                try:
                    # Ajustar parámetros según el número de documentos
                    num_docs = len(texts)
                    adjusted_min_df = max(1, min(2, num_docs // 3))
                    adjusted_max_df = min(0.95, max(0.4, 1.0 - (2.0 / num_docs)))
                    
                    vectorizer = TfidfVectorizer(
                        max_features=50,
                        stop_words=self._get_spanish_stopwords(),
                        ngram_range=(1, 2),
                        min_df=adjusted_min_df,
                        max_df=adjusted_max_df
                    )
                    
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Calcular similitudes entre términos
                    similarity_matrix = cosine_similarity(tfidf_matrix.T)
                    
                    # Crear nodos (términos importantes)
                    nodes = []
                    for i, term in enumerate(feature_names):
                        # Calcular importancia del término
                        importance = tfidf_matrix[:, i].sum()
                        
                        nodes.append({
                            'id': term,
                            'label': term,
                            'size': min(max(float(importance) * 20, 10), 50),
                            'color': f'hsl({hash(term) % 360}, 70%, 60%)'
                        })
                    
                    # Crear aristas (conexiones entre términos similares)
                    edges = []
                    threshold = 0.3  # Umbral de similitud
                    
                    for i in range(len(feature_names)):
                        for j in range(i + 1, len(feature_names)):
                            similarity = similarity_matrix[i, j]
                            if similarity > threshold:
                                edges.append({
                                    'from': feature_names[i],
                                    'to': feature_names[j],
                                    'weight': float(similarity),
                                    'width': max(similarity * 5, 1)
                                })
                    
                    # Estadísticas
                    stats = {
                        'total_nodes': len(nodes),
                        'total_edges': len(edges),
                        'avg_similarity': float(np.mean([edge['weight'] for edge in edges])) if edges else 0
                    }
                    
                    return {
                        'nodes': nodes,
                        'edges': edges,
                        'stats': stats,
                        'method': 'TF-IDF + Cosine Similarity'
                    }
                    
                except Exception as tfidf_error:
                    logger.warning(f"Error en TF-IDF para mapa conceptual: {tfidf_error}")
                    # Continuar con fallback básico
                    pass
                # Fallback básico
                all_text = " ".join(texts)
                words = re.findall(r'\b\w+\b', all_text.lower())
                word_freq = Counter(words)
                
                stop_words = set(self._get_spanish_stopwords())
                important_words = [
                    word for word, count in word_freq.most_common(20)
                    if len(word) > 3 and word not in stop_words
                ]
                
                nodes = []
                for word in important_words:
                    nodes.append({
                        'id': word,
                        'label': word,
                        'size': min(word_freq[word] * 2, 50),
                        'color': f'hsl({hash(word) % 360}, 70%, 60%)'
                    })
                
                # Conexiones básicas (palabras que aparecen juntas)
                edges = []
                for i, word1 in enumerate(important_words):
                    for j, word2 in enumerate(important_words[i+1:], i+1):
                        # Contar co-ocurrencias
                        cooccurrence = sum(1 for text in texts if word1 in text and word2 in text)
                        if cooccurrence > 1:
                            edges.append({
                                'from': word1,
                                'to': word2,
                                'weight': cooccurrence,
                                'width': min(cooccurrence, 5)
                            })
                
                stats = {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'method': 'Basic frequency + co-occurrence'
                }
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'stats': stats
                }
                
        except Exception as e:
            logger.error(f"Error generando mapa conceptual: {e}")
            return {'error': str(e)}

def render_advanced_dashboard(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Dashboard avanzado con métricas clave"""
    st.header("📊 Dashboard de Análisis")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Métricas generales
    st.subheader("📈 Métricas Generales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documentos", len(set(chunk.get('metadata', {}).get('source_file', 'Desconocido') for chunk in chunks)))
    
    with col2:
        st.metric("📝 Total Chunks", len(chunks))
    
    with col3:
        total_chars = sum(len(chunk.get('content', '')) for chunk in chunks)
        st.metric("🔤 Total Caracteres", f"{total_chars:,}")
    
    with col4:
        avg_chunk_size = total_chars // len(chunks) if chunks else 0
        st.metric("📏 Tamaño Promedio Chunk", avg_chunk_size)
    
    # Distribución por fuente
    st.subheader("📊 Distribución por Documento")
    
    source_stats = defaultdict(lambda: {'chunks': 0, 'characters': 0})
    for chunk in chunks:
        source = chunk.get('metadata', {}).get('source_file', 'Desconocido')
        source_stats[source]['chunks'] += 1
        source_stats[source]['characters'] += len(chunk.get('content', ''))
    
    # Crear DataFrame para visualización
    df_sources = pd.DataFrame([
        {
            'Documento': source,
            'Chunks': stats['chunks'],
            'Caracteres': stats['characters'],
            '% del Total': (stats['characters'] / total_chars) * 100 if total_chars > 0 else 0
        }
        for source, stats in source_stats.items()
    ])
    
    # Gráfico de barras
    fig = px.bar(
        df_sources,
        x='Documento',
        y='Caracteres',
        title="Distribución de Contenido por Documento",
        color='% del Total',
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de estadísticas
    st.dataframe(df_sources, use_container_width=True)

def render_advanced_themes(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Análisis avanzado de temas"""
    st.header("🎯 Análisis Avanzado de Temas")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Configuración
    col1, col2 = st.columns([3, 1])
    
    with col1:
        n_topics = st.slider("Número de temas a extraer:", 3, 15, 8)
    
    with col2:
        if st.button("🔄 Analizar Temas", type="primary"):
            if 'theme_analysis_cache' in st.session_state:
                del st.session_state.theme_analysis_cache
    
    # Realizar análisis
    if 'theme_analysis_cache' not in st.session_state:
        with st.spinner("Analizando temas..."):
            st.session_state.theme_analysis_cache = analyzer.extract_advanced_themes(chunks, n_topics)
    
    theme_analysis = st.session_state.theme_analysis_cache
    
    if 'topics' in theme_analysis:
        st.subheader("📊 Temas Identificados")
        
        # Mostrar temas en tarjetas
        for i, topic in enumerate(theme_analysis['topics']):
            with st.expander(f"**Tema {i+1}** - Coherencia: {topic.get('coherence', 0):.3f}"):
                st.write("**Palabras clave:**")
                st.write(", ".join(topic['words'][:8]))
                st.write(f"**Peso:** {topic['weight']:.3f}")
        
        # Visualización de temas
        if len(theme_analysis['topics']) > 1:
            # Preparar datos para visualización
            topic_data = []
            for topic in theme_analysis['topics']:
                topic_data.append({
                    'Tema': f"Tema {topic['id']+1}",
                    'Peso': topic['weight'],
                    'Coherencia': topic.get('coherence', 0),
                    'Palabras': ', '.join(topic['words'][:5])
                })
            
            df_topics = pd.DataFrame(topic_data)
            
            # Gráfico de burbujas
            fig = px.scatter(
                df_topics,
                x='Peso',
                y='Coherencia',
                size='Peso',
                hover_data=['Palabras'],
                title="Temas por Peso y Coherencia",
                labels={'Peso': 'Peso del Tema', 'Coherencia': 'Coherencia del Tema'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif 'themes' in theme_analysis:
        # Análisis básico
        st.subheader("📊 Temas Frecuentes")
        
        themes_data = [{'Tema': theme, 'Frecuencia': freq} 
                      for theme, freq in theme_analysis['themes'].items()]
        
        df_themes = pd.DataFrame(themes_data)
        
        fig = px.bar(
            df_themes.head(15),
            x='Frecuencia',
            y='Tema',
            orientation='h',
            title="Top 15 Temas por Frecuencia"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_clustering_analysis(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Análisis de clustering"""
    st.header("🔍 Análisis de Clustering")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    if not ADVANCED_ANALYSIS_AVAILABLE:
        st.error("Las funcionalidades de clustering requieren dependencias adicionales.")
        st.code("pip install scikit-learn")
        return
    
    # Configuración
    col1, col2 = st.columns([3, 1])
    
    with col1:
        n_clusters = st.slider("Número de clusters:", 2, min(10, len(chunks)//2), 5)
    
    with col2:
        if st.button("🔄 Ejecutar Clustering", type="primary"):
            if 'clustering_cache' in st.session_state:
                del st.session_state.clustering_cache
    
    # Realizar clustering
    if 'clustering_cache' not in st.session_state:
        with st.spinner("Ejecutando análisis de clustering..."):
            st.session_state.clustering_cache = analyzer.perform_clustering(chunks, n_clusters)
    
    clustering_result = st.session_state.clustering_cache
    
    if 'error' in clustering_result:
        st.error(f"Error en clustering: {clustering_result['error']}")
        return
    
    # Mostrar resultados
    st.subheader("📊 Resultados del Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Clusters", clustering_result['n_clusters'])
    
    with col2:
        method = clustering_result.get('method', 'K-means')
        st.metric("🔧 Método", method)
    
    # Visualización de clustering si hay embeddings
    if 'embeddings' in clustering_result and 'labels' in clustering_result:
        st.subheader("📊 Visualización de Clusters")
        
        embeddings = np.array(clustering_result['embeddings'])
        labels = clustering_result['labels']
        
        # Crear DataFrame para Plotly
        df_cluster = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'cluster': [f'Cluster {label}' for label in labels],
            'documento': [f'Doc {i+1}' for i in range(len(labels))]
        })
        
        fig = px.scatter(
            df_cluster,
            x='x', y='y',
            color='cluster',
            hover_data=['documento'],
            title="Distribución de Documentos en Clusters"
        )
        fig.update_layout(
            xaxis_title="Dimensión 1",
            yaxis_title="Dimensión 2"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detalles de clusters
    st.subheader("🔍 Detalles de Clusters")
    
    for cluster_id, cluster_docs in clustering_result['clusters'].items():
        with st.expander(f"**Cluster {cluster_id}** ({len(cluster_docs)} documentos)"):
            # Documentos en el cluster
            st.write("**Documentos:**")
            for doc in cluster_docs[:5]:  # Mostrar solo los primeros 5
                st.write(f"- **{doc['source']}**: {doc['text']}")
                
            if len(cluster_docs) > 5:
                st.write(f"... y {len(cluster_docs) - 5} documentos más")

def render_advanced_concept_map(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Mapa conceptual avanzado"""
    st.header("🗺️ Mapa Conceptual Avanzado")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Generar mapa
    if 'concept_map_cache' not in st.session_state:
        with st.spinner("Generando mapa conceptual avanzado..."):
            st.session_state.concept_map_cache = analyzer.generate_advanced_concept_map(chunks)
    
    map_data = st.session_state.concept_map_cache
    
    if 'error' in map_data:
        st.error(f"Error generando mapa: {map_data['error']}")
        return
    
    if not map_data["nodes"]:
        st.warning("No se pudieron generar datos para el mapa conceptual.")
        return
    
    # Estadísticas del mapa
    stats = map_data.get('stats', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔗 Nodos", stats.get('total_nodes', 0))
    
    with col2:
        st.metric("↔️ Conexiones", stats.get('total_edges', 0))
    
    with col3:
        density = (stats.get('total_edges', 0) * 2) / (stats.get('total_nodes', 1) * (stats.get('total_nodes', 1) - 1)) if stats.get('total_nodes', 0) > 1 else 0
        st.metric("📊 Densidad", f"{density:.2%}")
    
    # Crear visualización
    G = nx.Graph()
    
    # Agregar nodos
    for node in map_data["nodes"]:
        G.add_node(node["id"], **node)
    
    # Agregar aristas
    for edge in map_data["edges"]:
        G.add_edge(edge["from"], edge["to"], **edge)
    
    # Calcular layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.random_layout(G, seed=42)
    
    # Preparar trazas
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        text=[],
        textposition="middle center",
        marker=dict(
            size=[],
            color=[],
            line=dict(width=2)
        )
    )
    
    # Aristas
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    # Nodos
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node[1].get('label', node[0])])
        node_trace['marker']['size'] += tuple([node[1].get('size', 20)])
        node_trace['marker']['color'] += tuple([node[1].get('color', '#888')])
    
    # Crear figura
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Mapa Conceptual Avanzado',
            title_font_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Red de conceptos y similitudes entre documentos",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="#888", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_sentiment_analysis(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Análisis de sentimientos avanzado"""
    st.header("😊 Análisis de Sentimientos Avanzado")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Realizar análisis
    if 'sentiment_cache' not in st.session_state:
        with st.spinner("Analizando sentimientos..."):
            st.session_state.sentiment_cache = analyzer.advanced_sentiment_analysis(chunks)
    
    sentiment_result = st.session_state.sentiment_cache
    
    if 'error' in sentiment_result:
        st.error(f"Error en análisis de sentimientos: {sentiment_result['error']}")
        return
    
    # Mostrar método utilizado
    st.info(f"**Método de análisis:** {sentiment_result.get('method', 'Desconocido')}")
    
    # Estadísticas generales
    if 'overall_stats' in sentiment_result:
        st.subheader("📊 Estadísticas Generales")
        
        stats = sentiment_result['overall_stats']
        distribution = stats.get('distribution', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("😊 Positivo", distribution.get('positive', 0))
        
        with col2:
            st.metric("😐 Neutral", distribution.get('neutral', 0))
        
        with col3:
            st.metric("😞 Negativo", distribution.get('negative', 0))
        
        with col4:
            mean_score = stats.get('mean_score', 0)
            st.metric("📈 Score Promedio", f"{mean_score:.3f}")
        
        # Gráfico de distribución
        if distribution:
            fig = px.pie(
                values=list(distribution.values()),
                names=list(distribution.keys()),
                title="Distribución de Sentimientos",
                color_discrete_map={
                    'positive': '#2ecc71',
                    'neutral': '#95a5a6',
                    'negative': '#e74c3c'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Análisis por fuente
    st.subheader("📋 Análisis por Documento")
    
    source_data = []
    for source, data in sentiment_result['by_source'].items():
        total = data['positive'] + data['neutral'] + data['negative']
        if total > 0:
            avg_score = np.mean(data.get('scores', [0])) if data.get('scores') else 0
            source_data.append({
                'Documento': source,
                'Positivo': data['positive'],
                'Neutral': data['neutral'],
                'Negativo': data['negative'],
                'Total': total,
                'Score Promedio': avg_score,
                '% Positivo': (data['positive'] / total) * 100
            })
    
    if source_data:
        df_sentiment = pd.DataFrame(source_data)
        st.dataframe(df_sentiment, use_container_width=True)
        
        # Gráfico comparativo
        fig = px.bar(
            df_sentiment,
            x='Documento',
            y=['Positivo', 'Neutral', 'Negativo'],
            title="Sentimientos por Documento",
            color_discrete_map={
                'Positivo': '#2ecc71',
                'Neutral': '#95a5a6',
                'Negativo': '#e74c3c'
            }
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def render_word_cloud(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Nube de palabras"""
    st.header("☁️ Nube de Palabras")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    if not ADVANCED_ANALYSIS_AVAILABLE:
        st.error("La nube de palabras requiere dependencias adicionales.")
        st.code("pip install wordcloud matplotlib")
        return
    
    # Filtro por fuente
    sources = ['Todos'] + list(set(chunk.get('metadata', {}).get('source_file', 'Desconocido') for chunk in chunks))
    selected_source = st.selectbox("Filtrar por documento:", sources)
    
    source_filter = None if selected_source == 'Todos' else selected_source
    
    if st.button("🎨 Generar Nube de Palabras", type="primary"):
        with st.spinner("Generando nube de palabras..."):
            wordcloud_path = analyzer.generate_word_cloud(chunks, source_filter)
            
            if wordcloud_path and Path(wordcloud_path).exists():
                st.image(wordcloud_path, caption=f"Nube de palabras - {selected_source}")
                
                # Limpiar archivo temporal después de un tiempo
                try:
                    Path(wordcloud_path).unlink()
                except:
                    pass
            else:
                st.error("No se pudo generar la nube de palabras.")

def render_settings_tab(analyzer: AdvancedQualitativeAnalyzer):
    """Pestaña de configuración del análisis cualitativo"""
    st.header("⚙️ Configuración del Análisis")
    
    st.subheader("🎛️ Parámetros de Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Configuración de Temas")
        
        n_topics = st.slider(
            "Número de temas a extraer",
            min_value=3,
            max_value=20,
            value=st.session_state.get('qualitative_n_topics', 10),
            help="Número de temas principales a identificar en el análisis LDA"
        )
        st.session_state.qualitative_n_topics = n_topics
        
        min_topic_words = st.slider(
            "Palabras mínimas por tema",
            min_value=5,
            max_value=20,
            value=st.session_state.get('qualitative_min_topic_words', 10),
            help="Número mínimo de palabras representativas por tema"
        )
        st.session_state.qualitative_min_topic_words = min_topic_words
    
    with col2:
        st.markdown("### 🔍 Configuración de Clustering")
        
        n_clusters = st.slider(
            "Número de clusters",
            min_value=2,
            max_value=15,
            value=st.session_state.get('qualitative_n_clusters', 5),
            help="Número de grupos para el análisis de clustering"
        )
        st.session_state.qualitative_n_clusters = n_clusters
        
        clustering_method = st.selectbox(
            "Método de clustering",
            options=['K-means', 'DBSCAN'],
            index=0,
            help="Algoritmo de clustering a utilizar"
        )
        st.session_state.qualitative_clustering_method = clustering_method
    
    st.divider()
    
    # Configuración de cache
    st.subheader("💾 Gestión de Cache")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Limpiar Cache de Análisis", type="secondary"):
            # Limpiar cache específico del análisis cualitativo
            cache_keys = [key for key in st.session_state.keys() if key.startswith('qualitative_') or key.endswith('_cache')]
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Cache de análisis limpiado")
            st.rerun()
    
    with col2:
        if st.button("📊 Recargar Datos RAG", type="secondary"):
            if 'qualitative_chunks' in st.session_state:
                del st.session_state['qualitative_chunks']
            st.success("✅ Datos RAG recargados")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Limpiar Todo", type="secondary"):
            # Limpiar todo el cache relacionado
            cache_keys = [key for key in st.session_state.keys() if 'qualitative' in key or 'cache' in key]
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Todo el cache limpiado")
            st.rerun()
    
    st.divider()
    
    # Información del sistema
    st.subheader("ℹ️ Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Estado de Dependencias:**")
        if ADVANCED_ANALYSIS_AVAILABLE:
            st.success("✅ Análisis avanzado disponible")
            st.info("📦 Dependencias instaladas: scikit-learn, nltk, textblob, wordcloud")
        else:
            st.error("❌ Análisis avanzado no disponible")
            st.warning("⚠️ Instala las dependencias adicionales")
    
    with col2:
        st.markdown("**Cache del Sistema:**")
        try:
            cache_path = analyzer.cache_path
            if cache_path.exists():
                cache_size = cache_path.stat().st_size
                st.info(f"📁 Tamaño del cache: {cache_size:,} bytes")
                
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if 'chunks' in cache_data:
                        total_chunks = sum(len(chunk_list) for chunk_list in cache_data['chunks'].values())
                        st.info(f"📄 Chunks en cache: {total_chunks}")
            else:
                st.warning("⚠️ No hay cache disponible")
        except Exception as e:
            st.error(f"❌ Error leyendo cache: {e}")
    
    st.divider()
    
    # Configuración de exportación
    st.subheader("📤 Exportación de Resultados")
    
    export_format = st.selectbox(
        "Formato de exportación",
        options=['JSON', 'CSV', 'Excel'],
        help="Formato para exportar los resultados del análisis"
    )
    
    if st.button("💾 Exportar Análisis Actual", type="primary"):
        st.info("🚧 Funcionalidad de exportación en desarrollo")

def render_interactive_concept_map(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar mapa conceptual interactivo con PyVis"""
    st.header("🗺️ Mapa Conceptual Interactivo")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    if not PYVIS_AVAILABLE:
        st.error("PyVis no está disponible. Instala con: pip install pyvis")
        return
    
    # Configuración simplificada
    col1, col2 = st.columns([3, 1])
    
    with col1:
        layout_type = st.selectbox(
            "Tipo de Layout:",
            ["spring", "circular", "random", "shell"],
            index=0
        )
    
    with col2:
        if st.button("🔄 Generar Mapa", type="primary"):
            if 'concept_map_html' in st.session_state:
                del st.session_state.concept_map_html
    
    # Usar todos los chunks disponibles
    selected_chunks = chunks
    
    # Generar mapa
    if 'concept_map_html' not in st.session_state:
        with st.spinner("Generando mapa conceptual interactivo..."):
            html_file = analyzer.create_interactive_concept_map(selected_chunks, layout_type)
            st.session_state.concept_map_html = html_file
    
    html_file = st.session_state.concept_map_html
    
    if html_file and os.path.exists(html_file):
        # Leer y mostrar el HTML
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Usar un contenedor expandible para evitar cortes
        with st.container():
            st.components.v1.html(html_content, height=800, scrolling=True)
        
        # Información adicional
        st.info("💡 **Instrucciones:** Puedes hacer clic y arrastrar los nodos, hacer zoom, y pasar el cursor sobre los elementos para ver más información.")
        
        # Botón para descargar
        with open(html_file, 'rb') as f:
            st.download_button(
                label="📥 Descargar Mapa HTML",
                data=f.read(),
                file_name="mapa_conceptual.html",
                mime="text/html"
            )
    else:
        st.error("No se pudo generar el mapa conceptual.")

def render_interactive_mind_map(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar mapa mental interactivo con mejor espaciado y visualización"""
    st.header("🧠 Mapa Mental Interactivo")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    if not AGRAPH_AVAILABLE:
        st.error("streamlit-agraph no está disponible. Instala con: pip install streamlit-agraph")
        return
    
    # Configuración simplificada
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        node_spacing = st.slider(
            "Espaciado entre Nodos:",
            min_value=100,
            max_value=500,
            value=250,
            step=50,
            help="Controla la distancia entre los nodos del mapa mental"
        )
    
    with col2:
        physics_strength = st.slider(
            "Fuerza de Física:",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Controla la intensidad de las fuerzas físicas"
        )
    
    with col3:
        if st.button("🔄 Regenerar Mapa", type="primary"):
            if 'mind_map_data' in st.session_state:
                del st.session_state.mind_map_data
    
    # Usar todos los chunks disponibles
    selected_chunks = chunks
    
    # Generar mapa mental
    if 'mind_map_data' not in st.session_state:
        with st.spinner("Generando mapa mental interactivo..."):
            try:
                mind_map_data = analyzer.create_interactive_mind_map(
                    selected_chunks, 
                    node_spacing=node_spacing,
                    return_data=True
                )
                st.session_state.mind_map_data = mind_map_data
            except Exception as e:
                st.error(f"Error generando mapa mental: {e}")
                return
    
    mind_map_data = st.session_state.mind_map_data
    
    if mind_map_data and 'nodes' in mind_map_data and 'edges' in mind_map_data:
        # Mostrar estadísticas
        if 'stats' in mind_map_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Nodos", mind_map_data['stats']['total_nodes'])
            with col2:
                st.metric("🔗 Conexiones", mind_map_data['stats']['total_edges'])
            with col3:
                st.metric("🎯 Ramas Principales", mind_map_data['stats'].get('main_branches', 0))
            with col4:
                st.metric("💡 Conceptos Detallados", mind_map_data['stats'].get('detailed_concepts', 0))
        
        # Verificar disponibilidad de streamlit-agraph
        try:
            
            # Crear nodos con mejor configuración visual
            nodes = []
            for node_data in mind_map_data['nodes']:
                node = Node(
                    id=node_data['id'],
                    label=node_data['label'],
                    size=node_data.get('size', 20),
                    color=node_data.get('color', '#97C2FC'),
                    font={'size': node_data.get('font_size', 14), 'color': 'black'},
                    borderWidth=2,
                    shadow=True
                )
                nodes.append(node)
            
            # Crear aristas con mejor configuración
            edges = []
            for edge_data in mind_map_data['edges']:
                edge = Edge(
                    source=edge_data['from'],
                    target=edge_data['to'],
                    label=edge_data.get('label', ''),
                    color=edge_data.get('color', '#848484'),
                    width=edge_data.get('width', 2),
                    smooth=True
                )
                edges.append(edge)
            
            # Configuración mejorada del grafo
            config = Config(
                width=1000,
                height=700,
                directed=False,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=False,
                node={
                    'labelProperty': 'label',
                    'size': 300,
                    'highlightStrokeColor': 'blue',
                    'fontSize': 14,
                    'fontColor': 'black'
                },
                link={
                    'labelProperty': 'label',
                    'renderLabel': True,
                    'fontSize': 12,
                    'fontColor': 'black'
                },
                d3={
                    'alphaTarget': 0.05,
                    'gravity': -physics_strength * 100,
                    'linkDistance': node_spacing,
                    'linkStrength': physics_strength,
                    'disableLinkForce': False
                }
            )
            
            # Renderizar el grafo en un contenedor expandible
            with st.container():
                return_value = agraph(nodes=nodes, edges=edges, config=config)
            
            # Información adicional
            st.info("💡 **Instrucciones:** Haz clic en los nodos para explorar, arrastra para reorganizar, y usa la rueda del mouse para hacer zoom.")
            
        except ImportError:
            st.error("streamlit-agraph no está disponible. Instala con: pip install streamlit-agraph")
            return
        except Exception as e:
            st.error(f"Error renderizando el mapa mental: {e}")
            return
        
        # Opciones de descarga mejoradas
        st.subheader("📥 Opciones de Descarga")
        col1, col2 = st.columns(2)
        
        with col1:
            # Descargar datos JSON
            import json
            json_data = json.dumps(mind_map_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📊 Descargar Datos JSON",
                data=json_data,
                file_name=f"mapa_mental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Descarga los datos del mapa mental en formato JSON"
            )
        
        with col2:
            # Generar HTML básico para descarga
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Mapa Mental Interactivo</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                    .header {{ text-align: center; margin-bottom: 20px; }}
                    .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                    .stat {{ text-align: center; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .data {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🧠 Mapa Mental Interactivo</h1>
                    <p>Generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <h3>📊 Nodos</h3>
                        <p>{mind_map_data['stats']['total_nodes']}</p>
                    </div>
                    <div class="stat">
                        <h3>🔗 Conexiones</h3>
                        <p>{mind_map_data['stats']['total_edges']}</p>
                    </div>
                    <div class="stat">
                        <h3>🎯 Ramas</h3>
                        <p>{mind_map_data['stats'].get('main_branches', 0)}</p>
                    </div>
                </div>
                
                <div class="data">
                    <h2>Datos del Mapa Mental</h2>
                    <pre>{json.dumps(mind_map_data, indent=2, ensure_ascii=False)}</pre>
                </div>
            </body>
            </html>
            """
            
            st.download_button(
                label="📄 Descargar Reporte HTML",
                data=html_content,
                file_name=f"reporte_mapa_mental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                help="Descarga un reporte HTML con los datos del mapa mental"
            )
    
    else:
        st.error("❌ No se pudo generar el mapa mental. Verifica que los datos sean válidos.")
        
        # Información de debug
        with st.expander("🔧 Información de Debug"):
            st.write("Intentando generar mapa mental...")
            try:
                test_structure = analyzer._analyze_intelligent_mind_map_structure(chunks[:5])
                st.write("Estructura del mapa mental:")
                st.json(test_structure)
            except Exception as e:
                st.write(f"Error analizando estructura: {e}")
                
            # Verificar dependencias
            try:
                agraph  # Verificar si está disponible
                st.write("✅ streamlit-agraph disponible")
            except ImportError:
                st.write("❌ streamlit-agraph no disponible")
                st.write("Instala con: pip install streamlit-agraph")

def render_automatic_summary(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar resumen automático"""
    st.header("📝 Resumen Automático")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Configuración
    col1, col2 = st.columns([3, 1])
    
    with col1:
        max_length = st.slider("Longitud máxima del resumen:", 200, 1000, 500, 50)
    
    with col2:
        if st.button("🔄 Generar Resumen", type="primary"):
            if 'auto_summary' in st.session_state:
                del st.session_state.auto_summary
    
    # Generar resumen
    if 'auto_summary' not in st.session_state:
        with st.spinner("Generando resumen automático..."):
            summary = analyzer.generate_rag_summary(chunks, max_length)
            st.session_state.auto_summary = summary
    
    summary = st.session_state.auto_summary
    
    # Mostrar resumen
    st.subheader("📄 Resumen Generado")
    st.write(summary)
    
    # Estadísticas del resumen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Palabras", len(summary.split()))
    
    with col2:
        st.metric("📝 Caracteres", len(summary))
    
    with col3:
        st.metric("📄 Oraciones", len([s for s in summary.split('.') if s.strip()]))
    
    # Botón para copiar
    st.code(summary, language=None)

def render_triangulation_analysis(analyzer: AdvancedQualitativeAnalyzer, chunks: List[Dict]):
    """Renderizar análisis de triangulación"""
    st.header("🔺 Análisis de Triangulación")
    
    if not chunks:
        st.warning("No hay datos disponibles.")
        return
    
    # Realizar triangulación
    if 'triangulation_data' not in st.session_state:
        with st.spinner("Realizando análisis de triangulación..."):
            triangulation = analyzer.perform_triangulation_analysis(chunks)
            st.session_state.triangulation_data = triangulation
    
    triangulation = st.session_state.triangulation_data
    
    if 'error' in triangulation:
        st.error(f"Error en triangulación: {triangulation['error']}")
        return
    
    # Estadísticas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Fuentes Totales", triangulation['total_sources'])
    
    with col2:
        st.metric("🔍 Conceptos Totales", triangulation['total_concepts'])
    
    with col3:
        st.metric("✅ Conceptos Validados", triangulation['validated_concepts'])
    
    with col4:
        validation_rate = (triangulation['validated_concepts'] / triangulation['total_concepts']) * 100 if triangulation['total_concepts'] > 0 else 0
        st.metric("📊 Tasa de Validación", f"{validation_rate:.1f}%")
    
    # Lista de fuentes
    st.subheader("📚 Fuentes Analizadas")
    for i, source in enumerate(triangulation['sources'], 1):
        st.write(f"{i}. {source}")
    
    # Conceptos triangulados
    st.subheader("🔺 Conceptos Triangulados")
    
    if triangulation['triangulated_concepts']:
        # Crear DataFrame para visualización
        concepts_data = []
        for concept in triangulation['triangulated_concepts']:
            concepts_data.append({
                'Concepto': concept['concept'],
                'Fuentes': concept['source_count'],
                'Confiabilidad': f"{concept['reliability']:.2%}",
                'Puntuación Promedio': f"{concept['avg_score']:.3f}",
                'Fuentes Específicas': ', '.join(concept['sources'][:3]) + ('...' if len(concept['sources']) > 3 else '')
            })
        
        df_concepts = pd.DataFrame(concepts_data)
        st.dataframe(df_concepts, use_container_width=True)
        
        # Gráfico de confiabilidad
        if len(concepts_data) > 1:
            fig = px.scatter(
                df_concepts.head(15),
                x='Fuentes',
                y='Puntuación Promedio',
                size='Fuentes',
                hover_data=['Concepto', 'Confiabilidad'],
                title="Conceptos por Fuentes y Puntuación"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No se encontraron conceptos que aparezcan en múltiples fuentes.")

def render():
    """Función principal para renderizar el módulo de análisis cualitativo avanzado"""
    st.title("🔬 Análisis Cualitativo Avanzado")
    st.markdown("*Análisis profundo de contenido RAG con técnicas de NLP y mapas interactivos*")
    
    # Verificar disponibilidad de funcionalidades avanzadas
    if not ADVANCED_ANALYSIS_AVAILABLE:
        st.warning("⚠️ Algunas funcionalidades avanzadas no están disponibles.")
        st.info("Para habilitar todas las funcionalidades, instala las dependencias adicionales:")
        st.code("pip install -r requirements.txt")
    
    # Inicializar analizador
    analyzer = AdvancedQualitativeAnalyzer()
    
    # Botón para refrescar datos
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Actualizar Datos", type="secondary"):
            # Limpiar cache de análisis cualitativo
            for key in list(st.session_state.keys()):
                if key.startswith(('qualitative_', 'theme_analysis_', 'clustering_', 'concept_map_', 
                                 'mind_map_', 'auto_summary', 'triangulation_', 'sentiment_', 'wordcloud_')):
                    del st.session_state[key]
            st.rerun()
    
    # Cargar datos del RAG
    if 'qualitative_chunks' not in st.session_state:
        with st.spinner("Cargando datos del sistema RAG..."):
            st.session_state.qualitative_chunks = analyzer.load_rag_data()
    
    chunks = st.session_state.qualitative_chunks
    
    if not chunks:
        st.warning("⚠️ No hay datos disponibles en el sistema RAG.")
        st.info("Asegúrate de haber procesado documentos en la pestaña 'Procesamiento RAG' primero.")
        return
    
    # Mostrar información de datos cargados
    with col1:
        st.info(f"📊 Datos cargados: {len(chunks)} chunks de {len(set(chunk.get('metadata', {}).get('source_file', 'Desconocido') for chunk in chunks))} documentos")
    
    # Pestañas del análisis cualitativo avanzado - ACTUALIZADAS
    tabs = st.tabs([
        "📊 Dashboard",
        "🎯 Temas Avanzados",
        "🔍 Clustering", 
        "🗺️ Mapa Conceptual",
        "🧠 Mapa Mental",
        "📝 Resumen Automático",
        "🔺 Triangulación",
        "😊 Sentimientos",
        "☁️ Nube de Palabras",
        "⚙️ Configuración"
    ])
    
    with tabs[0]:
        render_advanced_dashboard(analyzer, chunks)
    
    with tabs[1]:
        render_advanced_themes(analyzer, chunks)
    
    with tabs[2]:
        render_clustering_analysis(analyzer, chunks)
    
    with tabs[3]:
        render_interactive_concept_map(analyzer, chunks)
    
    with tabs[4]:
        render_interactive_mind_map(analyzer, chunks)
    
    with tabs[5]:
        render_automatic_summary(analyzer, chunks)
    
    with tabs[6]:
        render_triangulation_analysis(analyzer, chunks)
    
    with tabs[7]:
        render_sentiment_analysis(analyzer, chunks)
    
    with tabs[8]:
        render_word_cloud(analyzer, chunks)
    
    with tabs[9]:
        render_settings_tab(analyzer)

if __name__ == "__main__":
    render()