"""
Asistente de Codificación Cualitativa

Objetivo: Sugerir códigos iniciales con fundamentación (citas) y permitir
refinamiento con LLM para nombres/definiciones/ejemplos coherentes y relaciones (merge/split sugeridos).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.config import AnalysisConfig
from ..core.citation_manager import CitationManager, Citation
from utils.ollama_client import ollama_client


@dataclass
class CodeSuggestion:
    code: str
    """Nombre del código sugerido"""

    definition: str
    """Definición breve del código"""

    examples: List[str] = field(default_factory=list)
    """Ejemplos breves/fragmentos explicativos (pueden incluir Explicación)"""

    citations: List[Citation] = field(default_factory=list)
    """Citas de respaldo a fragmentos originales"""

    confidence: float = 0.5
    """Confianza agregada del sistema (0-1)"""

    clusters: Optional[str] = None
    """Etiqueta de cluster/comunidad si aplica"""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'definition': self.definition,
            'examples': self.examples,
            'num_citations': len(self.citations),
            'confidence': self.confidence,
            'clusters': self.clusters,
            'timestamp': self.timestamp
        }


class QualitativeCodingAssistant:
    """
    Genera candidatos de códigos basados en n-gramas/TF-IDF y los enriquece con LLM.
    Integra un sistema de citación para trazabilidad y ejemplos.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.citation_manager = CitationManager()

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _candidates_with_tfidf(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("🔍 [CodingAssistant] Iniciando generación de candidatos (TF-IDF / frecuencia)...")
        if not SKLEARN_AVAILABLE:
            # Fallback simple: términos frecuentes (palabras de 3+ letras)
            print("⚠️ [CodingAssistant] scikit-learn no disponible, usando fallback por frecuencia")
            freq = Counter()
            for ch in chunks:
                words = re.findall(r'\b[a-záéíóúñü]{3,}\b', self._preprocess(ch.get('content', '')))
                freq.update(words)
            common = [w for w, _ in freq.most_common(self.config.max_code_suggestions * 3)]
            print(f"✅ [CodingAssistant] Candidatos por frecuencia: {len(common)} términos")
            return [{'term': t, 'score': 1.0} for t in common]

        nmin, nmax = self.config.coding_ngram_range if self.config.coding_use_ngrams else (1, 1)
        texts = [self._preprocess(ch.get('content', '')) for ch in chunks]
        if not texts:
            print("⚠️ [CodingAssistant] No hay textos para vectorizar")
            return []

        vectorizer = TfidfVectorizer(
            max_features=self.config.max_code_suggestions * 4,
            ngram_range=(nmin, nmax),
            min_df=2,
            max_df=0.9,
            token_pattern=r'(?u)\b[a-záéíóúñü]+\b'
        )

        print(f"🧮 [CodingAssistant] Vectorizando textos con TF-IDF (n-gramas {nmin}-{nmax})...")
        matrix = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
        scores = matrix.max(axis=0).toarray().flatten()

        items = [{'term': t, 'score': float(s)} for t, s in zip(terms, scores) if s > 0]
        items.sort(key=lambda x: x['score'], reverse=True)
        print(f"✅ [CodingAssistant] Candidatos TF-IDF generados: {len(items)}")
        return items[: self.config.max_code_suggestions * 2]

    def _attach_citations(self, term: str, chunks: List[Dict[str, Any]], max_cites: int = 5) -> Dict[str, Any]:
        # Nota: se invoca por término; evitamos logs ruidosos, pero contamos totales
        citations: List[Citation] = []
        examples: List[str] = []
        term_l = term.lower()
        seen = 0
        for idx, ch in enumerate(chunks):
            content = ch.get('content', '')
            meta = ch.get('metadata', {})
            source = meta.get('source_file', 'unknown')
            pos = content.lower().find(term_l)
            if pos != -1:
                start = max(0, pos - self.config.citation_context_chars)
                end = min(len(content), pos + len(term) + self.config.citation_context_chars)
                before = content[start:pos]
                exact = content[pos:pos + len(term)]
                after = content[pos + len(term):end]
                if self.config.enable_citations:
                    c = self.citation_manager.add_citation(
                        source_file=source,
                        chunk_id=idx,
                        content=exact,
                        context_before=before,
                        context_after=after,
                        page_number=meta.get('page_number'),
                        relevance_score=0.5
                    )
                    citations.append(c)
                examples.append(f"...{before}[{exact}]{after}...")
                seen += 1
                if seen >= max_cites:
                    break
        return { 'citations': citations, 'examples': examples[:3] }

    def _refine_with_llm(self, raw_terms: List[str], document_context: str) -> List[Dict[str, str]]:
        if not self.config.enable_coding_llm_refinement or not raw_terms:
            if not self.config.enable_coding_llm_refinement:
                print("ℹ️ [CodingAssistant] Refinamiento LLM desactivado, usando términos crudos")
            else:
                print("⚠️ [CodingAssistant] No hay términos crudos para refinar")
            return [{'code': t, 'definition': '', 'example': ''} for t in raw_terms[: self.config.max_code_suggestions]]

        preview = document_context[:2000] + "..." if len(document_context) > 2000 else document_context
        print(f"🤖 [CodingAssistant] Enviando {min(len(raw_terms), self.config.max_code_suggestions)} términos al LLM {self.config.coding_llm_model}...")
        prompt = f"""Eres un metodólogo experto en análisis cualitativo.
Tu tarea: convertir términos candidatos en un LIBRO DE CÓDIGOS INICIAL con nombres claros, definiciones 
operacionales y ejemplos sintéticos breves. Responde SOLO en JSON y en español.

CONTEXTO:
{preview}

TÉRMINOS CANDIDATOS:
{', '.join(raw_terms)}

REQUISITOS:
1) Cada código debe tener: nombre (2-4 palabras), definición (1-2 frases, operacional, no genérica), ejemplo (1 frase breve y coherente)
2) Los ejemplos deben ser sintéticos y representar claramente el fenómeno o concepto del código
3) Nombres claros y no redundantes; agrupa sinónimos, separa conceptos distintos
4) Enfoca procesos, experiencias, relaciones o fenómenos
5) Devuelve máximo {self.config.max_code_suggestions} códigos

FORMATO JSON ESTRICTO:
{{
  "codigos": [
    {{"nombre": "...", "definicion": "...", "ejemplo": "..."}}
  ]
}}
"""

        response = ollama_client.generate_response(
            model=self.config.coding_llm_model,
            prompt=prompt,
            format_json=True,
            temperature=self.config.coding_llm_temperature,
            seed=42
        )

        import json
        try:
            cleaned = self._clean_json_response(response)
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            json_str = cleaned[start:end]
            if not self._is_json_complete(json_str):
                json_str = self._repair_truncated_json(json_str)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Intento agresivo de limpieza
                json_str = self._aggressive_json_clean(json_str)
                data = json.loads(json_str)
            out: List[Dict[str, str]] = []
            for it in data.get('codigos', []):
                name = it.get('nombre') or ''
                defin = it.get('definicion') or ''
                example = it.get('ejemplo') or ''
                if len(name) > 3:
                    out.append({'code': name, 'definition': defin, 'example': example})
            print(f"✅ [CodingAssistant] LLM devolvió {len(out)} códigos refinados con ejemplos")
            return out[: self.config.max_code_suggestions]
        except Exception:
            print("⚠️ [CodingAssistant] Error parseando respuesta del LLM; usando términos crudos")
            return [{'code': t, 'definition': '', 'example': ''} for t in raw_terms[: self.config.max_code_suggestions]]

    def _clean_json_response(self, response: str) -> str:
        import re
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', response)
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        cleaned = re.sub(r' +', ' ', cleaned)
        return cleaned

    def _is_json_complete(self, json_str: str) -> bool:
        return (
            json_str.count('{') == json_str.count('}') and
            json_str.count('[') == json_str.count(']') and
            json_str.strip().endswith(('}', ']'))
        )

    def _repair_truncated_json(self, json_str: str) -> str:
        # Cierra array/objeto principal de forma segura
        if not json_str.strip().endswith('}'):  # esperamos objeto raíz
            last_complete = json_str.rfind('},')
            if last_complete != -1:
                json_str = json_str[: last_complete + 1] + '\n]}' if json_str.strip().endswith(']') else json_str[: last_complete + 1] + '\n}'
        return json_str

    def _aggressive_json_clean(self, json_str: str) -> str:
        import re
        # Escapar saltos de línea y comillas dentro de strings
        def esc(match):
            s = match.group(1)
            s = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
            s = re.sub(r'[^\x20-\x7E]', '', s)
            return '"' + s + '"'
        json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', esc, json_str)
        # Arreglar comas duplicadas y comas antes de cierre
        json_str = re.sub(r',\s*,', ',', json_str)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        return json_str

    def suggest_codes(self, chunks: List[Dict[str, Any]]) -> List[CodeSuggestion]:
        print("🚀 [CodingAssistant] Inicio del asistente de codificación")
        if not chunks:
            print("⚠️ [CodingAssistant] No se recibieron chunks")
            return []

        self.citation_manager.clear()

        # Candidatos iniciales
        items = self._candidates_with_tfidf(chunks)
        raw_terms = [it['term'] for it in items[: max(10, self.config.max_code_suggestions * 2)]]
        print(f"📎 [CodingAssistant] Términos crudos seleccionados: {len(raw_terms)}")

        # Contexto para el LLM
        doc_ctx = " ".join(self._preprocess(ch.get('content', '')) for ch in chunks)

        refined = self._refine_with_llm(raw_terms, doc_ctx)

        suggestions: List[CodeSuggestion] = []
        for r in refined:
            cites = self._attach_citations(r['code'], chunks)
            # Fallback: si el nombre refinado no aparece literal, intenta con su primera palabra sustantiva
            if not cites['citations']:
                tokens = re.findall(r'\b[a-záéíóúñü]{3,}\b', r['code'].lower())
                if tokens:
                    fallback_term = tokens[0]
                    alt_cites = self._attach_citations(fallback_term, chunks)
                    if alt_cites['citations']:
                        print(f"🔄 [CodingAssistant] Fallback de citación usando '{fallback_term}' para '{r['code']}'")
                        cites = alt_cites
            
            # Combinar ejemplo del LLM (si existe) con ejemplos de citaciones
            examples = []
            llm_example = r.get('example', '').strip()
            if llm_example:
                examples.append(llm_example)
            # Agregar ejemplos de citaciones (máximo 2 adicionales)
            examples.extend(cites['examples'][:2])
            
            suggestion = CodeSuggestion(
                code=r['code'],
                definition=r.get('definition', ''),
                examples=examples,
                citations=cites['citations'],
                confidence=0.6
            )
            suggestions.append(suggestion)

        # Orden básico por confianza/longitud del nombre
        suggestions.sort(key=lambda s: (s.confidence, len(s.code)), reverse=True)
        final = suggestions[: self.config.max_code_suggestions]
        total_citations = sum(len(s.citations) for s in final)
        print(f"🏁 [CodingAssistant] Sugerencias listas: {len(final)} códigos, {total_citations} citas totales")
        return final

    def export_codebook_csv(self, suggestions: List[CodeSuggestion]) -> str:
        import csv
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        # Formato compatible básico con NVivo/Atlas.ti (nombre, definición, ejemplo)
        writer.writerow(["Code", "Definition", "Example"])
        for s in suggestions:
            example = ''
            for ex in s.examples:
                if ex.strip():
                    example = ex.strip()
                    break
            writer.writerow([s.code, s.definition, example])
        return buf.getvalue()


