"""
Sistema de Gestión de Citas y Referencias
Asegura la trazabilidad y fundamentación de todos los resultados del análisis
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class Citation:
    """
    Representa una cita/referencia a una fuente original
    
    Este objeto garantiza que cada concepto, idea o dato extraído
    pueda rastrearse hasta su fuente original.
    """
    
    source_file: str
    """Nombre del archivo fuente"""
    
    chunk_id: int
    """ID del chunk donde se encontró la información"""
    
    content: str
    """Contenido exacto citado"""
    
    context_before: str
    """Contexto anterior (para entender mejor)"""
    
    context_after: str
    """Contexto posterior (para entender mejor)"""
    
    page_number: Optional[int] = None
    """Número de página (si está disponible)"""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """Momento en que se extrajo la cita"""
    
    relevance_score: float = 1.0
    """Score de relevancia (0.0 a 1.0)"""
    
    citation_id: str = field(init=False)
    """ID único de la cita"""
    
    def __post_init__(self):
        """Generar ID único basado en contenido y fuente"""
        citation_str = f"{self.source_file}:{self.chunk_id}:{self.content[:50]}"
        self.citation_id = hashlib.md5(citation_str.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            'citation_id': self.citation_id,
            'source_file': self.source_file,
            'chunk_id': self.chunk_id,
            'content': self.content,
            'context_before': self.context_before,
            'context_after': self.context_after,
            'page_number': self.page_number,
            'timestamp': self.timestamp,
            'relevance_score': self.relevance_score
        }
    
    def get_full_context(self, max_chars: int = 300) -> str:
        """
        Obtener contexto completo alrededor de la cita
        
        Args:
            max_chars: Máximo de caracteres a retornar
            
        Returns:
            Texto con contexto completo
        """
        full_text = f"{self.context_before} [{self.content}] {self.context_after}"
        
        if len(full_text) > max_chars:
            # Truncar manteniendo la cita en el centro
            half = max_chars // 2
            citation_start = len(self.context_before)
            citation_end = citation_start + len(self.content)
            
            start = max(0, citation_start - half)
            end = min(len(full_text), citation_end + half)
            
            truncated = full_text[start:end]
            if start > 0:
                truncated = "..." + truncated
            if end < len(full_text):
                truncated = truncated + "..."
            
            return truncated
        
        return full_text
    
    def format_citation(self, style: str = "academic") -> str:
        """
        Formatear cita según estilo especificado
        
        Args:
            style: Estilo de citación ("academic", "inline", "footnote")
            
        Returns:
            Cita formateada
        """
        if style == "academic":
            page_info = f", p. {self.page_number}" if self.page_number else ""
            return f"({self.source_file}{page_info})"
        
        elif style == "inline":
            return f"[{self.source_file}]"
        
        elif style == "footnote":
            return f"[{self.citation_id}] {self.source_file}"
        
        else:
            return self.source_file


class CitationManager:
    """
    Gestor centralizado de citas y referencias
    
    Este sistema garantiza que:
    1. Cada resultado tenga sus fuentes documentadas
    2. Se pueda rastrear de dónde viene cada información
    3. El investigador pueda verificar y validar los resultados
    """
    
    def __init__(self):
        self.citations: List[Citation] = []
        self.citation_index: Dict[str, Citation] = {}
    
    def add_citation(
        self,
        source_file: str,
        chunk_id: int,
        content: str,
        context_before: str = "",
        context_after: str = "",
        page_number: Optional[int] = None,
        relevance_score: float = 1.0
    ) -> Citation:
        """
        Agregar una nueva cita al sistema
        
        Args:
            source_file: Nombre del archivo fuente
            chunk_id: ID del chunk
            content: Contenido citado
            context_before: Contexto anterior
            context_after: Contexto posterior
            page_number: Número de página (opcional)
            relevance_score: Score de relevancia
            
        Returns:
            Objeto Citation creado
        """
        citation = Citation(
            source_file=source_file,
            chunk_id=chunk_id,
            content=content,
            context_before=context_before,
            context_after=context_after,
            page_number=page_number,
            relevance_score=relevance_score
        )
        
        # Evitar duplicados
        if citation.citation_id not in self.citation_index:
            self.citations.append(citation)
            self.citation_index[citation.citation_id] = citation
        
        return citation
    
    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Obtener cita por ID"""
        return self.citation_index.get(citation_id)
    
    def get_citations_by_source(self, source_file: str) -> List[Citation]:
        """Obtener todas las citas de una fuente específica"""
        return [c for c in self.citations if c.source_file == source_file]
    
    def get_top_citations(self, n: int = 10) -> List[Citation]:
        """Obtener las N citas más relevantes"""
        return sorted(self.citations, key=lambda c: c.relevance_score, reverse=True)[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas sobre las citas
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.citations:
            return {
                'total_citations': 0,
                'unique_sources': 0,
                'avg_relevance': 0.0
            }
        
        unique_sources = len(set(c.source_file for c in self.citations))
        avg_relevance = sum(c.relevance_score for c in self.citations) / len(self.citations)
        
        # Citas por fuente
        citations_by_source = {}
        for citation in self.citations:
            if citation.source_file not in citations_by_source:
                citations_by_source[citation.source_file] = 0
            citations_by_source[citation.source_file] += 1
        
        return {
            'total_citations': len(self.citations),
            'unique_sources': unique_sources,
            'avg_relevance': avg_relevance,
            'citations_by_source': citations_by_source,
            'most_cited_source': max(citations_by_source.items(), key=lambda x: x[1])[0] if citations_by_source else None
        }
    
    def generate_bibliography(self, style: str = "academic") -> List[str]:
        """
        Generar bibliografía completa
        
        Args:
            style: Estilo de citación
            
        Returns:
            Lista de referencias bibliográficas
        """
        # Agrupar por fuente
        sources = set(c.source_file for c in self.citations)
        
        bibliography = []
        for source in sorted(sources):
            source_citations = self.get_citations_by_source(source)
            count = len(source_citations)
            bibliography.append(f"{source} ({count} referencias)")
        
        return bibliography
    
    def export_citations(self) -> List[Dict[str, Any]]:
        """
        Exportar todas las citas como lista de diccionarios
        
        Returns:
            Lista de diccionarios con información de citas
        """
        return [c.to_dict() for c in self.citations]
    
    def clear(self):
        """Limpiar todas las citas"""
        self.citations.clear()
        self.citation_index.clear()
    
    def __len__(self) -> int:
        """Número total de citas"""
        return len(self.citations)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return f"CitationManager(citations={stats['total_citations']}, sources={stats['unique_sources']})"

