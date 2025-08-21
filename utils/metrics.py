"""
Sistema de métricas y monitoreo para CogniChat
Recolección y análisis de métricas de rendimiento
"""

import time
import psutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger()

@dataclass
class SystemMetrics:
    """Métricas del sistema"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    process_count: int

@dataclass
class RAGMetrics:
    """Métricas específicas del sistema RAG"""
    timestamp: str
    documents_processed: int
    total_chunks: int
    average_chunk_size: int
    embeddings_generated: int
    queries_processed: int
    average_response_time: float

class MetricsCollector:
    """Recolector de métricas del sistema"""
    
    def __init__(self):
        self.metrics_file = config.CACHE_DIR / "system_metrics.json"
        self.rag_metrics_file = config.CACHE_DIR / "rag_metrics.json"
        self._ensure_directories()
        self.start_time = time.time()
    
    def _ensure_directories(self):
        """Asegurar que los directorios necesarios existan"""
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Recopilar métricas del sistema"""
        try:
            # Métricas de CPU y memoria
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                process_count=len(psutil.pids())
            )
            
            return metrics
        except Exception as e:
            logger.error(f"Error al recopilar métricas del sistema: {e}")
            return None
    
    def save_system_metrics(self, metrics: SystemMetrics):
        """Guardar métricas del sistema"""
        try:
            # Cargar métricas existentes
            existing_metrics = []
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
            
            # Agregar nueva métrica
            existing_metrics.append(asdict(metrics))
            
            # Mantener solo las últimas 1000 métricas
            if len(existing_metrics) > 1000:
                existing_metrics = existing_metrics[-1000:]
            
            # Guardar
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metrics, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error al guardar métricas del sistema: {e}")
    
    def get_system_metrics_history(self, hours: int = 24) -> List[Dict]:
        """Obtener historial de métricas del sistema"""
        try:
            if not self.metrics_file.exists():
                return []
            
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                all_metrics = json.load(f)
            
            # Filtrar por tiempo
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_metrics = []
            
            for metric in all_metrics:
                metric_time = datetime.fromisoformat(metric['timestamp'])
                if metric_time >= cutoff_time:
                    filtered_metrics.append(metric)
            
            return filtered_metrics
            
        except Exception as e:
            logger.error(f"Error al obtener historial de métricas: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de rendimiento"""
        try:
            metrics_history = self.get_system_metrics_history(24)
            
            if not metrics_history:
                return {}
            
            # Calcular promedios
            avg_cpu = sum(m['cpu_percent'] for m in metrics_history) / len(metrics_history)
            avg_memory = sum(m['memory_percent'] for m in metrics_history) / len(metrics_history)
            max_cpu = max(m['cpu_percent'] for m in metrics_history)
            max_memory = max(m['memory_percent'] for m in metrics_history)
            
            return {
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'avg_cpu_percent': round(avg_cpu, 2),
                'avg_memory_percent': round(avg_memory, 2),
                'max_cpu_percent': round(max_cpu, 2),
                'max_memory_percent': round(max_memory, 2),
                'metrics_count': len(metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Error al generar resumen de rendimiento: {e}")
            return {}
    
    def clear_metrics(self):
        """Limpiar todas las métricas"""
        try:
            if self.metrics_file.exists():
                self.metrics_file.unlink()
            if self.rag_metrics_file.exists():
                self.rag_metrics_file.unlink()
            logger.info("Métricas limpiadas exitosamente")
        except Exception as e:
            logger.error(f"Error al limpiar métricas: {e}")

# Instancia global
metrics_collector = MetricsCollector()