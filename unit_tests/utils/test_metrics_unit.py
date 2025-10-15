"""
Pruebas unitarias para el módulo metrics
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta
from dataclasses import asdict

from utils.metrics import SystemMetrics, RAGMetrics, MetricsCollector


class TestSystemMetrics:
    """Pruebas para la clase SystemMetrics"""
    
    def test_system_metrics_creation(self):
        """Probar creación de SystemMetrics"""
        timestamp = datetime.now().isoformat()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=45.5,
            memory_percent=60.2,
            memory_used_mb=2048.5,
            disk_usage_percent=75.0,
            process_count=150
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 60.2
        assert metrics.memory_used_mb == 2048.5
        assert metrics.disk_usage_percent == 75.0
        assert metrics.process_count == 150
    
    def test_system_metrics_to_dict(self):
        """Probar conversión de SystemMetrics a diccionario"""
        timestamp = datetime.now().isoformat()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=45.5,
            memory_percent=60.2,
            memory_used_mb=2048.5,
            disk_usage_percent=75.0,
            process_count=150
        )
        
        metrics_dict = asdict(metrics)
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['timestamp'] == timestamp
        assert metrics_dict['cpu_percent'] == 45.5
        assert metrics_dict['memory_percent'] == 60.2


class TestRAGMetrics:
    """Pruebas para la clase RAGMetrics"""
    
    def test_rag_metrics_creation(self):
        """Probar creación de RAGMetrics"""
        timestamp = datetime.now().isoformat()
        metrics = RAGMetrics(
            timestamp=timestamp,
            documents_processed=10,
            total_chunks=500,
            average_chunk_size=256,
            embeddings_generated=500,
            queries_processed=25,
            average_response_time=1.5
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.documents_processed == 10
        assert metrics.total_chunks == 500
        assert metrics.average_chunk_size == 256
        assert metrics.embeddings_generated == 500
        assert metrics.queries_processed == 25
        assert metrics.average_response_time == 1.5
    
    def test_rag_metrics_to_dict(self):
        """Probar conversión de RAGMetrics a diccionario"""
        timestamp = datetime.now().isoformat()
        metrics = RAGMetrics(
            timestamp=timestamp,
            documents_processed=10,
            total_chunks=500,
            average_chunk_size=256,
            embeddings_generated=500,
            queries_processed=25,
            average_response_time=1.5
        )
        
        metrics_dict = asdict(metrics)
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['documents_processed'] == 10
        assert metrics_dict['total_chunks'] == 500
        assert metrics_dict['average_response_time'] == 1.5


class TestMetricsCollector:
    """Pruebas para la clase MetricsCollector"""
    
    def setup_method(self):
        """Configuración antes de cada prueba"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_config = Mock()
        self.mock_config.CACHE_DIR = self.temp_dir
    
    def teardown_method(self):
        """Limpieza después de cada prueba"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('utils.metrics.config')
    def test_metrics_collector_initialization(self, mock_config):
        """Probar inicialización de MetricsCollector"""
        mock_config.CACHE_DIR = self.temp_dir
        
        collector = MetricsCollector()
        
        assert collector.metrics_file == self.temp_dir / "system_metrics.json"
        assert collector.rag_metrics_file == self.temp_dir / "rag_metrics.json"
        assert hasattr(collector, 'start_time')
        assert self.temp_dir.exists()
    
    @patch('utils.metrics.config')
    @patch('utils.metrics.psutil')
    def test_collect_system_metrics_success(self, mock_psutil, mock_config):
        """Probar recolección exitosa de métricas del sistema"""
        mock_config.CACHE_DIR = self.temp_dir
        
        # Mock de psutil
        mock_psutil.cpu_percent.return_value = 45.5
        mock_memory = Mock()
        mock_memory.percent = 60.2
        mock_memory.used = 2048 * 1024 * 1024  # 2048 MB en bytes
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.percent = 75.0
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_psutil.pids.return_value = list(range(150))  # 150 procesos
        
        collector = MetricsCollector()
        metrics = collector.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 60.2
        assert metrics.memory_used_mb == 2048.0
        assert metrics.disk_usage_percent == 75.0
        assert metrics.process_count == 150
        assert metrics.timestamp is not None
    
    @patch('utils.metrics.config')
    @patch('utils.metrics.psutil')
    @patch('utils.metrics.logger')
    def test_collect_system_metrics_error(self, mock_logger, mock_psutil, mock_config):
        """Probar manejo de errores en recolección de métricas"""
        mock_config.CACHE_DIR = self.temp_dir
        mock_psutil.cpu_percent.side_effect = Exception("Error de prueba")
        
        collector = MetricsCollector()
        metrics = collector.collect_system_metrics()
        
        assert metrics is None
        # Verificar que se llamó logger.error con el mensaje correcto
        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert "Error al recopilar métricas del sistema" in args[0]
    
    @patch('utils.metrics.config')
    def test_save_system_metrics_new_file(self, mock_config):
        """Probar guardado de métricas en archivo nuevo"""
        mock_config.CACHE_DIR = self.temp_dir
        
        collector = MetricsCollector()
        metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=45.5,
            memory_percent=60.2,
            memory_used_mb=2048.5,
            disk_usage_percent=75.0,
            process_count=150
        )
        
        collector.save_system_metrics(metrics)
        
        # Verificar que el archivo se creó
        assert collector.metrics_file.exists()
        
        # Verificar contenido
        with open(collector.metrics_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 1
        assert saved_data[0]['cpu_percent'] == 45.5
        assert saved_data[0]['memory_percent'] == 60.2
    
    @patch('utils.metrics.config')
    def test_save_system_metrics_existing_file(self, mock_config):
        """Probar guardado de métricas en archivo existente"""
        mock_config.CACHE_DIR = self.temp_dir
        
        # Crear archivo con métricas existentes
        existing_data = [{
            'timestamp': '2023-01-01T00:00:00',
            'cpu_percent': 30.0,
            'memory_percent': 50.0,
            'memory_used_mb': 1024.0,
            'disk_usage_percent': 60.0,
            'process_count': 100
        }]
        
        metrics_file = self.temp_dir / "system_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f)
        
        collector = MetricsCollector()
        new_metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=45.5,
            memory_percent=60.2,
            memory_used_mb=2048.5,
            disk_usage_percent=75.0,
            process_count=150
        )
        
        collector.save_system_metrics(new_metrics)
        
        # Verificar que se agregó la nueva métrica
        with open(collector.metrics_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 2
        assert saved_data[1]['cpu_percent'] == 45.5
    
    @patch('utils.metrics.config')
    def test_save_system_metrics_limit_1000(self, mock_config):
        """Probar límite de 1000 métricas"""
        mock_config.CACHE_DIR = self.temp_dir
        
        # Crear archivo con 1000 métricas
        existing_data = []
        for i in range(1000):
            existing_data.append({
                'timestamp': f'2023-01-01T{i:02d}:00:00',
                'cpu_percent': float(i),
                'memory_percent': 50.0,
                'memory_used_mb': 1024.0,
                'disk_usage_percent': 60.0,
                'process_count': 100
            })
        
        metrics_file = self.temp_dir / "system_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f)
        
        collector = MetricsCollector()
        new_metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=999.0,
            memory_percent=60.2,
            memory_used_mb=2048.5,
            disk_usage_percent=75.0,
            process_count=150
        )
        
        collector.save_system_metrics(new_metrics)
        
        # Verificar que se mantienen solo 1000 métricas
        with open(collector.metrics_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 1000
        assert saved_data[-1]['cpu_percent'] == 999.0  # La nueva métrica está al final
        assert saved_data[0]['cpu_percent'] == 1.0  # Se eliminó la primera métrica original
    
    @patch('utils.metrics.config')
    @patch('utils.metrics.logger')
    def test_save_system_metrics_error(self, mock_logger, mock_config):
        """Probar manejo de errores en guardado de métricas"""
        mock_config.CACHE_DIR = self.temp_dir
        
        # Crear el collector primero
        collector = MetricsCollector()
        
        # Simular error al escribir el archivo
        with patch('builtins.open', side_effect=OSError("Error de permisos")):
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=45.5,
                memory_percent=60.2,
                memory_used_mb=2048.5,
                disk_usage_percent=75.0,
                process_count=150
            )
            
            collector.save_system_metrics(metrics)
        
        # Verificar que se llamó logger.error con el mensaje correcto
        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert "Error al guardar métricas del sistema" in args[0]
    
    @patch('utils.metrics.config')
    def test_get_system_metrics_history_empty(self, mock_config):
        """Probar obtención de historial con archivo inexistente"""
        mock_config.CACHE_DIR = self.temp_dir
        
        collector = MetricsCollector()
        history = collector.get_system_metrics_history()
        
        assert history == []
    
    @patch('utils.metrics.config')
    def test_get_system_metrics_history_with_data(self, mock_config):
        """Probar obtención de historial con datos"""
        mock_config.CACHE_DIR = self.temp_dir
        
        # Crear datos de prueba
        now = datetime.now()
        test_data = [
            {
                'timestamp': (now - timedelta(hours=2)).isoformat(),
                'cpu_percent': 30.0,
                'memory_percent': 50.0,
                'memory_used_mb': 1024.0,
                'disk_usage_percent': 60.0,
                'process_count': 100
            },
            {
                'timestamp': (now - timedelta(hours=1)).isoformat(),
                'cpu_percent': 45.0,
                'memory_percent': 60.0,
                'memory_used_mb': 2048.0,
                'disk_usage_percent': 70.0,
                'process_count': 120
            },
            {
                'timestamp': (now - timedelta(hours=25)).isoformat(),  # Muy antiguo
                'cpu_percent': 20.0,
                'memory_percent': 40.0,
                'memory_used_mb': 512.0,
                'disk_usage_percent': 50.0,
                'process_count': 80
            }
        ]
        
        metrics_file = self.temp_dir / "system_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        collector = MetricsCollector()
        history = collector.get_system_metrics_history(24)  # Últimas 24 horas
        
        # Solo debe devolver las 2 métricas recientes
        assert len(history) == 2
        assert history[0]['cpu_percent'] == 30.0
        assert history[1]['cpu_percent'] == 45.0
    
    @patch('utils.metrics.config')
    @patch('utils.metrics.logger')
    def test_get_system_metrics_history_error(self, mock_logger, mock_config):
        """Probar manejo de errores en obtención de historial"""
        mock_config.CACHE_DIR = self.temp_dir
        
        # Crear archivo con JSON inválido
        metrics_file = self.temp_dir / "system_metrics.json"
        with open(metrics_file, 'w') as f:
            f.write("json inválido")
        
        collector = MetricsCollector()
        history = collector.get_system_metrics_history()
        
        assert history == []
        mock_logger.error.assert_called_once()
    
    @patch('utils.metrics.config')
    @patch('time.time')
    def test_get_performance_summary_success(self, mock_time, mock_config):
        """Probar obtención exitosa de resumen de rendimiento"""
        mock_config.CACHE_DIR = self.temp_dir
        mock_time.return_value = 3600.0  # 1 hora después del inicio
        
        # Crear datos de prueba
        now = datetime.now()
        test_data = [
            {
                'timestamp': (now - timedelta(hours=2)).isoformat(),
                'cpu_percent': 30.0,
                'memory_percent': 50.0,
                'memory_used_mb': 1024.0,
                'disk_usage_percent': 60.0,
                'process_count': 100
            },
            {
                'timestamp': (now - timedelta(hours=1)).isoformat(),
                'cpu_percent': 40.0,
                'memory_percent': 60.0,
                'memory_used_mb': 2048.0,
                'disk_usage_percent': 70.0,
                'process_count': 120
            }
        ]
        
        metrics_file = self.temp_dir / "system_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        collector = MetricsCollector()
        collector.start_time = 0.0  # Tiempo de inicio simulado
        
        summary = collector.get_performance_summary()
        
        assert 'uptime_hours' in summary
        assert 'avg_cpu_percent' in summary
        assert 'avg_memory_percent' in summary
        assert 'max_cpu_percent' in summary
        assert 'max_memory_percent' in summary
        assert 'metrics_count' in summary
        
        assert summary['avg_cpu_percent'] == 35.0  # (30 + 40) / 2
        assert summary['avg_memory_percent'] == 55.0  # (50 + 60) / 2
        assert summary['max_cpu_percent'] == 40.0
        assert summary['max_memory_percent'] == 60.0
        assert summary['metrics_count'] == 2
    
    @patch('utils.metrics.config')
    def test_get_performance_summary_no_data(self, mock_config):
        """Probar resumen de rendimiento sin datos"""
        mock_config.CACHE_DIR = self.temp_dir
        
        collector = MetricsCollector()
        summary = collector.get_performance_summary()
        
        assert summary == {}
    
    @patch('utils.metrics.config')
    @patch('utils.metrics.logger')
    def test_clear_metrics_success(self, mock_logger, mock_config):
        """Probar limpieza exitosa de métricas"""
        mock_config.CACHE_DIR = self.temp_dir
        
        # Crear archivos de métricas
        metrics_file = self.temp_dir / "system_metrics.json"
        rag_metrics_file = self.temp_dir / "rag_metrics.json"
        
        metrics_file.write_text('{"test": "data"}')
        rag_metrics_file.write_text('{"test": "data"}')
        
        assert metrics_file.exists()
        assert rag_metrics_file.exists()
        
        collector = MetricsCollector()
        collector.clear_metrics()
        
        assert not metrics_file.exists()
        assert not rag_metrics_file.exists()
        mock_logger.info.assert_called_once_with("Métricas limpiadas exitosamente")
    
    @patch('utils.metrics.config')
    @patch('utils.metrics.logger')
    def test_clear_metrics_error(self, mock_logger, mock_config):
        """Probar manejo de errores en limpieza de métricas"""
        mock_config.CACHE_DIR = Path("/directorio/inexistente")
        
        # Crear archivos mock que existan pero fallen al eliminar
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.unlink', side_effect=OSError("Error de permisos")):
            collector = MetricsCollector()
            collector.clear_metrics()
        
        # Verificar que se llamó logger.error con el mensaje correcto
        mock_logger.error.assert_called_once()
        args, kwargs = mock_logger.error.call_args
        assert "Error al limpiar métricas" in args[0]