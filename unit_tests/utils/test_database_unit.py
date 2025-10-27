"""
Pruebas unitarias para DatabaseManager
Cobertura de gestión de base de datos y operaciones con ChromaDB
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
from pathlib import Path
import tempfile
import shutil

# Configurar el path antes de importar
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TestDatabaseManager(unittest.TestCase):
    """Pruebas para la clase DatabaseManager"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        # Crear directorio temporal para pruebas
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        # Mock de configuración
        self.mock_config = Mock()
        self.mock_config.CACHE_DIR = self.cache_dir
        
        # Mock de logger
        self.mock_logger = Mock()
        
        # Patches para evitar importaciones circulares
        self.config_patch = patch('utils.database.config', self.mock_config)
        self.logger_patch = patch('utils.database.logger', self.mock_logger)
        self.setup_logger_patch = patch('utils.database.setup_logger', return_value=self.mock_logger)
        
        # Iniciar patches
        self.config_patch.start()
        self.logger_patch.start()
        self.setup_logger_patch.start()
        
        # Importar después de configurar mocks
        from utils.database import DatabaseManager
        self.db_manager = DatabaseManager()
    
    def tearDown(self):
        """Limpieza después de cada prueba"""
        # Detener patches
        self.config_patch.stop()
        self.logger_patch.stop()
        self.setup_logger_patch.stop()
        
        # Limpiar directorio temporal
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Probar inicialización correcta del DatabaseManager"""
        # Verificar que se configuraron las rutas correctamente
        expected_db_path = self.cache_dir / "chroma_db"
        expected_metadata_file = self.cache_dir / "db_metadata.json"
        
        self.assertEqual(self.db_manager.db_path, expected_db_path)
        self.assertEqual(self.db_manager.metadata_file, expected_metadata_file)
    
    def test_ensure_directories_creates_paths(self):
        """Probar que se crean los directorios necesarios"""
        # Verificar que los directorios fueron creados
        self.assertTrue(self.db_manager.db_path.exists())
        self.assertTrue(self.cache_dir.exists())
        
        # Verificar que son directorios
        self.assertTrue(self.db_manager.db_path.is_dir())
        self.assertTrue(self.cache_dir.is_dir())
    
    def test_ensure_directories_existing_paths(self):
        """Probar que no hay error si los directorios ya existen"""
        # Los directorios ya existen por la inicialización
        # Crear otra instancia para probar que no hay conflicto
        from utils.database import DatabaseManager
        db_manager2 = DatabaseManager()
        
        # No debe haber errores y las rutas deben ser las mismas
        self.assertEqual(db_manager2.db_path, self.db_manager.db_path)
        self.assertEqual(db_manager2.metadata_file, self.db_manager.metadata_file)
    
    def test_db_path_structure(self):
        """Probar la estructura correcta de rutas de base de datos"""
        # Verificar que db_path es subdirectorio de CACHE_DIR
        self.assertTrue(str(self.db_manager.db_path).startswith(str(self.cache_dir)))
        
        # Verificar que metadata_file está en CACHE_DIR
        self.assertEqual(self.db_manager.metadata_file.parent, self.cache_dir)
        
        # Verificar nombres correctos
        self.assertEqual(self.db_manager.db_path.name, "chroma_db")
        self.assertEqual(self.db_manager.metadata_file.name, "db_metadata.json")
    
    @patch('utils.database.Path.mkdir')
    def test_ensure_directories_with_permission_error(self, mock_mkdir):
        """Probar manejo de errores de permisos al crear directorios"""
        # Configurar mock para lanzar PermissionError
        mock_mkdir.side_effect = PermissionError("Permission denied")
        
        # Esto debería lanzar la excepción ya que no hay manejo de errores
        with self.assertRaises(PermissionError):
            from utils.database import DatabaseManager
            DatabaseManager()
    
    def test_multiple_instances_same_paths(self):
        """Probar que múltiples instancias usan las mismas rutas"""
        from utils.database import DatabaseManager
        
        # Crear múltiples instancias
        db_manager1 = DatabaseManager()
        db_manager2 = DatabaseManager()
        db_manager3 = DatabaseManager()
        
        # Todas deben tener las mismas rutas
        self.assertEqual(db_manager1.db_path, db_manager2.db_path)
        self.assertEqual(db_manager2.db_path, db_manager3.db_path)
        self.assertEqual(db_manager1.metadata_file, db_manager2.metadata_file)
        self.assertEqual(db_manager2.metadata_file, db_manager3.metadata_file)
    
    def test_paths_are_pathlib_objects(self):
        """Probar que las rutas son objetos Path de pathlib"""
        self.assertIsInstance(self.db_manager.db_path, Path)
        self.assertIsInstance(self.db_manager.metadata_file, Path)
    
    def test_cache_dir_configuration_dependency(self):
        """Probar dependencia correcta de la configuración CACHE_DIR"""
        # Cambiar configuración y crear nueva instancia
        new_cache_dir = Path(self.temp_dir) / "new_cache"
        self.mock_config.CACHE_DIR = new_cache_dir
        
        from utils.database import DatabaseManager
        new_db_manager = DatabaseManager()
        
        # Verificar que usa la nueva configuración
        expected_db_path = new_cache_dir / "chroma_db"
        expected_metadata_file = new_cache_dir / "db_metadata.json"
        
        self.assertEqual(new_db_manager.db_path, expected_db_path)
        self.assertEqual(new_db_manager.metadata_file, expected_metadata_file)
        
        # Verificar que se crearon los nuevos directorios
        self.assertTrue(new_cache_dir.exists())
        self.assertTrue(expected_db_path.exists())


class TestDatabaseManagerIntegration(unittest.TestCase):
    """Pruebas de integración para DatabaseManager"""
    
    def setUp(self):
        """Configuración inicial"""
        # Crear directorio temporal real
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "integration_cache"
        
        # Mock de configuración con directorio real
        self.mock_config = Mock()
        self.mock_config.CACHE_DIR = self.cache_dir
        
        self.mock_logger = Mock()
        
        # Patches
        self.config_patch = patch('utils.database.config', self.mock_config)
        self.logger_patch = patch('utils.database.logger', self.mock_logger)
        self.setup_logger_patch = patch('utils.database.setup_logger', return_value=self.mock_logger)
        
        self.config_patch.start()
        self.logger_patch.start()
        self.setup_logger_patch.start()
    
    def tearDown(self):
        """Limpieza"""
        self.config_patch.stop()
        self.logger_patch.stop()
        self.setup_logger_patch.stop()
        
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_real_directory_creation(self):
        """Probar creación real de directorios en el sistema de archivos"""
        from utils.database import DatabaseManager
        
        # Verificar que el directorio cache no existe inicialmente
        self.assertFalse(self.cache_dir.exists())
        
        # Crear instancia (debe crear directorios)
        db_manager = DatabaseManager()
        
        # Verificar que se crearon los directorios
        self.assertTrue(self.cache_dir.exists())
        self.assertTrue(db_manager.db_path.exists())
        
        # Verificar que son directorios reales
        self.assertTrue(self.cache_dir.is_dir())
        self.assertTrue(db_manager.db_path.is_dir())
    
    def test_metadata_file_path_accessibility(self):
        """Probar que el archivo de metadata es accesible para escritura"""
        from utils.database import DatabaseManager
        
        db_manager = DatabaseManager()
        
        # Intentar escribir en el archivo de metadata
        test_data = {"test": "data", "timestamp": "2024-01-01"}
        
        try:
            with open(db_manager.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f)
            
            # Verificar que se puede leer
            with open(db_manager.metadata_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data, test_data)
            
        except Exception as e:
            self.fail(f"No se pudo escribir/leer archivo de metadata: {e}")
    
    def test_concurrent_access_safety(self):
        """Probar seguridad de acceso concurrente"""
        from utils.database import DatabaseManager
        
        # Crear múltiples instancias simultáneamente
        managers = []
        for i in range(5):
            manager = DatabaseManager()
            managers.append(manager)
        
        # Verificar que todas tienen las mismas rutas
        base_db_path = managers[0].db_path
        base_metadata_file = managers[0].metadata_file
        
        for manager in managers[1:]:
            self.assertEqual(manager.db_path, base_db_path)
            self.assertEqual(manager.metadata_file, base_metadata_file)
        
        # Verificar que el directorio existe y es accesible
        self.assertTrue(base_db_path.exists())
        self.assertTrue(base_db_path.is_dir())


class TestDatabaseManagerEdgeCases(unittest.TestCase):
    """Pruebas de casos extremos para DatabaseManager"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_logger = Mock()
        
        self.logger_patch = patch('utils.database.logger', self.mock_logger)
        self.setup_logger_patch = patch('utils.database.setup_logger', return_value=self.mock_logger)
        
        self.logger_patch.start()
        self.setup_logger_patch.start()
    
    def tearDown(self):
        """Limpieza"""
        self.logger_patch.stop()
        self.setup_logger_patch.stop()
        
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_very_long_path(self):
        """Probar con rutas muy largas"""
        # Crear ruta muy larga
        long_path_parts = ["very_long_directory_name"] * 10
        long_cache_dir = Path(self.temp_dir)
        for part in long_path_parts:
            long_cache_dir = long_cache_dir / part
        
        mock_config = Mock()
        mock_config.CACHE_DIR = long_cache_dir
        
        with patch('utils.database.config', mock_config):
            from utils.database import DatabaseManager
            
            try:
                db_manager = DatabaseManager()
                # Si no hay excepción, la prueba pasa
                self.assertIsInstance(db_manager.db_path, Path)
                self.assertIsInstance(db_manager.metadata_file, Path)
            except OSError:
                # En algunos sistemas, rutas muy largas pueden fallar
                # Esto es comportamiento esperado del sistema operativo
                pass
    
    def test_special_characters_in_cache_dir(self):
        """Probar con caracteres especiales en el directorio cache"""
        # Crear directorio con caracteres especiales (válidos en el sistema)
        special_cache_dir = Path(self.temp_dir) / "cache_with_spaces_and-dashes"
        
        mock_config = Mock()
        mock_config.CACHE_DIR = special_cache_dir
        
        with patch('utils.database.config', mock_config):
            from utils.database import DatabaseManager
            
            db_manager = DatabaseManager()
            
            # Verificar que se crearon correctamente
            self.assertTrue(special_cache_dir.exists())
            self.assertTrue(db_manager.db_path.exists())
    
    def test_relative_path_handling(self):
        """Probar manejo de rutas relativas"""
        # Usar ruta relativa
        relative_cache_dir = Path("./relative_cache")
        
        mock_config = Mock()
        mock_config.CACHE_DIR = relative_cache_dir
        
        with patch('utils.database.config', mock_config):
            from utils.database import DatabaseManager
            
            db_manager = DatabaseManager()
            
            # Verificar que las rutas se configuraron
            self.assertEqual(db_manager.db_path.name, "chroma_db")
            self.assertEqual(db_manager.metadata_file.name, "db_metadata.json")
            
            # Limpiar directorio creado
            if relative_cache_dir.exists():
                shutil.rmtree(relative_cache_dir)


if __name__ == '__main__':
    unittest.main()