"""
Pruebas unitarias para scripts/check_dependencies.py
Script de verificación de dependencias del proyecto
"""

import pytest
import sys
import importlib.metadata
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from io import StringIO

# Agregar el directorio raíz al path para las importaciones
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts import check_dependencies


class TestCheckPackageVersion:
    """Pruebas para la función check_package_version"""
    
    @patch('importlib.metadata.distribution')
    def test_check_package_version_installed_no_version_requirement(self, mock_distribution):
        """Test de paquete instalado sin requerimiento de versión"""
        # Configurar mock
        mock_dist = Mock()
        mock_dist.version = "1.5.0"
        mock_distribution.return_value = mock_dist
        
        status, version, message = check_dependencies.check_package_version("test_package")
        
        assert status == "✅"
        assert version == "1.5.0"
        assert message == "OK"
        mock_distribution.assert_called_once_with("test_package")
    
    @patch('importlib.metadata.distribution')
    @patch('packaging.version.parse')
    def test_check_package_version_installed_version_ok(self, mock_version_parse, mock_distribution):
        """Test de paquete instalado con versión correcta"""
        # Configurar mocks
        mock_dist = Mock()
        mock_dist.version = "2.0.0"
        mock_distribution.return_value = mock_dist
        
        # Mock para comparación de versiones
        mock_installed_version = Mock()
        mock_required_version = Mock()
        mock_installed_version.__ge__ = Mock(return_value=True)
        mock_version_parse.side_effect = [mock_installed_version, mock_required_version]
        
        status, version, message = check_dependencies.check_package_version("test_package", "1.5.0")
        
        assert status == "✅"
        assert version == "2.0.0"
        assert message == "OK"
    
    @patch('importlib.metadata.distribution')
    @patch('packaging.version.parse')
    def test_check_package_version_installed_version_old(self, mock_version_parse, mock_distribution):
        """Test de paquete instalado con versión antigua"""
        # Configurar mocks
        mock_dist = Mock()
        mock_dist.version = "1.0.0"
        mock_distribution.return_value = mock_dist
        
        # Mock para comparación de versiones
        mock_installed_version = Mock()
        mock_required_version = Mock()
        mock_installed_version.__ge__ = Mock(return_value=False)
        mock_version_parse.side_effect = [mock_installed_version, mock_required_version]
        
        status, version, message = check_dependencies.check_package_version("test_package", "1.5.0")
        
        assert status == "⚠️"
        assert version == "1.0.0"
        assert message == "Requiere >=1.5.0"
    
    @patch('importlib.metadata.distribution')
    def test_check_package_version_installed_no_packaging(self, mock_distribution):
        """Test de paquete instalado sin módulo packaging disponible"""
        # Configurar mock
        mock_dist = Mock()
        mock_dist.version = "1.5.0"
        mock_distribution.return_value = mock_dist
        
        # Simular ImportError para packaging
        with patch('packaging.version.parse', side_effect=ImportError):
            status, version, message = check_dependencies.check_package_version("test_package", "1.0.0")
        
        assert status == "✅"
        assert version == "1.5.0"
        assert message == "OK (sin verificación de versión)"
    
    @patch('importlib.metadata.distribution')
    def test_check_package_version_not_installed(self, mock_distribution):
        """Test de paquete no instalado"""
        # Configurar mock para lanzar excepción
        mock_distribution.side_effect = importlib.metadata.PackageNotFoundError
        
        status, version, message = check_dependencies.check_package_version("missing_package")
        
        assert status == "❌"
        assert version == "No instalado"
        assert message == "Faltante"


class TestMainFunction:
    """Pruebas para la función principal main()"""
    
    def setup_method(self):
        """Configuración para cada test"""
        self.sample_requirements = """# Core dependencies
streamlit>=1.28.0
requests>=2.31.0

# Machine Learning
scikit-learn>=1.3.0
nltk>=3.8.1

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0
"""
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.check_dependencies.check_package_version')
    def test_main_success_all_installed(self, mock_check_package, mock_file, mock_exists, mock_print):
        """Test de ejecución exitosa con todos los paquetes instalados"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = self.sample_requirements.split('\n')
        mock_check_package.return_value = ("✅", "1.28.0", "OK")
        
        check_dependencies.main()
        
        # Verificar que se imprimió el encabezado
        mock_print.assert_any_call("🔍 CogniChat - Verificador de Dependencias")
        mock_print.assert_any_call("🎉 ¡Todas las dependencias están instaladas correctamente!")
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.check_dependencies.check_package_version')
    def test_main_missing_packages(self, mock_check_package, mock_file, mock_exists, mock_print):
        """Test de ejecución con paquetes faltantes"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = self.sample_requirements.split('\n')
        
        # Simular algunos paquetes faltantes
        def side_effect(package_name, version=None):
            if package_name == "streamlit":
                return ("✅", "1.28.0", "OK")
            elif package_name == "requests":
                return ("❌", "No instalado", "Faltante")
            else:
                return ("✅", "1.0.0", "OK")
        
        mock_check_package.side_effect = side_effect
        
        check_dependencies.main()
        
        # Verificar que se mostraron los paquetes faltantes
        # Buscar en todas las llamadas a print
        print_calls = [str(call) for call in mock_print.call_args_list]
        found_missing_packages = any("PAQUETES FALTANTES" in call for call in print_calls)
        found_warning = any("Hay dependencias faltantes" in call for call in print_calls)
        
        assert found_missing_packages or found_warning
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    def test_main_requirements_file_not_found(self, mock_exists, mock_print):
        """Test de manejo cuando no se encuentra requirements.txt"""
        # Configurar mock para archivo no encontrado
        mock_exists.return_value = False
        
        check_dependencies.main()
        
        # Verificar que se mostró el error
        mock_print.assert_any_call("❌ Error: No se encontró el archivo requirements.txt")
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.check_dependencies.check_package_version')
    def test_main_version_warnings(self, mock_check_package, mock_file, mock_exists, mock_print):
        """Test de ejecución con advertencias de versión"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = self.sample_requirements.split('\n')
        
        # Simular paquetes con versiones incorrectas
        def side_effect(package_name, version=None):
            if package_name == "streamlit":
                return ("⚠️", "1.20.0", "Requiere >=1.28.0")
            else:
                return ("✅", "1.0.0", "OK")
        
        mock_check_package.side_effect = side_effect
        
        check_dependencies.main()
        
        # Verificar que se procesaron las advertencias
        assert mock_check_package.called
        assert mock_print.called
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.check_dependencies.check_package_version')
    def test_main_functionality_checks(self, mock_check_package, mock_file, mock_exists, mock_print):
        """Test de verificación de funcionalidades específicas"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = self.sample_requirements.split('\n')
        
        # Configurar respuestas para paquetes específicos
        def side_effect(package_name, version=None):
            advanced_packages = ['scikit-learn', 'nltk', 'textblob', 'wordcloud']
            viz_packages = ['plotly', 'matplotlib', 'seaborn']
            cluster_packages = ['umap-learn', 'hdbscan']
            
            if package_name in advanced_packages:
                return ("✅", "1.0.0", "OK")
            elif package_name in viz_packages:
                return ("✅", "1.0.0", "OK")
            elif package_name in cluster_packages:
                return ("❌", "No instalado", "Faltante")
            else:
                return ("✅", "1.0.0", "OK")
        
        mock_check_package.side_effect = side_effect
        
        check_dependencies.main()
        
        # Verificar que se verificaron las funcionalidades
        print_calls = [str(call) for call in mock_print.call_args_list]
        found_functionality_check = any("VERIFICACIÓN DE FUNCIONALIDADES" in call for call in print_calls)
        found_advanced_analysis = any("Análisis cualitativo avanzado" in call for call in print_calls)
        found_visualizations = any("Visualizaciones avanzadas" in call for call in print_calls)
        found_clustering = any("Clustering avanzado" in call for call in print_calls)
        
        assert found_functionality_check
        assert found_advanced_analysis or found_visualizations or found_clustering


class TestRequirementsFileParsing:
    """Pruebas para el análisis del archivo requirements.txt"""
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.check_dependencies.check_package_version')
    @patch('builtins.print')
    def test_parse_requirements_with_categories(self, mock_print, mock_check_package, mock_file, mock_exists):
        """Test de análisis de requirements.txt con categorías"""
        requirements_content = """# Core dependencies
streamlit>=1.28.0
requests>=2.31.0

# Machine Learning
scikit-learn>=1.3.0

# Optional packages
matplotlib
"""
        
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = requirements_content.split('\n')
        mock_check_package.return_value = ("✅", "1.0.0", "OK")
        
        check_dependencies.main()
        
        # Verificar que se procesaron las categorías
        print_calls = [str(call) for call in mock_print.call_args_list]
        found_core = any("Core dependencies" in call for call in print_calls)
        found_ml = any("Machine Learning" in call for call in print_calls)
        found_optional = any("Optional packages" in call for call in print_calls)
        
        assert found_core or found_ml or found_optional
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.check_dependencies.check_package_version')
    @patch('builtins.print')
    def test_parse_requirements_skip_builtin_packages(self, mock_print, mock_check_package, mock_file, mock_exists):
        """Test de que se omiten paquetes integrados en Python"""
        requirements_content = """pathlib
sqlite3
requests>=2.31.0
"""
        
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = requirements_content.split('\n')
        mock_check_package.return_value = ("✅", "1.0.0", "OK")
        
        check_dependencies.main()
        
        # Verificar que solo se verificó requests, no pathlib ni sqlite3
        check_calls = [call[0][0] for call in mock_check_package.call_args_list]
        assert "requests" in check_calls
        assert "pathlib" not in check_calls
        assert "sqlite3" not in check_calls


class TestModuleImports:
    """Pruebas para verificar las importaciones del módulo"""
    
    def test_required_imports_exist(self):
        """Test de que todas las importaciones requeridas existen"""
        # Verificar que las funciones principales están disponibles
        assert hasattr(check_dependencies, 'check_package_version')
        assert hasattr(check_dependencies, 'main')
        
        # Verificar que son callable
        assert callable(check_dependencies.check_package_version)
        assert callable(check_dependencies.main)
    
    def test_importlib_metadata_available(self):
        """Test de que importlib.metadata está disponible"""
        import importlib.metadata
        assert hasattr(importlib.metadata, 'distribution')
        assert hasattr(importlib.metadata, 'PackageNotFoundError')


if __name__ == "__main__":
    pytest.main([__file__])