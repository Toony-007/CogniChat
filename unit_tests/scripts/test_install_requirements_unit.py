"""
Pruebas unitarias para scripts/install_requirements.py
Script de instalación automática de dependencias
"""

import pytest
import sys
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback para Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

# Agregar el directorio raíz al path para las importaciones
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts import install_requirements


class TestCheckPackageInstalled:
    """Pruebas para la función check_package_installed"""
    
    @patch('scripts.install_requirements.version')
    def test_check_package_installed_success(self, mock_version):
        """Test de paquete instalado correctamente"""
        # Configurar mock para paquete encontrado
        mock_version.return_value = "1.0.0"
        
        result = install_requirements.check_package_installed("test_package")
        
        assert result is True
        mock_version.assert_called_once_with("test_package")
    
    @patch('scripts.install_requirements.version')
    def test_check_package_installed_not_found(self, mock_version):
        """Test de paquete no instalado"""
        # Configurar mock para lanzar excepción
        mock_version.side_effect = PackageNotFoundError
        
        result = install_requirements.check_package_installed("missing_package")
        
        assert result is False
        mock_version.assert_called_once_with("missing_package")


class TestInstallPackage:
    """Pruebas para la función install_package"""
    
    @patch('subprocess.check_call')
    def test_install_package_success(self, mock_check_call):
        """Test de instalación exitosa de paquete"""
        # Configurar mock para instalación exitosa
        mock_check_call.return_value = None
        
        result = install_requirements.install_package("test_package>=1.0.0")
        
        assert result is True
        mock_check_call.assert_called_once_with([
            sys.executable, "-m", "pip", "install", "test_package>=1.0.0"
        ])
    
    @patch('subprocess.check_call')
    def test_install_package_failure(self, mock_check_call):
        """Test de fallo en instalación de paquete"""
        # Configurar mock para lanzar excepción
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "pip")
        
        result = install_requirements.install_package("failing_package")
        
        assert result is False
        mock_check_call.assert_called_once_with([
            sys.executable, "-m", "pip", "install", "failing_package"
        ])


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

# Built-in packages (should be ignored)
pathlib
sqlite3
"""
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('scripts.install_requirements.install_package')
    @patch('subprocess.check_call')
    def test_main_all_packages_already_installed(self, mock_subprocess, mock_install, 
                                               mock_check_installed, mock_file, 
                                               mock_exists, mock_print):
        """Test de ejecución cuando todos los paquetes ya están instalados"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = self.sample_requirements.split('\n')
        mock_check_installed.return_value = True  # Todos los paquetes ya instalados
        
        # Mock para NLTK
        mock_nltk = MagicMock()
        with patch.dict('sys.modules', {'nltk': mock_nltk}):
            install_requirements.main()
        
        # Verificar que se imprimió el encabezado
        mock_print.assert_any_call("🚀 CogniChat - Instalador de Dependencias")
        mock_print.assert_any_call("   🎉 ¡Todas las dependencias instaladas correctamente!")
        
        # Verificar que no se intentó instalar nada (todos ya estaban instalados)
        mock_install.assert_not_called()
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('scripts.install_requirements.install_package')
    @patch('subprocess.check_call')
    def test_main_install_missing_packages(self, mock_subprocess, mock_install, 
                                         mock_check_installed, mock_file, 
                                         mock_exists, mock_print):
        """Test de instalación de paquetes faltantes"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = self.sample_requirements.split('\n')
        
        # Simular algunos paquetes faltantes
        def check_installed_side_effect(package_name):
            return package_name in ["streamlit"]  # Solo streamlit está instalado
        
        mock_check_installed.side_effect = check_installed_side_effect
        mock_install.return_value = True  # Instalación exitosa
        
        # Mock para NLTK
        mock_nltk = MagicMock()
        with patch.dict('sys.modules', {'nltk': mock_nltk}):
            install_requirements.main()
        
        # Verificar que se intentó instalar los paquetes faltantes
        assert mock_install.call_count > 0
        mock_print.assert_any_call("   🎉 ¡Todas las dependencias instaladas correctamente!")
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('scripts.install_requirements.install_package')
    @patch('subprocess.check_call')
    def test_main_installation_failures(self, mock_subprocess, mock_install, 
                                      mock_check_installed, mock_file, 
                                      mock_exists, mock_print):
        """Test de manejo de fallos en instalación"""
        # Configurar mocks
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = self.sample_requirements.split('\n')
        mock_check_installed.return_value = False  # Ningún paquete instalado
        
        # Simular fallos en instalación
        def install_side_effect(package):
            return "requests" not in package  # requests falla
        
        mock_install.side_effect = install_side_effect
        
        # Mock para NLTK
        mock_nltk = MagicMock()
        with patch.dict('sys.modules', {'nltk': mock_nltk}):
            install_requirements.main()
        
        # Verificar que se mostraron los paquetes fallidos
        mock_print.assert_any_call("   📋 Paquetes que fallaron:")
        mock_print.assert_any_call("💡 Intenta instalar manualmente los paquetes fallidos:")
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    def test_main_requirements_file_not_found(self, mock_exists, mock_print):
        """Test de manejo cuando no se encuentra requirements.txt"""
        # Configurar mock para archivo no encontrado
        mock_exists.return_value = False
        
        install_requirements.main()
        
        # Verificar que se mostró el error
        mock_print.assert_any_call("❌ Error: No se encontró el archivo requirements.txt")
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('subprocess.check_call')
    def test_main_nltk_resources_download(self, mock_subprocess, mock_check_installed, 
                                        mock_file, mock_exists, mock_print):
        """Test de descarga de recursos de NLTK"""
        # Configurar mocks básicos
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = ["streamlit>=1.28.0"]
        mock_check_installed.return_value = True
        
        # Mock para NLTK
        mock_nltk = MagicMock()
        with patch.dict('sys.modules', {'nltk': mock_nltk}):
            install_requirements.main()
        
        # Verificar que se intentó descargar recursos de NLTK
        mock_nltk.download.assert_any_call('punkt', quiet=True)
        mock_nltk.download.assert_any_call('stopwords', quiet=True)
        mock_nltk.download.assert_any_call('vader_lexicon', quiet=True)
        mock_nltk.download.assert_any_call('wordnet', quiet=True)
        mock_nltk.download.assert_any_call('omw-1.4', quiet=True)
        
        mock_print.assert_any_call("✅ Recursos de NLTK descargados")
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('subprocess.check_call')
    def test_main_nltk_not_available(self, mock_subprocess, mock_check_installed, 
                                   mock_file, mock_exists, mock_print):
        """Test de manejo cuando NLTK no está disponible"""
        # Configurar mocks básicos
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = ["streamlit>=1.28.0"]
        mock_check_installed.return_value = True
        
        # Simular que NLTK no está disponible usando ImportError en el import
        with patch('builtins.__import__', side_effect=lambda name, *args: 
                  __import__(name, *args) if name != 'nltk' else (_ for _ in ()).throw(ImportError())):
            install_requirements.main()
        
        # Verificar que se manejó la ausencia de NLTK
        mock_print.assert_any_call("⚠️ NLTK no disponible - saltando descarga de recursos")
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('subprocess.check_call')
    def test_main_spacy_model_installation(self, mock_subprocess, mock_check_installed, 
                                         mock_file, mock_exists, mock_print):
        """Test de instalación del modelo de spaCy"""
        # Configurar mocks básicos
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = ["streamlit>=1.28.0"]
        mock_check_installed.return_value = True
        
        # Mock para NLTK (para evitar ImportError)
        mock_nltk = MagicMock()
        
        with patch.dict('sys.modules', {'nltk': mock_nltk}):
            install_requirements.main()
        
        # Verificar que se intentó instalar el modelo de spaCy
        expected_call = [sys.executable, "-m", "spacy", "download", "es_core_news_sm"]
        mock_subprocess.assert_any_call(expected_call)
        mock_print.assert_any_call("✅ Modelo de spaCy instalado")
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('subprocess.check_call')
    def test_main_spacy_installation_failure(self, mock_subprocess, mock_check_installed, 
                                           mock_file, mock_exists, mock_print):
        """Test de manejo de fallo en instalación de spaCy"""
        # Configurar mocks básicos
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = ["streamlit>=1.28.0"]
        mock_check_installed.return_value = True
        
        # Configurar fallo en spaCy (solo para la llamada específica)
        def subprocess_side_effect(cmd):
            if "spacy" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return None
        
        mock_subprocess.side_effect = subprocess_side_effect
        
        # Mock para NLTK
        mock_nltk = MagicMock()
        
        with patch.dict('sys.modules', {'nltk': mock_nltk}):
            install_requirements.main()
        
        # Verificar que se manejó el fallo de spaCy
        mock_print.assert_any_call("⚠️ No se pudo instalar el modelo de spaCy - instálalo manualmente:")
        mock_print.assert_any_call("   python -m spacy download es_core_news_sm")


class TestRequirementsFileParsing:
    """Pruebas para el análisis del archivo requirements.txt"""
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('subprocess.check_call')
    def test_parse_requirements_skip_builtin_packages(self, mock_subprocess, mock_check_installed, 
                                                    mock_file, mock_exists, mock_print):
        """Test de que se omiten paquetes integrados en Python"""
        requirements_content = """pathlib
sqlite3
requests>=2.31.0
streamlit>=1.28.0
"""
        
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = requirements_content.split('\n')
        mock_check_installed.return_value = True
        
        # Mock para NLTK
        mock_nltk = MagicMock()
        
        with patch.dict('sys.modules', {'nltk': mock_nltk}):
            install_requirements.main()
        
        # Verificar que se procesaron solo 2 paquetes (no pathlib ni sqlite3)
        check_calls = [call[0][0] for call in mock_check_installed.call_args_list]
        assert "requests" in check_calls
        assert "streamlit" in check_calls
        assert "pathlib" not in check_calls
        assert "sqlite3" not in check_calls
    
    @patch('builtins.print')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('scripts.install_requirements.check_package_installed')
    @patch('subprocess.check_call')
    def test_parse_requirements_handle_different_version_specs(self, mock_subprocess, mock_check_installed, 
                                                             mock_file, mock_exists, mock_print):
        """Test de manejo de diferentes especificaciones de versión"""
        requirements_content = """requests>=2.31.0
streamlit==1.28.0
matplotlib
"""
        
        mock_exists.return_value = True
        mock_file.return_value.readlines.return_value = requirements_content.split('\n')
        mock_check_installed.return_value = True
        
        # Mock para NLTK
        mock_nltk = MagicMock()
        
        with patch.dict('sys.modules', {'nltk': mock_nltk}):
            install_requirements.main()
        
        # Verificar que se procesaron todos los tipos de especificaciones
        check_calls = [call[0][0] for call in mock_check_installed.call_args_list]
        assert "requests" in check_calls
        assert "streamlit" in check_calls
        assert "matplotlib" in check_calls


class TestModuleImports:
    """Pruebas para verificar las importaciones del módulo"""
    
    def test_required_imports_exist(self):
        """Test de que todas las importaciones requeridas existen"""
        # Verificar que las funciones principales están disponibles
        assert hasattr(install_requirements, 'check_package_installed')
        assert hasattr(install_requirements, 'install_package')
        assert hasattr(install_requirements, 'main')
        
        # Verificar que son callable
        assert callable(install_requirements.check_package_installed)
        assert callable(install_requirements.install_package)
        assert callable(install_requirements.main)
    
    def test_required_modules_available(self):
        """Test de que los módulos requeridos están disponibles"""
        import subprocess
        import sys
        from pathlib import Path
        
        try:
            from importlib.metadata import version, PackageNotFoundError
        except ImportError:
            from importlib_metadata import version, PackageNotFoundError
        
        # Verificar que los módulos están disponibles
        assert subprocess is not None
        assert sys is not None
        assert version is not None
        assert PackageNotFoundError is not None
        assert Path is not None


if __name__ == "__main__":
    pytest.main([__file__])