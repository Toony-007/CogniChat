"""Pruebas unitarias para el módulo document_upload"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import modules.document_upload as document_upload


class TestDocumentUpload:
    """Pruebas para las funciones de document_upload"""
    
    def setup_method(self):
        """Configuración antes de cada prueba"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file1 = self.temp_dir / "test1.txt"
        self.test_file2 = self.temp_dir / "test2.pdf"
        self.test_file1.write_text("Contenido de prueba 1")
        self.test_file2.write_text("Contenido de prueba 2")
    
    def teardown_method(self):
        """Limpieza después de cada prueba"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_get_uploaded_files_count(self):
        """Probar conteo de archivos cargados"""
        with patch.object(document_upload.config, 'UPLOADS_DIR', self.temp_dir):
            count = document_upload.get_uploaded_files_count()
            assert count == 2  # test1.txt y test2.pdf

    def test_get_uploaded_files_count_empty_directory(self):
        """Probar conteo con directorio vacío"""
        empty_dir = Path(tempfile.mkdtemp())
        
        try:
            with patch.object(document_upload.config, 'UPLOADS_DIR', empty_dir):
                count = document_upload.get_uploaded_files_count()
                assert count == 0
        finally:
            shutil.rmtree(empty_dir)

    def test_get_uploaded_files_count_error(self):
        """Probar conteo de archivos con directorio inexistente"""
        non_existent_dir = Path("/directorio/inexistente_dir")
        
        with patch.object(document_upload.config, 'UPLOADS_DIR', non_existent_dir):
            count = document_upload.get_uploaded_files_count()
            assert count == 0
            # No se espera que se llame a logger.error porque Path.glob no falla con directorios inexistentes

    def test_get_uploaded_files_list(self):
        """Probar obtención de lista de archivos"""
        with patch.object(document_upload.config, 'UPLOADS_DIR', self.temp_dir):
            files = document_upload.get_uploaded_files_list()
            assert len(files) == 2
            file_names = [f.name for f in files]
            assert "test1.txt" in file_names
            assert "test2.pdf" in file_names

    def test_get_uploaded_files_list_error(self):
        """Probar listado de archivos con directorio inexistente"""
        non_existent_dir = Path("/directorio/inexistente_dir")
        
        with patch.object(document_upload.config, 'UPLOADS_DIR', non_existent_dir):
            files = document_upload.get_uploaded_files_list()
            assert files == []

    def test_get_valid_uploaded_files(self):
        """Probar obtención de archivos válidos"""
        # Crear archivo oculto y directorio
        hidden_file = self.temp_dir / ".hidden"
        hidden_file.write_text("archivo oculto")
        test_dir = self.temp_dir / "subdirectorio"
        test_dir.mkdir()
        
        with patch.object(document_upload.config, 'UPLOADS_DIR', self.temp_dir):
            files = document_upload.get_valid_uploaded_files()
            assert len(files) == 2  # Solo test1.txt y test2.pdf
            file_names = [f.name for f in files]
            assert "test1.txt" in file_names
            assert "test2.pdf" in file_names
            assert ".hidden" not in file_names

    def test_get_valid_uploaded_files_error(self):
        """Probar obtención de archivos válidos con directorio inexistente"""
        non_existent_dir = Path("/directorio/inexistente_dir")
        
        with patch.object(document_upload.config, 'UPLOADS_DIR', non_existent_dir):
            files = document_upload.get_valid_uploaded_files()
            assert files == []

    @patch('streamlit.success')
    @patch('modules.document_upload.logger')
    def test_delete_file_success(self, mock_logger, mock_st_success):
        """Probar eliminación exitosa de archivo"""
        test_file = self.temp_dir / "archivo_a_eliminar.txt"
        test_file.write_text("contenido")
        
        assert test_file.exists()
        document_upload.delete_file(test_file)
        
        assert not test_file.exists()
        mock_st_success.assert_called_once()
        mock_logger.info.assert_called_once()

    @patch('modules.document_upload.error_handler')
    def test_delete_file_error(self, mock_error_handler):
        """Probar manejo de errores en eliminación de archivo"""
        non_existent_file = self.temp_dir / "no_existe.txt"
        
        document_upload.delete_file(non_existent_file)
        mock_error_handler.handle_error.assert_called_once()

    @patch('streamlit.success')
    @patch('modules.document_upload.logger')
    def test_clear_all_files_success(self, mock_logger, mock_st_success):
        """Probar limpieza exitosa de archivos"""
        with patch.object(document_upload.config, 'UPLOADS_DIR', self.temp_dir):
            document_upload.clear_all_files()
            
            mock_logger.info.assert_called_once()
            mock_st_success.assert_called_once()

    @patch('modules.document_upload.error_handler')
    def test_clear_all_files_error(self, mock_error_handler):
        """Probar manejo de errores en limpieza de archivos"""
        with patch('modules.document_upload.get_uploaded_files_list', side_effect=Exception("Error de prueba")):
            document_upload.clear_all_files()
            mock_error_handler.handle_error.assert_called_once()

    @patch('time.sleep')
    @patch('streamlit.empty')
    @patch('streamlit.progress')
    @patch('streamlit.success')
    @patch('modules.document_upload.logger')
    def test_process_uploaded_files_success(self, mock_logger, mock_st_success, mock_progress, mock_empty, mock_sleep):
        """Probar procesamiento exitoso de archivos"""
        # Crear archivos de prueba
        mock_file1 = MagicMock()
        mock_file1.name = "test1.txt"
        mock_file1.size = 1000
        mock_file1.getbuffer.return_value = b"contenido test1"
        
        mock_file2 = MagicMock()
        mock_file2.name = "test2.pdf"
        mock_file2.size = 2000
        mock_file2.getbuffer.return_value = b"contenido test2"
        
        mock_files = [mock_file1, mock_file2]
        
        # Mock de elementos de Streamlit
        mock_progress_bar = MagicMock()
        mock_status_text = MagicMock()
        mock_progress.return_value = mock_progress_bar
        mock_empty.return_value = mock_status_text
        
        with patch.object(document_upload.config, 'UPLOADS_DIR', self.temp_dir), \
             patch.object(document_upload.config, 'MAX_FILE_SIZE_MB', 10), \
             patch('builtins.open', mock_open()) as mock_file:
            
            document_upload.process_uploaded_files(mock_files)
            
            # Verificar que se guardaron los archivos
            assert mock_file.call_count == 2
            mock_logger.info.assert_called()
            mock_st_success.assert_called_once()

    @patch('modules.document_upload.error_handler')
    def test_process_uploaded_files_file_too_large(self, mock_error_handler):
        """Probar procesamiento con archivo muy grande"""
        # Crear archivo de prueba muy grande
        mock_file = MagicMock()
        mock_file.name = "archivo_grande.pdf"
        mock_file.size = 200 * 1024 * 1024  # 200MB
        
        with patch.object(document_upload.config, 'MAX_FILE_SIZE_MB', 10), \
             patch('streamlit.progress'), \
             patch('streamlit.empty'):
            
            document_upload.process_uploaded_files([mock_file])
            
            mock_error_handler.handle_warning.assert_called_once()

    @patch('modules.document_upload.error_handler')
    def test_process_uploaded_files_error(self, mock_error_handler):
        """Probar manejo de errores en procesamiento"""
        mock_files = [MagicMock(name="test.txt")]
        
        with patch('streamlit.progress', side_effect=Exception("Error de prueba")):
            document_upload.process_uploaded_files(mock_files)
            
            mock_error_handler.handle_error.assert_called_once()


class TestDocumentUploadRender:
    """Pruebas para la función render de document_upload"""
    
    @patch('modules.document_upload.get_valid_uploaded_files')
    @patch('streamlit.columns')
    @patch('streamlit.header')
    def test_render_basic_structure(self, mock_header, mock_columns, mock_get_files):
        """Probar estructura básica del render"""
        mock_get_files.return_value = []
        mock_col = MagicMock()
        mock_columns.return_value = [mock_col, mock_col, mock_col]
        
        with patch('streamlit.markdown'), \
             patch('streamlit.divider'), \
             patch('streamlit.subheader'), \
             patch('streamlit.expander'), \
             patch('streamlit.file_uploader'), \
             patch.object(document_upload.config, 'SUPPORTED_FORMATS', ['.pdf', '.txt']), \
             patch.object(document_upload.config, 'MAX_FILE_SIZE_MB', 10):
            
            document_upload.render()
            
            mock_header.assert_called_once_with("📄 Gestión de Documentos")
            mock_columns.assert_called()

    @patch('modules.document_upload.get_valid_uploaded_files')
    @patch('streamlit.file_uploader')
    def test_render_with_file_uploader(self, mock_file_uploader, mock_get_files):
        """Probar render con file uploader"""
        mock_get_files.return_value = []
        mock_file_uploader.return_value = None
        
        with patch('streamlit.header'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.markdown'), \
             patch('streamlit.divider'), \
             patch('streamlit.subheader'), \
             patch('streamlit.expander'), \
             patch.object(document_upload.config, 'SUPPORTED_FORMATS', ['.pdf', '.txt']), \
             patch.object(document_upload.config, 'MAX_FILE_SIZE_MB', 10):
            
            mock_col = MagicMock()
            mock_columns.return_value = [mock_col, mock_col, mock_col]
            
            document_upload.render()
            
            mock_file_uploader.assert_called()

    @patch('modules.document_upload.get_valid_uploaded_files')
    def test_render_with_uploaded_files(self, mock_get_files):
        """Probar render con archivos cargados"""
        # Simular archivos cargados
        mock_files = [Path("test1.txt"), Path("test2.pdf")]
        mock_get_files.return_value = mock_files
        
        with patch('streamlit.header'), \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.markdown'), \
             patch('streamlit.divider'), \
             patch('streamlit.subheader'), \
             patch('streamlit.expander'), \
             patch('streamlit.file_uploader'), \
             patch.object(document_upload.config, 'SUPPORTED_FORMATS', ['.pdf', '.txt']), \
             patch.object(document_upload.config, 'MAX_FILE_SIZE_MB', 10):
            
            mock_col = MagicMock()
            mock_columns.return_value = [mock_col, mock_col, mock_col]
            
            document_upload.render()
            
            mock_get_files.assert_called()