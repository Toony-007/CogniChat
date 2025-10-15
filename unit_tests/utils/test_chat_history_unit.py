"""
Pruebas Unitarias - Módulo Chat History

Este archivo contiene pruebas unitarias específicas para el módulo chat_history,
enfocándose en la gestión del historial de conversaciones y exportación de mensajes.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from io import BytesIO

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.chat_history import ChatHistoryManager, ChatExporter

class TestChatHistoryManagerUnit(unittest.TestCase):
    """Pruebas unitarias para ChatHistoryManager."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        with patch('config.settings.config') as mock_config:
            mock_config.DATA_DIR = Path(tempfile.mkdtemp())
            self.chat_manager = ChatHistoryManager()
    
    def test_chat_history_manager_initialization(self):
        """Prueba la inicialización correcta del gestor de historial."""
        self.assertIsInstance(self.chat_manager.history_dir, Path)
        self.assertTrue(self.chat_manager.history_dir.exists())
    
    @patch('utils.chat_history.logger')
    def test_save_conversation_with_custom_name(self, mock_logger):
        """Prueba guardar conversación con nombre personalizado."""
        messages = [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "¡Hola! ¿Cómo puedo ayudarte?"}
        ]
        conversation_name = "test_conversation"
        
        # Mock para el archivo JSON
        with patch('builtins.open', mock_open()) as mock_file:
            result = self.chat_manager.save_conversation(messages, conversation_name)
        
        # Verificar resultado
        self.assertEqual(result, f"{conversation_name}.json")
        
        # Verificar que se intentó escribir el archivo
        mock_file.assert_called_once()
        
        # Verificar log
        mock_logger.info.assert_called_once()
    
    @patch('utils.chat_history.logger')
    def test_save_conversation_auto_name(self, mock_logger):
        """Prueba guardar conversación con nombre automático."""
        messages = [{"role": "user", "content": "Test"}]
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('utils.chat_history.datetime') as mock_datetime:
                # Mock del datetime.now() y strftime
                mock_now = mock_datetime.now.return_value
                mock_now.strftime.return_value = "20240101_120000"
                mock_now.isoformat.return_value = "2024-01-01T12:00:00"
                
                result = self.chat_manager.save_conversation(messages)
        
        # Verificar que se generó nombre automático
        self.assertEqual(result, "conversacion_20240101_120000.json")
        
        # Verificar que se escribió el archivo
        mock_file.assert_called_once()
        
        # Verificar que se registró en el log
        mock_logger.info.assert_called_once()
    
    @patch('utils.chat_history.logger')
    def test_save_conversation_error_handling(self, mock_logger):
        """Prueba el manejo de errores al guardar conversación."""
        messages = [{"role": "user", "content": "Test"}]
        
        # Simular error de escritura
        with patch('builtins.open', side_effect=OSError("Error de escritura")):
            with self.assertRaises(Exception):
                self.chat_manager.save_conversation(messages, "test")
        
        # Verificar que se registró el error
        mock_logger.error.assert_called_once()
    
    @patch('utils.chat_history.logger')
    def test_load_conversation_success(self, mock_logger):
        """Prueba cargar conversación exitosamente."""
        # Datos de prueba
        test_data = {
            "metadata": {"name": "Test", "created_at": "2024-01-01"},
            "messages": [{"role": "user", "content": "Test"}]
        }
        
        # Mock del archivo
        with patch('builtins.open', mock_open(read_data=json.dumps(test_data))):
            with patch.object(Path, 'exists', return_value=True):
                result = self.chat_manager.load_conversation("test.json")
        
        # Verificar resultado
        self.assertEqual(result, test_data)
        mock_logger.info.assert_called_once()
    
    @patch('utils.chat_history.logger')
    def test_load_conversation_file_not_found(self, mock_logger):
        """Prueba cargar conversación cuando el archivo no existe."""
        with patch.object(Path, 'exists', return_value=False):
            result = self.chat_manager.load_conversation("nonexistent.json")
        
        # Verificar que retorna None
        self.assertIsNone(result)
        
        # Verificar warning
        mock_logger.warning.assert_called_once()
    
    @patch('utils.chat_history.logger')
    def test_load_conversation_json_error(self, mock_logger):
        """Prueba cargar conversación con JSON inválido."""
        # Mock archivo con JSON inválido
        with patch('builtins.open', mock_open(read_data="invalid json")):
            with patch.object(Path, 'exists', return_value=True):
                result = self.chat_manager.load_conversation("invalid.json")
        
        # Verificar que retorna None
        self.assertIsNone(result)
        
        # Verificar que se registró el error
        mock_logger.error.assert_called_once()
    
    @patch('utils.chat_history.logger')
    def test_list_conversations(self, mock_logger):
        """Prueba listar conversaciones."""
        # Mock de archivos JSON
        mock_files = [
            Mock(name="conv1.json", stem="conv1"),
            Mock(name="conv2.json", stem="conv2")
        ]
        
        # Datos de prueba para cada archivo
        test_data_1 = {
            "metadata": {
                "name": "Conversación 1",
                "created_at": "2024-01-01T10:00:00",
                "total_messages": 5
            }
        }
        test_data_2 = {
            "metadata": {
                "name": "Conversación 2", 
                "created_at": "2024-01-02T10:00:00",
                "total_messages": 3
            }
        }
        
        # Mock del stat para obtener tamaño de archivo
        for mock_file in mock_files:
            mock_file.stat.return_value.st_size = 1024
        
        # Mock glob para retornar archivos
        with patch.object(Path, 'glob', return_value=mock_files):
            # Mock open para cada archivo
            with patch('builtins.open', mock_open()) as mock_file_open:
                mock_file_open.side_effect = [
                    mock_open(read_data=json.dumps(test_data_1)).return_value,
                    mock_open(read_data=json.dumps(test_data_2)).return_value
                ]
                
                result = self.chat_manager.list_conversations()
        
        # Verificar resultado
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        
        # Verificar que están ordenados por fecha (más reciente primero)
        self.assertEqual(result[0]["name"], "Conversación 2")
        self.assertEqual(result[1]["name"], "Conversación 1")
    
    @patch('utils.chat_history.logger')
    def test_list_conversations_empty(self, mock_logger):
        """Prueba listar conversaciones cuando no hay archivos."""
        with patch.object(Path, 'glob', return_value=[]):
            result = self.chat_manager.list_conversations()
        
        # Verificar que retorna lista vacía
        self.assertEqual(result, [])
    
    @patch('utils.chat_history.logger')
    def test_delete_conversation_success(self, mock_logger):
        """Prueba eliminación exitosa de conversación."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_file.unlink.return_value = None
        
        # Usar patch para mockear el método __truediv__ del Path
        with patch.object(Path, '__truediv__', return_value=mock_file):
            result = self.chat_manager.delete_conversation("test.json")
        
        # Verificar resultado
        self.assertTrue(result)
        
        # Verificar que se llamó unlink
        mock_file.unlink.assert_called_once()

    @patch('utils.chat_history.logger')
    def test_delete_conversation_not_found(self, mock_logger):
        """Prueba eliminación de conversación inexistente."""
        mock_file = Mock()
        mock_file.exists.return_value = False
        
        # Usar patch para mockear el método __truediv__ del Path
        with patch.object(Path, '__truediv__', return_value=mock_file):
            result = self.chat_manager.delete_conversation("nonexistent.json")
        
        # Verificar resultado
        self.assertFalse(result)
        
        # Verificar warning
        mock_logger.warning.assert_called_once()

    @patch('utils.chat_history.logger')
    def test_delete_conversation_error(self, mock_logger):
        """Prueba manejo de errores al eliminar conversación."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_file.unlink.side_effect = OSError("Error de permisos")
        
        # Usar patch para mockear el método __truediv__ del Path
        with patch.object(Path, '__truediv__', return_value=mock_file):
            result = self.chat_manager.delete_conversation("test.json")
        
        # Verificar resultado
        self.assertFalse(result)
        
        # Verificar que se registró el error
        mock_logger.error.assert_called_once()


class TestChatExporterUnit(unittest.TestCase):
    """Pruebas unitarias para ChatExporter."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        with patch('config.settings.config') as mock_config:
            mock_config.DATA_DIR = Path(tempfile.mkdtemp())
            self.chat_exporter = ChatExporter()
    
    def test_chat_exporter_initialization(self):
        """Prueba la inicialización correcta del exportador."""
        self.assertIsInstance(self.chat_exporter.temp_dir, Path)
        self.assertTrue(self.chat_exporter.temp_dir.exists())
    
    def test_get_message_text_for_clipboard_with_metadata(self):
        """Prueba obtener texto de mensaje para portapapeles con metadatos."""
        message = {
            "role": "user",
            "content": "¿Cuál es la capital de Francia?",
            "timestamp": "2024-01-01T10:00:00"
        }
        
        result = self.chat_exporter.get_message_text_for_clipboard(message, include_metadata=True)
        
        # Verificar que contiene el contenido y metadatos
        self.assertIn("¿Cuál es la capital de Francia?", result)
        self.assertIn("Usuario", result)
        self.assertIn("2024-01-01T10:00:00", result)
    
    def test_get_message_text_for_clipboard_without_metadata(self):
        """Prueba obtener texto de mensaje para portapapeles sin metadatos."""
        message = {
            "role": "assistant",
            "content": "La capital de Francia es París.",
            "timestamp": "2024-01-01T10:01:00"
        }
        
        result = self.chat_exporter.get_message_text_for_clipboard(message, include_metadata=False)
        
        # Verificar que solo contiene el contenido
        self.assertEqual(result.strip(), "La capital de Francia es París.")
        self.assertNotIn("Asistente", result)
        self.assertNotIn("2024-01-01T10:01:00", result)
    
    @patch('utils.chat_history.DOCX_AVAILABLE', True)
    @patch('utils.chat_history.Document')
    @patch('utils.chat_history.BytesIO')
    @patch('utils.chat_history.WD_ALIGN_PARAGRAPH')
    def test_export_message_to_docx_available(self, mock_align, mock_bytesio, mock_document):
        """Prueba exportar mensaje a DOCX cuando está disponible."""
        # Configurar mocks
        mock_doc = Mock()
        mock_document.return_value = mock_doc
        mock_buffer = Mock(spec=BytesIO)
        mock_bytesio.return_value = mock_buffer
        mock_align.CENTER = 1
        
        # Configurar métodos del documento
        mock_doc.add_heading.return_value = Mock()
        mock_doc.add_paragraph.return_value = Mock()
        mock_doc.save = Mock()
        
        # Configurar tabla y celdas
        mock_table = Mock()
        mock_row = Mock()
        mock_cell1 = Mock()
        mock_cell2 = Mock()
        mock_cell1.text = ""
        mock_cell2.text = ""
        mock_row.cells = [mock_cell1, mock_cell2]
        mock_table.add_row.return_value = mock_row
        mock_doc.add_table.return_value = mock_table
        
        message = {
            "role": "assistant",
            "content": "Test content",
            "timestamp": "2024-01-01T10:00:00"
        }
        
        result = self.chat_exporter.export_message_to_docx(message)
        
        # Verificar que se creó el documento
        mock_document.assert_called_once()
        
        # Verificar que se retornó el buffer
        self.assertEqual(result, mock_buffer)
    
    @patch('utils.chat_history.DOCX_AVAILABLE', False)
    def test_export_message_to_docx_not_available(self):
        """Prueba exportar mensaje a DOCX cuando no está disponible."""
        message = {"role": "user", "content": "Test"}
        
        with self.assertRaises(ImportError):
            self.chat_exporter.export_message_to_docx(message)
    
    @patch('utils.chat_history.PDF_AVAILABLE', True)
    @patch('utils.chat_history.SimpleDocTemplate')
    @patch('utils.chat_history.getSampleStyleSheet')
    @patch('utils.chat_history.BytesIO')
    @patch('utils.chat_history.ParagraphStyle')
    @patch('utils.chat_history.Paragraph')
    @patch('utils.chat_history.Spacer')
    @patch('utils.chat_history.HexColor')
    def test_export_message_to_pdf_available(self, mock_hex, mock_spacer, mock_paragraph, mock_style, mock_bytesio, mock_styles, mock_doc):
        """Prueba exportar mensaje a PDF cuando está disponible."""
        # Configurar mocks
        mock_buffer = Mock(spec=BytesIO)
        mock_bytesio.return_value = mock_buffer
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance
        mock_styles.return_value = {'Normal': Mock(), 'Heading1': Mock(), 'Heading2': Mock()}
        mock_style.return_value = Mock()
        mock_paragraph.return_value = Mock()
        mock_spacer.return_value = Mock()
        mock_hex.return_value = Mock()
        
        message = {
            "role": "assistant",
            "content": "Test response",
            "timestamp": "2024-01-01T10:00:00"
        }
        
        result = self.chat_exporter.export_message_to_pdf(message)
        
        # Verificar que se creó el documento PDF
        mock_doc.assert_called_once()
        
        # Verificar que se retornó el buffer
        self.assertEqual(result, mock_buffer)
    
    @patch('utils.chat_history.PDF_AVAILABLE', False)
    def test_export_message_to_pdf_not_available(self):
        """Prueba exportar mensaje a PDF cuando no está disponible."""
        message = {"role": "assistant", "content": "Test"}
        
        with self.assertRaises(ImportError):
            self.chat_exporter.export_message_to_pdf(message)
    
    @patch('utils.chat_history.DOCX_AVAILABLE', True)
    @patch('utils.chat_history.Document')
    @patch('utils.chat_history.BytesIO')
    @patch('utils.chat_history.WD_ALIGN_PARAGRAPH')
    def test_export_conversation_to_docx(self, mock_align, mock_bytesio, mock_document):
        """Prueba exportar conversación completa a DOCX."""
        # Configurar mocks
        mock_doc = Mock()
        mock_document.return_value = mock_doc
        mock_buffer = Mock(spec=BytesIO)
        mock_bytesio.return_value = mock_buffer
        mock_align.CENTER = 1
        
        # Configurar métodos del documento
        mock_doc.add_heading.return_value = Mock()
        mock_doc.add_paragraph.return_value = Mock()
        mock_doc.add_page_break.return_value = Mock()
        mock_doc.save = Mock()
        
        # Configurar tabla y celdas
        mock_table = Mock()
        mock_row = Mock()
        mock_cell1 = Mock()
        mock_cell2 = Mock()
        mock_cell1.text = ""
        mock_cell2.text = ""
        mock_row.cells = [mock_cell1, mock_cell2]
        mock_table.add_row.return_value = mock_row
        mock_doc.add_table.return_value = mock_table
        
        conversation = [
            {"role": "user", "content": "Pregunta", "timestamp": "2024-01-01T10:00:00"},
            {"role": "assistant", "content": "Respuesta", "timestamp": "2024-01-01T10:01:00"}
        ]
        
        result = self.chat_exporter.export_conversation_to_docx(conversation)
        
        # Verificar que se creó el documento
        mock_document.assert_called_once()
        
        # Verificar que se retornó el buffer
        self.assertEqual(result, mock_buffer)

    @patch('utils.chat_history.DOCX_AVAILABLE', True)
    @patch('utils.chat_history.Document')
    @patch('utils.chat_history.BytesIO')
    @patch('utils.chat_history.WD_ALIGN_PARAGRAPH')
    def test_export_single_message_to_docx(self, mock_align, mock_bytesio, mock_document):
        """Prueba exportar un solo mensaje a DOCX."""
        # Configurar mocks
        mock_doc = Mock()
        mock_document.return_value = mock_doc
        mock_buffer = Mock(spec=BytesIO)
        mock_bytesio.return_value = mock_buffer
        mock_align.CENTER = 1
        
        # Configurar métodos del documento
        mock_doc.add_heading.return_value = Mock()
        mock_doc.add_paragraph.return_value = Mock()
        mock_doc.save = Mock()
        
        # Configurar tabla y celdas
        mock_table = Mock()
        mock_row = Mock()
        mock_cell1 = Mock()
        mock_cell2 = Mock()
        mock_cell1.text = ""
        mock_cell2.text = ""
        mock_row.cells = [mock_cell1, mock_cell2]
        mock_table.add_row.return_value = mock_row
        mock_doc.add_table.return_value = mock_table
        
        message = {"role": "assistant", "content": "Test content"}
        
        result = self.chat_exporter.export_message_to_docx(message)
        
        # Verificar que se creó el documento
        mock_document.assert_called_once()
        
        # Verificar que se retornó el buffer
        self.assertEqual(result, mock_buffer)
    
    def test_message_role_formatting(self):
        """Prueba el formateo correcto de roles de mensajes."""
        user_message = {"role": "user", "content": "Test user"}
        assistant_message = {"role": "assistant", "content": "Test assistant"}
        
        user_text = self.chat_exporter.get_message_text_for_clipboard(user_message, True)
        assistant_text = self.chat_exporter.get_message_text_for_clipboard(assistant_message, True)
        
        # Verificar que los roles se formatean correctamente
        self.assertIn("Usuario", user_text)
        self.assertIn("Asistente", assistant_text)


if __name__ == '__main__':
    unittest.main()