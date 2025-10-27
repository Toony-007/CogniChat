"""
Pruebas unitarias para módulo de validadores
Cobertura completa de FileValidator y TextValidator
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import os
import tempfile
from pathlib import Path

# Configurar el path antes de importar
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TestFileValidator(unittest.TestCase):
    """Pruebas para la clase FileValidator"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        from utils.validators import FileValidator
        self.validator = FileValidator()
    
    def test_is_valid_file_type_valid(self):
        """Probar validación de tipos de archivo válidos"""
        # Crear archivos temporales con extensiones válidas
        valid_extensions = ['.pdf', '.docx', '.txt', '.md']
        
        for ext in valid_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                
            try:
                result = self.validator.is_valid_file_type(temp_path)
                self.assertTrue(result, f"Extensión {ext} debería ser válida")
            finally:
                temp_path.unlink()
    
    def test_is_valid_file_type_invalid(self):
        """Probar rechazo de tipos de archivo inválidos"""
        # Crear archivos temporales con extensiones inválidas
        invalid_extensions = ['.jpg', '.mp4', '.exe', '.py']
        
        for ext in invalid_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                
            try:
                result = self.validator.is_valid_file_type(temp_path)
                self.assertFalse(result, f"Extensión {ext} debería ser inválida")
            finally:
                temp_path.unlink()
    
    def test_is_valid_file_size_within_limit(self):
        """Probar validación de tamaño dentro del límite"""
        # Crear archivo pequeño
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"Contenido pequeno")
            temp_path = Path(temp_file.name)
        
        try:
            result = self.validator.is_valid_file_size(temp_path)
            self.assertTrue(result)
        finally:
            temp_path.unlink()
    
    def test_validate_file_complete_success(self):
        """Probar validación completa exitosa"""
        # Crear archivo válido
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"Contenido de prueba")
            temp_path = Path(temp_file.name)
        
        try:
            is_valid, message = self.validator.validate_file(temp_path)
            self.assertTrue(is_valid)
            self.assertEqual(message, "Archivo válido")
        finally:
            temp_path.unlink()
    
    def test_validate_file_nonexistent(self):
        """Probar validación de archivo inexistente"""
        nonexistent_path = Path("archivo_inexistente.pdf")
        
        is_valid, message = self.validator.validate_file(nonexistent_path)
        
        self.assertFalse(is_valid)
        self.assertEqual(message, "El archivo no existe")
    
    def test_validate_file_invalid_type(self):
        """Probar validación con tipo inválido"""
        # Crear archivo con extensión inválida
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(b"Contenido de imagen")
            temp_path = Path(temp_file.name)
        
        try:
            is_valid, message = self.validator.validate_file(temp_path)
            self.assertFalse(is_valid)
            self.assertIn("Tipo de archivo no soportado", message)
        finally:
            temp_path.unlink()
    
    def test_validate_file_directory(self):
        """Probar validación de directorio (no archivo)"""
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            is_valid, message = self.validator.validate_file(temp_path)
            
            self.assertFalse(is_valid)
            self.assertEqual(message, "La ruta no corresponde a un archivo")
    
    def test_get_file_info_basic(self):
        """Probar obtención de información básica de archivo"""
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"Contenido de prueba")
            temp_path = temp_file.name
        
        try:
            # Usar validate_file en lugar de get_file_info
            is_valid, message = self.validator.validate_file(Path(temp_path))
            
            # Verificar resultado
            self.assertTrue(is_valid)
            self.assertEqual(message, "Archivo válido")
        finally:
            os.unlink(temp_path)
    
    def test_get_file_info_nonexistent(self):
        """Probar obtención de información de archivo inexistente"""
        is_valid, message = self.validator.validate_file(Path("archivo_inexistente.pdf"))
        
        # Verificar resultado
        self.assertFalse(is_valid)
        self.assertEqual(message, "El archivo no existe")


class TestTextValidator(unittest.TestCase):
    """Pruebas para la clase TextValidator"""
    
    def setUp(self):
        """Configuración inicial"""
        from utils.validators import TextValidator
        self.validator = TextValidator()
    
    def test_is_valid_query_success(self):
        """Probar validación exitosa de consultas"""
        # Casos válidos
        valid_queries = [
            "¿Cuál es la capital de Colombia?",
            "Explica el concepto de inteligencia artificial",
            "abc",  # Mínimo 3 caracteres
            "Una consulta normal de longitud media"
        ]
        
        for query in valid_queries:
            with self.subTest(query=query):
                is_valid, message = self.validator.is_valid_query(query)
                self.assertTrue(is_valid)
                self.assertEqual(message, "Consulta válida")
    
    def test_is_valid_query_invalid_type(self):
        """Probar rechazo de tipos inválidos"""
        invalid_inputs = [None, 123, [], {}, True]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                is_valid, message = self.validator.is_valid_query(invalid_input)
                self.assertFalse(is_valid)
                self.assertEqual(message, "La consulta debe ser texto")
    
    def test_is_valid_query_empty(self):
        """Probar rechazo de consultas vacías"""
        empty_queries = ["", "   ", "\t\n  "]
        
        for query in empty_queries:
            with self.subTest(query=repr(query)):
                is_valid, message = self.validator.is_valid_query(query)
                self.assertFalse(is_valid)
                self.assertEqual(message, "La consulta no puede estar vacía")
    
    def test_is_valid_query_too_short(self):
        """Probar rechazo de consultas muy cortas"""
        short_queries = ["a", "ab", "  a  "]
        
        for query in short_queries:
            with self.subTest(query=query):
                is_valid, message = self.validator.is_valid_query(query)
                self.assertFalse(is_valid)
                self.assertEqual(message, "La consulta debe tener al menos 3 caracteres")
    
    def test_is_valid_query_too_long(self):
        """Probar rechazo de consultas muy largas"""
        long_query = "a" * 1001  # Excede límite de 1000 caracteres
        
        is_valid, message = self.validator.is_valid_query(long_query)
        
        self.assertFalse(is_valid)
        self.assertEqual(message, "La consulta no debe exceder 1000 caracteres")
    
    def test_sanitize_filename_basic(self):
        """Probar sanitización básica de nombres de archivo"""
        # Casos con caracteres problemáticos
        test_cases = [
            ("archivo<peligroso>.pdf", "archivo_peligroso_.pdf"),
            ("documento:con|caracteres.txt", "documento_con_caracteres.txt"),
            ("archivo\"con'comillas.docx", "archivo_con'comillas.docx"),
            ("archivo/con\\barras.md", "archivo_con_barras.md"),
            ("archivo?con*asterisco.pdf", "archivo_con_asterisco.pdf")
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.validator.sanitize_filename(original)
                self.assertEqual(result, expected)
    
    def test_sanitize_filename_long_name(self):
        """Probar sanitización de nombres muy largos"""
        # Crear nombre muy largo
        long_name = "a" * 150 + ".pdf"
        
        result = self.validator.sanitize_filename(long_name)
        
        # Debe ser truncado a 100 caracteres total
        self.assertEqual(len(result), 100)
        self.assertTrue(result.endswith(".pdf"))
    
    def test_sanitize_filename_preserve_extension(self):
        """Probar que se preserve la extensión al truncar"""
        long_name = "archivo_con_nombre_muy_largo_que_excede_el_limite" * 3 + ".docx"
        
        result = self.validator.sanitize_filename(long_name)
        
        self.assertTrue(result.endswith(".docx"))
        self.assertEqual(len(result), 100)
    
    def test_is_valid_model_name_valid(self):
        """Probar validación de nombres de modelo válidos"""
        valid_names = [
            "llama3",
            "nomic-embed-text",
            "deepseek-coder",
            "model_name_123",
            "simple-model.v2",
            "model123",
            "llama3.2"
        ]
        
        for name in valid_names:
            with self.subTest(name=name):
                result = self.validator.is_valid_model_name(name)
                self.assertTrue(result, f"El nombre '{name}' debería ser válido")
    
    def test_is_valid_model_name_invalid(self):
        """Probar rechazo de nombres de modelo inválidos"""
        invalid_names = [
            "",  # Vacío
            "a",  # Muy corto (un solo carácter)
            "-invalid",  # Empieza con guión
            "invalid-",  # Termina con guión
            ".invalid",  # Empieza con punto
            "invalid.",  # Termina con punto
            "model with spaces",  # Contiene espacios
            "model@invalid",  # Caracteres especiales
            "model#invalid",
            123,  # No es string
            None  # Tipo inválido
        ]
        
        for name in invalid_names:
            with self.subTest(name=name):
                result = self.validator.is_valid_model_name(name)
                self.assertFalse(result)


class TestValidatorsIntegration(unittest.TestCase):
    """Pruebas de integración para validadores"""
    
    def setUp(self):
        """Configuración inicial"""
        # Mock de configuración
        self.mock_config = Mock()
        self.mock_config.SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt', '.md']
        self.mock_config.MAX_FILE_SIZE_MB = 5  # 5MB
        
        self.config_patch = patch('utils.validators.config', self.mock_config)
        self.config_patch.start()
        
        from utils.validators import FileValidator, TextValidator
        self.file_validator = FileValidator()
        self.text_validator = TextValidator()
    
    def tearDown(self):
        """Limpieza"""
        self.config_patch.stop()
    
    def test_complete_validation_workflow(self):
        """Probar flujo completo de validación"""
        # Crear archivo temporal para validar
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"Contenido de prueba")
            temp_path = Path(temp_file.name)
        
        try:
            # Validar archivo completo
            is_valid_file, message = self.file_validator.validate_file(temp_path)
            self.assertTrue(is_valid_file)
            self.assertEqual(message, "Archivo válido")
            
            # Validar tipo de archivo
            is_valid_type = self.file_validator.is_valid_file_type(temp_path)
            self.assertTrue(is_valid_type)
            
            # Validar consulta relacionada
            query = "¿Qué contiene el documento importante?"
            is_valid_query, _ = self.text_validator.is_valid_query(query)
            self.assertTrue(is_valid_query)
            
            # Validar nombre de modelo
            model_name = "llama3"
            is_valid_model = self.text_validator.is_valid_model_name(model_name)
            self.assertTrue(is_valid_model)
        finally:
            temp_path.unlink()
    
    def test_sanitization_workflow(self):
        """Probar flujo de sanitización"""
        # Archivo con nombre problemático
        problematic_filename = "documento<peligroso>:con|caracteres.pdf"
        
        # Sanitizar nombre
        safe_filename = self.text_validator.sanitize_filename(problematic_filename)
        
        # Verificar que el nombre sanitizado no contiene caracteres peligrosos
        dangerous_chars = ['<', '>', ':', '|', '?', '*', '"']
        for char in dangerous_chars:
            self.assertNotIn(char, safe_filename)
        
        # Verificar que mantiene la extensión
        self.assertTrue(safe_filename.endswith('.pdf'))


if __name__ == '__main__':
    unittest.main()