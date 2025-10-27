"""Tests automatizados para los módulos de utilidades
Pruebas de funcionalidad de validators, logger, error_handler, etc.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import tempfile
from utils.validators import FileValidator, ConfigValidator, TextValidator
from utils.logger import setup_logger
from utils.error_handler import ErrorHandler

class TestFileValidator(unittest.TestCase):
    """Tests para FileValidator"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.validator = FileValidator()
        self.test_dir = Path(tempfile.mkdtemp())
        
    def test_file_validator_exists(self):
        """Test que FileValidator existe y tiene métodos esperados"""
        self.assertTrue(hasattr(FileValidator, 'is_valid_file_type'))
        self.assertTrue(hasattr(FileValidator, 'is_valid_file_size'))
        self.assertTrue(hasattr(FileValidator, 'validate_file'))
        
    def test_config_validator_exists(self):
        """Test que ConfigValidator existe y tiene métodos esperados"""
        self.assertTrue(hasattr(ConfigValidator, 'validate_chunk_size'))
        self.assertTrue(hasattr(ConfigValidator, 'validate_chunk_overlap'))
        self.assertTrue(hasattr(ConfigValidator, 'validate_similarity_threshold'))
        
    def test_text_validator_exists(self):
        """Test que TextValidator existe y tiene métodos esperados"""
        self.assertTrue(hasattr(TextValidator, 'is_valid_query'))
        self.assertTrue(hasattr(TextValidator, 'sanitize_filename'))
        self.assertTrue(hasattr(TextValidator, 'is_valid_model_name'))
        
    def test_logger_setup(self):
        """Test que setup_logger funciona correctamente"""
        logger = setup_logger()
        self.assertIsNotNone(logger)
        
    def test_error_handler_initialization(self):
        """Test que ErrorHandler se inicializa correctamente"""
        error_handler = ErrorHandler()
        self.assertIsNotNone(error_handler)





if __name__ == '__main__':
    unittest.main()