"""
Tests para los validadores de CogniChat
"""

import pytest
from pathlib import Path
from utils.validators import FileValidator, ConfigValidator, TextValidator

class TestFileValidator:
    """Tests para FileValidator"""
    
    def test_is_valid_file_type(self):
        """Test validación de tipos de archivo"""
        # Casos válidos
        assert FileValidator.is_valid_file_type(Path("test.pdf"))
        assert FileValidator.is_valid_file_type(Path("test.docx"))
        assert FileValidator.is_valid_file_type(Path("test.txt"))
        
        # Casos inválidos
        assert not FileValidator.is_valid_file_type(Path("test.exe"))
        assert not FileValidator.is_valid_file_type(Path("test.bat"))

class TestConfigValidator:
    """Tests para ConfigValidator"""
    
    def test_validate_chunk_size(self):
        """Test validación de tamaño de chunk"""
        # Casos válidos
        valid, msg = ConfigValidator.validate_chunk_size(1000)
        assert valid
        
        # Casos inválidos
        valid, msg = ConfigValidator.validate_chunk_size(50)
        assert not valid
        
        valid, msg = ConfigValidator.validate_chunk_size(15000)
        assert not valid
    
    def test_validate_similarity_threshold(self):
        """Test validación de umbral de similitud"""
        # Casos válidos
        valid, msg = ConfigValidator.validate_similarity_threshold(0.5)
        assert valid
        
        valid, msg = ConfigValidator.validate_similarity_threshold(0.0)
        assert valid
        
        valid, msg = ConfigValidator.validate_similarity_threshold(1.0)
        assert valid
        
        # Casos inválidos
        valid, msg = ConfigValidator.validate_similarity_threshold(-0.1)
        assert not valid
        
        valid, msg = ConfigValidator.validate_similarity_threshold(1.5)
        assert not valid

class TestTextValidator:
    """Tests para TextValidator"""
    
    def test_is_valid_query(self):
        """Test validación de consultas"""
        # Casos válidos
        valid, msg = TextValidator.is_valid_query("¿Qué es la inteligencia artificial?")
        assert valid
        
        # Casos inválidos
        valid, msg = TextValidator.is_valid_query("")
        assert not valid
        
        valid, msg = TextValidator.is_valid_query("ab")
        assert not valid
        
        valid, msg = TextValidator.is_valid_query("a" * 1001)
        assert not valid
    
    def test_sanitize_filename(self):
        """Test sanitización de nombres de archivo"""
        # Casos con caracteres especiales
        result = TextValidator.sanitize_filename("archivo<>:test.txt")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        
        # Casos con nombres muy largos
        long_name = "a" * 150 + ".txt"
        result = TextValidator.sanitize_filename(long_name)
        assert len(result) <= 100