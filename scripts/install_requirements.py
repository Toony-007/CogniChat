#!/usr/bin/env python3
"""
Script de instalación automática de dependencias para CogniChat
Verifica e instala todas las dependencias necesarias para el análisis cualitativo avanzado
"""

import subprocess
import sys
from pathlib import Path

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback para Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

def check_package_installed(package_name):
    """Verificar si un paquete está instalado"""
    try:
        version(package_name)
        return True
    except PackageNotFoundError:
        return False

def install_package(package):
    """Instalar un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Función principal de instalación"""
    print("🚀 CogniChat - Instalador de Dependencias")
    print("=" * 50)
    
    # Leer requirements.txt (en el directorio padre)
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: No se encontró el archivo requirements.txt")
        return
    
    with open(requirements_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extraer paquetes (ignorar comentarios y líneas vacías)
    packages = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Extraer nombre del paquete (antes de >= o ==)
            package_name = line.split('>=')[0].split('==')[0].strip()
            if package_name != 'pathlib' and package_name != 'sqlite3':  # Estos vienen con Python
                packages.append((package_name, line))
    
    print(f"📦 Se verificarán {len(packages)} paquetes...")
    print()
    
    # Verificar e instalar paquetes
    installed_count = 0
    failed_packages = []
    
    for package_name, full_requirement in packages:
        print(f"🔍 Verificando {package_name}...", end=" ")
        
        if check_package_installed(package_name):
            print("✅ Ya instalado")
            installed_count += 1
        else:
            print("❌ No encontrado - Instalando...", end=" ")
            if install_package(full_requirement):
                print("✅ Instalado correctamente")
                installed_count += 1
            else:
                print("❌ Error en instalación")
                failed_packages.append(package_name)
    
    print()
    print("=" * 50)
    print(f"📊 Resumen de instalación:")
    print(f"   ✅ Paquetes instalados: {installed_count}/{len(packages)}")
    
    if failed_packages:
        print(f"   ❌ Paquetes fallidos: {len(failed_packages)}")
        print("   📋 Paquetes que fallaron:")
        for pkg in failed_packages:
            print(f"      - {pkg}")
        print()
        print("💡 Intenta instalar manualmente los paquetes fallidos:")
        print(f"   pip install {' '.join(failed_packages)}")
    else:
        print("   🎉 ¡Todas las dependencias instaladas correctamente!")
    
    print()
    print("🔧 Instalación de recursos adicionales...")
    
    # Instalar recursos de NLTK
    try:
        import nltk
        print("📚 Descargando recursos de NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ Recursos de NLTK descargados")
    except ImportError:
        print("⚠️ NLTK no disponible - saltando descarga de recursos")
    
    # Instalar modelo de spaCy para español
    try:
        print("🌍 Instalando modelo de spaCy para español...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
        print("✅ Modelo de spaCy instalado")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ No se pudo instalar el modelo de spaCy - instálalo manualmente:")
        print("   python -m spacy download es_core_news_sm")
    
    print()
    print("🎯 ¡Instalación completada!")
    print("   Ahora puedes ejecutar: streamlit run app.py")

if __name__ == "__main__":
    main()