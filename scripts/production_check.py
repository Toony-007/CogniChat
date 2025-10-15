#!/usr/bin/env python3
"""
Script de verificación para producción - CogniChat
Verifica que todos los componentes estén listos para despliegue en producción.
"""

import os
import sys
import subprocess
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Imprime un header formateado"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(message: str):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    """Imprime mensaje informativo"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def check_python_version() -> bool:
    """Verifica la versión de Python"""
    print_info("Verificando versión de Python...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} - Se requiere Python 3.11+")
        return False

def check_required_files() -> bool:
    """Verifica que existan los archivos necesarios"""
    print_info("Verificando archivos requeridos...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'Dockerfile',
        '.dockerignore',
        'railway.json',
        'docker-compose.yml',
        '.env.example',
        '.env.production'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print_success(f"{file} encontrado")
    
    if missing_files:
        for file in missing_files:
            print_error(f"{file} no encontrado")
        return False
    
    return True

def check_directory_structure() -> bool:
    """Verifica la estructura de directorios"""
    print_info("Verificando estructura de directorios...")
    
    required_dirs = [
        'modules',
        'utils',
        'config',
        'scripts',
        'data',
        'data/uploads',
        'data/processed',
        'data/cache',
        'data/chat_history',
        'data/temp_exports',
        'unit_tests',
        'docs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print_success(f"Directorio {dir_path}/ encontrado")
    
    if missing_dirs:
        for dir_path in missing_dirs:
            print_error(f"Directorio {dir_path}/ no encontrado")
        return False
    
    return True

def check_dependencies() -> bool:
    """Verifica que las dependencias estén instaladas"""
    print_info("Verificando dependencias críticas...")
    
    critical_deps = [
        'streamlit',
        'langchain',
        'chromadb',
        'ollama',
        'pandas',
        'numpy',
        'plotly',
        'nltk',
        'sklearn',  # scikit-learn se importa como sklearn
        'wordcloud',
        'networkx'
    ]
    
    missing_deps = []
    for dep in critical_deps:
        try:
            importlib.import_module(dep)
            if dep == 'sklearn':
                print_success("scikit-learn instalado")
            else:
                print_success(f"{dep} instalado")
        except ImportError:
            if dep == 'sklearn':
                missing_deps.append('scikit-learn')
                print_error("scikit-learn no instalado")
            else:
                missing_deps.append(dep)
                print_error(f"{dep} no instalado")
    
    if missing_deps:
        print_error(f"Dependencias faltantes: {', '.join(missing_deps)}")
        return False
    
    return True

def check_environment_variables() -> bool:
    """Verifica variables de entorno críticas"""
    print_info("Verificando configuración de variables de entorno...")
    
    # Leer .env.production
    env_file = Path('.env.production')
    if not env_file.exists():
        print_error("Archivo .env.production no encontrado")
        return False
    
    critical_vars = [
        'OLLAMA_BASE_URL',
        'DEFAULT_LLM_MODEL',
        'DEFAULT_EMBEDDING_MODEL',
        'CHUNK_SIZE',
        'MAX_RETRIEVAL_DOCS'
    ]
    
    env_vars = {}
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
    
    missing_vars = []
    for var in critical_vars:
        if var in env_vars and env_vars[var]:
            print_success(f"{var}={env_vars[var]}")
        else:
            missing_vars.append(var)
            print_error(f"{var} no configurado")
    
    if missing_vars:
        return False
    
    return True

def check_docker_configuration() -> bool:
    """Verifica la configuración de Docker"""
    print_info("Verificando configuración de Docker...")
    
    # Verificar Dockerfile
    dockerfile = Path('Dockerfile')
    if not dockerfile.exists():
        print_error("Dockerfile no encontrado")
        return False
    
    dockerfile_content = dockerfile.read_text(encoding='utf-8')
    
    # Verificar elementos críticos del Dockerfile
    checks = [
        ('FROM python:3.11-slim as builder', 'Multi-stage build configurado'),
        ('USER cognichat', 'Usuario no-root configurado'),
        ('HEALTHCHECK', 'Healthcheck configurado'),
        ('EXPOSE ${PORT}', 'Puerto expuesto'),
        ('--mount=type=cache', 'Cache mount configurado')
    ]
    
    all_good = True
    for check, description in checks:
        if check in dockerfile_content:
            print_success(description)
        else:
            print_error(f"{description} - no encontrado: {check}")
            all_good = False
    
    return all_good

def run_unit_tests() -> bool:
    """Ejecuta las pruebas unitarias"""
    print_info("Ejecutando pruebas unitarias...")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'unit_tests/', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print_success("Todas las pruebas unitarias pasaron")
            return True
        else:
            print_error("Algunas pruebas unitarias fallaron")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Timeout ejecutando pruebas unitarias")
        return False
    except Exception as e:
        print_error(f"Error ejecutando pruebas: {e}")
        return False

def check_security_configuration() -> bool:
    """Verifica configuraciones de seguridad"""
    print_info("Verificando configuraciones de seguridad...")
    
    security_checks = []
    
    # Verificar que no hay archivos .env en el repositorio (solo advertencia)
    if Path('.env').exists():
        print_warning("Archivo .env encontrado - asegúrate de que no esté en el repositorio")
        # No agregamos esto como fallo crítico, solo advertencia
    else:
        print_success("No hay archivo .env en el directorio raíz")
    
    # Verificar .gitignore
    gitignore = Path('.gitignore')
    if gitignore.exists():
        gitignore_content = gitignore.read_text(encoding='utf-8')
        if '.env' in gitignore_content:
            print_success("Archivos .env están en .gitignore")
            security_checks.append(True)
        else:
            print_error("Archivos .env no están en .gitignore")
            security_checks.append(False)
    else:
        print_error("Archivo .gitignore no encontrado")
        security_checks.append(False)
    
    # Verificar configuración de usuario no-root en Dockerfile
    dockerfile = Path('Dockerfile')
    if dockerfile.exists():
        dockerfile_content = dockerfile.read_text(encoding='utf-8')
        if 'USER cognichat' in dockerfile_content:
            print_success("Usuario no-root configurado en Dockerfile")
            security_checks.append(True)
        else:
            print_error("Usuario no-root no configurado en Dockerfile")
            security_checks.append(False)
    
    return all(security_checks)

def generate_production_report() -> Dict[str, Any]:
    """Genera un reporte completo de preparación para producción"""
    print_header("GENERANDO REPORTE DE PRODUCCIÓN")
    
    checks = {
        'python_version': check_python_version(),
        'required_files': check_required_files(),
        'directory_structure': check_directory_structure(),
        'dependencies': check_dependencies(),
        'environment_variables': check_environment_variables(),
        'docker_configuration': check_docker_configuration(),
        'unit_tests': run_unit_tests(),
        'security_configuration': check_security_configuration()
    }
    
    # Información adicional
    try:
        if os.name == 'nt':  # Windows
            timestamp = subprocess.run(['powershell', 'Get-Date'], capture_output=True, text=True).stdout.strip()
        else:  # Unix/Linux
            timestamp = subprocess.run(['date'], capture_output=True, text=True).stdout.strip()
    except:
        from datetime import datetime
        timestamp = datetime.now().isoformat()
    
    info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': sys.platform,
        'cwd': os.getcwd(),
        'timestamp': timestamp
    }
    
    # Obtener timestamp compatible con Windows
    try:
        if os.name == 'nt':  # Windows
            timestamp = subprocess.run(['powershell', 'Get-Date'], capture_output=True, text=True).stdout.strip()
        else:  # Unix/Linux
            timestamp = subprocess.run(['date'], capture_output=True, text=True).stdout.strip()
    except:
        from datetime import datetime
        timestamp = datetime.now().isoformat()
    
    report = {
        'status': 'READY' if all(checks.values()) else 'NOT_READY',
        'checks': checks,
        'info': info,
        'ready_for_production': all(checks.values()),
        'timestamp': timestamp
    }
    
    return report

def main():
    """Función principal"""
    print_header("VERIFICACIÓN DE PREPARACIÓN PARA PRODUCCIÓN - COGNICHAT")
    
    # Cambiar al directorio del proyecto
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)
    
    print_info(f"Directorio de trabajo: {os.getcwd()}")
    
    # Generar reporte
    report = generate_production_report()
    
    # Guardar reporte
    report_file = Path('production_readiness_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print_header("RESUMEN FINAL")
    
    if report['ready_for_production']:
        print_success("🎉 CogniChat está LISTO para producción!")
        print_info("Puedes proceder con el despliegue en Railway.")
    else:
        print_error("❌ CogniChat NO está listo para producción.")
        print_info("Revisa los errores anteriores y corrígelos antes del despliegue.")
        
        failed_checks = [check for check, passed in report['checks'].items() if not passed]
        print_error(f"Verificaciones fallidas: {', '.join(failed_checks)}")
    
    print_info(f"Reporte completo guardado en: {report_file}")
    
    return 0 if report['ready_for_production'] else 1

if __name__ == "__main__":
    sys.exit(main())