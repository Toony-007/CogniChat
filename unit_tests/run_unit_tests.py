#!/usr/bin/env python3
"""
Script de Ejecución de Pruebas Unitarias - CogniChat

Este script ejecuta todas las pruebas unitarias del proyecto CogniChat
y genera reportes detallados de cobertura y resultados.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_unit_tests(verbose=False, coverage=False, specific_module=None):
    """
    Ejecuta las pruebas unitarias con opciones configurables.
    
    Args:
        verbose: Si mostrar salida detallada
        coverage: Si generar reporte de cobertura
        specific_module: Módulo específico a probar
    """
    print("🧪 Iniciando Pruebas Unitarias - CogniChat")
    print("=" * 50)
    
    # Configurar comando base
    cmd = ["python", "-m", "pytest", "unit_tests/"]
    
    # Agregar opciones
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Agregar marcadores para pruebas unitarias
    cmd.extend(["-m", "unit"])
    
    # Módulo específico
    if specific_module:
        cmd.append(f"unit_tests/{specific_module}")
    
    # Configurar cobertura
    if coverage:
        cmd.extend([
            "--cov=modules",
            "--cov=utils", 
            "--cov=config",
            "--cov-report=html:unit_tests/reports/coverage_html",
            "--cov-report=json:unit_tests/reports/coverage.json",
            "--cov-report=term-missing"
        ])
    
    # Generar reporte JUnit
    cmd.extend([
        "--junit-xml=unit_tests/reports/junit_report.xml",
        "--json-report",
        "--json-report-file=unit_tests/reports/test_results.json"
    ])
    
    try:
        print(f"Ejecutando: {' '.join(cmd)}")
        print("-" * 50)
        
        # Crear directorio de reportes si no existe
        reports_dir = Path("unit_tests/reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Ejecutar pruebas
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Mostrar resultados
        print(result.stdout)
        if result.stderr:
            print("ERRORES:")
            print(result.stderr)
        
        # Generar resumen
        generate_test_summary(result.returncode, coverage)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error ejecutando pruebas: {e}")
        return False

def generate_test_summary(return_code, coverage_enabled):
    """
    Genera un resumen de los resultados de las pruebas.
    
    Args:
        return_code: Código de retorno de pytest
        coverage_enabled: Si la cobertura estaba habilitada
    """
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS UNITARIAS")
    print("=" * 50)
    
    # Estado general
    if return_code == 0:
        print("✅ TODAS LAS PRUEBAS PASARON")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON")
    
    # Información de archivos generados
    reports_dir = Path("unit_tests/reports")
    
    if reports_dir.exists():
        print(f"\n📁 Reportes generados en: {reports_dir}")
        
        # Listar archivos de reporte
        for report_file in reports_dir.glob("*"):
            if report_file.is_file():
                print(f"   - {report_file.name}")
    
    # Información de cobertura
    if coverage_enabled:
        coverage_file = reports_dir / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                    print(f"\n📈 Cobertura Total: {total_coverage:.1f}%")
            except Exception as e:
                print(f"⚠️  No se pudo leer el reporte de cobertura: {e}")
    
    # Timestamp
    print(f"\n🕒 Ejecutado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_specific_test_suite(suite_name):
    """
    Ejecuta una suite específica de pruebas.
    
    Args:
        suite_name: Nombre de la suite (modules, utils, config)
    """
    valid_suites = ['modules', 'utils', 'config']
    
    if suite_name not in valid_suites:
        print(f"❌ Suite inválida. Opciones: {', '.join(valid_suites)}")
        return False
    
    print(f"🎯 Ejecutando suite: {suite_name}")
    return run_unit_tests(verbose=True, specific_module=suite_name)

def check_test_environment():
    """
    Verifica que el entorno esté configurado correctamente para las pruebas.
    """
    print("🔍 Verificando entorno de pruebas...")
    
    # Verificar pytest
    try:
        import pytest
        print(f"✅ pytest instalado: {pytest.__version__}")
    except ImportError:
        print("❌ pytest no está instalado")
        return False
    
    # Verificar pytest-cov para cobertura
    try:
        import pytest_cov
        print("✅ pytest-cov disponible")
    except ImportError:
        print("⚠️  pytest-cov no está instalado (cobertura no disponible)")
    
    # Verificar estructura de directorios
    required_dirs = ['unit_tests/modules', 'unit_tests/utils', 'unit_tests/fixtures']
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directorio encontrado: {dir_path}")
        else:
            print(f"❌ Directorio faltante: {dir_path}")
            return False
    
    print("✅ Entorno de pruebas configurado correctamente")
    return True

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Ejecutor de Pruebas Unitarias - CogniChat")
    
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Salida detallada")
    parser.add_argument("-c", "--coverage", action="store_true",
                       help="Generar reporte de cobertura")
    parser.add_argument("-s", "--suite", choices=['modules', 'utils', 'config'],
                       help="Ejecutar suite específica")
    parser.add_argument("--check-env", action="store_true",
                       help="Verificar entorno de pruebas")
    
    args = parser.parse_args()
    
    # Verificar entorno si se solicita
    if args.check_env:
        return 0 if check_test_environment() else 1
    
    # Verificar entorno automáticamente
    if not check_test_environment():
        print("❌ El entorno no está configurado correctamente")
        return 1
    
    # Ejecutar suite específica
    if args.suite:
        success = run_specific_test_suite(args.suite)
        return 0 if success else 1
    
    # Ejecutar todas las pruebas
    success = run_unit_tests(
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())