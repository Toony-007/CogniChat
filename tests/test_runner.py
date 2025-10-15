"""
Script ejecutor de tests automatizados para CogniChat
Ejecuta todas las pruebas y genera reportes de cobertura
"""

import pytest
import sys
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime

def run_all_tests():
    """Ejecuta todas las pruebas automatizadas"""
    
    # Directorio de tests
    tests_dir = Path(__file__).parent
    
    # Lista de archivos de test
    test_files = [
        "test_validators.py",
        "test_chatbot.py", 
        "test_document_processor.py",
        "test_qualitative_analysis.py",
        "test_document_upload.py",
        "test_alerts.py",
        "test_settings.py",
        "test_utils.py"
    ]
    
    print("🚀 Iniciando ejecución de tests automatizados para CogniChat")
    print("=" * 60)
    
    results = {}
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        
        if not test_path.exists():
            print(f"⚠️  Archivo de test no encontrado: {test_file}")
            continue
        
        print(f"\n📋 Ejecutando: {test_file}")
        print("-" * 40)
        
        try:
            # Ejecutar pytest para el archivo específico
            result = pytest.main([
                str(test_path),
                "-v",
                "--tb=short",
                "--no-header"
            ])
            
            if result == 0:
                print(f"✅ {test_file}: TODOS LOS TESTS PASARON")
                status = "PASSED"
            else:
                print(f"❌ {test_file}: ALGUNOS TESTS FALLARON")
                status = "FAILED"
            
            results[test_file] = {
                "status": status,
                "exit_code": result
            }
            
        except Exception as e:
            print(f"💥 Error ejecutando {test_file}: {str(e)}")
            results[test_file] = {
                "status": "ERROR",
                "error": str(e)
            }
    
    # Generar reporte final
    print("\n" + "=" * 60)
    print("📊 REPORTE FINAL DE TESTS")
    print("=" * 60)
    
    for test_file, result in results.items():
        status_icon = "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "💥"
        print(f"{status_icon} {test_file}: {result['status']}")
    
    # Estadísticas
    passed_count = sum(1 for r in results.values() if r["status"] == "PASSED")
    failed_count = sum(1 for r in results.values() if r["status"] == "FAILED")
    error_count = sum(1 for r in results.values() if r["status"] == "ERROR")
    
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   ✅ Tests exitosos: {passed_count}")
    print(f"   ❌ Tests fallidos: {failed_count}")
    print(f"   💥 Tests con error: {error_count}")
    print(f"   📊 Total archivos: {len(results)}")
    
    # Calcular porcentaje de éxito
    if len(results) > 0:
        success_rate = (passed_count / len(results)) * 100
        print(f"   🎯 Tasa de éxito: {success_rate:.1f}%")
    
    return results

def run_specific_module_tests(module_name):
    """Ejecuta tests para un módulo específico"""
    
    test_file = f"test_{module_name}.py"
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"❌ Archivo de test no encontrado: {test_file}")
        return False
    
    print(f"🎯 Ejecutando tests para módulo: {module_name}")
    print("-" * 40)
    
    try:
        result = pytest.main([
            str(test_path),
            "-v",
            "--tb=long"
        ])
        
        if result == 0:
            print(f"✅ Todos los tests de {module_name} pasaron")
            return True
        else:
            print(f"❌ Algunos tests de {module_name} fallaron")
            return False
            
    except Exception as e:
        print(f"💥 Error ejecutando tests de {module_name}: {str(e)}")
        return False

def run_coverage_report():
    """Genera reporte de cobertura de código"""
    
    print("📊 Generando reporte de cobertura...")
    
    try:
        # Ejecutar pytest con coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=modules",
            "--cov=utils", 
            "--cov-report=html",
            "--cov-report=term-missing",
            "tests/"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Reporte de cobertura generado exitosamente")
            print("📁 Reporte HTML disponible en: htmlcov/index.html")
        else:
            print("❌ Error generando reporte de cobertura")
            print(result.stderr)
            
    except Exception as e:
        print(f"💥 Error ejecutando coverage: {str(e)}")

def generate_test_report(results):
    """Genera reporte detallado en JSON"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files": len(results),
            "passed": sum(1 for r in results.values() if r["status"] == "PASSED"),
            "failed": sum(1 for r in results.values() if r["status"] == "FAILED"),
            "errors": sum(1 for r in results.values() if r["status"] == "ERROR")
        },
        "details": results
    }
    
    # Guardar reporte
    report_path = Path(__file__).parent / "test_report.json"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte detallado guardado en: {report_path}")
        
    except Exception as e:
        print(f"⚠️  Error guardando reporte: {str(e)}")

def main():
    """Función principal del ejecutor de tests"""
    
    if len(sys.argv) > 1:
        # Ejecutar tests para módulo específico
        module_name = sys.argv[1]
        run_specific_module_tests(module_name)
    else:
        # Ejecutar todos los tests
        results = run_all_tests()
        
        # Generar reporte detallado
        generate_test_report(results)
        
        # Preguntar si generar reporte de cobertura
        print("\n🤔 ¿Desea generar reporte de cobertura? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes', 's', 'si', 'sí']:
                run_coverage_report()
        except KeyboardInterrupt:
            print("\n👋 Ejecución cancelada por el usuario")

if __name__ == "__main__":
    main()