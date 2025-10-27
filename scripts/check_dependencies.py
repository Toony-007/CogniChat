#!/usr/bin/env python3
"""
Script para verificar el estado de las dependencias de CogniChat
Muestra qué paquetes están instalados y cuáles faltan
"""

import importlib.metadata
import sys
from pathlib import Path
from collections import defaultdict

def check_package_version(package_name, required_version=None):
    """Verificar si un paquete está instalado y su versión"""
    try:
        installed = importlib.metadata.distribution(package_name)
        if required_version:
            try:
                from packaging import version
                if version.parse(installed.version) >= version.parse(required_version):
                    return "✅", installed.version, "OK"
                else:
                    return "⚠️", installed.version, f"Requiere >={required_version}"
            except ImportError:
                # Si packaging no está disponible, solo verificar que esté instalado
                return "✅", installed.version, "OK (sin verificación de versión)"
        else:
            return "✅", installed.version, "OK"
    except importlib.metadata.PackageNotFoundError:
        return "❌", "No instalado", "Faltante"

def main():
    """Función principal de verificación"""
    print("🔍 CogniChat - Verificador de Dependencias")
    print("=" * 60)
    
    # Leer requirements.txt (en el directorio padre)
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Error: No se encontró el archivo requirements.txt")
        return
    
    with open(requirements_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Categorizar paquetes
    categories = defaultdict(list)
    current_category = "General"
    
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            if line.startswith('# '):
                current_category = line[2:].strip()
        elif line and not line.startswith('#'):
            # Extraer nombre y versión
            if '>=' in line:
                package_name, version = line.split('>=')
                package_name = package_name.strip()
                version = version.strip()
            else:
                package_name = line.strip()
                version = None
            
            if package_name not in ['pathlib', 'sqlite3']:  # Estos vienen con Python
                categories[current_category].append((package_name, version, line))
    
    # Verificar cada categoría
    total_packages = 0
    installed_packages = 0
    warning_packages = 0
    missing_packages = []
    
    for category, packages in categories.items():
        print(f"\n📂 {category}")
        print("-" * 40)
        
        for package_name, required_version, full_line in packages:
            total_packages += 1
            status, current_version, message = check_package_version(package_name, required_version)
            
            print(f"{status} {package_name:<20} {current_version:<15} {message}")
            
            if status == "✅":
                installed_packages += 1
            elif status == "⚠️":
                warning_packages += 1
                installed_packages += 1  # Está instalado pero versión incorrecta
            else:
                missing_packages.append(full_line)
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE DEPENDENCIAS")
    print("=" * 60)
    print(f"📦 Total de paquetes:     {total_packages}")
    print(f"✅ Instalados:           {installed_packages}")
    print(f"⚠️  Versión incorrecta:   {warning_packages}")
    print(f"❌ Faltantes:            {len(missing_packages)}")
    
    # Calcular porcentaje
    if total_packages > 0:
        percentage = (installed_packages / total_packages) * 100
        print(f"📈 Completitud:          {percentage:.1f}%")
    
    # Mostrar paquetes faltantes
    if missing_packages:
        print(f"\n❌ PAQUETES FALTANTES:")
        print("-" * 30)
        for pkg in missing_packages:
            print(f"   {pkg}")
        
        print(f"\n💡 Para instalar los paquetes faltantes:")
        print(f"   pip install {' '.join([pkg.split('>=')[0].strip() for pkg in missing_packages])}")
        print(f"\n🚀 O ejecuta el instalador automático:")
        print(f"   python install_requirements.py")
    
    # Verificar funcionalidades específicas
    print(f"\n🔧 VERIFICACIÓN DE FUNCIONALIDADES")
    print("-" * 40)
    
    # Análisis avanzado
    advanced_packages = ['scikit-learn', 'nltk', 'textblob', 'wordcloud']
    advanced_available = all(check_package_version(pkg)[0] == "✅" for pkg in advanced_packages)
    
    if advanced_available:
        print("✅ Análisis cualitativo avanzado: DISPONIBLE")
    else:
        print("❌ Análisis cualitativo avanzado: NO DISPONIBLE")
        missing_advanced = [pkg for pkg in advanced_packages if check_package_version(pkg)[0] != "✅"]
        print(f"   Faltan: {', '.join(missing_advanced)}")
    
    # Visualizaciones
    viz_packages = ['plotly', 'matplotlib', 'seaborn']
    viz_available = all(check_package_version(pkg)[0] == "✅" for pkg in viz_packages)
    
    if viz_available:
        print("✅ Visualizaciones avanzadas: DISPONIBLE")
    else:
        print("❌ Visualizaciones avanzadas: NO DISPONIBLE")
        missing_viz = [pkg for pkg in viz_packages if check_package_version(pkg)[0] != "✅"]
        print(f"   Faltan: {', '.join(missing_viz)}")
    
    # Clustering y reducción dimensional
    cluster_packages = ['umap-learn', 'hdbscan']
    cluster_available = all(check_package_version(pkg)[0] == "✅" for pkg in cluster_packages)
    
    if cluster_available:
        print("✅ Clustering avanzado: DISPONIBLE")
    else:
        print("⚠️ Clustering avanzado: LIMITADO")
        print("   (Funciona con K-means básico)")
    
    print(f"\n{'='*60}")
    
    if len(missing_packages) == 0:
        print("🎉 ¡Todas las dependencias están instaladas correctamente!")
        print("   Puedes ejecutar: streamlit run app.py")
    else:
        print("⚠️  Hay dependencias faltantes. Instálalas para funcionalidad completa.")

if __name__ == "__main__":
    main()