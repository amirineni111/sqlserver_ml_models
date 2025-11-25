#!/usr/bin/env python3
"""
Quick Start Script for SQL Server ML Project
This script helps you get started with the project setup and initial testing.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("🚀 SQL Server ML Project - Quick Start")
    print("=" * 60)

def check_environment():
    """Check if required environment is set up."""
    print("\n📋 Checking Environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Please copy .env.example to .env and configure your database credentials.")
        return False
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Python 3.8+ required.")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor} detected.")
    
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing Dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test if key modules can be imported."""
    print("\n🧪 Testing Key Imports...")
    
    modules = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("sqlalchemy", "SQLAlchemy"),
        ("pyodbc", "pyodbc"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("jupyter", "jupyter")
    ]
    
    failed = []
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            failed.append(name)
    
    return len(failed) == 0

def create_gitignore():
    """Create a comprehensive .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env

# Database
*.db
*.sqlite3

# Logs
logs/
*.log

# Reports
reports/*.html
reports/*.pdf

# Model files
*.pkl
*.joblib
*.h5

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (add specific patterns as needed)
data/raw/*.csv
data/raw/*.xlsx
data/raw/*.json
data/processed/*.csv
data/processed/*.xlsx
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✅ .gitignore created.")

def show_next_steps():
    """Show next steps to the user."""
    print("\n🎯 Next Steps:")
    print("1. Configure your .env file with SQL Server credentials")
    print("2. Test database connection: python -c \"from src.database.connection import SQLServerConnection; conn = SQLServerConnection(); conn.test_connection()\"")
    print("3. Open Jupyter Lab: jupyter lab --notebook-dir=notebooks")
    print("4. Start with the 01_database_connection.ipynb notebook")
    print("5. Explore your data in 02_data_exploration.ipynb")
    print("6. Build models in 03_model_development.ipynb")
    print("\n📚 Available VS Code Tasks (Ctrl+Shift+P > Tasks: Run Task):")
    print("   - Install Dependencies")
    print("   - Start Jupyter Lab")
    print("   - Test Database Connection")
    print("   - Format Code with Black")
    print("   - Lint with Flake8")

def main():
    """Main function to run the quick start process."""
    print_banner()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed.")
        return
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some modules failed to import. Check your installation.")
    
    # Create .gitignore
    create_gitignore()
    
    # Show next steps
    show_next_steps()
    
    print("\n🎉 Quick start complete! Happy coding!")

if __name__ == "__main__":
    main()
