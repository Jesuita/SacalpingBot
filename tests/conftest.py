"""Fixtures compartidas para todos los tests."""
import sys, os

# Agregar root del proyecto al path para poder importar modulos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
