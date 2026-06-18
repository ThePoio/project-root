"""Configuracion de pytest para el paquete de tests.

Inserta la raiz del proyecto en ``sys.path`` para que los tests del directorio
``test/`` puedan importar ``src.predict``, ``src.data_loader``, etc. sin
necesidad de instalar el paquete.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
