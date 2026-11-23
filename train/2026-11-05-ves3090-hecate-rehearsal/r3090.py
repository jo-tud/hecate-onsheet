"""VES-3090 shared constants and the module loader (named r3090, not common: F-457 / ves2241 import their own
`common` through sys.path, memory `experiment-run-py-import-name-clash`)."""
import importlib.util
import os
import sys
from pathlib import Path

OUT = Path(os.environ.get('ONSHEET_DATA', 'data')) / 'ves3090'
OUT522 = Path(os.environ.get('ONSHEET_DATA', 'data')) / 'ves3067'
OUT517 = Path(os.environ.get('ONSHEET_DATA', 'data')) / 'ves3034'
DATA0139 = Path(os.environ.get('ONSHEET_DATA', 'data')) / 'ves2241'        # ves2241 prep.py: 0139 dev-five 32-plane blocks
REHEARSE = ('w040', 'w044')       # 0139 segments whose blocks are rehearsed (the two with most blocks: 48, 31)
GUARD = ('w035', 'w039', 'w041')  # 0139 segments whose F-395 sites form the guard (never trained on here)


def load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m
