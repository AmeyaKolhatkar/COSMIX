from pathlib import Path

# 1. Anchor this specific file
_THIS_FILE = Path(__file__).resolve()  # src/cosmix/core/paths.py

# 2. Traverse up exactly 2 levels to hit the repository root
# paths.py -> core -> cosmix -> src -> COSMIX (Root)
ROOT_DIR = _THIS_FILE.parent.parent

# 3. Define all your global shared directories here ONCE
DATA_DIR = ROOT_DIR / "data"