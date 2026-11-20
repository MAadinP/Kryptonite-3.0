"""Execute notebooks in place, saving their outputs (the coursework requires notebooks pushed with outputs).

    python run_notebook.py notebooks/H1_learnability.ipynb [more notebooks ...]
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

for arg in sys.argv[1:]:
    path = Path(arg).resolve()
    t0 = time.time()
    print(f"running {path.name} ...", flush=True)
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
    finally:
        nbformat.write(nb, path)  # keep partial outputs if a cell fails
    print(f"done {path.name} in {time.time() - t0:.0f}s", flush=True)
