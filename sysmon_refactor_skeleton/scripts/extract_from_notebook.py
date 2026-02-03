"""extract_from_notebook.py

This helper script is for YOUR current monolith notebook.
Because your notebook contains multiple duplicate definitions, this script:
  1) parses the notebook
  2) extracts function definitions
  3) reports duplicates
  4) writes a 'functions_dump.py' so you can copy/paste selectively into modules

Usage:
  python scripts/extract_from_notebook.py /path/to/Sysmon_v12_enhanced.ipynb

NOTE:
  I couldn't access your notebook file in this runtime, so this script is provided
  as a robust starting point to do the extraction locally (or after you place the
  notebook into this folder).
"""

from __future__ import annotations
import sys, json, re, pathlib
from collections import defaultdict

FUNC_RE = re.compile(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.M)

def read_ipynb(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def iter_code_cells(nb: dict):
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            yield i, src

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_from_notebook.py /path/to/notebook.ipynb")
        raise SystemExit(2)

    nb_path = pathlib.Path(sys.argv[1]).expanduser()
    nb = read_ipynb(nb_path)

    funcs = defaultdict(list)
    ordered = []
    for idx, src in iter_code_cells(nb):
        for m in FUNC_RE.finditer(src):
            name = m.group(1)
            funcs[name].append(idx)
        ordered.append((idx, src))

    dupes = {k:v for k,v in funcs.items() if len(v) > 1}
    print(f"Found {sum(len(v) for v in funcs.values())} function defs across {len(funcs)} unique names.")
    if dupes:
        print("\nDUPLICATES:")
        for k,v in sorted(dupes.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"  {k}: cells {v}")
    else:
        print("No duplicates detected.")

    # Dump all code (for manual extraction / diffing)
    out_path = nb_path.with_suffix("").name + "_functions_dump.py"
    out_file = pathlib.Path(out_path)
    with out_file.open("w", encoding="utf-8") as f:
        f.write("# Auto-dumped from notebook\n\n")
        for idx, src in ordered:
            if "def " in src:
                f.write(f"\n\n# ---- cell {idx} ----\n")
                f.write(src)
                f.write("\n")

    print(f"\nWrote: {out_file.resolve()}")

if __name__ == "__main__":
    main()
