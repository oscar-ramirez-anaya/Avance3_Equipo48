"""Verifica que los archivos preparados para commit cumplan docs/POLITICA_DATOS.md.

Se ejecuta desde el hook .githooks/pre-commit o a mano:
    python3 scripts/verificar_publicacion.py            # archivos en staging
    python3 scripts/verificar_publicacion.py RUTA ...   # archivos indicados
"""
import json
import re
import subprocess
import sys

# Cadenas que identifican infraestructura interna de Bnext; no deben publicarse.
PATRONES_PROHIBIDOS = [
    r"172\.30\.\d+\.\d+",
    r"usrplanograma",
    r"vmbnext",
    r"dev\.azure\.com/bnext",
    r"/opt/workspace",
    r"vpn-psp\.bnext",
    r"externo3",
]
EXTENSIONES_BINARIAS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}


def archivos_en_staging():
    salida = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [r for r in salida.splitlines() if r]


def texto_de(ruta):
    if any(ruta.lower().endswith(e) for e in EXTENSIONES_BINARIAS):
        return ""
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return ""


def problemas_notebook(ruta):
    """Los notebooks de trabajo en notebooks/ se versionan sin salidas."""
    if not (ruta.startswith("notebooks/") and ruta.endswith(".ipynb")):
        return []
    with open(ruta, encoding="utf-8") as f:
        nb = json.load(f)
    con_salida = [i for i, c in enumerate(nb.get("cells", [])) if c.get("outputs")]
    if con_salida:
        return [f"{ruta}: celdas con salidas {con_salida[:10]} (limpiar antes de commit)"]
    return []


def main(rutas):
    errores = []
    for ruta in rutas:
        if ruta.endswith("verificar_publicacion.py") or ruta.endswith("POLITICA_DATOS.md"):
            continue
        texto = texto_de(ruta)
        for patron in PATRONES_PROHIBIDOS:
            if re.search(patron, texto):
                errores.append(f"{ruta}: contiene el patrón restringido '{patron}'")
        errores += problemas_notebook(ruta)
    for e in errores:
        print("BLOQUEADO", e)
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or archivos_en_staging()))
