from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

print("=" * 58)
print(" MACHINE LEARNING - VERIFICACION DEL ENTORNO")
print("=" * 58)

print(f"{'Python':20} {platform.python_version():15} OK")

packages = [
    ("NumPy", "numpy"),
    ("pandas", "pandas"),
    ("Matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
]

failed = False

for label, module_name in packages:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "desconocida")
        print(f"{label:20} {version:15} OK")
    except Exception as exc:
        failed = True
        print(f"{label:20} {'ERROR':15} {exc}")

print("-" * 58)

datasets = [
    DATA / "candy-data.csv",
    DATA / "winequality-red.csv",
]

for path in datasets:
    exists = path.is_file()
    print(f"{path.name:30} {'OK' if exists else 'NO ENCONTRADO'}")
    failed |= not exists

print("-" * 58)

out = ROOT / "resultados"
try:
    out.mkdir(exist_ok=True)
    test = out / ".write_test"
    test.write_text("ok", encoding="utf-8")
    test.unlink()
    print(f"{'resultados/ escritura':30} OK")
except Exception as exc:
    failed = True
    print(f"{'resultados/ escritura':30} ERROR: {exc}")

print("=" * 58)

if failed:
    print("ENTORNO INCOMPLETO")
    sys.exit(1)

print("ENTORNO LISTO")
