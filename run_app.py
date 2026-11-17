from pathlib import Path
import subprocess, sys, shutil
ROOT = Path(__file__).resolve().parent
def main():
    if importlib.util.find_spec("streamlit") is None:
        print("Streamlit is not installed. Run: python -m pip install -r requirements.txt")
        raise SystemExit(1)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], cwd=ROOT, check=False)
if __name__ == "__main__":
    import importlib.util
    main()
