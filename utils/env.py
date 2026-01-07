from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_env() -> bool:
    """Load .env from repo root if present."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return False

    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return False

    load_dotenv(env_path)
    return True
