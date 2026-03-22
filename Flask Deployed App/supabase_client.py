import os
import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.info("python-dotenv not installed, skipping .env file loading")
except Exception as e:
    logger.warning("Failed to load .env file: %s", e)

_supabase = None

def get_supabase():
    global _supabase
    if _supabase is not None:
        return _supabase

    url = os.getenv('SUPABASE_URL', '')
    key = os.getenv('SUPABASE_KEY', '')

    if not url or not key:
        logger.warning(
            "Supabase credentials missing: SUPABASE_URL=%s, SUPABASE_KEY=%s. "
            "Falling back to SQLite.",
            "set" if url else "NOT SET",
            "set" if key else "NOT SET",
        )
        return None

    try:
        from supabase import create_client
        _supabase = create_client(url, key)
        logger.info("Supabase client initialized successfully")
        return _supabase
    except Exception as e:
        logger.error("Supabase init error: %s", e)
        return None


def test_connection():
    client = get_supabase()
    if client is None:
        return False, "Supabase not configured (using SQLite fallback)"
    try:
        client.table('alerts').select('*').limit(1).execute()
        return True, "Supabase connected"
    except Exception as e:
        return False, f"Supabase connection failed: {e}"
