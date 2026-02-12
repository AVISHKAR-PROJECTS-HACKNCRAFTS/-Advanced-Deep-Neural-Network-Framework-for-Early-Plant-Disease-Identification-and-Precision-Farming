import os
from dotenv import load_dotenv

load_dotenv()

_supabase = None

def get_supabase():
    global _supabase
    if _supabase is not None:
        return _supabase

    url = os.getenv('SUPABASE_URL', '')
    key = os.getenv('SUPABASE_KEY', '')

    if not url or not key:
        return None

    try:
        from supabase import create_client
        _supabase = create_client(url, key)
        return _supabase
    except Exception as e:
        print(f"Supabase init error: {e}")
        return None
