"""
Supabase client singleton.
Reads SUPABASE_URL and SUPABASE_KEY from the environment (already loaded by
dotenv in app.py before this module is first imported).
"""

import os
from supabase import create_client, Client

_client: Client | None = None


def get_client() -> Client:
    """Return a reusable Supabase client instance."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in the .env file"
            )
        _client = create_client(url, key)
    return _client
