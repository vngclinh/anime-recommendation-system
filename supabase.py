from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import os
from dotenv import load_dotenv

load_dotenv()

def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

DATABASE_URL = URL.create(
    "postgresql+psycopg2",
    username=_required_env("SUPABASE_DB_USER"),
    password=_required_env("SUPABASE_DB_PASSWORD"),
    host=_required_env("SUPABASE_DB_HOST"),
    port=int(_required_env("SUPABASE_DB_PORT")),
    database=_required_env("SUPABASE_DB_NAME"),
    query={"sslmode": "require"},
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
