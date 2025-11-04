import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- Import metadata from your models ---
from app.db.base import Base  # <-- adjust this if your Base is elsewhere
target_metadata = Base.metadata

# Build DB URL
db_url = os.getenv("DB_URL", "sqlite+aiosqlite:///./aspira_ai_db.db")
db_url = db_url.replace("aiosqlite", "pysqlite")  # use sync driver for Alembic

def run_migrations_offline():
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        {"sqlalchemy.url": db_url},
        prefix="",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
