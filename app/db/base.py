from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# --- IMPORT ALL MODELS HERE so Alembic detects them ---
# (keep your existing ones + add these)
from app.db.models_extra import Resume, Job, Embedding  # noqa: F401
# from app.db.models_user import User  # example
# from app.db.models_career import Career
# from app.db.models_survey import SurveyResponse
