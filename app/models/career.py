from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Text, DateTime, func
from app.db.base import Base

class Career(Base):
    __tablename__ = "careers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    onet_code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    skills_vector: Mapped[str | None] = mapped_column(Text)  # JSON array as text for SQLite
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
