# app/main.py
from app.api.routes import router as api_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.session import engine
from app.db.base import Base

# Import models so SQLAlchemy knows about them (for create_all)
from app.models.user import User  # noqa: F401
from app.models.survey import SurveyResponse  # noqa: F401
from app.models.career import Career  # noqa: F401

def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    # CORS
    origins = [o.strip() for o in settings.BACKEND_CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure tables exist (uses your SQLite file)
    Base.metadata.create_all(bind=engine)

    app.include_router(api_router)


    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app

app = create_app()
