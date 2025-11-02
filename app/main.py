# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.db.session import engine, SessionLocal
from app.db.base import Base

# Import models so SQLAlchemy knows about them (for create_all)
from app.models.user import User  # noqa: F401
from app.models.survey import SurveyResponse  # noqa: F401
from app.models.career import Career  # noqa: F401

# Routers
from app.api.routes import router as api_router
from app.api.survey_routes import router as survey_router
from app.api.career_routes import router as career_router

# Seeder
from app.db.seed import seed_careers


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    # Root -> redirect to Swagger UI
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    # CORS
    origins = [o.strip() for o in settings.BACKEND_CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure tables exist (uses your SQLite file from .env)
    Base.metadata.create_all(bind=engine)

    # Auto-seed careers if empty
    with SessionLocal() as db:
        inserted = seed_careers(db)
        if inserted:
            print(f"[seed] inserted {inserted} career rows")

    # API routes
    app.include_router(api_router)       # /health, /auth/register, /auth/login
    app.include_router(survey_router)    # /survey/*
    app.include_router(career_router)

    return app


app = create_app()
