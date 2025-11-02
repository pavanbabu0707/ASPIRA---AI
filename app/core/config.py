from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    APP_NAME: str = "AspiraAI"
    APP_ENV: str = "dev"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    SECRET_KEY: str = "change_me_please"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # SQLite connection string (read from .env)
    DATABASE_URL: str = "sqlite:///./aspira_ai_db.db?check_same_thread=false"

    BACKEND_CORS_ORIGINS: str = "http://localhost:5173"

settings = Settings()
