# app/schemas/survey.py
from pydantic import BaseModel
from typing import Any

class SurveySubmit(BaseModel):
    user_id: int
    answers: Any  # can be a list or dict

class SurveyOut(BaseModel):
    id: int
    user_id: int
    answers: Any

    class Config:
        from_attributes = True
