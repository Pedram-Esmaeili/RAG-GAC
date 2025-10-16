from pydantic import BaseModel, Field, validator
from typing import List


class UuidResponse(BaseModel):
    uuid: str=Field(..., example="123e4567-e89b-12d3-a456-426614174000")


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is an EMR?", description="The natural language query to process.")
    user_id: str = Field(..., example="user123", description="Unique identifier for the user.")