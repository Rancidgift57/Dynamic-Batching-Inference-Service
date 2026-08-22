from pydantic import BaseModel


class InferRequest(BaseModel):
    text: str


class InferResponse(BaseModel):
    embedding: list[float]
    cached: bool = False


class ErrorResponse(BaseModel):
    detail: str
