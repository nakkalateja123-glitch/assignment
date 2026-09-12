from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent import SupportAgent
from src.config import settings


app = FastAPI(
    title="Hiver Customer Support Agent",
    version="1.0.0",
)

agent = SupportAgent(
    brand=settings.brand,
    use_llm=False,
)


class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Incoming customer message.",
    )


@app.get("/")
def root():
    return {
        "name": "Hiver Customer Support Agent",
        "brand": settings.brand,
        "status": "running",
    }


@app.post("/respond")
def respond(request: MessageRequest):
    result = agent.respond(request.message)
    return result.to_dict()
