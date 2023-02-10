from fastapi import FastAPI
from embedders.labse import LaBSE
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "This is the vector embeddings API for Serica!"}


class Query(BaseModel):
    query_params: str


@app.post("/get/vector/")
async def say_hello(query: Query):

    print(query.query_params)

    model = LaBSE()

    return {"vector": model(query.query_params)[0, :].tolist()}
