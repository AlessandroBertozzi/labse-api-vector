from fastapi import FastAPI
from embedders.labse import LaBSE
from pydantic import BaseModel
from elasticsearch import Elasticsearch, helpers
import stanza
import os
from serica.write import exist_document

app = FastAPI()

# Get environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "9200")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")
doc_index = os.getenv("doc_index", "sentences")
# Database connection
client = Elasticsearch(
    "http://"+DB_HOST+":"+DB_PORT,
    basic_auth=(DB_USER, DB_PASS),
    verify_certs=True,
    request_timeout=10
)

# Initialize model for embeddings
model = LaBSE()

# Initialize Stanza pipeline
nlp = stanza.Pipeline(lang="la", processors="tokenize")


@app.get("/")
async def root():
    return {"message": "This is the vector embeddings API for Serica!"}


class Query(BaseModel):
    query_params: str


@app.post("/get/vector/")
async def vector(query: Query):

    return {"vector": model(query.query_params)[0, :].tolist()}


class Transcription(BaseModel):
    title: str
    document_id: int
    slug: str
    text: str

@app.put("/insertion/")
async def insertion(transcription: Transcription):

    document_id = transcription.document_id
    title = transcription.title
    slug = transcription.slug
    text = transcription.text
    output = {"status": 200, "message": "Sentences created"}

    if client.indices.exists(index=doc_index):
        if exist_document(client, doc_index, 'document_id', document_id):

            # Delete the document data for document_id
            client.delete_by_query(
                index=doc_index, query={"term": {"document_id": document_id}}
            )

            output = {"status": 200, "message": "Sentences deleted and re-created"}

        parsed = nlp(text)
        sentences = [sentence.text for sentence in parsed.sentences]
        embeddings = model(sentences)

        actions = list()

        for i, sentence in enumerate(zip(sentences, embeddings)):
            doc = {
                    "_index": "sentences",
                    "_source": {
                        "title": title,
                        "slug": slug,
                        "document_id": document_id,
                        "number": i,
                        "sentence": sentence[0],
                        "LaBSE_features": sentence[1]
                    }
                }

            actions.append(doc)

        helpers.bulk(client, actions)
    else:
        output["status"] = 404
        output["message"] = "Index not found"

    return output


@app.delete("/deletion/{document_id}/")
async def deletion(document_id):

    output = {"status": 200, "message": "Document deleted"}
    if client.indices.exists(index=doc_index):
        if exist_document(client, doc_index, 'document_id', document_id):

            # Delete the document data for document_id
            resp = client.delete_by_query(
                index=doc_index, query={"term": {"document_id": document_id}}
            )
        else:
            output["status"] = 404
            output["message"] = "Document not found"
    else:
        output["status"] = 404
        output["message"] = "Index not found"


    return output


