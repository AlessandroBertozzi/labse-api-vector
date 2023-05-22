from fastapi import FastAPI
from embedders.labse import LaBSE
from elasticsearch import Elasticsearch, helpers
import stanza
import os
from serica.write import exist_document
from serica.conf import create_index, Transcription, Query
import threading
from logging.config import dictConfig
import logging
import requests
import time
from dotenv import load_dotenv

# take environment variables from .env.
load_dotenv()

# take logs configuration
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Initialize the app
app = FastAPI()

# Get environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
INDEX = os.getenv("INDEX")
HTTP = os.getenv("HTTP")
NLP_API_HTTP = os.getenv("NLP_API_HTTP")
NLP_API_HOST = os.getenv("NLP_API_HOST")
NLP_API_PORT = os.getenv("NLP_API_PORT")
NLP_API_ENDPOINT = os.getenv("NLP_API_ENDPOINT")

# Database connection
client = Elasticsearch(
    HTTP + DB_HOST + ":" + DB_PORT,
    basic_auth=(DB_USER, DB_PASS),
    verify_certs=True,
    request_timeout=10
)

# Initialize model for embeddings
model = LaBSE()

# Initialize Stanza pipeline
nlp = stanza.Pipeline(lang="la", processors="tokenize")

# Create index in elastic if not exist
tries = 10
start_trial = 0
while not client.indices.exists(index="sentences"):
    create_index(client, "LaBSE", model.dim)
    if not client.indices.exists(index="sentences"):
        start_trial += 1
        time.sleep(10)
        if start_trial > tries:
            logger.error(f"cannot create new sentences index")
    else:
        logger.error(f"create new sentences index")


@app.get("/")
async def root():
    return {"message": "This is the vector embeddings API for Serica!"}


@app.post("/vectorize/")
async def vector(query: Query):
    return {"vector": model(query.query_params)[0, :].tolist()}


def insertion_pipeline(text, title, slug, document_id, transcription_url):
    text = text
    title = title
    slug = slug
    document_id = document_id
    transcription_url = transcription_url

    bulk_list = list()

    cleaned_text_response = requests.post(NLP_API_HTTP + NLP_API_HOST + ":" + NLP_API_PORT + NLP_API_ENDPOINT,
                                          json={"mrc_xml_transcription_texts_json": text})

    cleaned_text = cleaned_text_response.json()["mrc_xml_transcription_texts_json"]

    logger.info({"status": cleaned_text_response.status_code, 'message': 'text cleaned'})

    logger.info(f"INFO: start process")

    for n_section, section in enumerate(cleaned_text, start=1):

        logger.info({"message": f"start parsing section number {n_section}"})

        parsed = nlp(section['xml_text'])

        sentences = [sentence.text for sentence in parsed.sentences]

        logger.info({"message": f"{len(sentences)} sentences prepared"})

        while len(sentences) > 0:

            sentence_batch = sentences[:500]
            del sentences[:500]

            embeddings = model(sentence_batch)

            for i, sentence in enumerate(zip(sentence_batch, embeddings)):
                doc = {
                    "_index": "sentences",
                    "_source": {
                        "title": title,
                        "slug": slug,
                        "document_id": document_id,
                        "transcription_url": transcription_url,
                        "number": i,
                        "_path": section["_path"],
                        "sentence": sentence[0],
                        "LaBSE_features": sentence[1]
                    }
                }

                bulk_list.append(doc)

            # index if is last section or the number of sentences if grater than 500
            if len(bulk_list) >= 500 or n_section == len(text):
                logger.info(f"indexing {len(bulk_list)} sentences")
                helpers.bulk(client, bulk_list)
                bulk_list.clear()

    logger.info({"status_code": 200, "message": "indexing completed"})


# TODO: close endpoint with password
@app.put("/insertion")
async def insertion(transcription: Transcription):
    document_id = transcription.document_id
    title = transcription.title
    slug = transcription.slug
    text = transcription.xml_to_json
    transcription_url = transcription.transcription_url
    output = {"status": 200, "message": "Sentences created"}

    if client.indices.exists(index=INDEX):
        if exist_document(client, INDEX, 'document_id', document_id):
            output = {"status": 403, "message": "Sentences already exist"}

        insertion_pipeline(text, title, slug, document_id, transcription_url)

    else:
        output["status"] = 404
        output["message"] = "Index not found"

    return output


# TODO: close endpoint with password
@app.delete("/deletion/{document_id}/")
async def deletion(document_id):
    output = {"status": 200, "message": "Document deleted"}
    if client.indices.exists(index=INDEX):
        if exist_document(client, INDEX, 'document_id', document_id):

            # Delete the document data for document_id
            resp = client.delete_by_query(
                index=INDEX, query={"term": {"document_id": document_id}}
            )
        else:
            output["status"] = 404
            output["message"] = "Document not found"
    else:
        output["status"] = 404
        output["message"] = "Index not found"

    return output
