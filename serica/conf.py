from pydantic import BaseModel


def create_index(es, target, model_dimension):
    es.indices.create(
        index="sentences",
        mappings={
            "properties": {
                "sentence": {
                    "type": "text"
                },
                "document": {
                    "type": "text"
                },
                "document_id": {
                    "type": "long"
                },
                "title": {
                    "type": "text"
                },
                "slug": {
                    "type": "text"
                },
                "number": {
                    "type": "long"
                },
                "n_chunk": {
                    "type": "long"
                },
                f"{target}_features": {
                    "type": "dense_vector",
                    "dims": model_dimension
                }
            }
        }
    )


class Query(BaseModel):
    query_params: str


class Transcription(BaseModel):
    title: str
    document_id: int
    slug: str
    xml_to_json: list
    transcription_url: str
    n_iteration: int
