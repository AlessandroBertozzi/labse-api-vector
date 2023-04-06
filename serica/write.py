


def exist_document(client, index, field_name: str, field_value):

    resp = client.search(index=index, query={"term": {field_name: field_value}})

    count_resp = resp['hits']['total']['value']

    return count_resp != 0


