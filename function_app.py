import os
import logging
import json
import jsonschema

import azure.functions as func

from text_chunker import TextChunker
from text_embedder import TextEmbedder

# TODO: Try to use this from the indexer

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
TEXT_CHUNKER = TextChunker()
TEXT_EMBEDDER = TextEmbedder()
CHUNK_TOKEN_SIZE = int(os.getenv("CHUNK_TOKEN_SIZE"))

@app.route(route="kbbot_embedder")
def kbbot_embedder(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('kbbot_embedder: Processing request ...')

    request = req.get_json()

    try:
        jsonschema.validate(request, schema=get_request_schema())
    except jsonschema.exceptions.ValidationError as e:
        return func.HttpResponse("Invalid request: {0}".format(e), status_code=400)
    logging.info('kbbot_embedder: Request validated ...')

    doc_no = 1
    values = []
    for value in request['values']:
        recordId = value['recordId']
        document_name = value['data']['name']
        text = value['data']['text']
        logging.info(f'kbbot_embedder-main: processing document: {document_name}')
        
        logging.info(f'kbbot_embedder: processing text: {text[:50]} ...')
        vector = TEXT_EMBEDDER.embed_content(text)
        logging.info(f'kbbot_embedder-main: vector created {len(vector)}.')
        values.append(
            {
                "recordId": recordId,
                "data": {'vector': vector},
                "errors": "",
                "warnings": ""
            }
        )
        doc_no += 1
    response_body = { "values": values }
    response = func.HttpResponse(json.dumps(response_body, default=lambda obj: obj.__dict__))
    response.headers['Content-Type'] = 'application/json'    
    logging.info('kbbot_embedder-main: returning response')
    return response

def get_request_schema():
    return {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "recordId": {"type": "string"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "text": {"type": "string", "minLength": 1}
                            },
                            "required": ["name", "text"],
                        },
                    },
                    "required": ["recordId", "data"],
                },
            }
        },
        "required": ["values"],
    }
