
import os
import re
import logging
import openai

class TextEmbedder():

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

    def clean_text(self, text, text_limit=7000):
        # Clean up text (e.g. line breaks, )    
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\n\r]+', ' ', text).strip()
        # Truncate text if necessary (e.g. for, ada-002, 4095 tokens ~ 7000 chracters)    
        if len(text) > text_limit:
            logging.warning("kbbot-embedder-text-embedder: Token limit reached exceeded maximum length, truncating...")
            text = text[:text_limit]
        return text
    
    def embed_content(self, text, clean_text=True):
        logging.info(f'kbbot-embedder-text-embedder: embedding: {text[:50]} ...')
        embedding_precision = 9
        if clean_text:
            text = self.clean_text(text)

        openai.api_key = self.OPENAI_API_KEY
        response = openai.Embedding.create(input=text,
                                           deployment_id=self.OPENAI_EMBEDDING_MODEL)
        embedding = [round(x, embedding_precision) for x in response['data'][0]['embedding']]
        logging.info(f'kbbot-embedder-text-embedder: vector: {embedding[:3]} ...')
        return embedding
