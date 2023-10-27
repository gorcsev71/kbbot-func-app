
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from token_estimator import TokenEstimator
from text_embedder import TextEmbedder

class TextChunker():

    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = ['\n', '\t', '}', '{', ']', '[', ')', '(', ' ', ':', ';', ',']
    TEXT_EMBEDDER = TextEmbedder()
    TOKEN_ESTIMATOR = TokenEstimator()

    def chunk_content(self, doc_source:str, doc_content:str, num_tokens: int = 2048):
        # logging.info(f'Log - Splitter - Splitting: {content}')
        chunks = []
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                separators=self.SENTENCE_ENDINGS + self.WORDS_BREAKS,
                chunk_size=num_tokens, chunk_overlap=0)
        chunked_content_list = splitter.split_text(doc_content)
        i = 1
        for chunk in chunked_content_list:
            # logging.info(f'Log - Splitter - Chunk: {chunk}')
            vector = self.TEXT_EMBEDDER.embed_content(text=chunk)
            chunks.append(
                {
                    "source": doc_source,
                    "page_no": str(i),
                    "content_text": chunk,
                    "length": self.TOKEN_ESTIMATOR.estimate_tokens(chunk),
                    "content_vector": vector,
                }
            )
            i += 1
        return chunks


