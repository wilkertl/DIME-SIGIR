from sentence_transformers import SentenceTransformer

from .AbstractDenseModel import AbstractDenseModel


class BidirLM1BEmbedding(AbstractDenseModel):

    def __init__(self, *args, model_hgf="BidirLM/BidirLM-1B-Embedding", max_seq_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_hgf = model_hgf
        self.model = SentenceTransformer(model_hgf, trust_remote_code=True)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.name = "bidirlm1bembedding"
        self.embeddings_dim = 1152

    def encode_queries(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def encode_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def start_multi_process_pool(self):
        return self.model.start_multi_process_pool()

    def stop_multi_process_pool(self, pool):
        self.model.stop_multi_process_pool(pool)

    def get_model(self):
        return self.model
