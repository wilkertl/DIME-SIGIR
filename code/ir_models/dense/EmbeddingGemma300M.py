from sentence_transformers import SentenceTransformer

from .AbstractDenseModel import AbstractDenseModel


class EmbeddingGemma300M(AbstractDenseModel):

    def __init__(self, *args, model_hgf="google/embeddinggemma-300m", max_seq_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_hgf = model_hgf
        self.model = SentenceTransformer(model_hgf)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.name = "embeddinggemma300m"
        self.embeddings_dim = 768

    def encode_queries(self, texts):
        return self.model.encode_query(texts, normalize_embeddings=True)

    def encode_documents(self, texts):
        return self.model.encode_document(texts, normalize_embeddings=True)

    def start_multi_process_pool(self):
        return self.model.start_multi_process_pool()

    def stop_multi_process_pool(self, pool):
        self.model.stop_multi_process_pool(pool)

    def get_model(self):
        return self.model
