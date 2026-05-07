from sentence_transformers import SentenceTransformer

from .AbstractDenseModel import AbstractDenseModel


class Qwen3Embedding06B(AbstractDenseModel):

    def __init__(self, *args, model_hgf="Qwen/Qwen3-Embedding-0.6B", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_hgf = model_hgf
        self.model = SentenceTransformer(model_hgf, tokenizer_kwargs={"padding_side": "left"})
        self.name = "qwen3embedding06b"
        self.embeddings_dim = 1024

    def encode_queries(self, texts):
        return self.model.encode(texts, prompt_name="query", normalize_embeddings=True)

    def encode_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def start_multi_process_pool(self):
        return self.model.start_multi_process_pool()

    def stop_multi_process_pool(self, pool):
        self.model.stop_multi_process_pool(pool)

    def get_model(self):
        return self.model
