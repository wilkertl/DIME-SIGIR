from .Contriever import Contriever
from .Ance import Ance
from .Tasb import Tasb
from .TctColbert import TctColbert
from .Dragon import Dragon
from .MiniLM import MiniLM
from .Minilml12 import Minilml12
from .Starbucks import Starbucks
from .Qwen3Embedding06B import Qwen3Embedding06B
from .BidirLM1BEmbedding import BidirLM1BEmbedding
from .BgeM3 import BgeM3
from .EmbeddingGemma300M import EmbeddingGemma300M


DENSE_MODELS = {
    "ance": Ance,
    "bidirlm1bembedding": BidirLM1BEmbedding,
    "bgem3": BgeM3,
    "contriever": Contriever,
    "dragon": Dragon,
    "embeddinggemma300m": EmbeddingGemma300M,
    "minilm": MiniLM,
    "minilml12": Minilml12,
    "qwen3embedding06b": Qwen3Embedding06B,
    "starbucks": Starbucks,
    "tasb": Tasb,
    "tctcolbert": TctColbert,
}


def get_dense_model(name):
    try:
        return DENSE_MODELS[name.lower()]()
    except KeyError:
        available = ", ".join(sorted(DENSE_MODELS))
        raise ValueError(f"encoder not recognized: {name}. Available encoders: {available}")