# Code directly from ChromaDB cookbook https://cookbook.chromadb.dev/embeddings/bring-your-own-embeddings/

import importlib
from typing import Optional, cast

import numpy as np
import numpy.typing as npt
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class TransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            model_name: str = "w601sxs/b1ade-embed",
            cache_dir: Optional[str] = None,
            use_cuda: bool = False,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer

            self._torch = importlib.import_module("torch")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            self._use_cuda = use_cuda and self._torch.cuda.is_available()
            if self._use_cuda:
                self._model.cuda()
        except ImportError:
            raise ValueError(
                "The transformers and/or pytorch python package is not installed. Please install it with "
                "`pip install transformers` or `pip install torch`"
            )

    @staticmethod
    def _normalize(vector: npt.NDArray) -> npt.NDArray:
        """Normalizes a vector to unit length using L2 norm."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def __call__(self, input: Documents) -> Embeddings:
        inputs = self._tokenizer(
            input, padding=True, truncation=True, return_tensors="pt"
        )
        if self._use_cuda:
            inputs = inputs.to('cuda')
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        if self._use_cuda:
            embeddings = embeddings.cpu()
        return [e.tolist() for e in self._normalize(embeddings)]