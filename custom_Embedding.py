from transformers import AutoTokenizer, AutoModel  # pylint: disable=C0413
import torch  # pylint: disable=C0413
import numpy as np
from typing import Any, List
from gptcache.utils import import_huggingface, import_torch
from gptcache.embedding.base import BaseEmbedding

import_torch()
import_huggingface()


class CustomEmbedding(BaseEmbedding):
    """Generate sentence embedding for given text using pretrained models from Huggingface transformers.

    :param model: model name, defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
    :type model: str

    Example:
        .. code-block:: python

            from gptcache.embedding import Huggingface

            test_sentence = 'Hello, world.'
            encoder = Huggingface(model='gpt-neo-2.7B')
            embed = encoder.to_embeddings(test_sentence)
    """

    def __init__(self, model: str = "gpt-neo-2.7B"):
        self.model = AutoModel.from_pretrained(model)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = "[PAD]"
        try:
            self.__dimension = self.model.config.hidden_size
        except Exception:  # pylint: disable=W0703
            from transformers import AutoConfig  # pylint: disable=C0415

            config = AutoConfig.from_pretrained(model)
            self.__dimension = config.hidden_size

    def to_embeddings(self, data, **_):
        """Generate embedding given text input

        :param data: text in string.
        :type data: str

        :return: a text embedding in shape of (dim,).
        """
        if not isinstance(data, list):
            data = [data]
        inputs = self.tokenizer(
            data, padding=True, truncation=True, return_tensors="pt"
        )
        outs = self.model(**inputs).last_hidden_state
        emb = self.post_proc(outs, inputs).squeeze(0).detach().numpy()
        return np.array(emb).astype("float32")

    def post_proc(self, token_embeddings, inputs):
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
        )
        sentence_embs = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embs

    @property
    def dimension(self):
        """Embedding dimension.

        :return: embedding dimension
        """
        return self.__dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = [self.to_embeddings(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        return self.to_embeddings(text)
