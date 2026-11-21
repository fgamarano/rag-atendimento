"""
ETAPA 2 - EMBEDDINGS: transformar texto em vetores numéricos.

Um embedding é uma lista de números (aqui, 384) em que textos com SIGNIFICADO parecido
ficam com vetores parecidos, mesmo sem palavras em comum:
"limite de diária" ~ "quanto posso gastar com hotel".

Usamos um modelo local multilíngue (sentence-transformers):
- roda na sua máquina, sem custo por chamada e sem enviar as políticas para terceiros;
- é multilíngue, o que importa porque nossos documentos estão em português;
- baixa ~470 MB na primeira execução.

Os vetores saem NORMALIZADOS (tamanho 1). Assim, similaridade de cosseno = produto
escalar, que é uma única multiplicação de matrizes no numpy.
"""

import numpy as np

NOME_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
_modelo = None


def _carregar_modelo():
    global _modelo
    if _modelo is None:
        # import tardio: só paga o custo (e o download) quando realmente precisa
        from sentence_transformers import SentenceTransformer

        _modelo = SentenceTransformer(NOME_MODELO)
    return _modelo


def embed(textos: list[str]) -> np.ndarray:
    """Retorna matriz (n_textos, dimensao) com vetores normalizados."""
    vetores = _carregar_modelo().encode(textos, normalize_embeddings=True)
    return np.asarray(vetores, dtype=np.float32)
