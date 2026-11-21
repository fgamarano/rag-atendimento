"""
ETAPA 3 - VECTOR STORE: guardar os vetores e buscar os mais parecidos.

Esta versão é propositalmente simples: uma matriz numpy em memória e busca por
força bruta. Para 18 chunks é instantâneo e deixa a lógica visível:

    similaridade = vetores_dos_chunks @ vetor_da_pergunta     (cosseno, pois normalizados)
    top-k        = índices dos k maiores scores

Quando migrar para um banco vetorial (pgvector, Qdrant, etc.)?
- quando houver milhares+ de chunks (busca aproximada/índice HNSW);
- quando precisar persistir entre execuções, filtrar por metadados ou ter vários usuários.
A interface (`adicionar` e `buscar`) é a mesma; só a implementação muda.
"""

from dataclasses import dataclass

import numpy as np

from src.chunking import Chunk


@dataclass(frozen=True)
class Resultado:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vetores: np.ndarray | None = None

    def adicionar(self, chunks: list[Chunk], vetores: np.ndarray) -> None:
        self._chunks = list(chunks)
        self._vetores = vetores

    def buscar(self, vetor_pergunta: np.ndarray, k: int = 3) -> list[Resultado]:
        if self._vetores is None:
            raise RuntimeError("Vector store vazio: chame adicionar() primeiro.")
        scores = self._vetores @ vetor_pergunta
        melhores = np.argsort(scores)[::-1][:k]
        return [Resultado(self._chunks[i], float(scores[i])) for i in melhores]
