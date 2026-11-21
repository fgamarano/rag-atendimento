"""
ETAPA 5 - GUARDRAILS: limites de segurança antes e depois do modelo.

LLMs respondem com confiança mesmo quando não sabem. Em RAG, o risco principal é uma
resposta bem escrita que NÃO está nas políticas (alucinação). Duas defesas:

1. GUARDRAIL DE ENTRADA (escopo): se o melhor chunk recuperado tem score baixo, a
   pergunta provavelmente está fora das políticas ("receita de lasanha"). Recusamos
   SEM chamar o modelo: mais barato, mais rápido e sem risco de inventar.

2. GUARDRAIL DE SAÍDA (grounding): toda resposta deve citar a fonte no formato
   [fonte: arquivo > seção]. Resposta sem citação é tratada como não sustentada
   pelos documentos e substituída por uma recusa.

O limiar abaixo precisa ser CALIBRADO com seu modelo: rode `python -m src.rag`, veja os
scores no [debug] de perguntas boas e ruins e ajuste. Esse ajuste é parte do trabalho.
"""

import re

SCORE_MINIMO_RELEVANCIA = 0.30  # medido: perguntas válidas 0.41-0.78; fora do escopo 0.15-0.17
MENSAGEM_RECUSA = (
    "Não encontrei essa informação nas políticas de viagem. "
    "Posso ajudar com voos, hospedagem e despesas de viagem."
)
_PADRAO_CITACAO = re.compile(r"\[fonte:\s*[^\]]+\]", re.IGNORECASE)


def pergunta_dentro_do_escopo(melhor_score: float) -> bool:
    return melhor_score >= SCORE_MINIMO_RELEVANCIA


def resposta_tem_citacao(resposta: str) -> bool:
    return bool(_PADRAO_CITACAO.search(resposta))
