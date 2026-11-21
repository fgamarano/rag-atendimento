"""
EVALS - como testar um sistema não determinístico.

Em vez de um único "a resposta está certa?", separamos em CAMADAS. Quando algo falha,
você sabe ONDE (é a pirâmide de testes aplicada a IA):

  Camada 1 - RETRIEVAL  : o trecho certo (arquivo + seção) foi recuperado no top-k? (sem LLM, rápido, grátis)
  Camada 2 - FATOS      : a resposta contém os fatos-chave esperados? (usa o LLM)
  Camada 3 - GUARDRAILS : perguntas fora do escopo são recusadas? respostas citam a fonte?

Uso:
    python -m evals.eval                 # as 3 camadas (precisa de ANTHROPIC_API_KEY)
    python -m evals.eval --so-retrieval  # só camada 1 e a parte de escopo da camada 3 (sem API)

Se o retrieval falha, não adianta culpar o prompt. Se o retrieval passa e os fatos
falham, o problema está na geração. Essa separação é o argumento central de QA.
"""

import argparse
import sys
import unicodedata
from dataclasses import dataclass

from src import guardrails
from src.rag import AssistenteRAG


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFD", texto)
    sem_acento = "".join(c for c in sem_acento if unicodedata.category(c) != "Mn")
    return sem_acento.lower()


@dataclass(frozen=True)
class Caso:
    pergunta: str
    arquivo_esperado: str | None  # None = pergunta fora do escopo
    secao_esperada: str | None = None  # trecho exato que deve ser recuperado
    fatos: tuple[str, ...] = ()  # trechos que DEVEM aparecer na resposta


CASOS = [
    Caso("Qual o limite de diária de hotel em São Paulo?", "politica_hospedagem.md", "Limites de diária por cidade", ("550",)),
    Caso("Posso voar de executiva num voo internacional de 10 horas?", "politica_voos.md", "Classe de voo permitida", ("8 horas",)),
    Caso("Com quantos dias de antecedência devo comprar uma passagem internacional?", "politica_voos.md", "Antecedência de compra", ("21",)),
    Caso("Qual o prazo para prestar contas depois da viagem?", "politica_despesas.md", "Prestação de contas", ("5 dias uteis",)),
    Caso("Posso pedir reembolso de bebida alcoólica num jantar com cliente?", "politica_despesas.md", "Bebidas alcoólicas", ("nao",)),
    Caso("A empresa paga mala despachada?", "politica_voos.md", "Bagagem", ("23",)),
    Caso("Qual a melhor receita de lasanha?", None),
    Caso("Quem ganhou a Copa do Mundo de 2022?", None),
]


def rodar(so_retrieval: bool) -> int:
    print("Carregando modelo de embeddings e indexando políticas...\n")
    assistente = AssistenteRAG()
    falhas = 0

    for i, caso in enumerate(CASOS, 1):
        print(f"{i}. {caso.pergunta}")
        resultados = assistente.recuperar(caso.pergunta)
        melhor = resultados[0]
        escopo_ok = guardrails.pergunta_dentro_do_escopo(melhor.score)
        linhas: list[tuple[str, bool, str]] = []

        if caso.arquivo_esperado:
            # Camada 1: o TRECHO esperado (arquivo + seção) está entre os k recuperados?
            achou = any(
                r.chunk.arquivo == caso.arquivo_esperado and r.chunk.secao == caso.secao_esperada
                for r in resultados
            )
            linhas.append(("retrieval", achou, f"top-{assistente.k}: " + ", ".join(r.chunk.fonte for r in resultados)))
            # Camada 3 (parte A): pergunta legítima não pode ser bloqueada pelo guardrail
            linhas.append(("guardrail: não bloqueia pergunta válida", escopo_ok, f"melhor score {melhor.score:.2f}"))
            if not so_retrieval:
                r = assistente.perguntar(caso.pergunta)
                resposta_norm = normalizar(r.resposta)
                faltando = [f for f in caso.fatos if normalizar(f) not in resposta_norm]
                linhas.append(("fatos", not faltando, f"faltando: {faltando}" if faltando else "ok"))
                linhas.append(("guardrail: cita a fonte", guardrails.resposta_tem_citacao(r.resposta) and not r.recusada, r.resposta[:90]))
        else:
            # Camada 3 (parte B): fora do escopo deve ser recusado ANTES do LLM
            linhas.append(("guardrail: recusa fora do escopo", not escopo_ok, f"melhor score {melhor.score:.2f}"))

        for nome, ok, detalhe in linhas:
            print(f"   [{'PASS' if ok else 'FAIL'}] {nome} - {detalhe}")
            falhas += 0 if ok else 1
        print()

    total = sum(1 for _ in CASOS)
    print(f"{total} casos, {falhas} verificações com falha.")
    return 1 if falhas else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--so-retrieval", action="store_true", help="não chama a API do LLM")
    args = parser.parse_args()
    sys.exit(rodar(args.so_retrieval))
