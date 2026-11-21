"""
ETAPA 4 - O PIPELINE RAG: conecta tudo.

    pergunta -> embedding -> busca top-k -> guardrail de entrada
             -> prompt com os trechos -> LLM (API da Anthropic)
             -> guardrail de saída -> resposta com fonte

RAG (Retrieval-Augmented Generation) = em vez de confiar na memória do modelo,
buscamos os trechos relevantes e os colocamos no prompt. O modelo vira um "leitor":
responde apenas com base no que recebeu. Isso reduz alucinação e permite atualizar o
conhecimento trocando documentos, sem treinar nada.

Parâmetros da chamada que caem em entrevista:
- system vs messages: `system` define regras e papel; `messages` leva a pergunta e o contexto.
- max_tokens: limita custo e tamanho da resposta.
- temperature: não é enviada aqui, porque a versão do SDK/modelo usada não aceita esse parâmetro.

Uso: python -m src.rag     (precisa de ANTHROPIC_API_KEY no ambiente)
"""

import os
from dataclasses import dataclass, field

from src import guardrails
from src.chunking import Chunk, carregar_chunks
from src.embeddings import embed
from src.vector_store import Resultado, VectorStore

MODELO = os.environ.get("RAG_MODELO", "claude-haiku-5-5")
TOP_K = 3

SYSTEM_PROMPT = """Você é um assistente de políticas de viagem corporativa.
Responda SOMENTE com base nos trechos de política fornecidos em <contexto>.
Regras:
1. Se os trechos não contêm a resposta, diga exatamente: "Não encontrei essa informação nas políticas de viagem."
2. Não use conhecimento externo e não invente valores, prazos ou exceções.
3. Cite a fonte de cada informação usada no formato [fonte: arquivo > seção], copiando o identificador do trecho.
4. Seja direto: 1 a 3 frases."""


@dataclass
class RespostaRAG:
    resposta: str
    recusada: bool
    motivo: str = ""
    recuperados: list[Resultado] = field(default_factory=list)


def montar_contexto(resultados: list[Resultado]) -> str:
    blocos = [f"[fonte: {r.chunk.fonte}]\n{r.chunk.texto}" for r in resultados]
    return "\n\n".join(blocos)


class AssistenteRAG:
    def __init__(self, chunks: list[Chunk] | None = None, k: int = TOP_K) -> None:
        self.k = k
        self.chunks = chunks or carregar_chunks()
        self.store = VectorStore()
        self.store.adicionar(self.chunks, embed([c.texto_indexado for c in self.chunks]))
        self._cliente = None

    def recuperar(self, pergunta: str) -> list[Resultado]:
        return self.store.buscar(embed([pergunta])[0], self.k)

    def _llm(self, pergunta: str, resultados: list[Resultado]) -> str:
        if self._cliente is None:
            import anthropic  # import tardio: evals de retrieval não precisam da API

            self._cliente = anthropic.Anthropic()
        resposta = self._cliente.messages.create(
            model=MODELO,
            max_tokens=1500,  # folga: o modelo pode gastar tokens raciocinando antes de responder
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"<contexto>\n{montar_contexto(resultados)}\n</contexto>\n\nPergunta: {pergunta}",
                }
            ],
        )
        # A resposta pode trazer blocos de raciocínio (thinking) antes do texto: junta só os blocos de texto.
        return "".join(b.text for b in resposta.content if getattr(b, "type", "") == "text").strip()

    def perguntar(self, pergunta: str) -> RespostaRAG:
        resultados = self.recuperar(pergunta)

        # Guardrail de entrada: fora do escopo -> recusa sem chamar o modelo
        if not guardrails.pergunta_dentro_do_escopo(resultados[0].score):
            return RespostaRAG(guardrails.MENSAGEM_RECUSA, True, "fora_do_escopo", resultados)

        texto = self._llm(pergunta, resultados)

        # Recusa legítima do próprio modelo
        if texto.startswith("Não encontrei essa informação"):
            return RespostaRAG(guardrails.MENSAGEM_RECUSA, True, "modelo_sem_resposta", resultados)

        # Guardrail de saída: resposta sem citação não é confiável
        if not guardrails.resposta_tem_citacao(texto):
            return RespostaRAG(guardrails.MENSAGEM_RECUSA, True, "sem_citacao", resultados)

        return RespostaRAG(texto, False, "", resultados)


def main() -> None:
    print("Carregando modelo de embeddings e indexando políticas...")
    assistente = AssistenteRAG()
    print(f"{len(assistente.chunks)} chunks indexados. Digite 'sair' para encerrar.\n")
    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue
        r = assistente.perguntar(pergunta)
        print(f"\nAssistente: {r.resposta}")
        debug = ", ".join(f"{x.chunk.fonte} ({x.score:.2f})" for x in r.recuperados)
        print(f"[debug] recusada={r.recusada} {r.motivo} | recuperados: {debug}\n")


if __name__ == "__main__":
    main()
