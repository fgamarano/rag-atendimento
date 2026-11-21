# rag-atendimento

Assistente de políticas de viagem corporativa com **RAG** (Retrieval-Augmented Generation), em Python.
Responde perguntas sobre voos, hospedagem e despesas **citando a fonte** e recusa o que está fora do escopo.

> Os documentos em `data/politicas/` são fictícios, criados para demonstrar o pipeline.

## Arquitetura

```
pergunta → embedding → busca top-k → guardrail de entrada (escopo)
         → prompt com trechos → LLM (API da Anthropic)
         → guardrail de saída (citação obrigatória) → resposta
```

| Arquivo | Etapa |
|---|---|
| `src/chunking.py` | Divide as políticas em chunks por seção (com divisão por tamanho + overlap como rede de segurança) |
| `src/embeddings.py` | Embeddings locais multilíngues (sentence-transformers) |
| `src/vector_store.py` | Busca por similaridade de cosseno em memória (numpy) |
| `src/guardrails.py` | Recusa fora do escopo e exige citação de fonte |
| `src/rag.py` | Orquestra o pipeline e chama a API da Anthropic |
| `evals/eval.py` | Avaliação em 3 camadas: retrieval, fatos, guardrails |

## Como rodar

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sua-chave"   # Windows PowerShell: $env:ANTHROPIC_API_KEY="sua-chave"

python -m src.chunking            # ver os chunks
python -m src.rag                 # conversar com o assistente
python -m evals.eval              # rodar as avaliações (3 camadas)
python -m evals.eval --so-retrieval   # sem chamar a API
```

Na primeira execução, o modelo de embeddings (~470 MB) é baixado uma vez.
O modelo do LLM pode ser trocado com `RAG_MODELO` (padrão: `claude-haiku-5-5`).

## Calibre o guardrail (importante)

`SCORE_MINIMO_RELEVANCIA` em `src/guardrails.py` (0.30) foi calibrado com o modelo de embeddings real: perguntas válidas
ficaram entre 0.41 e 0.78 e perguntas fora do escopo entre 0.15 e 0.17. É uma amostra pequena; se mudar o modelo ou os
documentos, meça de novo com `python -m evals.eval --so-retrieval`.

## Limitação conhecida

A pergunta "A empresa paga **mala** despachada?" não recupera o trecho *Bagagem* no top-3: a busca semântica não liga a
palavra coloquial "mala" ao termo "bagagem". O teste da camada 1 falha por isso, e re-ranking é a correção natural.

## Avaliação em camadas

1. **Retrieval:** o trecho esperado (arquivo e seção) aparece no top-k? Sem LLM, rápido e determinístico.
2. **Fatos:** a resposta contém os fatos-chave (ex.: "550")? Mede a geração.
3. **Guardrails:** perguntas fora do escopo são recusadas antes do LLM; respostas válidas citam a fonte.

Quando um caso falha, a camada mostra onde procurar o problema (busca, prompt ou guardrail).
