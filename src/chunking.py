"""
ETAPA 1 - CHUNKING: transformar documentos em pedaços (chunks) pesquisáveis.

Por que dividir? O modelo de embeddings e o prompt têm limite de tamanho, e um vetor
que representa um documento inteiro "dilui" o significado. Pedaços pequenos e coesos
geram vetores mais específicos, então a busca encontra a regra certa.

Decisão de design: nossos documentos já são estruturados em seções (`## Título`).
Cada seção é uma regra de negócio completa, então dividimos POR SEÇÃO.
- Vantagem: o chunk é semanticamente coeso e fácil de citar como fonte.
- Limite: se uma seção ficar muito grande, `dividir_secao_longa` quebra por tamanho
  com sobreposição (overlap), para que uma frase cortada no meio não perca contexto.

Pergunta clássica de entrevista: "como você escolheu o tamanho do chunk?"
Resposta: chunk pequeno = busca precisa mas pouco contexto; chunk grande = muito
contexto mas busca ruidosa. Aqui, a estrutura do documento decide; o tamanho
máximo é só uma rede de segurança.
"""

from dataclasses import dataclass
from pathlib import Path

PASTA_POLITICAS = Path(__file__).resolve().parent.parent / "data" / "politicas"
TAMANHO_MAXIMO = 800  # caracteres
OVERLAP = 100  # caracteres repetidos entre chunks vizinhos


@dataclass(frozen=True)
class Chunk:
    arquivo: str  # ex.: politica_voos.md
    secao: str  # ex.: Classe de voo permitida
    texto: str

    @property
    def fonte(self) -> str:
        """Identificador que o modelo deve citar na resposta."""
        return f"{self.arquivo} > {self.secao}"

    @property
    def texto_indexado(self) -> str:
        """Texto que vira embedding: o título da seção ajuda a busca."""
        return f"{self.secao}. {self.texto}"


def dividir_secao_longa(texto: str, tamanho: int = TAMANHO_MAXIMO, overlap: int = OVERLAP) -> list[str]:
    """Quebra texto longo em janelas com sobreposição."""
    if len(texto) <= tamanho:
        return [texto]
    pedacos, inicio = [], 0
    while inicio < len(texto):
        pedacos.append(texto[inicio : inicio + tamanho])
        inicio += tamanho - overlap
    return pedacos


def carregar_chunks(pasta: Path = PASTA_POLITICAS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for caminho in sorted(pasta.glob("*.md")):
        secao_atual, linhas = None, []

        def fechar_secao():
            if secao_atual and linhas:
                corpo = " ".join(l.strip() for l in linhas if l.strip())
                for pedaco in dividir_secao_longa(corpo):
                    chunks.append(Chunk(caminho.name, secao_atual, pedaco))

        for linha in caminho.read_text(encoding="utf-8").splitlines():
            if linha.startswith("## "):
                fechar_secao()
                secao_atual, linhas = linha[3:].strip(), []
            elif linha.startswith("# "):
                continue  # título do documento
            else:
                linhas.append(linha)
        fechar_secao()
    return chunks


if __name__ == "__main__":
    todos = carregar_chunks()
    print(f"{len(todos)} chunks carregados\n")
    for c in todos:
        print(f"- [{c.fonte}] ({len(c.texto)} chars)")
