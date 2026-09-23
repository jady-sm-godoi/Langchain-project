# =============================================================================
# Human in the loop (aprovacao humana) — exemplo didatico
# -----------------------------------------------------------------------------
# Demonstra o `HumanInTheLoopMiddleware`: um middleware que PAUSA o agente
# (por meio do `interrupt()` do LangGraph) sempre que ele decide chamar uma
# ferramenta sensivel/potencialmente perigosa, e aguarda a decisao de um humano
# (aprovar ou rejeitar) antes de realmente executar a ferramenta.
#
# Cenario do exemplo:
#   - `consultar_client`   -> ferramenta apenas de leitura (auto-aprovada).
#   - `cancelar_plano`     -> ferramenta DESTRUTIVA (exige aprovacao humana).
#
# Fluxo:
#   1) O agente recebe "Cancela o plano do cliente 4471".
#   2) O modelo decide chamar a tool `cancelar_plano`.
#   3) O middleware intercepta (hook `after_model`) e gera um INTERRUPT,
#      pausando o grafo ANTES de executar a tool.
#   4) Nosso codigo detecta `result.interrupts` e pergunta ao humano:
#      [a]provar ou [r]ejeitar.
#   5) A decisao e enviada de volta via `Command(resume={"decisions": [...]})`.
#   6) O grafo RETOMA: se "approve" executa a tool; se "reject" NAO executa
#      (devolve uma ToolMessage de erro ao modelo e ele responde ao usuario).
#   7) O loop repete enquanto houver novos interrupts.
#
# O exemplo usa o modelo Google Gemini ("google_genai").
# =============================================================================

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()


MODEL = init_chat_model("gemini-3.6-flash", model_provider="google_genai")

# -----------------------------------------------------------------------------
# Checkpointer: NECESSARIO para o human in the loop.
# A pausa do `interrupt()` salva o estado do grafo (as mensagens, as tool_calls
# pendentes) em um "checkpoint". Sem o checkpointer, nao haveria como RETOMAR
# o grafo de onde parou. Aqui usamos o `InMemorySaver` (tudo em RAM, ideal
# para demos). Em producao, usar um saver persistente (SqliteSaver, etc).
# -----------------------------------------------------------------------------
checkpointer = InMemorySaver()

# `thread_id`: identifica UMA conversa/sessao. O estado (incluindo a pausa do
# interrupt) fica isolado por `thread_id` — e por isso conseguimos retomar a
# MESMA execucao depois da resposta do humano.
config = {"configurable": {"thread_id": "sessao-1"}}

# Mapa de decisoes que o humano pode digitar no terminal:
decisoes = {
    "a": {"type": "approve"},
    "r": {"type": "reject"},
}


# -----------------------------------------------------------------------------
# Tools de exemplo.
# -----------------------------------------------------------------------------
@tool
def consultar_client(client_id):
    """Consulta informações do cliente com base no ID."""
    return "John Doe, plano Fibra 500MB, ativo desde 2023"


@tool
def cancelar_plano(client_id):
    """Cancela o plano do cliente com base no ID."""
    print(f">>>> Cancelou o plano do cliente {client_id}")
    return f"Plano do cliente {client_id} cancelado com sucesso."


# -----------------------------------------------------------------------------
# Agente com HumanInTheLoopMiddleware.
# -----------------------------------------------------------------------------
agente = create_agent(
    model=MODEL,
    tools=[consultar_client, cancelar_plano],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # `cancelar_plano`: tool que exige aprovacao. O humano pode
                "cancelar_plano": {"allowed_decisions": ["approve", "reject"]},
                # Sem precisar parar o agente para cada consulta simples.
                "consultar_client": False,
            }
        )
    ],
    checkpointer=checkpointer,
)

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1a. invocacao: o grafo roda ate pedir aprovacao da tool `cancelar_plano`.
    # Quando `interrupt()` e chamado pelo middleware, o `invoke` NAO retorna a
    # resposta final — retorna um objeto com `.interrupts` preenchido.
    # `version="v2"` ativa a nova API de retorno (`.value` / `.interrupts`).
    # -------------------------------------------------------------------------
    r = agente.invoke(
        {
            "messages": [
                {"role": "user", "content": "Cancela o plano do cliente 4471"}
            ]
        },
        config,
        version="v2",
    )

    print(r.value["messages"][-1].content)

    # -------------------------------------------------------------------------
    # 2. Loop de HUMAN IN THE LOOP.
    # `r.interrupts` e uma lista (vazia quando o agente terminou). Enquanto
    # houver um interrupt pendente, o agente esta "pausado" esperando um humano.
    # -------------------------------------------------------------------------
    while r.interrupts:
        print("agente pausado...")

        # Pedimos a decisao ao humano no terminal.
        decisao = input("[a]provar  [r]ejeitar > ").strip().lower()

        # Retomamos o grafo enviando a decisao via `Command(resume=...)`.
        # O ambiente `{"decisions": [...]}` alimenta o `interrupt()` que fez a
        # pausa — o par chave/valor precisa casar com o que foi interceptado.
        r = agente.invoke(
            Command(resume={"decisions": [decisoes[decisao]]}),
            config=config,
            version="v2",
        )

    print(r.value["messages"][-1].content)
