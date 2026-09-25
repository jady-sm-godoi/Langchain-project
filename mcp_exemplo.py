import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
model = init_chat_model("gemini-3.6-flash", model_provider="google_genai")

# ─────────────────────────────── CONFIGURAÇÃO DOS SERVIDORES MCP ───────────────────────────────
# O MultiServerMCPClient gerencia múltiplos servidores MCP simultaneamente.
# Cada servidor é definido por um nome + configuração de transporte.
client = MultiServerMCPClient(
    {
        # ── Transporte stdio ──────────────────────────────────────────────────
        # Roda o servidor como um processo local na máquina.
        # O "command" + "args" define o executável/pacote MCP a ser iniciado.
        "busca": {
            "transport": "stdio",       # comunicação via stdin/stdout do subprocesso
            "command": "uvx",           # uvx executa pacotes Python sem instalá-los
            "args": ["duckduckgo-mcp-server"],  # servidor MCP de busca DuckDuckGo
        },
        # ── Transporte http (Streamable HTTP) ────────────────────────────────
        # Conecta a um servidor MCP remoto hospedado (não precisa rodar local).
        # O "url" aponta para o endpoint do servidor.
        "docs": {
            "transport": "http",        # comunicação via requisições HTTP
            "url": "https://mcp.context7.com/mcp",  # servidor público de documentação
        },
    }
)

# Converte as ferramentas expostas pelos servidores MCP em tools do LangChain
tools = asyncio.run(client.get_tools())

# Cria o agente LangChain com as ferramentas vindas dos servidores MCP
agente = create_agent(
    model=model,
    tools=tools,
)

# ═══════════════════════════════ TESTES ═══════════════════════════════
# Os blocos abaixo são mutuamente exclusivos: descomente UM de cada vez.

# ── Teste 1: Listar ferramentas disponíveis ──────────────────────────
# async def main():
#     tools = await client.get_tools()
#     print(f"{len(tools)} tools chegaram:\n")
#     for t in tools:
#         print(f"  {t.name}: {t.description[:70]}")
# asyncio.run(main())

# ── Teste 2: Executar o agente com uma pergunta ─────────────────────
# async def main():
#     r = await agente.ainvoke({
#         "messages": [{"role": "user", "content": "Como fazer tofu?"}]
#     })
#     # A resposta do modelo está em r["messages"][-1].content.
#     # Quando a tool retorna múltiplos blocos (texto, imagens, etc.),
#     # filtra-se pelo tipo desejado:
#     resposta = next(
#         b["text"] for b in r["messages"][-1].content if b["type"] == "text"
#     )
#     print(resposta)
# asyncio.run(main())