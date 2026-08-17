# LangChain Project

Projeto de estudos do curso de LangChain. O objetivo é aprender a construir aplicações com **LLMs** (Grandes Modelos de Linguagem) usando o ecossistema LangChain, com o modelo **Google Gemini** como provedor principal.

Este README é **didático** e será **aprimorado conforme o curso evolui** — cada módulo novo de estudo deve adicionar/atualizar as seções correspondentes.

## Índice

- [Visão geral](#visão-geral)
- [Pré-requisitos](#pré-requisitos)
- [Configuração do ambiente](#configuração-do-ambiente)
- [Como rodar](#como-rodar)
- [Conteúdo do curso](#conteúdo-do-curso)
- [Criando ferramentas para agentes](#criando-ferramentas-para-agentes)
- [Agente com banco de dados (tools nativas)](#agente-com-banco-de-dados-tools-nativas)
- [Gerenciando o contexto com SummarizationMiddleware](#gerenciando-o-contexto-com-summarizationmiddleware)
- [Ferramentas extras — LangGraph Studio/CLI](#ferramentas-extras--langgraph-studiocli)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Dependências](#dependências)
- [Roadmap do curso](#roadmap-do-curso)
- [Referências](#referências)

## Visão geral

**LangChain** é um framework para construir aplicações com LLMs. Em vez de "chamar a API do modelo" direto, você monta **cadeias de componentes** (prompts, modelos, parsers, ferramentas) que se conectam entre si.

O projeto hoje demonstra **quatro** conceitos centrais do LangChain:

1. **Pipeline (cadeia)** — uma entrada de texto passa por etapas encadeadas até gerar uma resposta formatada, tudo conectado com o operador `|`:
   ```
   entrada → prompt → modelo (Gemini) → parser → pós-processamento → saída
   ```
2. **Agente com LangGraph** — um agente que usa **ferramentas** (busca web via Tavily) para responder, com grafo servido localmente pela CLI do LangGraph.
3. **Ferramentas para agentes** — a evolução da criação de ferramentas **próprias**: de uma tool simples (conversão de temperatura) até uma tool com **validação de entrada** e **tratamento de erros** (busca de CEP na ViaCEP).
4. **Agente com banco de dados** — um agente que consulta um banco SQLite em linguagem natural usando o `SQLDatabaseToolkit`, com ferramentas que inspecionam o esquema, validam e executam consultas SQL.

## Pré-requisitos

- **Python 3.13** (versão gerenciada pelo arquivo `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — gerenciador de dependências e ambientes (rápido, usa o `pyproject.toml`)
- **Conta Google** com acesso ao **Gemini API** (para gerar a chave `GEMINI_API_KEY`)
- **Chave do Tavily** (`TAVILY_API_KEY`) — usada pela tool de busca web do agente. Gere em [tavily.com](https://tavily.com)

## Configuração do ambiente

1. **Instalar as dependências** do `pyproject.toml`:

   ```bash
   uv sync
   ```

2. **Configurar as variáveis de ambiente**: o arquivo `.env` (que **não é commitado**) contém as chaves. Crie-o a partir do exemplo:

   ```bash
   cp .env.example .env
   ```

3. **Preencher as chaves** no `.env`:

   ```
   GEMINI_API_KEY=suas_chave_aqui
   TAVILY_API_KEY=sua_chave_aqui
   ```

   > A `GEMINI_API_KEY` é obrigatória para rodar a pipeline e o agente. Gere em [Google AI Studio](https://aistudio.google.com/apikey). A `TAVILY_API_KEY` é necessária para a busca web do agente.

## Como rodar

O código do curso fica no notebook `main.ipynb`. Para abri-lo:

```bash
uv run jupyter notebook main.ipynb
```

Em seguida, execute as células na ordem. O pipeline já está configurado e pronto para uso:

```python
pipeline.invoke("o que é a inteligência artificial")
```

Saída esperada (resposta do Gemini em caixa alta, pois há um passo de pós-processamento):

```
A INTELIGÊNCIA ARTIFICIAL (IA) É UM CAMPO DA TECNOLOGIA DEDICADO À CRIAÇÃO DE SISTEMAS E MÁQUINAS CAPAZES DE SIMULAR A CAPACIDADE HUMANA DE RACIOCINAR, APRENDER E TOMAR DECISÕES. ...
```

### Rodando o agente

O agente fica em `agent.py`. Exemplo de invocação (com memória por `thread_id`):

```python
from agent import agente_jady

config = {"configurable": {"thread_id": "1"}}
resposta = agente_jady.invoke(
    {"messages": [{"role": "user", "content": "Qual a temperatura média em São Paulo hoje?"}]},
    config=config,
)
print(resposta["messages"][-1].text)
```

Para visualizar e depurar o grafo do agente na UI (LangGraph Studio):

```bash
langgraph dev
```

O servidor sobe em `http://localhost:2024`, lendo o `langgraph.json`.

> Hoje o `langgraph dev` carrega o agente de banco de dados (`./agente_banco_v2.py:agente_banco`). Para testar os outros agentes, altere o `langgraph.json` (ex.: `./agente_banco.py:agente_banco`, `./tool_busca_cep.py:agente_cep` ou `./too_celsius_fahrenheit.py:agente_clima`).

## Conteúdo do curso

Seção evolutiva: registra o que já foi estudado e servirá de índice para os próximos módulos.

### Pipeline básica com LangChain

Construção da primeira pipeline, ligando 4 componentes com o operador `|`:

```python
pipeline = prompt | model | parser | passo_extra
```

Componentes:

| Componente | Papel |
| --- | --- |
| `ChatPromptTemplate` | Define o template da pergunta com a variável `{assunto}` e formata a entrada |
| `init_chat_model(...)` | Instancia o modelo Gemini (`google_genai`) |
| `StrOutputParser` | Converte a resposta do modelo em texto simples |
| `RunnableLambda` | Passo extra de pós-processamento (ex.: deixar tudo em maiúsculas) |

**Fluxo dos dados:**

1. **`prompt`** recebe o dicionário/string de entrada e monta a `ChatPromptValue` com o `{assunto}` preenchido;
2. **`model`** recebe o prompt e devolve um `AIMessage` com a resposta do Gemini;
3. **`parser`** extrai o texto puro do `AIMessage`;
4. **`passo_extra`** aplica transformações finais (ex.: `.upper()`).

### A ordem importa (nota didática)

A pipeline é executada **da esquerda para a direita**. O erro clássico é montar `model | prompt | parser`: nesse caso, o modelo recebe a entrada crua e o parser recebe um `ChatPromptValue` (objeto, não texto), estourando um `ValidationError`.

> A ordem correta é sempre **`prompt | model | parser`**: o prompt primeiro, o modelo no meio e o parser por último. Depois deles, adicione quantos `RunnableLambda` quiser.

### Executando a pipeline

Para passar valores ao template, use um dicionário com a chave da variável:

```python
resultado = pipeline.invoke({"assunto": "o que é a inteligência artificial"})
print(resultado)
```

### Agente com LangGraph e busca Tavily

Um **agente** vai além da pipeline fixa: ele decide, em loop, se usa **ferramentas** antes de responder. O `agent.py` cria um agente que consulta a web via Tavily quando precisa:

```python
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

agente_jady = create_agent(
    model=model,                                    # Gemini
    system_prompt="Você é um assistente útil e prestativo...",
    tools=[TavilySearch()],                         # tool de busca web
)
```

Conceitos-chave:

| Conceito | Papel |
| --- | --- |
| `create_agent` | Monta o agente: modelo + system prompt + ferramentas |
| `TavilySearch` | Tool de busca na web; o agente a chama para buscar informações atuais |
| `langgraph.json` | Configuração do grafo para a CLI — aponta `agent` para `./agent.py:agente_jady` e carrega o `.env` |
| `langgraph dev` | Sobe o servidor de desenvolvimento com hot reload em `http://localhost:2024` |

**Como o agente funciona:** o Gemini responde; se a pergunta exige informação externa, o modelo decide chamar a `TavilySearch`, recebe o resultado da busca e então monta a resposta final com base nesses dados.

O agente também é exposto como grafo do LangGraph — é o que o `langgraph.json` faz ao declarar a variável `agente_jady` como entrada do grafo.

### Criando ferramentas para agentes

Um agente só é útil se tiver **ferramentas** (tools) para agir. O `agent.py` já usava a `TavilySearch` **pronta**; agora evoluímos para **criar nossas próprias ferramentas**. Esta seção documenta essa evolução em **duas fases**.

#### Fase 1 — a tool simples (`too_celsius_fahrenheit.py`)

A primeira ferramenta foi construída com o decorador `@tool`. Ela converte uma temperatura de Celsius para Fahrenheit:

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def converter_temperatura(celsius: float) -> str:
    """
    Converte uma temperatura de Celsius para Fahrenheit.
    Use quando o usuário perguntar sobre conversão de temperatura.

    Args:
        celsius (float): A temperatura em graus Celsius.

    returns:
        str: A temperatura convertida em Fahrenheit, formatada como uma string.
    """
    fahrenheit = (celsius * 9/5) + 32
    return f"{fahrenheit:.2f} °F"

agente_clima = create_agent(
    model=model,
    system_prompt="...",
    tools=[converter_temperatura],
)
```

**O contrato de uma tool boa:** para o modelo saber **quando** e **como** usar a ferramenta, ela precisa de um "contrato" claro:

| Elemento | Papel |
| --- | --- |
| Nome da função | `converter_temperatura` — descreve a ação de forma concisa |
| Doc-string | Explica o que a função faz, o uso indicado, os parâmetros e o retorno |
| Type hints | `celsius: float -> str` — informa os tipos de entrada e saída |
| Descrição | Frase curta indicando **quando usar** (o modelo lê isso para decidir) |

**Responsabilidade única:** a tool faz **uma coisa só** — converter temperatura. Ferramentas genéricas confundem o modelo e ficam difíceis de testar.

**Como o agente usa:** quando o usuário pergunta "quanto é 100°C em Fahrenheit?", o Gemini decide chamar `converter_temperatura(100)`, recebe `212.00 °F` e monta a resposta final com esse valor.

#### Fase 2 — a tool com camadas de proteção (`tool_busca_cep.py`)

A segunda ferramenta vai além: ela consulta uma **API externa** (ViaCEP) e adiciona **camadas de proteção** para o mundo real. Três novidades:

**1. Validação de entrada com Pydantic (`args_schema`)**

Um esquema Pydantic descreve os argumentos esperados e **valida** o que o modelo enviar **antes** de a função rodar:

```python
from pydantic import BaseModel, Field, field_validator

class cep_input(BaseModel):  # classe para validação de entrada de dados com Pydantic
    cep: str = Field(..., description="O CEP a ser consultado no padrão brasileiro com 8 dígitos.")

    @field_validator("cep")
    @classmethod
    def validate_cep(cls, v: str) -> str:
        cep_limpo = v.replace("-", "").strip()
        if not cep_limpo.isdigit() or len(cep_limpo) != 8:
            raise ValueError("CEP inválido. Deve conter apenas números e ter 8 dígitos.")
        return cep_limpo

@tool(args_schema=cep_input)  # o esquema de validação é passado à tool
def busca_cep(cep: str) -> str:
    """..."""
    url = f"https://viacep.com.br/ws/{cep}/json/"
    response = requests.get(url)
    data = response.json()
    return data
```

- `Field(..., description=...)` descreve o parâmetro para o modelo — enriquece o contrato da tool.
- `field_validator("cep")` roda antes da execução: remove hífens/espaços e garante **8 dígitos numéricos**. Se inválido, lança `ValueError` e a tool **nem chega a chamar a API**.

**2. Tratamento de erros com `@wrap_tool_call`**

Erros (API fora do ar, CEP inexistente, falha de rede) não podem derrubar o agente. O middleware `wrap_tool_call` envolve a execução e devolve um `ToolMessage` amigável ao modelo:

```python
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
async def tratar_erros(request, handler):
    try:
        return await handler(request)
    except Exception as e:
        tool_call_id = request.tool_call["id"]
        return ToolMessage(
            content=f"Erro ao executar ferramenta: {request.tool_call['name']}. Detalhes do erro: {str(e)}",
            tool_call_id=tool_call_id,
        )

agente_cep = create_agent(
    model=model,
    system_prompt="...",
    tools=[busca_cep],
    middleware=[tratar_erros],   # intercepta falhas e devolve mensagem de erro
)
```

**3. Chamada a uma API real**

A tool usa `requests` para consultar a ViaCEP e devolve os dados do endereço.

**Comparativo didático — Fase 1 × Fase 2:**

| Aspecto | Fase 1 — `converter_temperatura` | Fase 2 — `busca_cep` |
| --- | --- | --- |
| Cálculo local | ✅ fórmula direta | ❌ consulta a uma API externa |
| Validação de entrada | Não (entrada numérica simples) | ✅ Pydantic (`args_schema` + `field_validator`) |
| Tratamento de erros | Implícito | ✅ middleware `@wrap_tool_call` |
| Risco de falha | Baixo | Alto (rede, API, dados inexistentes) |
| Complexidade | Mínima | Média |

**A lição:** tools simples não precisam de muita proteção; tools que tocam o **mundo externo** (APIs, bancos, arquivos) **exigem** validação e tratamento de erros. É o que os comentários no fim do `tool_busca_cep.py` chamam de **camadas de proteção** — validação de entrada, tratamento de erros, logging, testes, monitoramento, segurança, entre outras.

### Agente com banco de dados (tools nativas)

Um agente pode ir além de ferramentas que chamam APIs e interagir diretamente com um **banco de dados relacional**. O projeto tem duas versões desse agente:

- **`agente_banco.py` (v1, referência didática):** usa o `SQLDatabaseToolkit` do `langchain-community`, que monta automaticamente as 4 ferramentas SQL. Mantido para estudo/consulta.
- **`agente_banco_v2.py` (v2, atual):** implementa as **mesmas 4 ferramentas manualmente** com `@tool` + `sqlite3`, sem depender do `langchain-community`.

**Por que o v2 existe:** o pacote `langchain-community` foi **aposentado (sunset) em maio de 2026** — não recebe mais correções e seu repositório foi arquivado. Como não há pacote sucessor oficial para o `SQLDatabaseToolkit`, a orientação da própria LangChain é implementar as tools SQL diretamente no código da aplicação, usando o driver nativo do banco. É exatamente o que o v2 faz.

**As quatro ferramentas (idênticas em ambas as versões):**

| Ferramenta | Papel |
| --- | --- |
| `sql_db_list_tables` | Lista as tabelas existentes no banco |
| `sql_db_schema` | Retorna o esquema de uma tabela (colunas e amostras de dados) |
| `sql_db_query` | Executa uma consulta SQL e retorna o resultado |
| `sql_db_query_checker` | Valida a consulta e corrige erros comuns **antes** de executar |

No v1, elas eram obtidas de uma linha só:

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()   # as 4 ferramentas prontas
```

No v2, cada uma vira uma função anotada com `@tool` que abre e fecha a conexão com `sqlite3` a cada chamada (padrão `try/finally`):

```python
import sqlite3
from langchain.tools import tool

@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return ", ".join(row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_"))
    finally:
        conn.close()
```

Detalhes da implementação do v2:

- **`sql_db_schema`** consulta o `sqlite_master` para obter o `CREATE TABLE` e traz **3 linhas de amostra** de cada tabela (dá "exemplos" do formato dos dados ao modelo);
- **`sql_db_query`** captura exceções e devolve o erro como string — o modelo **lê o erro, corrige a query e tenta de novo** (padrão de auto-correção);
- **`sql_db_query_checker`** usa o próprio `model` para revisar a query gerada contra uma lista de erros comuns (NOT IN com NULL, UNION em vez de UNION ALL, BETWEEN exclusivo, tipos, joins...) e devolver a query corrigida;
- nomes de tabelas são **validados contra a lista real do banco** e escapados com aspas antes de montar o SQL (evita injeção).

O agente é criado com `create_agent`, como nos módulos anteriores:

```python
agente_banco = create_agent(
    model=model,
    system_prompt=system_prompt,   # SELECT apenas, LIMIT 10, responder em português
    tools=tools,
)
```

**Como funciona:** o usuário faz uma pergunta em linguagem natural (ex.: "quais os 5 produtos mais caros?"). O modelo:
1. inspeciona as tabelas (`sql_db_list_tables`) e o esquema (`sql_db_schema`);
2. gera a consulta SQL;
3. valida com o `sql_db_query_checker`;
4. executa com `sql_db_query` e monta a resposta final em português.

**Regras de segurança no system prompt** (a camada de proteção para bancos):

- inspecionar tabelas/esquema **antes** de gerar qualquer query (evita inventar colunas ou tabelas);
- limitar resultados a no máximo **10 registros** (`LIMIT 10`);
- **apenas `SELECT`** — nenhum comando DML (INSERT, UPDATE, DELETE, DROP);
- se não encontrar o dado, avisar em vez de inventar.

> **Atenção (didático):** essa proteção vive no **prompt**. Em produção, o ideal é usar um **usuário do banco com permissão somente de leitura** — o prompt orienta o modelo, mas o banco é quem garante a segurança de verdade.

### Modelos são stateless (sem memória)

Demonstração feita rodando o `agent.py` em modo de conversa:

```
Digite sua pergunta: Meu nome é Jady
Resposta: Prazer em conhecê-la, Jady! Como posso te ajudar hoje?
Digite sua pergunta: Qual o meu nome?
Resposta: Caralho, eu não sei! Você não me disse o seu nome ainda.
```

**Por que isso acontece:** cada `invoke` é uma chamada **independente** à API. O modelo não guarda estado entre requisições — ele apenas "enxerga" o que é enviado naquela chamada. Na segunda pergunta, o contexto (o nome) **não foi reenviado**, então o modelo não tem como saber.

**Conceito:** LLMs são **stateless** (sem estado). A "memória" **não é intrínseca ao modelo** — é um **acessório** que precisamos implementar por fora. O padrão é reenviar o **histórico de mensagens** a cada chamada:

```
"Meu nome é Jady"  →  "Qual o meu nome?"
```

passando as duas mensagens juntas na invocação. No LangChain/LangGraph, isso é feito com o histórico em `messages` e mecanismos de estado/checkpoint — que é exatamente o que implementamos na seção seguinte.

### Implementando memória com checkpoints (LangGraph)

Para dar "memória" ao agente, usamos o mecanismo de **checkpoint** do LangGraph. Primeiro experimentamos o `InMemorySaver` (tudo em RAM) e depois evoluímos para **persistência em SQLite** — o `checkpoints.db`. O `agent.py` atual usa esta versão:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)  # 1. banco em arquivo
checkpoint = SqliteSaver(conn)

agente_jady = create_agent(
    model=model,
    system_prompt="Você é um assistente útil e prestativo...",
    tools=[TavilySearch()],
    checkpointer=checkpoint,   # 2. conecta a memória ao agente
)

config = {"configurable": {"thread_id": "novo_thread"}}  # 3. identifica a conversa

resposta = agente_jady.invoke(
    {"messages": [{"role": "user", "content": pergunta}]},
    config=config,
)
```

| Peça | Papel |
| --- | --- |
| `sqlite3.connect("checkpoints.db")` | Abre a conexão com o **banco em disco** (arquivo `checkpoints.db`) |
| `SqliteSaver(conn)` | Checkpointer que grava o **estado do grafo** (histórico de mensagens) no SQLite |
| `checkpointer=` | Liga a memória ao agente: cada execução salva o estado |
| `thread_id` | Identifica **uma conversa**; o estado é isolado por `thread_id` |
| `config=config` | Informa na invocação qual conversa está em andamento |

**A demonstração** (mesmo script de antes, agora com memória):

```
Digite sua pergunta: Olá, meu nome é Jadysse
Resposta: Olá, Jadysse! É um prazer falar com você. Como posso te ajudar hoje?
Digite sua pergunta: Qual é meu nome mesmo?
Resposta: Seu nome é Jadysse!
```

**Como funciona:** na primeira pergunta, o histórico ("Olá, meu nome é Jadysse" + a resposta) é **salvo no checkpoint** da thread `novo_thread`. Na segunda pergunta, o agente **recupera esse histórico** e o reenvia ao modelo junto da nova mensagem — por isso o modelo agora sabe o nome.

**`InMemorySaver` vs `SqliteSaver`** (por que evoluímos):

| | `InMemorySaver` | `SqliteSaver` |
| --- | --- | --- |
| Onde guarda | Memória RAM | Arquivo SQLite (`checkpoints.db`) |
| Sobrevive a reiniciar o processo? | ❌ perde tudo | ✅ estado permanece |
| Uso ideal | Testes rápidos/demos | Persistência real (aulas seguintes: `PostgresSaver`) |

> **Atenção (didático):**
> - O `thread_id` está **fixo** no código (`# TODO: id dinâmico`) — em produção, cada usuário/sessão teria um id próprio. Threads diferentes não compartilham memória.
> - O arquivo `checkpoints.db*` é **dado de runtime** (não versionado — está no `.gitignore`).

### Gerenciando o contexto com SummarizationMiddleware

A memória com checkpoints resolve o problema do *stateless*, mas cria outro: a cada `invoke`, **todo o histórico** da thread é reenviado ao modelo. Em conversas longas, o histórico cresce sem limite e **estoura a janela de contexto** (o limite de tokens que o modelo aceita por chamada).

O `SummarizationMiddleware` resolve isso **condensando o histórico antigo**: em vez de reenviar mensagem por mensagem, ele gera um **resumo** das conversas passadas e reenvia só o que importa, mantendo o contexto essencial sem consumir a janela do modelo.

**Configuração atual do `agent.py`:**

```python
from langchain.agents.middleware import SummarizationMiddleware

agente_jady = create_agent(
    model=model,
    system_prompt="...",
    tools=[TavilySearch()],
    checkpointer=checkpoint,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 3000),   # quando a conversa passa de 3k tokens...
            keep=("messages", 10),      # ...mantém as 10 últimas mensagens intactas
        )
    ],
)
```

| Configuração | Papel |
| --- | --- |
| `middleware=[...]` | Lista de middlewares que interceptam a execução do agente (aqui, resumo de contexto) |
| `trigger` | **Gatilho**: dispara a condensação quando o histórico atinge um limite — por contagem de mensagens (`("messages", N)`) ou por volume de tokens (`("tokens", N)`) |
| `keep` | **Retenção**: quantas mensagens recentes permanecem intactas após a condensação |
| `model=model` | Modelo usado para **gerar o resumo** (uma chamada extra, feita em segundo plano) |

**Como funciona:** ao atingir o `trigger`, o middleware comprime as mensagens antigas num resumo, preserva as `keep` mais recentes e segue adiante. O resumo (estado `summary`) é persistido junto ao checkpoint, então a informação condensada não se perde entre execuções. A contrapartida: cada condensação gera **uma chamada adicional ao modelo**.

#### Recomendações e Boas Práticas

* **Aumentar o gatilho (`trigger`):** O ideal é dar margem para o agente acumular contexto útil antes de comprimi-lo. Dependendo do tamanho do modelo, valores como `("messages", 20)` ou baseados em tokens (ex: `("tokens", 3000)` a `("tokens", 4000)`) são muito mais comuns e eficientes.
* **Aumentar a retenção (`keep`):** Recomenda-se manter pelo menos as últimas **10 a 20 mensagens** recentes intactas (`keep=("messages", 20)`), garantindo que o fluxo imediato da conversa e o uso recente de ferramentas não sejam perdidos na condensação.

#### Exemplo de configuração mais equilibrada:
```python
SummarizationMiddleware(
    model=model, 
    trigger=("messages", 20), # ou ("tokens", 4000)
    keep=("messages", 10)       # mantém um histórico recente saudável
)
```

## Ferramentas extras — LangGraph Studio/CLI

Para visualizar e depurar grafos de forma visual, o ecossistema LangChain oferece o **LangGraph Studio**, cujo backend local é gerenciado pela CLI oficial do LangGraph.

Instale a CLI:

```bash
uv add "langgraph-cli[inmem]"
```

**Explicação do comando:**

- **`langgraph-cli`** é a interface de linha de comando oficial do [LangGraph](https://langchain-ai.github.io/langgraph/) — framework do LangChain para construir **agentes e grafos de estados**. A CLI cria projetos (`langgraph new`), sobe servidor de desenvolvimento (`langgraph dev`), roda em Docker (`langgraph up`) e gera build de produção (`langgraph build`).
- **`[inmem]`** é um *extra* que instala o runtime em **memória**, eliminando a necessidade de Docker. É ele que habilita o modo de desenvolvimento `langgraph dev`.
- **`langgraph dev`** sobe um servidor local com **hot reload** em `http://localhost:2024`, lendo o `langgraph.json` (que declara dependências, grafos e variáveis de ambiente — podendo apontar o `env` para `./.env`). Esse servidor é o que alimenta a UI do LangGraph Studio.
- **Quando usar**: quando o curso chegar nos módulos de agentes/grafos, este ambiente será usado para visualizar o grafo, testar e depurar a aplicação em tempo real.

## Estrutura do projeto

```
Langchain-project/
├── .env                  # Chaves de API (NÃO versionado)
├── .env.example          # Modelo das variáveis de ambiente
├── .gitignore
├── .langgraph_api/       # Artefatos de runtime do LangGraph (ignorado)
├── .python-version       # Versão do Python (3.13)
├── agent.py              # Agente com LangGraph (Gemini + Tavily + memória SQLite)
├── agente_banco.py       # v1: agente de banco com SQLDatabaseToolkit (referência didática)
├── agente_banco_v2.py    # v2: agente de banco com tools nativas (@tool + sqlite3, sem langchain-community)
├── checkpoints.db        # Banco da memória do agente (runtime, ignorado)
├── langgraph.json        # Configuração do grafo para a CLI (aponta para `agente_banco_v2.py:agente_banco`)
├── loja.sqlite           # Banco da loja fictícia consultado pelo agente (runtime, ignorado)
├── main.ipynb            # Notebook principal do curso
├── main.py               # Entry point simples do projeto
├── pyproject.toml        # Definição do projeto e dependências
├── README.md
├── tool_busca_cep.py     # Fase 2: tool com validação Pydantic + tratamento de erros (agente_cep)
├── too_celsius_fahrenheit.py  # Fase 1: tool simples de conversão de temperatura (agente_clima)
└── uv.lock               # Lockfile de dependências (uv)
```

## Dependências

Definidas em `pyproject.toml`:

| Pacote | Papel |
| --- | --- |
| `langchain` | Framework principal para construir cadeias/aplicações com LLMs |
| `langchain-core` | Núcleo: interfaces de prompts, modelos, parsers e `Runnable`s |
| `langchain-community` | Integrações da comunidade. **Aposentado em maio/2026** — mantido no projeto apenas para o `agente_banco.py` (v1) de referência executar |
| `langchain-google-genai` | Integração com os modelos Google Gemini |
| `langchain-groq` | Integração com modelos Groq (provedor alternativo) |
| `langchain-openai` | Integração com modelos OpenAI (provedor alternativo) |
| `langgraph` | Framework de grafos/estados para agentes |
| `langgraph-checkpoint-sqlite` | Checkpointer persistente em SQLite (fornece o `SqliteSaver`) |
| `langgraph-cli[inmem]` | CLI do LangGraph com runtime em memória (sem Docker) |
| `langchain-tavily` | Tool de busca web (Tavily) usada pelo agente |
| `dotenv` | Carrega variáveis de ambiente do arquivo `.env` |

O projeto usa **uv** para gerenciar as dependências (`uv.lock` garante reprodutibilidade). Para adicionar novos pacotes:

```bash
uv add nome-do-pacote
```

## Roadmap do curso

> A atualizar conforme as aulas avançam.

- [x] Setup do projeto com uv e variáveis de ambiente
- [x] Primeira pipeline: `prompt | model | parser | RunnableLambda`
- [x] Agente com LangGraph: `create_agent` + tool `TavilySearch`
- [x] Ferramentas para agentes — Fase 1: `@tool` simples (`converter_temperatura`)
- [x] Ferramentas para agentes — Fase 2: validação Pydantic + tratamento de erros (`busca_cep`)
- [x] Agente com banco de dados: `SQLDatabaseToolkit` (`agente_banco.py`)
- [x] Migração pós-sunset: agente de banco com tools nativas (`agente_banco_v2.py`, sem `langchain-community`)
- [x] Demonstração: modelos são stateless (sem memória)
- [x] Memória com checkpoints: `InMemorySaver` + `thread_id`
- [x] Persistência real da memória: `SqliteSaver` (arquivo `checkpoints.db`)
- [x] Resumo de contexto: `SummarizationMiddleware` (`trigger` + `keep`)
- [ ] LangGraph Studio (visualização e depuração)
- [ ] RAG (recuperação de informação) com Tavily
- [ ] TBD...

## Referências

- [Documentação do LangChain (Python)](https://python.langchain.com/docs/introduction/)
- [LangGraph CLI](https://reference.langchain.com/python/langgraph-cli)
- [uv — gerenciador de projetos](https://docs.astral.sh/uv/)
- [Google AI Studio (chave da API Gemini)](https://aistudio.google.com/apikey)
