# LangChain Project

Projeto de estudos do curso de LangChain. O objetivo é aprender a construir aplicações com **LLMs** (Grandes Modelos de Linguagem) usando o ecossistema LangChain, com o modelo **Google Gemini** como provedor principal.

Este README é **didático** e será **aprimorado conforme o curso evolui** — cada módulo novo de estudo deve adicionar/atualizar as seções correspondentes.

## Índice

- [Visão geral](#visão-geral)
- [Pré-requisitos](#pré-requisitos)
- [Configuração do ambiente](#configuração-do-ambiente)
- [Como rodar](#como-rodar)
- [Conteúdo do curso](#conteúdo-do-curso)
- [Ferramentas extras — LangGraph Studio/CLI](#ferramentas-extras--langgraph-studiocli)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Dependências](#dependências)
- [Roadmap do curso](#roadmap-do-curso)
- [Referências](#referências)

## Visão geral

**LangChain** é um framework para construir aplicações com LLMs. Em vez de "chamar a API do modelo" direto, você monta **cadeias de componentes** (prompts, modelos, parsers, ferramentas) que se conectam entre si.

O projeto hoje demonstra dois conceitos centrais do LangChain:

1. **Pipeline (cadeia)** — uma entrada de texto passa por etapas encadeadas até gerar uma resposta formatada, tudo conectado com o operador `|`:
   ```
   entrada → prompt → modelo (Gemini) → parser → pós-processamento → saída
   ```
2. **Agente com LangGraph** — um agente que usa **ferramentas** (busca web via Tavily) para responder, com grafo servido localmente pela CLI do LangGraph.

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

Para dar "memória" ao agente, usamos o mecanismo de **checkpoint** do LangGraph. O `agent.py` foi atualizado com 3 peças:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpoint = InMemorySaver()   # 1. "memória" do agente (em memória)

agente_jady = create_agent(
    model=model,
    system_prompt="Você é um assistente útil e prestativo...",
    tools=[TavilySearch()],
    checkpointer=checkpoint,   # 2. conecta a memória ao agente
)

config = {"configurable": {"thread_id": "1"}}  # 3. identifica a conversa

resposta = agente_jady.invoke(
    {"messages": [{"role": "user", "content": pergunta}]},
    config=config,
)
```

| Peça | Papel |
| --- | --- |
| `InMemorySaver` | Checkpointer que guarda o **estado do grafo** (histórico de mensagens) em memória |
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

**Como funciona:** na primeira pergunta, o histórico ("Olá, meu nome é Jadysse" + a resposta) é **salvo no checkpoint** da thread `1`. Na segunda pergunta, o agente **recupera esse histórico** e o reenvia ao modelo junto da nova mensagem — por isso o modelo agora sabe o nome.

> **Atenção (didático):**
> - O `thread_id` está **fixo** em `"1"` no código (`# TODO: id dinâmico`) — em produção, cada usuário/sessão teria um id próprio. Threads diferentes não compartilham memória.
> - O `InMemorySaver` guarda tudo **em memória RAM**: o estado se perde ao reiniciar o processo. Para persistência entre execuções, usa-se um saver com banco de dados (ex.: `SqliteSaver`, `PostgresSaver`).

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
├── agent.py              # Agente com LangGraph (Gemini + Tavily)
├── langgraph.json        # Configuração do grafo para a CLI
├── main.ipynb            # Notebook principal do curso
├── main.py               # Entry point simples do projeto
├── pyproject.toml        # Definição do projeto e dependências
├── README.md
└── uv.lock               # Lockfile de dependências (uv)
```

## Dependências

Definidas em `pyproject.toml`:

| Pacote | Papel |
| --- | --- |
| `langchain` | Framework principal para construir cadeias/aplicações com LLMs |
| `langchain-core` | Núcleo: interfaces de prompts, modelos, parsers e `Runnable`s |
| `langchain-google-genai` | Integração com os modelos Google Gemini |
| `langchain-groq` | Integração com modelos Groq (provedor alternativo) |
| `langchain-openai` | Integração com modelos OpenAI (provedor alternativo) |
| `langchain-tavily` | Tool de busca web (Tavily) usada pelo agente |
| `langgraph-cli[inmem]` | CLI do LangGraph com runtime em memória (sem Docker) |
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
- [x] Demonstração: modelos são stateless (sem memória)
- [x] Memória com checkpoints: `InMemorySaver` + `thread_id`
- [ ] Persistência real da memória (banco de dados, ex.: `SqliteSaver`/`PostgresSaver`)
- [ ] LangGraph Studio (visualização e depuração)
- [ ] RAG (recuperação de informação) com Tavily
- [ ] TBD...

## Referências

- [Documentação do LangChain (Python)](https://python.langchain.com/docs/introduction/)
- [LangGraph CLI](https://reference.langchain.com/python/langgraph-cli)
- [uv — gerenciador de projetos](https://docs.astral.sh/uv/)
- [Google AI Studio (chave da API Gemini)](https://aistudio.google.com/apikey)
