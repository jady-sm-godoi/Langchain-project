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

O projeto hoje demonstra o conceito central do LangChain: a **pipeline** (cadeia). Uma entrada de texto passa por etapas encadeadas até gerar uma resposta formatada — tudo conectado com o operador `|`.

```
entrada → prompt → modelo (Gemini) → parser → pós-processamento → saída
```

## Pré-requisitos

- **Python 3.13** (versão gerenciada pelo arquivo `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — gerenciador de dependências e ambientes (rápido, usa o `pyproject.toml`)
- **Conta Google** com acesso ao **Gemini API** (para gerar a chave `GEMINI_API_KEY`)
- (opcional) Chave do **Tavily** (`TAVILY_API_KEY`) — usada em funcionalidades futuras de busca

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

   > A `GEMINI_API_KEY` é obrigatória para rodar a pipeline. Gere em [Google AI Studio](https://aistudio.google.com/apikey).

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
├── .env                # Chaves de API (NÃO versionado)
├── .env.example        # Modelo das variáveis de ambiente
├── .gitignore
├── .python-version     # Versão do Python (3.13)
├── main.ipynb          # Notebook principal do curso
├── main.py             # Entry point simples do projeto
├── pyproject.toml      # Definição do projeto e dependências
├── README.md
└── uv.lock             # Lockfile de dependências (uv)
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
| `dotenv` | Carrega variáveis de ambiente do arquivo `.env` |

O projeto usa **uv** para gerenciar as dependências (`uv.lock` garante reprodutibilidade). Para adicionar novos pacotes:

```bash
uv add nome-do-pacote
```

## Roadmap do curso

> A atualizar conforme as aulas avançam.

- [x] Setup do projeto com uv e variáveis de ambiente
- [x] Primeira pipeline: `prompt | model | parser | RunnableLambda`
- [ ] Conversa com histórico (mensagens)
- [ ] Ferramentas / agentes com LangGraph
- [ ] LangGraph Studio (visualização e depuração)
- [ ] RAG (recuperação de informação) com Tavily
- [ ] TBD...

## Referências

- [Documentação do LangChain (Python)](https://python.langchain.com/docs/introduction/)
- [LangGraph CLI](https://reference.langchain.com/python/langgraph-cli)
- [uv — gerenciador de projetos](https://docs.astral.sh/uv/)
- [Google AI Studio (chave da API Gemini)](https://aistudio.google.com/apikey)
