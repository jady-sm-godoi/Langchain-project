# Comparativo de RAG: LangChain vs. Agno

No **LangChain** (com LangGraph), o RAG é tratado como um **pipeline modular e um grafo de controle de estado**, onde você molda e decide cada etapa manualmente. No **Agno**, o RAG é tratado como um **módulo de conhecimento (Knowledge Base) acoplado diretamente ao Agente**, abstraindo a complexidade do pipeline em poucas linhas de código.

---

## 1. RAG no LangChain / LangGraph (Abordagem Modular e Baseada em Estados)

No ecossistema LangChain, o RAG clássico segue a arquitetura **LCEL (LangChain Expression Language)** ou, para casos avançados, o **LangGraph** (RAG corretivo/adaptativo). Você precisa configurar manualmente cada peça do quebra-cabeça.

```
[Documento] ➔ [Document Loader] ➔ [Text Splitter] ➔ [Embeddings] ➔ [Vector Store]
                                                                                    ⬇
[Usuário] ➔ [Prompt Template] ➔ [LLM] ➔ [Output Parser] ⬅ [Retriever]
```

### Como Funciona Passo a Passo:
1. **Carregamento e Divisão:** Você usa um `DocumentLoader` específico (PDF, Web, Notion) e define manualmente um `TextSplitter` (ex: `RecursiveCharacterTextSplitter`) configurando o tamanho do chunk e o overlap.
2. **Vetorização e Armazenamento:** Você escolhe o modelo de `Embeddings` e o banco vetorial (`VectorStore`), enviando os blocos de texto explicitamente.
3. **Recuperação (Retriever):** Transforma-se o banco vetorial em um `Retriever`. Aqui você pode aplicar técnicas avançadas como *Multi-Query Retriever*, *Contextual Compression*, ou *Re-ranking* (via Cohere, por exemplo).
4. **No LangGraph (RAG Avançado):** O processo vira um grafo com nós (*nodes*) e arestas condicionais (*conditional edges*). 
   * **Nó 1 (Retrieve):** Busca os documentos.
   * **Nó 2 (Grade Documents):** Uma LLM avalia se os documentos são relevantes.
   * **Aresta Condicional:** Se forem relevantes, vai para o **Nó de Geração**. Se não forem, vai para o **Nó de Busca na Web** para complementar o contexto antes de responder.

---

## 2. RAG no Agno (Abordagem Declarativa e Nativa do Agente)

No Agno, o RAG é integrado ao ciclo de vida do agente por meio do conceito de **Knowledge Base** (Base de Conhecimento). O framework automatiza a ingestão e a busca por baixo dos panos.

```markdown
[Documento/URL] ➔ [Agno Knowledge Base] (Auto-split + Auto-embed) ➔ [Vector Database]
                                                                        ⬇
[Usuário] ➔ [Agno Agent (com Knowledge Base Ativada)] ➔ [Resposta com Contexto]
```

### Como Funciona Passo a Passo:
1. **Definição da Base:** Você instancia um objeto de conhecimento (ex: `PDFKnowledgeBase` ou `WebsiteKnowledgeBase`). Você passa o caminho dos arquivos e o banco vetorial desejado (ex: pgvector, Pinecone).
2. **Ingestão Simplificada:** O Agno possui leitores e splitters padrão integrados. Com um comando como `knowledge_base.load(recreate=True)`, ele lê, divide em chunks, gera os embeddings e salva no banco de dados automaticamente.
3. **Associação ao Agente:** Você passa essa `knowledge_base` diretamente como um argumento na criação do agente (`Agent(knowledge_base=knowledge_base)`).
4. **Execução:** Ao fazer uma pergunta ao agente, ele executa autonomamente a busca vetorial por trás das cortinas, injeta o contexto recuperado no prompt do sistema e entrega a resposta final formatada. Você não precisa desenhar a cadeia de busca.

---

## Comparação Direta de Código (Conceitual)

### Exemplo no Agno (Foco em Velocidade)
```python
from agno.agent import Agent
from agno.knowledge.pdf import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector

# 1. Configura a base de conhecimento e o banco vetorial
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://exemplo.com/manual.pdf"],
    vector_db=PgVector(table_name="manual_rules", db_url="postgresql+psycopg://...")
)
knowledge_base.load(recreate=True) # Ingestão automática

# 2. Conecta a base diretamente ao agente
agent = Agent(
    knowledge_base=knowledge_base,
    search_knowledge=True, # Ativa o RAG automático nas interações
    markdown=True
)

agent.print_response("O que o manual diz sobre reembolsos?")
```

### Exemplo no LangChain (Foco em Controle)
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. Pipeline manual de ingestão
loader = PyPDFLoader("manual.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Configura banco vetorial e retriever
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Desenha a cadeia de execução (LCEL)
prompt = ChatPromptTemplate.from_template("Responda com base no contexto:

{context}

Pergunta: {question}")
llm = ChatOpenAI(model="gpt-4o")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

response = rag_chain.invoke("O que o manual diz sobre reembolsos?")
```

---

## Resumo da Escolha para RAG

* Escolha o **Agno** se você quer que o seu agente tenha acesso a documentos PDF, sites ou tabelas de forma rápida, sem perder tempo configurando encadeamentos manuais ou gerenciando o fluxo de texto.
* Escolha o **LangChain/LangGraph** se o seu RAG falha com buscas simples e você precisa criar um fluxo dinâmico (ex: testar se o documento recuperado presta, reescrever a pergunta se a busca falhar ou aplicar filtros avançados de metadados baseados no tempo).
