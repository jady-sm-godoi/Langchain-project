# RAG no LangChain — Teoria por trás do `rag_pdf.ipynb`

Este documento explica a **lógica que fica invisível** no notebook `rag_pdf.ipynb`: o que cada método do RAG faz por baixo dos panos, por que ele existe e como as peças se conectam. É a teoria que sustenta o código.

O notebook real: `rag_pdf.ipynb` — carrega um livro de Flutter em PDF, divide em chunks, vetoriza com embeddings, persiste num vector store Chroma e gera respostas com o Gemini usando apenas o contexto recuperado.

---

## 1. O que é RAG e por que existe

**RAG** (*Retrieval-Augmented Generation* — Geração Aumentada por Recuperação) é o padrão de **buscar documentos relevantes** em um corpus próprio **antes** de pedir a resposta ao modelo.

Um LLM sozinho tem três limitações estruturais:

| Limitação | Consequência |
| --- | --- |
| **Conhecimento congelado** | Sabe só o que foi treinado (corte de data); não conhece eventos novos. |
| **Dados privados** | Não viu seus documentos, seu banco, sua base interna. |
| **Alucinação** | Quando não sabe, *inventa* — e inventa com confiança. |

O RAG ataca as três de uma vez: em vez de o modelo responder "do nada", ele recebe **trechos reais do seu corpus** como contexto e responde **fundamentado neles** (grounding).

> **Ideia central:** o RAG não muda o modelo. Ele muda o **prompt**. O modelo continua o mesmo; o que muda é que a pergunta agora chega acompanhada de evidências recuperadas da sua base.

---

## 2. As duas fases do RAG (a lógica invisível)

O notebook esconde que o RAG tem **dois momentos distintos**:

### Fase A — Indexação (roda uma vez, offline)

```
[PDF] → [Loader] → [Splitter] → [Embeddings] → [Vector Store]
```

É a fase de **preparar a base**: transformar o documento bruto em vetores armazenados. No notebook isso acontece nas células que criam o `chroma_db/`. O nome técnico é *ingestão* ou *indexação*.

### Fase B — Consulta (roda a cada pergunta, online)

```
[Pergunta] → [Embeddings da pergunta] → [Busca por similaridade] → [Contexto] → [LLM] → [Resposta]
```

É a fase de **responder**: a pergunta é convertida em vetor, busca-se no vector store os trechos mais parecidos, junta-se tudo como contexto e o LLM gera a resposta. No notebook, essa é a célula com `retriver.invoke(pergunta)` e `llm.invoke(...)`.

> **Por que essa separação importa:** a indexação é **cara** (embedda milhares de chunks, uma chamada de API por texto). A consulta é **barata** (uma chamada para a pergunta + uma para o LLM). É por isso que o notebook **persiste** o vector store — para pagar o custo de indexação uma vez só.

---

## 3. As peças, o que fazem por baixo dos panos

### 3.1 `PyPDFLoader(...).load()` — carregando o documento

```python
documents = PyPDFLoader("arquivos/Flutter_for_Beginners_by_Alessandro_Biessek_(z-lib.org).pdf").load()
# Documento carregado. Total de documents: 498
```

**O que parece:** "lê o PDF".

**O que acontece de verdade:** extrair texto de um PDF **não é ler texto**. O PDF armazena conteúdo como **objetos** (fontes, glifos, coordenadas de página, imagens, streams comprimidos) — o texto está "desenhado" na página, não guardado como um arquivo `.txt`. O `PyPDFLoader` (da `langchain-community`, que usa a biblioteca `pypdf`) **percorre cada página, descompacta os streams, extrai e reordena os caracteres** para reconstruir o texto legível.

**O que ele devolve:** uma **lista de `Document`** — 498 Documents, **um por página**. Um `Document` do LangChain é um pacote com dois campos:

- `page_content` — o texto extraído;
- `metadata` — dados sobre a origem (nº da página, fonte, etc.). Invisível no notebook, mas útil para citar a fonte depois.

> **Por que usar um loader dedicado em vez de `open().read()`?** Cada formato exige extração diferente (PDF, web, Notion, banco...). O loader isola essa complexidade e entrega sempre o mesmo tipo de objeto: `Document`.

### 3.2 Outros tipos de arquivo — a família de loaders

O `PyPDFLoader` é só um entre dezenas. No LangChain, **cada formato tem um loader**, mas todos seguem o mesmo contrato: `load()` → `list[Document]`.

**Por que não usar `open().read()`?** `open().read()` devolve uma string bruta — e, na maioria dos formatos reais, o texto **não existe como string legível no arquivo**:

| Formato | O que `open().read()` devolve | Por quê |
| --- | --- | --- |
| PDF | bytes binários inúteis | texto "desenhado" com glifos, dentro de streams comprimidos |
| DOCX | XML cheio de tags | texto misturado com instruções de formatação (`<w:p>...`) |
| HTML | tags, scripts, CSS | conteúdo útil embutido no markup |
| XLSX/CSV | linhas cruas sem estrutura | texto espalhado em linhas/colunas/planilhas |
| Notion/web/banco | nada — não é arquivo | o conteúdo mora numa **API ou banco**, não em disco |

Mesmo quando a string sai legível (`.txt`), ela vem **sem contexto** — sem página, sem fonte, sem metadados.

**O que o loader faz:** extrai o texto do formato específico, **limpa** o que não interessa (tags, scripts, estilos) e **empacota** tudo em `Document(page_content, metadata)` — enriquecendo os metadados do seu jeito (página no PDF, linha no CSV, URL/título na web). Como o restante do pipeline (splitter → embeddings → Chroma → retriever) **só conhece `Document`**, trocar a fonte é trocar **apenas o loader** — todo o código depois continua idêntico.

**Exemplos de loaders por formato:**

```python
from langchain_community.document_loaders import (
    TextLoader,                      # .txt / .md
    Docx2txtLoader,                  # .docx (Word)
    UnstructuredExcelLoader,         # .xlsx (Excel)
    CSVLoader,                       # .csv (1 Document por linha)
    UnstructuredPowerPointLoader,    # .pptx
    WebBaseLoader,                   # URL: baixa a página e remove tags
    NotionDirectoryLoader,           # Notion via API
    SQLDatabaseLoader,               # banco: query → linhas viram Documents
    UnstructuredFileLoader,          # auto-detecta o tipo
)

docs = Docx2txtLoader("manual.docx").load()
docs = UnstructuredExcelLoader("planilha.xlsx").load()
docs = WebBaseLoader("https://exemplo.com/artigo").load()
docs = TextLoader("notas.txt").load()
```

No `rag_pdf.ipynb`, trocar de PDF para DOCX seria só a primeira linha e o caminho do arquivo — do splitter em diante, nada muda:

```python
# de
documents = PyPDFLoader("arquivos/...pdf").load()
# para
documents = Docx2txtLoader("arquivos/...docx").load()
```

> **Duas observações:**
> - Muitos loaders vêm do ecossistema **Unstructured** (parsers comuns para ~20 formatos) — o `UnstructuredFileLoader` é o "coringa" que detecta o tipo sozinho.
> - O `langchain-community` foi aposentado em maio/2026; os loaders estão migrando para pacotes dedicados (`langchain-unstructured` etc.). O padrão `load()` → `Document` permanece, então a migração é quase transparente.

### 3.3 `RecursiveCharacterTextSplitter` — dividindo em chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
docs = text_splitter.split_documents(documents)
# total de chunks criados: 972
```

**O que parece:** "divide o texto em pedaços de 1000 caracteres".

**Por que dividir?** Três motivos:

1. **Janela de contexto:** o LLM aceita um limite de tokens por chamada. Um livro inteiro não cabe.
2. **Custo:** cada chunk vira um embedding — uma chamada de API paga. Menos chunks = menos custo.
3. **Granularidade de recuperação:** na busca, queremos devolver o trecho *relevante* (ex.: "a página que fala de widgets"), não o livro inteiro. Chunks pequenos = precisão maior.

**O que acontece de verdade — a parte "recursive":** o splitter não corta cegamente no caractere 1000. Ele tenta quebrar o texto por **separadores em ordem hierárquica**:

```
parágrafo (\n\n) → linha (\n) → espaço (" ") → caractere ("")
```

Ele primeiro tenta dividir por parágrafos inteiros; se um parágrafo ainda passa de `chunk_size`, tenta por linhas; depois por frases (espaços); e só por último corta no meio de um caractere. O resultado: **os cortes caem em fronteiras naturais** (fim de parágrafo/frase), preservando a coesão do texto — e 972 chunks saem de 498 páginas.

**O papel do `chunk_overlap`:** a borda entre dois chunks é uma zona cega — uma ideia que começa no fim do chunk 1 e termina no começo do chunk 2 pode se perder. O overlap de 200 caracteres **repete o fim de cada chunk no começo do próximo**, costurando a fronteira e mantendo o contexto contínuo.

> **Trade-off (a intuição):** `chunk_size` pequeno = recuperação precisa mas contexto cortado no meio de ideias; `chunk_size` grande = contexto completo mas busca imprecisa e custo maior. O overlap equilibra as bordas, mas **aumenta o total de tokens indexados** (texto repetido). Não existe valor "certo" — é um parâmetro a calibrar (ver seção 5).

### 3.4 `OpenAIEmbeddings(model="text-embedding-3-large")` — vetorizando

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

**O que parece:** "prepara o modelo que vai guardar o texto".

**O que acontece de verdade:** embeddings convertem **texto em vetor de números** — no caso do `text-embedding-3-large`, um vetor de **3072 dimensões**. Cada texto vira um ponto em um espaço de 3072 dimensões, e a mágica é que:

- textos com **significado parecido** ficam **perto** nesse espaço;
- textos com **significado diferente** ficam **longe**.

Isso vale mesmo com palavras diferentes: "dias de férias" e "período de descanso anual" ficam próximos, porque o modelo aprendeu a *semântica*, não as palavras exatas.

**O que fica invisível:** quando o `Chroma.from_documents` roda, ele chama a API da OpenAI **uma vez por chunk** — uma requisição HTTP com o texto, que volta como uma lista de números. **O modelo LLM não entende texto; ele só entende números.** O embedding é a ponte que transforma linguagem em matemática para a busca funcionar.

> **Por que não buscar por palavras exatas?** Porque uma busca por texto bruto (keyword) exige as palavras exatas da pergunta. A busca **vetorial** entende *intenção e significado* — a pergunta e o documento não precisam compartilhar uma única palavra.

### 3.5 `Chroma` / `persist_directory` — o vector store

```python
persist_directory = "./chroma_db"

if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
```

**O que parece:** "cria um banco para guardar os vetores".

**O que acontece de verdade:** um **vector store** guarda, para cada chunk: o **vetor** (números), o **texto original** (`page_content`) e os **metadados**. O Chroma persiste isso **em disco** em `chroma_db/` (internamente um banco SQLite — o `chroma.sqlite3` — mais os diretórios de coleção).

**O ponto técnico invisível — busca aproximada:** com milhares de vetores, comparar a pergunta com todos (busca exaustiva) fica lento. O Chroma constrói um **índice de vizinhança aproximada (ANN, *Approximate Nearest Neighbor*)** — uma estrutura que agrupa vetores parecidos em "vizinhanças", permitindo achar os mais próximos em tempo sublinear, com uma pequena perda de exatidão em troca de velocidade. É isso que torna o RAG prático em bases grandes.

**O padrão `if os.path.exists(...)` — o que ele resolve:** o notebook quer **não re-embeddar** os 972 chunks toda vez. A lógica é *idempotente*:

- `Chroma.from_documents(...)` → **cria** a base e embedda tudo (primeira execução);
- `Chroma(persist_directory=...)` → **carrega** a base já existente (execuções seguintes).

Como os embeddings são **pagos por chamada**, essa checagem evita gastar novamente o custo de 972 chamadas. O mesmo padrão se aplica a qualquer vector store persistente (Chroma, Qdrant, pgvector, Pinecone...).

> **Memória vs disco:** o `exemplo_rag.ipynb` usa o `InMemoryVectorStore` — tudo na RAM, **perdido ao reiniciar o kernel**, ideal para testes. O Chroma **persiste em arquivo**, sobrevivendo a reinícios. Numa aplicação real, o padrão é banco vetorial persistente (ver seção 4).

### 3.6 `as_retriever(search_kwargs={"k": 2})` — a busca

```python
retriver = vector_store.as_retriever(search_kwargs={"k": 2})
```

**O que parece:** "converte a base em um buscador".

**O que acontece de verdade:** o `retriever` é uma **interface** que padroniza a busca: você dá uma pergunta e recebe `Document`s. Por baixo, o `invoke` do retriever:

1. **embedda a pergunta** com o **mesmo** modelo usado na indexação (os vetores só são comparáveis se vierem do mesmo espaço);
2. calcula a **similaridade** da pergunta contra todos os vetores da base (via cosseno ou outra métrica de distância);
3. **ordena** do mais parecido ao menos parecido;
4. devolve os **k melhores** — aqui, `k=2` → os 2 chunks mais relevantes.

**Por que `k` importa:** contexto pequeno demais (`k=1`) pode perder informação; grande demais (`k=5+`) enche a janela de contexto com ruído e aumenta o custo. `k=2` é um bom ponto de partida para um livro.

> **A interface esconde a estratégia:** um retriever pode ser mais que "top-k por similaridade" — *Multi-Query Retriever* (reescreve a pergunta em várias variações), *Contextual Compression* (filtra/compacta o que recuperou), *reranking* (reordena com um segundo modelo). Tudo isso implementa a **mesma interface** `as_retriever`.

### 3.7 A geração — grounding vs alucinação

```python
pergunta = input("Sou o oráculo do Flutter. Faça sua pergunta: ")
similar_docs = retriver.invoke(pergunta)
contexto = "\n\n".join([doc.page_content for doc in similar_docs])
resposta = llm.invoke(f"com base no seguinte contexto \n\n {contexto} \n\n responda a seguinte pergunta \n\n {pergunta}")
```

**O que parece:** "o LLM responde".

**O que acontece de verdade — a peça central do RAG:** o modelo recebe um prompt com **três blocos**: o contexto recuperado, a pergunta e uma instrução. Ele não responde "sabendo a resposta" — responde **usando o contexto como fonte**. Isso é o **grounding** (fundamentação): a resposta é ancorada em evidências reais, não na memória do modelo.

**A demonstração real do notebook:** quando a pergunta pede algo que **não está** no livro ("qual a versão do Flutter?"), o modelo responde:

> "Com base no contexto fornecido, **não é mencionada a versão específica do Flutter**... o texto apenas apresenta a introdução ao framework..."

Esse comportamento é o **oposto da alucinação**: em vez de inventar um número, o modelo **reconhece a ausência** da informação. É o sinal de que o RAG está funcionando — a resposta é limitada pelo contexto.

> **Ato invisível:** o RAG **não garante** zero alucinação — o modelo pode ainda ignorar o contexto e inventar. Ele **reduz drasticamente** a chance, porque a evidência está ali. Garantia real exige camadas extras (avaliar se o contexto recuperado é relevante antes de gerar, filtrar, etc. — ver seção 6).

**Detalhe de formato:** `resposta.content[0]["text"]` — a resposta do Gemini via `langchain-google-genai` vem em estrutura de conteúdo (lista de blocos com texto), não como string simples. É por isso que o notebook acessa `[0]["text"]` para imprimir a resposta.

---

## 4. Comparativo dos dois notebooks de RAG

| | `exemplo_rag.ipynb` | `rag_pdf.ipynb` |
| --- | --- | --- |
| **Corpus** | 4 fatos fictícios de RH | Livro real de Flutter (24 MB, 498 páginas) |
| **Loader** | Nenhum (Documentos escritos à mão) | `PyPDFLoader` |
| **Splitter** | Nenhum | `RecursiveCharacterTextSplitter` (1000/200) |
| **Vector store** | `InMemoryVectorStore` (RAM) | `Chroma` persistente (`chroma_db/`) |
| **Sobrevive ao reinício?** | ❌ perde tudo | ✅ persiste em disco |
| **Custo de indexação** | Baixo (4 documentos) | Alto (972 chamadas de embedding) |
| **Geração** | ❌ só recuperação | ✅ LLM gera resposta com o contexto |
| **Escopo** | Entender retrieval (a busca) | RAG completo de ponta a ponta |

O primeiro notebook isola o **retrieval** para estudo; o segundo fecha o ciclo completo. Juntos cobrem as duas fases do RAG (seção 2).

---

## 5. Trade-offs de chunking

O `chunk_size` e o `chunk_overlap` são os parâmetros mais sensíveis de um RAG — mudam a qualidade da resposta sem você tocar no modelo.

| Escolha | Ganha | Perde |
| --- | --- | --- |
| **Chunk pequeno** (~200–500) | Precisão da busca (trecho cirúrgico) | Contexto cortado no meio de uma ideia |
| **Chunk grande** (~1000–2000) | Contexto completo e coeso | Busca imprecisa (muito ruído por chunk), custo maior |
| **Overlap alto** (~200–400) | Fronteiras costuradas (ideias não se perdem) | Texto repetido = mais tokens indexados, mais custo |

**Intuições práticas:**
- Conteúdo denso (código, termos técnicos) tolera chunks menores; conteúdo narrativo pede chunks maiores.
- O overlap deve ser maior que a maior "unidade de ideia" que você não quer cortar — 200 é um ponto de partida razoável para texto técnico.
- A regra de ouro: **chunk pequeno o suficiente para ser específico, grande o suficiente para fazer sentido sozinho**.

**Alternativas de splitter (só a intuição):**
- **Character splitter** — corta por contagem bruta de caracteres; simples, mas quebra palavras/ideias no meio.
- **Recursive (o usado aqui)** — tenta separadores em hierarquia (parágrafo → linha → espaço → caractere); preserva coesão, bom padrão geral.
- **Semantic splitter** — usa embeddings para dividir **onde o assunto muda**; resultado melhor, custo maior (chama o modelo durante a divisão).
- **Agentic splitting** — usa um LLM para decidir os cortes; mais inteligente, mais lento e mais caro.

---

## 6. Próximos passos do RAG

O notebook faz o RAG **"vanilla"** (simples): recupera top-k, joga no prompt, gera. As evoluções conhecidas:

| Técnica | Problema que resolve |
| --- | --- |
| **Reranking** | O top-k por similaridade nem sempre é o mais relevante; um segundo modelo reordena os candidatos. |
| **Multi-Query Retriever** | Uma pergunta mal formulada gera uma busca ruim; o retriever gera variações da pergunta e busca todas. |
| **Contextual Compression** | Chunks recuperados podem trazer ruído; comprime/filtra o que foi buscado antes de gerar. |
| **RAG corretivo (Self-RAG / CRAG)** | E se o contexto recuperado **não for relevante**? O grafo vira a lógica: recupera → **avalia** com o LLM → se ruim, busca na web ou refaz a pergunta. |
| **RAG adaptativo (LangGraph)** | Transforma o RAG em **grafo de estados**: nós de retrieve/grade/generate e arestas condicionais decidindo o fluxo dinamicamente. |

> O [RAG_Agno_versus_Langchain.md](RAG_Agno_versus_Langchain.md) já discute a ponte do RAG simples para o RAG como **grafo do LangGraph** com correção/recuperação adaptativa, e compara a abordagem modular do LangChain com a declarativa do Agno.

---

## 7. Glossário

| Método/Conceito | O que faz | Por que existe |
| --- | --- | --- |
| `PyPDFLoader().load()` | Extrai o texto de cada página do PDF e devolve uma lista de `Document` (498) | Traduz um formato binário complexo em objetos de texto padronizados |
| `RecursiveCharacterTextSplitter` | Divide os documentos em chunks (972) com `chunk_size` e `chunk_overlap` | Respeita a janela de contexto, o custo e a granularidade da busca |
| `OpenAIEmbeddings` | Converte cada chunk em um vetor de 3072 dimensões | Transforma texto em números para a busca por similaridade semântica |
| `Chroma.from_documents` | Cria o vector store e embedda os chunks em disco | Persiste a indexação para não repetir o custo |
| `Chroma(persist_directory=...)` | Carrega o vector store já existente | Padrão idempotente: reutiliza o que já foi indexado |
| `as_retriever(search_kwargs={"k": 2})` | Cria a interface de busca que devolve os k chunks mais similares | Esconde a estratégia de recuperação atrás de uma API comum |
| `retriever.invoke(pergunta)` | Embedda a pergunta, ordena por similaridade e devolve os documentos | Converte a pergunta em números e compara com a base |
| `llm.invoke(contexto + pergunta)` | Gera a resposta fundamentada no contexto | É o coração do RAG: responde com evidência, não de memória |
| **Embedding** | Vetor numérico que captura o significado do texto | É a "língua" que o modelo e a busca compartilham |
| **Vector store** | Banco que guarda vetores + textos + metadados | É a memória do RAG: onde a base indexada mora |
| **Similaridade (cosseno)** | Medida de proximidade entre dois vetores | É o critério que decide "o que é relevante" |
| **Grounding** | Resposta ancorada no contexto fornecido | Reduz alucinação: o modelo responde com fonte |