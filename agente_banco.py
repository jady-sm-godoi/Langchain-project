"""
Ferramentas do SQLDatabaseToolkit:
- sql_db_list_tables    -> Lista as tabelas do banco de dados.
- sql_db_schema         -> Retorna o esquema (tabelas, colunas e amostras) do banco.
- sql_db_query          -> Executa uma consulta SQL e retorna o resultado.
- sql_db_query_checker  -> Valida a consulta SQL e corrige erros comuns antes de executar.
"""
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

load_dotenv()

model = init_chat_model(
    model="gemini-3.5-flash-lite", model_provider="google_genai", temperature=0
) # temperature=0 para respostas mais determinísticas, sem variação

DB_PATH = os.path.abspath("loja.sqlite")
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

system_prompt = """

Regras obrigatórias:
- Sempre inspecione as tabelas disponíveis e o esquema ANTES de gerar qualquer query.
- Limite resultados a no máximo 10 registros (use LIMIT 10).
- NUNCA execute comandos DML (INSERT, UPDATE, DELETE, DROP). Somente SELECT.
- Se não encontrar a informação, diga que os dados não foram encontrados.
- Responda sempre em português.

"""

agente_banco = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=tools
)
