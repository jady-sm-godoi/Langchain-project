"""
Agente de banco de dados com tools nativas (v2).

Versão migrada do agente_banco.py (v1), que usava o SQLDatabaseToolkit do
langchain-community (aposentado em maio de 2026). Aqui as quatro ferramentas
são implementadas manualmente com @tool + sqlite3, sem depender do pacote
deprecado:

- sql_db_list_tables    -> Lista as tabelas do banco de dados.
- sql_db_schema         -> Retorna o esquema (tabelas, colunas e amostras) do banco.
- sql_db_query          -> Executa uma consulta SQL e retorna o resultado.
- sql_db_query_checker  -> Valida a consulta SQL e corrige erros comuns antes de executar.
"""
import os
import sqlite3

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

model = init_chat_model(
    model="gemini-3.5-flash-lite", model_provider="google_genai", temperature=0
)  # temperature=0 para respostas mais determinísticas, sem variação

DB_PATH = os.path.abspath("loja.sqlite")


def conectar() -> sqlite3.Connection:
    """Abre uma conexão nova com o banco. Fechada pela tool que a chamou."""
    return sqlite3.connect(DB_PATH)


def listar_tabelas(conn: sqlite3.Connection) -> list[str]:
    """Retorna os nomes das tabelas reais do banco (ignora as internas do sqlite)."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]


@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the database."""
    conn = conectar()
    try:
        return ", ".join(listar_tabelas(conn))
    finally:
        conn.close()


@tool
def sql_db_schema(table_names: str) -> str:
    """Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3"""
    conn = conectar()
    try:
        cursor = conn.cursor()
        tabelas_validas = set(listar_tabelas(conn))

        resultados = []
        for nome in table_names.split(","):
            nome = nome.strip()
            if nome not in tabelas_validas:
                resultados.append(f"Error: table_names {{{nome!r}}} not found in database")
                continue

            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (nome,)
            )
            linha = cursor.fetchone()
            if linha:
                resultados.append(linha[0])

                try:
                    nome_entre_aspas = '"' + nome.replace('"', '""') + '"'
                    cursor.execute(f"SELECT * FROM {nome_entre_aspas} LIMIT 3;")
                    linhas = cursor.fetchall()
                    if linhas:
                        colunas = [descricao[0] for descricao in cursor.description]
                        amostras = "\n".join(
                            "\t".join(str(valor) for valor in linha) for linha in linhas
                        )
                        resultados.append(
                            f"/*\n3 rows from {nome} table:\n"
                            + "\t".join(colunas)
                            + "\n"
                            + amostras
                            + "\n*/"
                        )
                except Exception as e:
                    resultados.append(f"Error fetching sample rows: {e}")

        return "\n\n".join(resultados)
    finally:
        conn.close()


@tool
def sql_db_query(query: str) -> str:
    """Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields."""
    conn = conectar()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        resultado = cursor.fetchall()
        return str(resultado)
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()


@tool
def sql_db_query_checker(query: str) -> str:
    """Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query!"""
    trigger_prompt = f"""{query}
Double check the sqlite query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """

    response = model.invoke(trigger_prompt)
    return response.text.strip()


tools = [sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker]

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