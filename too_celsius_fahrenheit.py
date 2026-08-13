from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

"""
contexto da função para AI:
    - doc-string: logo abaixo do nome da função, descrevendo o que a função faz, seus parâmetros e o que ela retorna
    - type hinting
    - nome da função

Tool boa tem:
    - um contrato bom: nome e descrição claros, doc-string, type hinting
    - responsabilidade única: faz uma coisa só, e faz bem feito
    - tratamento de erros: lida com entradas inválidas e retorna mensagens de erro claras

"""
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

model = init_chat_model(
    model="gemini-3.5-flash-lite", model_provider="google_genai"
)

agente_clima = create_agent(
    model=model,
    system_prompt="Você é um assistente útil e prestativo. Responda de forma clara e concisa. Caso não saiba a resposta, solte um palavrão e diga que não sabe. Não invente respostas.",
    tools=[converter_temperatura],
)