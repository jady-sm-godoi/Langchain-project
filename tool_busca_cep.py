from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool
import requests
from pydantic import BaseModel, Field, field_validator
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

load_dotenv()

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

class cep_input(BaseModel): # classe para validação de entrada de dados com Pydantic
    cep: str = Field(..., description="O CEP a ser consultado no padrão brasileiro com 8 dígitos.")

    @field_validator("cep")
    @classmethod
    def validate_cep(cls, v: str) -> str:
        cep_limpo = v.replace("-", "").strip()
        if not cep_limpo.isdigit() or len(cep_limpo) != 8:
            raise ValueError("CEP inválido. Deve conter apenas números e ter 8 dígitos.")
        return cep_limpo


@tool(args_schema=cep_input)
def busca_cep(cep: str) -> str:
    """
    Busca o endereço correspondente a um CEP brasileiro(Código de Endereçamento Postal).
    Use quando o usuário perguntar sobre endereços ou CEPs.

    Args:
        cep (str): O CEP a ser consultado.

    returns:
        str: O endereço correspondente ao CEP, formatado como uma string.
    """

    url = f"https://viacep.com.br/ws/{cep}/json/"
    response = requests.get(url)
    data = response.json()
    return data



model = init_chat_model(
    model="gemini-3.5-flash-lite", model_provider="google_genai"
)

agente_cep = create_agent(
    model=model,
    system_prompt="Você é um assistente útil e prestativo. Responda de forma clara e concisa. Caso não saiba a resposta, solte um palavrão e diga que não sabe. Não invente respostas.",
    tools=[busca_cep],
    middleware=[tratar_erros]
)



"""camadas de proteção:

    - pydantic: validação de tipos e formatos de dados, garantindo que os dados recebidos e enviados estejam corretos.
    - validação de entrada: verifica se os dados fornecidos pelo usuário estão no formato esperado e dentro dos limites aceitáveis, evitando erros e comportamentos inesperados.
    - tratamento de erros: captura exceções e retorna mensagens de erro claras, evitando que o sistema quebre ou retorne respostas inesperadas.
    - logging: registra eventos e erros, permitindo a análise e depuração do sistema, além de fornecer informações sobre o comportamento do agente.
    - testes unitários: verificam o funcionamento correto das funções e ferramentas, garantindo que elas se comportem conforme o esperado e que mudanças futuras não quebrem funcionalidades existentes.
    - monitoramento: acompanha o desempenho e a saúde do sistema, permitindo a detecção precoce de problemas e a tomada de medidas corretivas antes que eles afetem os usuários finais.
    - documentação: fornece informações claras sobre o funcionamento do sistema, suas ferramentas e APIs, facilitando a compreensão e utilização por desenvolvedores e usuários finais.
    - controle de versão: mantém um histórico das alterações no código, permitindo reverter para versões anteriores em caso de problemas e facilitando a colaboração entre desenvolvedores.
    - segurança: implementa medidas de proteção contra ataques e vulnerabilidades, garantindo a integridade e confidencialidade dos dados e do sistema.
    - escalabilidade: projeta o sistema para lidar com aumento de carga e crescimento do número de usuários, garantindo que ele continue funcionando de forma eficiente à medida que a demanda aumenta.
    - feedback do usuário: coleta opiniões e sugestões dos usuários, permitindo melhorias contínuas no sistema e garantindo que ele atenda às necessidades e expectativas dos usuários finais.
    - integração contínua e entrega contínua (CI/CD): automatiza o processo de construção, teste e implantação do sistema, garantindo que as alterações sejam rapidamente disponibilizadas aos usuários finais e que o sistema esteja sempre atualizado e funcionando corretamente.
    - análise de métricas: monitora indicadores de desempenho e uso do sistema, permitindo identificar áreas de melhoria e otimizar a experiência do usuário.
    - testes de carga: avalia o desempenho do sistema sob diferentes condições de carga, garantindo que ele possa lidar com picos de tráfego e uso intenso sem falhas ou degradação significativa da experiência do usuário.
"""
