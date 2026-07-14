from langchain_classic.evaluation.criteria.prompt import template
from langgraph.graph import StateGraph,START,END
#START: 시작 노드,END:종료 노드
from typing import Annotated,TypedDict
from langgraph.graph.message import add_messages#발생한 메시지를 누적 관리==>리듀서
from langchain_core.messages import HumanMessage,AIMessage
import networkx as nx#그래프 시각화
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
load_dotenv()
#전역 그래프 상태 설계가능
class ChatGraphState(TypedDict):
    messages:Annotated[list,add_messages]#list 형태로 계속 누적 'massege':[Aimessega(content=''),HumanMessage(content='')]
prompt=PromptTemplate(
    template='''주어진 질문에 최대 3문장 이내로 대답하세요
답을 모르면 모른다고 대답하세요
질문:{message}''',
    input_variables=['message']
)
#특정 노드의 역할을 함수로 구현

def ChatbotNode(state:ChatGraphState):
    message=state.get('messages',[])#'messages'키가 있으면 해당 value를 반환, 없으면 []
    # print('ChatbotNode에 전달된 메시지:',message)
    #간단하게 llm을 활용해서 질문에 답을 하는 로직을 설계해주자
    #실제 llm을 동작시켜서 질문에 대한 답을 response로 저장해서 전달

    # llm=ChatOpenAI(model='gpt-4.0-mini',temperature=0)
    llm=OllamaLLM(model='gemma4',base_url='http://192.168.10.20:11434')
    outputparser=StrOutputParser()
    chain=prompt|llm|outputparser
    response=chain.invoke(message)

    # response='llm이 요청에 대해 응답한 내용입니다'
    #해당노드가 종료될때 리턴값을 자동으로 상태데이터를 업데이트 시키는 역할
    return {'messages':[AIMessage(content=response)]}#전달받은 상태에 새로운 메세지를 추가(누적) 시킬때 사용하는 문법
#전체 langgraph 에이전트 흐름 워크 플로우 설계
graph=StateGraph(ChatGraphState)#각 노드들을 추가하고 연결해주는 전체 틀 역할

#해당 전체 워크 플로우 그래프에 추가할 노드 설계
#첫번째 문자열 ==> 노드의 이름을 명시
#두번째=> 함수 명=> 와야하고 해당 노드가 동작할때 수행해야할 함수명
graph.add_node('chatbotnode',ChatbotNode)#특정 노트가 이런 개념(함수)을 수행해라고 명시
#수행햐야할 노드를 연결 해주는 역할
graph.add_edge(START,'chatbotnode')
graph.add_edge('chatbotnode',END)

simplechatapp=graph.compile()#연결된 그래프를 수행시킬 준비 완료


# init_state=ChatGraphState()
while True:

    query=input('질문의 내용을 입력(종료:exit')
    if query=='exit':
        break
    response=simplechatapp.invoke({'messages':[HumanMessage(content=query)]})
    # response=simplechatapp.invoke({})x`


    print('======ansewer=======')
    print(response)
# query=input('질문의 내용을 입력')
# response=simplechatapp.invoke({'messages':[HumanMessage(content=query)]})
# # response=simplechatapp.invoke({})
#
#
# print('======ansewer=======')
# print(response)
#
# query=input('질문의 내용을 입력')
# response=simplechatapp.invoke({'messages':[HumanMessage(content=query)]})
# # response=simplechatapp.invoke({})
#
#
# print('======ansewer=======')
# print(response)
#
# query=input('질문의 내용을 입력')
# response=simplechatapp.invoke({'messages':[HumanMessage(content=query)]})
# # response=simplechatapp.invoke({})
#
#
# print('======ansewer=======')
# print(response)


#그래프 시각화 함수 정의
def visualize_graph(graph: StateGraph):
    G = nx.DiGraph()

    # 노드와 엣지 추가
    for node in graph.nodes:
        G.add_node(node)
    for edge in graph.edges:
        G.add_edge(edge[0], edge[1])

    # 그래프 시각화
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrowsize=20)
    plt.title("StateGraph Visualization")
    plt.show()

# 그래프 시각화 실행
# visualize_graph(graph)
