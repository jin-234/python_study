#상태 데이터 설계시 필요한 모듈 추가
from mpmath import workdps
from typing import Annotated,TypedDict
#메세지를 누적추가하는 리듀서를 추가
from langgraph.graph.message import add_messages
import matplotlib.pyplot as plt
import networkx as nx
class ChatGraphState(TypedDict):
    messages:Annotated[list,add_messages]

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()



#각 노드 수행함수
def Node_A(state:ChatGraphState):
    #input 함수를 ㅇ활용해 메시지 ㅏㅎ나 입려
    nord_choice=input('Nord B 또는 Nord C 선택')
    #메시지는 ==> Nord A로 분기하고싶어 또는 Nord B로 분기하고싶어
    return {'messages':[nord_choice]}

#각 노드 수행함수
def Node_B(state:ChatGraphState):
    print('Node_B 수행')
    return {'messages':['오늘 점심은 뭔가요?']}
#각 노드 수행함수
def Node_C(state:ChatGraphState):
    print('Node_C 수행')
    return {'messages':['오늘 날씨는 어떤가요?']}

#분기노드 (Route Node) 설계
import random
from typing import Literal#[]형태의 시퀀스 객체
def Route_Node(state:ChatGraphState)->['Nord_B','Nord_C']:
    prompt=PromptTemplate(template='''
    당신은 라우터입니다
    사용자에게 받은 질문을 Nord B또는 Nord C를 입력 받은 뒤 하나만 선택하세요
    -Nord B
    -Nord C
    규칙
    1,반드시 'Nord B'또는 'Nord C'중 하나만 출력하세요
    2.다른 설명은 절대 출력하지마세요.
    3.이유를 설명하지 마세요.
    4.공백이나 줄바꿈 없이 정확히 하나만 출력하세요.
    사용자 질문:
    {question}
    ''',
                          input_variables=['question'])
    llm= ChatOpenAI(model='gpt-4o-mini', temperature=0)
    output_parser = StrOutputParser()
    user_msg = state['messages'][-1]
    print(user_msg.content)
    # LLM을 통해서 LLM의 수행 결과를
    # Node_B또는 Node_C로만 LLM수행 결과를 제한해서
    # 수행 결과를 가지고분기(return)

    chain = prompt | llm | output_parser
    result = chain.invoke({'question': 'Node C'})
    # 0.0< random.random()<1.0 값으로 반환
    if result == 'Nord B':
        return 'Node_B'  # Node_B로 조건 분기

    return "Node_C"  # Node C로 분기해라

#그래프 정의
from langgraph.graph import StateGraph,START,END

workflow=StateGraph(ChatGraphState)

#노드  정의
workflow.add_node('Node_A',Node_A)
workflow.add_node('Node_B',Node_B)
workflow.add_node('Node_C',Node_C)

#노드 연결
workflow.add_edge(START,'Node_A')
#source= 어떤 노드 수행 결과물을 가지고 조건 분기 할거냐?
# path==>어떤 함수가 조건분기를 결정할지 라우트 역할의 함수를 명시
# path_map=> 앞으로 어떤 노드로 분기할거야를 명시
workflow.add_conditional_edges(source='Node_A',path=Route_Node,path_map=['Node_B','Node_C'])

workflow.add_edge('Node_B',END)
workflow.add_edge('Node_C',END)
#구성 완료된 그래프 컴파일
chatapp=workflow.compile()# 두성된 그래프를 실행 할 수 있는 환경으로 설정
response=chatapp.invoke({'messages':[]})#그래프 실행 시작
for msg in response['messages']:
    msg.pretty_print()



#그래프 시각화 함수 정의
# def visualize_graph(graph: StateGraph):
#     G = nx.DiGraph()
#
#     # 노드와 엣지 추가
#     for node in graph.nodes:
#         G.add_node(node)
#     for edge in graph.edges:
#         G.add_edge(edge[0], edge[1])
#
#     # 그래프 시각화
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrowsize=20)
#     plt.title("StateGraph Visualization")
#     plt.show()

# 그래프 시각화 실행
# visualize_graph(chatapp)