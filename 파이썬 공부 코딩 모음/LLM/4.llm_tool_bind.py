from dotenv import load_dotenv
print(load_dotenv())
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import ToolCall
from langchain.tools import tool
from langchain_tavily import TavilySearch


llm=ChatOpenAI(model='gpt-4o-mini',temperature=0)
#사용자 정의툴 구현
@tool
def multiply(a:int,b:int)-> int:#->는 리텀 함수의 값
    '''a와 b를 곱해서 반환하는 함수'''#docstring
    return a*b
@tool
def add_oper(a:int,b:int)->int:
    '''a와 b를 더해서 반환하는 함수'''
    return a+b
tvsearch=TavilySearch(max_results=3)
#llm에 tool 집어넣기
llm_with_tools=llm.bind_tools([multiply,add_oper,tvsearch])

#툴 사용 안한 llm
result=llm.invoke([HumanMessage(content='손흥민 선수 최근 소속팀은?')])
print(result)
print('='*80)
#툴 사용한 llm
tool_result=llm_with_tools.invoke([HumanMessage(content='손흥민 선수 최근 소속팀은?')])

print(tool_result)

tool_calls=tool_result.tool_calls
if tool_calls: #tool_calls 의 내용이 있다면
    args=tool_calls[0]['args']
    print('='*80)
    if tool_calls[0]['name']=='multiply':
        tool_output=multiply.invoke(args)#tool로 정의함 함수의 실제동작
    elif tool_calls[0]['name']=='add_oper':
        tool_output=add_oper.invoke(args)
    elif tool_calls[0]['name']=='tavily_search':
        tool_output=tvsearch.invoke(args['query'])
    print(tool_output) #42
    tool_msg=ToolMessage(tool_call_id=tool_calls[0]['id'],content=str(tool_output))#tool 사용 결과를 ToolMessage/실제 동작한 메시지
    print(tool_msg)
    final_result=llm_with_tools.invoke([HumanMessage(content='손흥민 선수 최근 소속팀은?'),
                    tool_result,tool_msg])
    print(final_result.content)