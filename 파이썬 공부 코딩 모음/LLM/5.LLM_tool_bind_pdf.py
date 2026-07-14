from dotenv import load_dotenv
print(load_dotenv())
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import ToolCall
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders.pdf import PyPDFLoader
import os
from langchain_chroma import Chroma

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings



llm=ChatOpenAI(model='gpt-4o-mini',temperature=0)
#사용자 정의툴 구현
@tool
def pdf_load_process(filepath:str,userquery:str)-> str:#->는 리텀 함수의 값
    '''filepath로 제공한 PDF파일의 내용을 찾아서 반환하는 함수'''#docstring
    print(filepath)
    print(userquery)
    #filepath로 제공해준 pdf파일을 로딩하고
    pdf_loader=PyPDFLoader('C:/python_project/sessac_agent_1/sessac_AIagent/20260714/yolov1_paper.pdf')
    textdocs=pdf_loader.load()
    #파일 분할
    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
    )
    texts=text_splitter.split_documents(textdocs)
     #임베딩
    PERSIST_DIR = "./chroma_db"

    # 기존 DB 폴더 삭제
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print("기존 chroma_db 삭제 완료")

    # 새로 벡터 저장
    vector_store = Chroma.from_documents(
        documents=texts,#위,에서 분할한 doc list를 전달
        embedding=OpenAIEmbeddings(),#open AI embedding 모델을 활용해서 위 dogument를 활용해서 전달한 doc를 지정한 vectordv에 지정한다
        persist_directory=PERSIST_DIR
    )

    print("새로운 chroma_db 생성(적재) 완료")
#   쿼리를 이용해서 해당 관련 내용을 리트리버로 찾아와서 전달(return)
    retriever=vector_store.as_retriever(
    search_kwargs={'k':2}#관련성이 높은 상위 k를 가져와

    )

    return_doc=retriever.invoke(userquery)
    return return_doc
#llm에 tool 집어넣기
llm_with_tools=llm.bind_tools([pdf_load_process])


#툴 사용한 llm
tool_result=llm_with_tools.invoke([HumanMessage(content='yolov1_paper.pdf 파일에서 네트워크 구조는?')])

print(tool_result)

tool_calls=tool_result.tool_calls
if tool_calls: #tool_calls 의 내용이 있다면
    args=tool_calls[0]['args']
    print('='*80)
    if tool_calls[0]['name']=='pdf_load_process':
        tool_output=pdf_load_process.invoke(args)#tool로 정의함 함수의 실제동작

    print(tool_output) #42
    tool_msg=ToolMessage(tool_call_id=tool_calls[0]['id'],content=str(tool_output))#tool 사용 결과를 ToolMessage/실제 동작한 메시지
    print(tool_msg)
    print('final_result : ')
    final_result=llm_with_tools.invoke([HumanMessage(content='yolov1_paper.pdf 파일에서 네트워크 구조는?'),
                    tool_result,tool_msg])
    print(final_result.content)