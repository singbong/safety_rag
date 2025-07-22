import vertexai
from google.oauth2 import service_account
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from typing_extensions import TypedDict, Annotated
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.messages import random_uuid
from langchain_core.runnables import RunnableConfig
from langchain.memory import ConversationSummaryMemory
from langchain.schema import Document
from langchain_core.runnables import Runnable, RunnableLambda
from vertexai.generative_models import GenerativeModel, GenerationConfig
from langchain_google_community.vertex_check_grounding import VertexAICheckGroundingWrapper
from datetime import datetime
from typing import Dict, List, Optional
import httpx
from pathlib import Path
import hashlib
import sys
import uuid
import json
from tqdm import tqdm
import re
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from vector.definition.vectore_store import store_vector_db
from pydantic import BaseModel



os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
project_id = os.getenv("PROJECT_ID")
google_api_key = os.getenv("GOOGLE_API_KEY")
credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_file(credential_path)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["PROJECT_ID"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "asia-northeast1"


vector_store = store_vector_db()
print(f"vector_store: {vector_store}")
vector_store.load_vector_store("./vector/data/vector_store/")

# 세션별 메모리 관리를 위한 딕셔너리
lc_memories: Dict[str, ConversationSummaryMemory] = {}

def get_session_memory(session_id: str) -> ConversationSummaryMemory:
    """세션 ID에 해당하는 ConversationSummaryMemory를 가져오거나 생성합니다."""
    if session_id not in lc_memories:
        # 새 세션을 위한 메모리 객체 생성
        lc_memories[session_id] = ConversationSummaryMemory(
            llm=ChatVertexAI(
                model_name="gemini-1.5-flash",
                api_transport="rest",
                temperature=0
            ),
            memory_key="chat_history",
            input_key="query",
            output_key="answer",
            summary_prompt=ChatPromptTemplate.from_messages([
                ("system", "다음 대화 내역을 간결히 요약하세요. 옳고 그름은 판단하지 마세요."),
                ("user", "{chat_history}")
            ]),
            k=3
        )
    return lc_memories[session_id]


vertexai.init(
    project=project_id,
    location="asia-northeast1",
    credentials=credentials
)

generate_llm = ChatVertexAI(
    model_name="gemini-2.5-flash",
    api_transport="rest",        # REST 호출 권장
    temperature=0.1,
    verbose=True,
)


fast_llm = ChatVertexAI(
    model_name="gemini-1.5-flash",
    api_transport="rest",        # REST 호출 권장
    temperature=0.1,
    max_output_tokens=512,       # 충분히 큰 값 설정
    verbose=True,
)

checker = VertexAICheckGroundingWrapper(        
    project_id = project_id,
    location_id = "asia-northeast1",
    grounding_config="default_grounding_config",   # 기본값
    citation_threshold=0.5,
    credentials=credentials,
)

re_write_system = """
당신의 역할은 사용자의 query를 키워드 중심으로 재작성하는 것입니다.
절대 답변을 생성하거나, chat_history을 요약·설명하거나, query 외의 어떤 정보도 추가하지 마세요.

- query에 있는 대명사/지시어(이것, 그것, 그 보직 등)는 chat history에서 언급된 명사(특기/보직명 등)로 치환하세요.
- query가 이미 명확하면 수정하지 말고 그대로 반환하세요.
- 새로운 정보, 불필요한 설명, 반복, 수식어, 예시, 답변, 안내문구는 절대 추가하지 마세요.
- 오직 재작성된 query만, 한 문장으로 반환하세요.
"""

question_decomposer_system = """
너는 이제부터 사용자의 복잡한 질문을 독립적으로 답변할 수 있는 여러 개의 단순한 질문으로 분해하는 역할을 수행한다.

규칙:
1. 각각의 분해된 질문은 그 자체로 완전한 의미를 가져야 한다.
2. 다른 설명이나 문장을 추가해서는 안 된다.


**예시 1**
* 입력: "오늘 구리시 날씨랑 미세먼지 농도 어때?"
* 출력: ["오늘 구리시 날씨","오늘 구리시 미세먼지 농도"]

**예시 2**
* 입력: "홍수 발생 시 연락해야하는 전화번호 알려주고 대피 장소 알려줘"
* 출력: ["홍수 발생 시 연락해야 하는 곳의 전화번호","홍수 발생 시 대피 장소"]

**예시 3**
* 입력: "교통사고 발생 시 주의사항, 대피 방법, 사고처리 방법에 대해 알려줘"
* 출력: ["교통사고 발생 시 주의사항", "교통사고 발생 시 대피 방법", "교통사고 발생 시 사고처리 방법"]
"""


generate_prompt_system = """
너는 소방청의 안전 지키미 AI 챗봇이야.
너는 지금부터 내가 제공하는 [document]를 참고해서 [query]에 답변해야 해.
"""

# 1) ChatPromptTemplate 정의
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", re_write_system),
    ("user",   "chat history: \n {chat_history} \n\n query: \n {query}")
])

question_decomposer_prompt = ChatPromptTemplate.from_messages([
    ("system", question_decomposer_system),
    ("user",   "query: {query}")
])

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", generate_prompt_system),
    ("user", "document: {document}, query: {query}")
])


# 3) Runnable 체인 구성
rewrite_chain = rewrite_prompt | fast_llm | StrOutputParser() | RunnableLambda(lambda text: text.strip())

question_decomposer_chain = question_decomposer_prompt | fast_llm | StrOutputParser() | RunnableLambda(lambda text: text.strip())

generate_chain = generate_prompt | generate_llm | StrOutputParser()

# 그래프 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "User question"]
    chat_history: Annotated[str, "Chat history for context"]
    document: Annotated[List[str], "Combined documents"]
    generation: Annotated[str, "LLM generated answer"]
    hallu_check: Annotated[str, "Hallucination check"]
    session_id: Annotated[Optional[str], "Session ID for memory management"]


def re_writer(state):
    """질문 재작성"""
    question = state["question"]
    chat_history = state.get("chat_history", "None")

    rewritten = rewrite_chain.invoke({"chat_history": chat_history, "query": question})
    return {"question": rewritten}


def question_decomposer(state):
    """질문 분해"""
    decomposed = question_decomposer_chain.invoke({"query": state["question"]})
    return {
        **state,
        "question": decomposed
    }


def search_document(state):
    """문서 검색"""
    searched_documents = vector_store.search(state["question"], k=150)
    result= vector_store.document_rerank(state["question"], searched_documents,k=20)
    return {
        **state,
        "document": result
    }


def generate_answer(state):
    """답변 생성"""
    answer = generate_chain.invoke({"document": state["document"], "query": state["question"]})
    return {
        **state,
        "generation": answer
    }


def hallu_checker(state):
    """허위 정보 검사"""
    hallucination_check = checker.invoke(
        state["generation"],
        config = {"configurable": {"documents": state["document"]}}
    )
    return {
        **state,
        "hallu_check": hallucination_check
    }

def check_hallu_complete_chat(state):
    """허위 정보 검사 완료 여부 확인 및 메모리 저장"""
    check = state.get("hallu_check", "None")
    if check.support_score > 0.5:
        session_id = state.get("session_id")
        if session_id:
            memory = get_session_memory(session_id)
            memory.save_context({"query": state["question"]}, {"answer": state["generation"]})
        return "END"
    else:
        return "re_writer"

def check_hallu_complete_make_form(state):
    """허위 정보 검사 완료 여부 확인"""
    check = state.get("hallu_check", "None")
    if check.support_score > 0.5:
        return "END"
    else:
        return "re_writer"

# LangGraph 워크플로우 생성
chat_workflow = StateGraph(GraphState)
chat_workflow.add_node("re_writer", re_writer)
chat_workflow.add_node("question_decomposer", question_decomposer)
chat_workflow.add_node("search_document", search_document)
chat_workflow.add_node("generate_answer", generate_answer)
chat_workflow.add_node("hallu_checker", hallu_checker)


chat_workflow.add_edge(START, "re_writer")
chat_workflow.add_edge("re_writer", "question_decomposer")
chat_workflow.add_edge("question_decomposer", "search_document")
chat_workflow.add_edge("search_document", "generate_answer")
chat_workflow.add_edge("generate_answer", "hallu_checker")
chat_workflow.add_conditional_edges(
    "hallu_checker",
    check_hallu_complete_chat,
    {
        "END": END,
        "re_writer": "re_writer"
    }
)
chat_flow = chat_workflow.compile(checkpointer=MemorySaver())

# LangGraph 워크플로우 생성
make_form_workflow = StateGraph(GraphState)
make_form_workflow.add_node("re_writer", re_writer)
make_form_workflow.add_node("question_decomposer", question_decomposer)
make_form_workflow.add_node("search_document", search_document)
make_form_workflow.add_node("generate_answer", generate_answer)
make_form_workflow.add_node("hallu_checker", hallu_checker)


make_form_workflow.add_edge(START, "re_writer")
make_form_workflow.add_edge("re_writer", "question_decomposer")
make_form_workflow.add_edge("question_decomposer", "search_document")
make_form_workflow.add_edge("search_document", "generate_answer")
make_form_workflow.add_edge("generate_answer", "hallu_checker")
make_form_workflow.add_conditional_edges(
    "hallu_checker",
    check_hallu_complete_make_form,
    {
        "END": END,
        "re_writer": "re_writer"
    }
)
make_form_flow = make_form_workflow.compile(checkpointer=MemorySaver())


def make_form_rag(question: str, session_id: Optional[str] = None) -> str:
    config = RunnableConfig(
                recursion_limit=12,
                configurable={"thread_id": random_uuid()},
                timeout=120,  # 2분 타임아웃
                max_retries=3  # 최대 3회 재시도
            )
    inputs = {"question": question, "chat_history": "", "session_id": None}
    final_state = make_form_flow.invoke(inputs, config)
    return final_state.get("generation", "에러: 답변을 생성할 수 없습니다.")


def chat_rag(question: str, session_id: str) -> str:
    config = RunnableConfig(
                recursion_limit=12,
                configurable={"thread_id": session_id},
                timeout=120,  # 2분 타임아웃
                max_retries=3  # 최대 3회 재시도
            )

    # 세션에 맞는 메모리 가져오기
    memory = get_session_memory(session_id)
    chat_history = memory.load_memory_variables({"query": ""}).get("chat_history", "")

    inputs = {"question": question, "chat_history": chat_history, "session_id": session_id}
    final_state = chat_flow.invoke(inputs, config)

    return final_state.get("generation", "에러: 답변을 생성할 수 없습니다!") ,final_state.get("hallu_check", "에러 발생!")


# FastAPI 애플리케이션 설정
app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# API 요청 모델 정의
class ApiRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

# API 엔드포인트 생성
@app.post("/api/generate_form")
async def generate_form_endpoint(request: ApiRequest):
    """
    질문을 받아 form 형식의 답변을 생성하는 API 엔드포인트입니다.
    이 엔드포인트는 상태를 유지하지 않는 make_form_rag 함수를 사용합니다.
    """
    try:
        answer = make_form_rag(request.question, request.session_id)
        return {"answer": answer}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/api/chat")
async def chat_endpoint(request: ApiRequest):
    """
    질문을 받아 채팅 형식의 답변을 생성하는 API 엔드포인트입니다.
    이 엔드포인트는 대화 기록을 사용하는 chat_rag 함수를 사용합니다.
    """
    try:
        if not request.session_id:
            # 채팅은 세션 ID가 필수입니다.
            raise HTTPException(status_code=400, detail="session_id is required for chat.")
        answer, hallu_check = chat_rag(request.question, request.session_id)
        return {"return_answer": answer, "hallu_check": hallu_check}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
