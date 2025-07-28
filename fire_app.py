import vertexai
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_teddynote.messages import random_uuid
from langchain_core.runnables import RunnableConfig
import httpx
from tqdm import tqdm
import re
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from vector.definition.chain_for_chat import chat_chain
from vector.definition.chain_for_form_chat import form_chat_chain
from vector.definition.chain_for_form import form_chain
from typing import Optional
from pydantic import BaseModel
from langchain.memory import ConversationSummaryMemory
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict


# 세션별 메모리 관리를 위한 딕셔너리
lc_memories: Dict[str, ConversationSummaryMemory] = {}
def get_session_memory(session_id: str) -> ConversationSummaryMemory:
    """세션 ID에 해당하는 ConversationSummaryMemory를 가져오거나 생성합니다."""
    if session_id not in lc_memories:
        # 새 세션을 위한 메모리 객체 생성
        lc_memories[session_id] = ConversationSummaryMemory(
            llm=ChatVertexAI(
                model_name="gemini-2.5-flash-lite",
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

chat_flow = chat_chain().link_nodes()
form_chat_flow = form_chat_chain().link_nodes()
make_form_flow = form_chain().link_nodes()


def chat_rag(question: str, session_id: str) -> str:
    config = RunnableConfig(
                recursion_limit=12,
                configurable={"thread_id": session_id},
                timeout=120,  # 2분 타임아웃
                max_retries=2  # 최대 2회 재시도
            )
    # 세션에 맞는 메모리 가져오기
    memory = get_session_memory(session_id)
    chat_history = memory.load_memory_variables({"query": ""}).get("chat_history", "")
    inputs = {"question": question, "chat_history": chat_history}
    final_state = chat_flow.invoke(inputs, config)
    memory.save_context({"query": final_state["question"]}, {"answer": final_state.get("generation", "에러: 답변을 생성할 수 없습니다!")})
    return final_state


def make_form(place_name: str, type: str, region: str, period: str, description: Optional[str] = None, category: Optional[str] = None, related_documents: Optional[str] = None, emergency_contact_name: Optional[str] = None, emergency_contact_phone: Optional[str] = None) -> str:
    config = RunnableConfig(
                recursion_limit=12,
                timeout=120,  # 2분 타임아웃
                max_retries=2,  # 최대 2회 재시도
                configurable={"thread_id": random_uuid()}
            )
    inputs = {
        "place_name": place_name,
        "type": type,
        "region": region,
        "period": period,
        "description": description,
        "category": category,
        "related_documents": related_documents,
        "emergency_contact_name": emergency_contact_name,
        "emergency_contact_phone": emergency_contact_phone
    }
    final_state = make_form_flow.invoke(inputs, config)
    return final_state

def make_form_chat(generated_form: Optional[str] = None, query: str = None, session_id: str = None) -> str:
    config = RunnableConfig(
                recursion_limit=12,
                configurable={"thread_id": session_id},
                timeout=120,  # 2분 타임아웃
                max_retries=2  # 최대 2회 재시도
            )
    memory = get_session_memory(session_id)
    chat_history = memory.load_memory_variables({"query": ""}).get("chat_history", "")
    inputs = {"generated_form": generated_form, "query": query, "chat_history": chat_history}
    final_state = form_chat_flow.invoke(inputs, config)
    memory.save_context({"query": final_state["query"]}, {"answer": final_state.get("generation", "에러: 답변을 생성할 수 없습니다!")})
    return final_state



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

class FormChatRequest(BaseModel):
    generated_form: Optional[str] = None
    query: str
    session_id: Optional[str] = None


class FormRequest(BaseModel):
    place_name: str
    type: str
    region: str
    period: str
    description: Optional[str] = None
    category: Optional[str] = None
    related_documents: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None



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
        final_state = chat_rag(request.question, request.session_id)
        return final_state
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")



#API 엔드포인트 생성
@app.post("/api/generate_form")
async def generate_form_endpoint(request: FormRequest):

    try:
        final_state = make_form(
            place_name=request.place_name,
            type=request.type,
            region=request.region,
            period=request.period,
            description=request.description,
            category=request.category,
            related_documents=request.related_documents,
            emergency_contact_name=request.emergency_contact_name,
            emergency_contact_phone=request.emergency_contact_phone
        )
        return final_state
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")



@app.post("/api/form_chat")
async def form_chat_endpoint(request: FormChatRequest):
    try:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="session_id is required for form chat.")
        final_state = make_form_chat(request.generated_form, request.query, request.session_id)
        return final_state
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
