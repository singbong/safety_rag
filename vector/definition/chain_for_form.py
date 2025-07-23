import vertexai
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_community.vertex_check_grounding import VertexAICheckGroundingWrapper
from langgraph.graph import END, StateGraph, START
from google.oauth2 import service_account
from typing import List, Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from .vector_store import store_vector_db
import os
import warnings
import json

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["PROJECT_ID"] = os.getenv("PROJECT_ID")
os.environ["GOOGLE_CLOUD_LOCATION"] = "asia-northeast1"
credential = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
warnings.filterwarnings("ignore")

# Vertex AI 초기화

vertexai.init(project=os.getenv("PROJECT_ID"), location="asia-northeast1")

# Vector Store 로드
vector_store = store_vector_db()
vector_store.load_vector_store("./vector/data/vector_store/")



query_generator_system = """
    [역할 및 목표 정의]
    너는 사용자의 입력을 분석하여 벡터 저장소(Vector Store)에서 가장 정확한 안전 정보를 검색하기 위한 '검색어 생성 전문 AI'다.
    너의 임무는 아래 **[사용자 입력 정보]**를 다각적으로 분석하고 추론하여, 관련 문서를 효과적으로 찾아낼 수 있는 여러 개의 구체적인 검색어(질문 형태)들을 생성하는 것이다.

    [사용자 입력 정보]
        - 운영 기간: [operation_period]
        - 운영 시간: [operation_time]
        - 사용자 추가 요청: [user_additional_request]

    [작업 수행 지침]
    1. 입력 정보 분석 및 컨텍스트 추론:
        - 운영기간을 보고 계절(여름, 겨울 등), 시기적 특성(장마철, 휴가철, 명절 등)을 추론한다.
        - 운영시간을 보고 시간대(주간, 야간, 새벽)에 따른 환경적 특성을 추론한다.
        - 사용자 추가 요청을 보고 핵심 키워드와 사용자가 가장 중요하게 생각하는 안전 이슈를 정확히 파악한다. (예: 야외 행사, 물놀이, 화재 예방 등)

    2. 다각적 검색어 생성:
        - 위 분석 내용을 바탕으로, 아래 세 가지 유형의 검색어를 각각 생성한다. 각 검색어는 단일 키워드가 아닌, 자연스러운 질문 또는 구체적인 서술 형태여야 한다.

        A. 시기적/환경적 검색어 (1~3개):
        - 추론된 계절, 날씨, 시간대와 관련된 포괄적인 안전 수칙을 묻는 질문을 생성한다.
        - (예시) "여름철(7월, 8월)에 주로 발생하는 안전사고 종류와 예방 수칙은 무엇인가?"
        - (예시) "야간 시간대 야외 활동 시 주의해야 할 안전 문제는?"

        B. 상황/유형별 검색어 (1~2개):
        - [user_additional_request]에 언급된 상황(예: '야외 행사', '건설 현장')과 관련된 일반적인 안전 매뉴얼이나 가이드를 찾는 질문을 생성한다.
        - (예시) "대규모 야외 행사를 진행할 때 반드시 점검해야 할 안전 관리 항목은?"
        - (예시) "건축 공사장 폭염 대비 근로자 안전 가이드라인은?"

        C. 사용자 요청 기반 검색어 (1~3개):
        - [user_additional_request]에서 사용자가 직접 '강조'해달라고 요청한 핵심 내용에 대해 가장 구체적이고 직접적인 질문을 생성한다. 이 유형의 검색어가 가장 중요하다.
        - (예시) "폭염 발생 시 온열질환 예방 및 응급처치 방법은?"
        - (예시) "여름철 식중독 예방을 위한 식품 보관 및 조리 수칙은?"
        - (예시) "물놀이 시 발생할 수 있는 익사 사고 예방 수칙은 무엇인가?"

    3. 최종 출력 형식 준수:

    **아래와 같이 JSON 형태로 출력하라. 반드시 JsonOutputParser()로 파싱 가능한 형태여야 한다.**

    {{
        "query": ["생성된 검색어 1", "생성된 검색어 2", "생성된 검색어 3",..., "생성된 검색어 N"]
    }}

"""




generate_prompt_system = """
[입력 정보 요약]
운영 기간: [operation_period]
운영 시간: [operation_time]
안내 방식: [instruction_type] (예: 상세 안내문, 카드뉴스, 문자 메시지 등)
사용자 추가 요청: [user_additional_request]
검색된 안전 정보 문서들: [documents]

[작업 수행 단계]

1. 컨텍스트 분석:
* 먼저 [운영 기간]과 [운영 시간]을 분석하여 시기적, 시간적 특성을 파악해.
* (예: 7~8월 → 여름철, 폭염, 장마, 태풍 / 오전 10시~오후 6시 → 낮 시간대 활동)
* 이 컨텍스트에서 발생할 확률이 가장 높은 안전사고 유형을 예측해봐.

2. 핵심 정보 선별:
* [검색된 안전 정보 문서들] 중에서, 위에서 분석한 시기적/시간적 컨텍스트와 가장 관련성이 높은 정보들을 선별해.
* [사용자 추가 요청]이 있다면, 해당 내용과 관련된 정보를 최우선으로 선별하고 가장 중요한 내용으로 다루어야 해.

3. 안내 방식에 맞춰 재구성:
* 선별된 핵심 정보들을 [안내 방식]에 지정된 형식에 맞춰 재구성하고 초안을 작성해.
* '상세 안내문'일 경우: 제목, 개요, 본문(소제목과 글머리 기호 사용), 당부 말씀 등 격식 있고 체계적인 구조로 작성.
* '방송'일 경우: 시민들이 쉽게 이해할 수 있도록 명확하고 간결한 문장으로, 주의사항과 행동 요령을 중심으로 방송문을 작성해. (예: "지금은 폭염특보가 발효 중입니다. 야외활동을 자제하고, 충분한 수분을 섭취하세요.")
* '공지판'일 경우: 한눈에 들어오는 제목, 핵심 요점(글머리표 사용), 주의사항 등으로 구성된 공지문 형태로 작성해. (예: "폭염 시 행동 요령 - 1. 외출 자제 2. 물 자주 마시기 3. 노약자 각별히 주의")
* '문자 메시지'일 경우: 글자 수 제한을 고려하여 가장 중요한 내용만을 짧고 명확하게 요약해서 작성해. (예: "폭염주의! 외출 자제, 물 자주 마시세요.")

4. 최종 생성 및 검토:
* 작성된 초안을 바탕으로, 최종 안전 안내문을 생성해.
* 전체 내용이 [운영 기간], [운영 시간], [사용자 추가 요청] 등 모든 조건을 충실히 반영했는지 최종적으로 검토하고, 문장이 자연스러운지 확인해.

[중요 규칙]
근거 기반 생성: 반드시 [검색된 안전 정보 문서들]에 있는 내용을 활용하여 안내문을 작성해야 한다.
창작 금지: 없는 사실이나 안전 수칙을 절대로 만들어내서는 안 된다.
목적 부합성: 생성된 안내문은 사용자가 즉시 활용할 수 있을 만큼 명확하고, 실용적이며, 이해하기 쉬워야 한다.
어조: 신뢰감을 주는 전문가의 어조를 유지하되, 시민들이 경각심을 가질 수 있도록 명료하고 단호한 표현을 사용해라.
"""


class GraphState(TypedDict):
    generation: Annotated[str, "LLM generated answer"]
    hallu_check: Annotated[dict, "Hallucination check result"]
    query_list: Annotated[List[str], "Query list"]
    
    instruction_type: Annotated[str, "Instruction type"]
    user_additional_request: Annotated[str, "User additional request"]
    operation_period: Annotated[str, "Operation period"]
    operation_time: Annotated[str, "Operation time"]
    searched_documents: Annotated[List[str], "Documents"]

class form_chain():
    def __init__(self):
        # 빠른 응답 및 간단한 작업용 LLM
        self.fast_llm = ChatVertexAI(
            model_name="gemini-1.5-flash",
            temperature=0.1,
            max_output_tokens=512,
            verbose=True,
        )
        # 답변 생성 및 Grounding 확인용 LLM
        self.generate_llm = ChatVertexAI(
            model_name="gemini-2.5-flash", # Grounding을 지원하는 최신 모델 사용 권장
            temperature=0.4,
            verbose=True,
        )
        self.graph = StateGraph(GraphState)
        # Grounding Wrapper 설정
        self.checker = VertexAICheckGroundingWrapper(        
            project_id = os.getenv("PROJECT_ID"),
            location_id = "asia-northeast1",
            grounding_config="default_grounding_config",   # 기본값
            citation_threshold=0.5,
            credentials=credential,
        )

    def query_generate(self, operation_period: str, operation_time: str, user_additional_request: Optional[str] = None) -> str:
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", query_generator_system),
            ("user",   "chat operation_period: {operation_period} \n operation_time: {operation_time} \n user_additional_request: {user_additional_request}")
        ])
        chain = generate_prompt | self.generate_llm | JsonOutputParser()
        query_list = chain.invoke({"operation_period": operation_period, "operation_time": operation_time, "user_additional_request": user_additional_request})
        return query_list.get("query")
    
    def generate(self, operation_period: Optional[str] = None, operation_time: Optional[str] = None, instruction_type: Optional[str] = None, user_additional_request: Optional[str] = None, documents: List[str] = None) -> str:
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", generate_prompt_system),
            ("user",   "operation_period: {operation_period} \n operation_time: {operation_time} \n instruction_type: {instruction_type} \n user_additional_request: {user_additional_request} \n documents: {documents}")
        ])
        chain = generate_prompt | self.generate_llm | StrOutputParser()
        generation = chain.invoke({"operation_period": operation_period, "operation_time": operation_time, "instruction_type": instruction_type, "user_additional_request": user_additional_request, "documents": documents})
        return generation


    def query_generator(self, state: GraphState) -> GraphState:
        query_list = self.query_generate(state["operation_period"], state["operation_time"], state["user_additional_request"])
        return {**state, "query_list": query_list}

    def search_document(self, state: GraphState) -> GraphState:
        documents = []
        for query in state["query_list"]:
            searched_documents = vector_store.search(query, k=50)
            reranked_documents = vector_store.document_rerank(query, searched_documents, k=10)
            documents.extend(reranked_documents)
        # 중복 문서 제거
        unique_documents = []
        seen_contents = set()
        for doc in documents:
            if doc['document'] not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc['document'])
        return {**state, "searched_documents": unique_documents}

    def generator(self, state: GraphState) -> GraphState:
        generation = self.generate(state["operation_period"], state["operation_time"], state["instruction_type"], state["user_additional_request"], state["searched_documents"])
        return {**state, "generation": generation}

    def hallu_checker(self, state: GraphState) -> GraphState:
        hallu_check = self.checker.invoke(
            state["generation"],
            config = {"configurable": {"documents": state["documents"]}}
        )
        return {**state, "hallu_check": hallu_check}

    def check_hallu_complete_chat(self, state: GraphState) -> str:
        """
        Grounding 확인 결과를 바탕으로 다음 단계를 결정합니다.
        """
        result = state["hallu_check"]
        score = result.support_score
        # 평균 점수가 0.5를 넘으면 종료, 아니면 재시도
        if score >= 0.5:
            return END
        else:
            return "re_writer"

    def link_nodes(self):
        """
        LangGraph의 노드들을 연결하여 그래프를 구성합니다.
        """
        self.graph.add_node("re_writer", self.re_writer)
        self.graph.add_node("question_decomposer", self.question_decomposer)
        self.graph.add_node("search_document", self.search_document)
        self.graph.add_node("generator", self.generator)
        self.graph.add_node("hallu_checker", self.hallu_checker)

        self.graph.add_edge(START, "re_writer")
        self.graph.add_edge("re_writer", "question_decomposer")
        self.graph.add_edge("question_decomposer", "search_document")
        self.graph.add_edge("search_document", "generator")
        self.graph.add_edge("generator", "hallu_checker")

        self.graph.add_conditional_edges(
            "hallu_checker",
            self.check_hallu_complete_chat, # 함수 객체 자체를 전달
            {
                END: END,
                "re_writer": "re_writer"
            }
        )
        # 컴파일된 그래프 반환
        return self.graph.compile(checkpointer=MemorySaver())