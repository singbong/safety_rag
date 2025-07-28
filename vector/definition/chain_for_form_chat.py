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
# .env 파일 로드

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["PROJECT_ID"] = os.getenv("PROJECT_ID")
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_file(credential_path)
warnings.filterwarnings("ignore")

# Vertex AI 초기화

vertexai.init(project=os.getenv("PROJECT_ID"), location="us-central1")

# Vector Store 로드
vector_store = store_vector_db()
vector_store.load_vector_store("./vector/data/vector_store/")


re_write_system = \
"""
[역할 및 목표]
너는 '안전 안내문' [generated_form]에 대한 사용자의 후속 질문 [query]를, RAG 검색에 용이하도록 명확하게 재작성하는 AI다.

[작업 지침]
1.  **맥락 파악:** [generated_form]와 [chat_history] 기반으로 내용을 파악하고, 사용자의 [query]가 어느 부분에 대한 질문인지 이해한다.
2.  **질문 재작성:**
    -   [query]에 포함된 "거기서", "그 부분", "그 내용은"과 같은 대명사나 지시어를 [chat_history]의 구체적인 내용(예: "폭염 시 행동 요령", "식중독 예방 수칙")으로 바꿔서 명확한 질문으로 만든다.
    -   사용자의 질문이 이미 명확하다면, 수정하지 않고 그대로 반환한다.
3.  **출력 형식:**
    -   오직 재작성된 질문(query)만을 한 문장으로 간결하게 출력한다.
    -   절대 질문에 대한 답변을 하거나, 설명을 덧붙이거나, 안내 문구를 추가하지 않는다.

[예시]
-   [generated_form]: "화재 발생 시 행동 요령: 1. 신속히 대피 2. 119 신고하기..."
-   [chat_history]: "Q: 화재가 발생하면 어떻게 해야 하나요? A: 먼저 신속히 대피하고, 119에 신고해야 합니다..."
-   [query]: "거기서 첫 번째가 왜 가장 중요한가요?"
-   [출력]: "화재 발생 시 신속한 대피가 왜 가장 중요한가요?"
"""

query_generator_system = \
"""
[역할 및 목표]
너는 사용자의 [query]와 [generated_form]을 바탕으로, 더 풍부한 RAG 검색을 위한 서브 쿼리들을 생성하는 AI다.

[작업 지침]
1. **맥락 파악:**
    - [generated_form]의 전체 내용과 맥락을 이해한다
    - 사용자의 [query]가 [generated_form]의 어느 부분과 관련된 질문인지 파악한다

2. **서브 쿼리 생성:**
    - [query]의 핵심 키워드와 의도를 파악한다
    - 다음 관점들을 고려하여 서브 query를 생성한다:
        * 직접적 답변을 찾기 위한 query
        * 배경 정보를 찾기 위한 query
        * 관련된 안전 수칙이나 예방법을 찾기 위한 query
        * 유사 사례나 상황을 찾기 위한 query

3. **출력 규칙:**
    - 각 서브 query는 독립적으로 검색 가능한 완전한 문장이어야 한다
    - 최소 3개에서 최대 5개의 서브 query를 생성한다
    - JSON 형식으로 출력한다

[예시]
- [generated_form]: "폭염 시 행동요령: 1. 외출자제 2. 물 자주 마시기..."
- [query]: "물을 자주 마셔야 하는 이유가 뭔가요?"
- [출력]:
{{
    "query": [
        "폭염시 수분 섭취가 중요한 이유",
        "더운 날씨에서 탈수 증상과 위험성",
        "폭염 대비 올바른 수분 섭취 방법",
        "폭염으로 인한 온열질환 예방법"
    ]
}}
"""

generate_prompt_system = \
"""
[역할 및 목표]
너는 대한민국 소방청의 '안전 지키미 AI' 챗봇이다.
너의 임무는 이전에 제공된 [generated_form](최초 안전 안내문)에 대한 사용자의 [query](후속 질문)에 대해, [searched_documents](추가 검색된 문서)를 참고하여 상세하고 정확한 답변을 제공하는 것이다.

[답변 생성 가이드라인]
1.  **맥락 파악:** 먼저 [generated_form]를 통해 전체 대화의 맥락을 파악한다. 사용자의 [query]가 내용에 대한 후속 질문인지 이해하는 것이 중요하다.
2.  **답변 생성:**
    -   [query]에 대한 핵심 답변을 반드시 [searched_documents]을 기반으로 하여 답변을 생성한다.

[중요 규칙]
-   **근거 중심:** 모든 답변은 반드시 [searched_documents]에 명시된 내용을 근거로 해야 한다.
-   **추측 금지:** [document]에 없는 내용은 절대로 추측하거나 임의로 생성해서는 안 된다. 만약 관련 내용이 없다면, "제공된 문서에서는 해당 정보를 찾을 수 없습니다."라고 명확히 밝혀준다.
-   **친절한 전문가 어조:** '안전 지키미'로서 전문적이면서도, 누구나 이해하기 쉬운 친절하고 명확한 말투를 사용한다.
"""


final_answer_system = \
"""
당신은 텍스트를 깔끔하게 정리하는 AI입니다.
주어진 [generation]는 내용과 citation([숫자])은 정확하지만, 줄바꿈이나 글머리 기호 같은 서식이 모두 사라진 상태입니다.

당신의 임무는 [generation]의 내용과 citation을 그대로 유지하면서, 사용자가 읽기 쉽도록 서식을 복원하는 것입니다.

**[규칙]**
1.  **내용 보존:** 원본 텍스트의 모든 단어와 citation([숫자])을 빠짐없이 사용해야 합니다.
2.  **서식 복원:** 문맥에 맞게 줄바꿈, 글머리 기호(*), 제목(##) 등의 마크다운 서식을 자연스럽게 추가해주세요.
3.  **내용 수정 금지:** 원본에 없는 내용을 추가하거나, 기존 내용을 변경하지 마세요.
4.  **출력:** 다른 설명 없이, 서식이 복원된 최종 텍스트만 출력합니다.
"""

class GraphState(TypedDict):
    generated_form: Annotated[str, "Generated form"]
    query: Annotated[str, "User question"]
    generated_queries: Annotated[List[str], "Generated queries"]
    searched_documents: Annotated[List[str], "Combined documents"]
    generation: Annotated[str, "LLM generated answer"]
    hallu_check: Annotated[dict, "Hallucination check result"]
    final_answer: Annotated[str, "Final answer with citations"]
    chat_history: Annotated[str, "Chat history"]

class form_chat_chain():
    def __init__(self):
        # 빠른 응답 및 간단한 작업용 LLM
        self.fast_llm = ChatVertexAI(
            model_name="gemini-2.5-flash-lite",
            temperature=0.1,
            max_output_tokens=512,
            verbose=True,
        )
        # 답변 생성 및 Grounding 확인용 LLM
        self.generate_llm = ChatVertexAI(
            model_name="gemini-2.5-flash-lite", # Grounding을 지원하는 최신 모델 사용 권장
            temperature=0.4,
            verbose=True,
        )
        self.final_llm = ChatVertexAI(
            model_name="gemini-2.5-flash-lite",
            temperature=0,
            verbose=True,
        )
        self.graph = StateGraph(GraphState)
        # Grounding Wrapper 설정
        self.checker = VertexAICheckGroundingWrapper(        
            project_id = os.getenv("PROJECT_ID"),
            location_id = "us-central1",
            grounding_config="default_grounding_config",   # 기본값
            citation_threshold=0.5,
            credentials=credentials,
        )

########################################################################################################################
# 체인 정의
########################################################################################################################
    def re_write(self, generated_form: Optional[str] = None, chat_history: Optional[str] = None, query: str = None) -> str:
        """
        사용자의 질문을 키워드 중심으로 재작성하는 체인
        """
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", re_write_system),
            ("user",   "generated_form: \n {generated_form} \n\n chat_history: \n {chat_history} \n\n query: \n {query}")
        ])
        chain = rewrite_prompt | self.fast_llm | StrOutputParser() | RunnableLambda(lambda text: text.strip())
        return chain.invoke({"generated_form": generated_form, "chat_history": chat_history, "query": query})

    def query_generate(self, generated_form: Optional[str], query: Optional[str] = None) -> List[str]:
        """
        사용자의 질문을 독립적으로 답변할 수 있는 여러 개의 단순한 질문으로 분해하는 체인
        """
        question_decomposer_prompt = ChatPromptTemplate.from_messages([
            ("system", query_generator_system),
            ("user",   "generated_form: {generated_form} \n query: {query}")
        ])
        chain = question_decomposer_prompt | self.fast_llm | JsonOutputParser()
        result = chain.invoke({"generated_form": generated_form, "query": query})
        return result.get("query")
    
    def generate(self, generated_form: Optional[str] = None, query: str = None, searched_documents: Optional[List[str]] = None) -> str:
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", generate_prompt_system),
            ("user",   "[generated_form]: {generated_form} \n [query]: {query} \n [searched_documents]: {searched_documents}")
        ])
        chain = generate_prompt | self.generate_llm | StrOutputParser()
        return chain.invoke({"generated_form": generated_form, "query": query, "searched_documents": searched_documents})

    def clean_up_answer_with_citations(self, generation: str) -> str:
        prompt = ChatPromptTemplate.from_messages([("system", final_answer_system), ("user", "[generation]: \n{generation}")])
        chain = prompt | self.final_llm | StrOutputParser()
        return chain.invoke({"generation": generation})

########################################################################################################################
# 그래프 노드 정의
########################################################################################################################
    def re_writer(self, state: GraphState) -> GraphState:
        rewritten = self.re_write(state["generated_form"], state["query"])
        return {**state, "question": rewritten}

    def query_generator(self, state: GraphState) -> GraphState:
        generated_queries = self.query_generate(state["generated_form"], state["query"])
        return {**state, "generated_queries": generated_queries}

    def search_document(self, state: GraphState) -> GraphState:
        documents = []
        for query in state["generated_queries"]:
            searched_documents = vector_store.search(query, k=50)
            reranked_documents = vector_store.document_rerank(query, searched_documents, k=10)
            documents.extend(reranked_documents)
        # 중복 문서 제거
        unique_documents = []
        seen_contents = set()
        for doc in documents:
            if doc.page_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc.page_content)
        return {**state, "searched_documents": unique_documents}

    def generator(self, state: GraphState) -> GraphState:
        generation = self.generate(state["generated_form"], state["query"], state["searched_documents"])
        return {**state, "generation": generation}

    def hallu_checker(self, state: GraphState) -> GraphState:
        hallu_check = self.checker.invoke(
            state["generation"],
            config = {"configurable": {"documents": state["searched_documents"]}}
        )
        return {**state, "hallu_check": hallu_check}

    def check_hallu_complete_chat(self, state: GraphState) -> str:
        """
        Grounding 확인 결과를 바탕으로 다음 단계를 결정합니다.
        """
        score = state["hallu_check"].support_score
        # 평균 점수가 0.5를 넘으면 종료, 아니면 재시도
        if score >= 0.5:
            return "clean"
        else:
            return "re_writer"

    def answer_beautifier(self, state: GraphState) -> GraphState:
        final_answer = state["hallu_check"].answer_with_citations
        final_answer = self.clean_up_answer_with_citations(final_answer)
        return {**state, "final_answer": final_answer}

########################################################################################################################
# 그래프 노드 연결
########################################################################################################################

    def link_nodes(self):
        """
        LangGraph의 노드들을 연결하여 그래프를 구성합니다.
        """
        self.graph.add_node("re_writer", self.re_writer)
        self.graph.add_node("query_generator", self.query_generator)
        self.graph.add_node("search_document", self.search_document)
        self.graph.add_node("generator", self.generator)
        self.graph.add_node("hallu_checker", self.hallu_checker)
        self.graph.add_node("answer_beautifier", self.answer_beautifier)

        self.graph.add_edge(START, "re_writer")
        self.graph.add_edge("re_writer", "query_generator")
        self.graph.add_edge("query_generator", "search_document")
        self.graph.add_edge("search_document", "generator")
        self.graph.add_edge("generator", "hallu_checker")
        self.graph.add_edge("hallu_checker", "answer_beautifier")
        # self.graph.add_conditional_edges(
        #     "hallu_checker",
        #     self.check_hallu_complete_chat, # 함수 객체 자체를 전달
        #     {
        #         "clean": "answer_beautifier",
        #         "re_writer": "re_writer"
        #     }
        # )
        self.graph.add_edge("answer_beautifier", END)
        # 컴파일된 그래프 반환
        return self.graph.compile(checkpointer=MemorySaver())