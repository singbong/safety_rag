import vertexai
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_community.vertex_check_grounding import VertexAICheckGroundingWrapper
from langgraph.graph import END, StateGraph, START
from google.oauth2 import service_account
from typing import List
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
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_file(credential_path)
warnings.filterwarnings("ignore")

# Vertex AI 초기화
vertexai.init(project=os.getenv("PROJECT_ID"), location="us-central1")

# Vector Store 로드
vector_store = store_vector_db()
vector_store.load_vector_store("./vector/data/vector_store/")

query_generator_system = """
[역할 및 목표]
너는 사용자의 행사 등록 정보를 분석하여, 발생 가능한 모든 안전 위협을 예측하고 관련 정보를 검색하기 위한 '다각적 검색어 생성 AI'다.
너의 임무는 아래 **[사용자 등록 정보]**를 바탕으로, Vector Store에서 가장 정확하고 유용한 안전 문서를 찾아낼 수 있는 구체적이고 다양한 검색어(질문 형태)들을 생성하는 것이다.

[사용자 등록 정보]
- 행사명: {place_name}
- 행사 유형: {type}
- 개최 지역: {region}
- 행사 기간: {period}
- 행사 설명: {description}
- 카테고리: {category}
- 주최측 제공 자료: {related_documents}

[작업 수행 지침]
1. **정보 분석 및 위험 요소 추론:**
    - **행사 정보:** '{place_name}'({type}, {category})의 특성과 '{description}'에 명시된 주요 활동을 분석하여 잠재적 위험 요소를 예측한다.
    - **개최 지역:** '{region}'의 지리적, 환경적 특성을 고려하여 발생 가능한 안전사고 유형을 추론한다.
    - **행사 기간:** '{period}'를 분석하여 계절적(예: 폭염, 한파, 장마) 또는 시기적(예: 휴가철, 명절) 위험 요소를 예측한다.
    - **주최측 자료:** '{related_documents}'의 내용을 분석하여 핵심 안전 키워드를 추출하고, 검색어에 반영한다.

2. **다각적 검색어 생성:**
    - 위 분석 내용을 종합하여, 아래 4가지 관점의 검색어를 **최소 5개 이상** 생성한다. 각 검색어는 독립적으로 검색 가능한 완전한 질문 형태여야 한다.

    A. **[행사 유형 + 카테고리] 기반 검색어 (1~2개):**
        - (예시) "콘서트(음악) 행사 안전 관리 수칙"
        - (예시) "스포츠 경기장 방문객 안전 가이드라인"
        - (예시) "지역 축제(문화/예술) 안전사고 예방 대책"

    B. **[주요 활동 + 잠재위험] 기반 검색어 (2~3개):**
        - (예시) "행사장 내 체험 부스 시설물 안전 점검 항목"
        - (예시) "불꽃놀이 행사 시 화재 및 화상 예방 수칙"
        - (예시) "여름철 야외 행사 식중독 예방 방법"

    C. **[응급상황] 기반 검색어 (1~2개):**
        - (예시) "행사장에서 응급 환자 발생 시 대처 요령"
        - (예시) "다중 밀집 행사에서 압사 사고 예방 수칙"
        - (예시) "재난 상황 발생 시 행사 참여자 대피 안내 방법"

    D. **[지역 + 기간] 기반 검색어 (1개 이상):**
        - (예시) "해안가 지역 여름철 행사 진행 시 기상 악화 대처법"
        - (예시) "겨울철 야외 축제 방문객 한랭질환 예방 안내"

3. **최종 출력 형식 준수:**
    - 아래와 같이 JSON 형태로 출력하라. 반드시 `JsonOutputParser()`로 파싱 가능한 형태여야 한다.
    ```json
    {{
        "query": ["생성된 검색어 1", "생성된 검색어 2", "생성된 검색어 3", ..., "생성된 검색어 N"]
    }}
    ```
"""

generate_prompt_system = """
[역할 및 목표]
너는 대한민국 최고의 안전 전문가로서, 행사 주최자가 등록한 정보를 바탕으로 '방문객을 위한 맞춤형 안전 안내문'을 작성하는 AI다.
너의 임무는 아래 **[사용자 등록 정보]**와 **[검색된 안전 정보 문서]**를 종합하여, 방문객들이 행사를 안전하고 즐겁게 경험할 수 있도록 실용적이고 상세한 안전 가이드를 제공하는 것이다.

[사용자 등록 정보]
- 행사명: {place_name}
- 행사 유형: {type}
- 개최 지역: {region}
- 행사 기간: {period}
- 행사 설명: {description}
- 카테고리: {category}
- 주최측 제공 자료: {related_documents}
- 비상 연락 담당자: {emergency_contact_name}
- 비상 연락처: {emergency_contact_phone}

[검색된 안전 정보 문서]
{searched_documents}


[안내문 작성 가이드라인]

1. **제목:**
   - '`{place_name}` 방문객을 위한 안전 안내'와 같이 행사명을 포함하여 명확하게 작성한다.

2. **행사 개요 및 핵심 안전 수칙:**
   - `{place_name}` 행사는 `{period}` 동안 `{region}`에서 열리는 `{category}` 행사입니다. ({description} 내용 요약)
   - 방문객이 가장 먼저 인지해야 할 핵심 안전 수칙 2~3가지를 요약하여 강조한다.

3. **상세 안전 가이드:**
   A. **공통 안전 수칙:**
      - (예: 질서 유지, 개인 소지품 관리, 지정된 장소 외 흡연 금지)
   B. **행사 유형({type}) 및 카테고리({category})별 안전 수칙:**
      - 행사의 특성을 반영한 맞춤 안전 수칙을 제공한다. (예: 공연 - 압사 사고 예방, 축제 - 식중독 예방, 체험 - 시설물 안전)
   C. **장소/시설 관련 안전 수칙:**
      - 행사장 내 특정 구역(예: 무대 근처, 체험 부스)에서의 주의사항을 안내한다.
      - 비상 대피로, 소화기, 의무실 위치를 설명한다.
   D. **행사 기간({period}) 관련 안전 수칙:**
      - 행사 기간의 계절적, 시기적 요인(예: 여름, 겨울, 야간)에 따른 대비책을 안내한다.

4. **비상시 행동 요령:**
   - 화재, 응급환자 발생, 시설물 붕괴 등 비상 상황 시 대처 방법을 구체적으로 설명한다.
   - **비상 연락처:** 위급 상황 발생 시 즉시 `{emergency_contact_name}`({emergency_contact_phone})에게 연락하십시오.
   - 주변 경찰서, 소방서 등 공공 안전기관 연락처를 함께 안내하면 좋습니다.

5. **특별 고려 대상 안내:**
   - 어린이, 노약자, 장애인 등 안전 취약 계층을 위한 편의시설 및 유의사항을 안내한다.

[중요 규칙]
- **정확성:** 모든 정보는 **[사용자 등록 정보]**와 **[검색된 안전 정보 문서]**에 근거하여 작성한다.
- **명확성:** 누구나 쉽게 이해할 수 있는 간결하고 명확한 표현을 사용한다.
- **실용성:** 방문객이 실제 상황에서 즉시 활용할 수 있는 구체적인 행동 지침을 제공한다.
- **신뢰성:** 행사 주최측의 공식 안내문으로서 신뢰도를 줄 수 있는 어조를 사용한다.
- **구조화:** 제목, 소제목, 글머리 기호 등을 활용하여 가독성을 높인다.

[출력 형식]
- Markdown을 사용하여 명확하게 구조화된 안내문을 생성한다.
"""

final_answer_system = \
"""
당신은 텍스트를 깔끔하게 정리하는 AI입니다.
주어진 [generation]는 내용과 citation([숫자])은 정확하지만, 줄바꿈이나 글머리 기호 같은 서식이 모두 사라진 상태입니다.

당신의 임무는 [generation]의 내용과 citation을 유지하면서, 사용자가 읽기 쉽도록 서식을 복원하는 것입니다.

**[규칙]**
1.  **내용 보존:** 원본 텍스트의 모든 단어와 citation([숫자])을 빠짐없이 사용해야 합니다.
2.  **서식 복원:** 문맥에 맞게 줄바꿈, 글머리 기호(*), 제목(##) 등의 마크다운 서식을 자연스럽게 추가해주세요.
3.  **내용 수정 금지:** 원본에 없는 내용을 추가하거나, 기존 내용을 변경하지 마세요.
4.  **출력:** 다른 설명 없이, 서식이 복원된 최종 텍스트만 출력합니다.
5.  **인용 위치:** citation([숫자])은 문장이나 문단의 끝에만 표시하고, 마지막 문단이나 결론 부분에는 citation을 표시하지 않습니다.
"""


class GraphState(TypedDict):
    generation: Annotated[str, "LLM generated answer"]
    hallu_check: Annotated[dict, "Hallucination check result"]
    query_list: Annotated[List[str], "Query list"]
    final_answer: Annotated[str, "Final answer"]

    place_name: Annotated[str, "Place name"]
    type: Annotated[str, "Type of place/event"]
    region: Annotated[str, "Region"]
    period: Annotated[str, "Period of event"]
    description: Annotated[str, "Description"]
    category: Annotated[str, "Category of event"]
    related_documents: Annotated[str, "Related documents from user"]
    emergency_contact_name: Annotated[str, "Emergency contact name"]
    emergency_contact_phone: Annotated[str, "Emergency contact phone"]
    searched_documents: Annotated[List[str], "Documents from vector store"]

class form_chain():
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
            model_name="gemini-2.5-flash", # Grounding을 지원하는 최신 모델 사용 권장
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
    def query_generate(self, place_name: str, type: str, region: str, period: str, description: str, category: str, related_documents: str) -> str:
        generate_prompt = ChatPromptTemplate.from_template(query_generator_system)
        chain = generate_prompt | self.generate_llm | JsonOutputParser()
        query_list = chain.invoke({
            "place_name": place_name,
            "type": type,
            "region": region,
            "period": period,
            "description": description,
            "category": category,
            "related_documents": related_documents
        })
        return query_list.get("query")
    
    def generate(self, place_name: str, type: str, region: str, period: str, description: str, category: str, related_documents: str, emergency_contact_name: str, emergency_contact_phone: str, searched_documents: List[str]) -> str:
        generate_prompt = ChatPromptTemplate.from_template(generate_prompt_system)
        chain = generate_prompt | self.generate_llm | StrOutputParser()
        generation = chain.invoke({
            "place_name": place_name,
            "type": type,
            "region": region,
            "period": period,
            "description": description,
            "category": category,
            "related_documents": related_documents,
            "emergency_contact_name": emergency_contact_name,
            "emergency_contact_phone": emergency_contact_phone,
            "searched_documents": searched_documents
        })
        return generation

    def clean_up_answer_with_citations(self, generation: str) -> str:
        prompt = ChatPromptTemplate.from_messages([("system", final_answer_system), ("user", "[generation]: \n{generation}")])
        chain = prompt | self.final_llm | StrOutputParser()
        return chain.invoke({"generation": generation})


########################################################################################################################
# 그래프 노드 정의
########################################################################################################################
 
    def query_generator(self, state: GraphState) -> GraphState:
        query_list = self.query_generate(
            state["place_name"],
            state["type"],
            state["region"],
            state["period"],
            state["description"],
            state["category"],
            state["related_documents"]
        )
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
            if doc.page_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc.page_content)
        return {**state, "searched_documents": unique_documents}

    def generator(self, state: GraphState) -> GraphState:
        generation = self.generate(
            state["place_name"],
            state["type"],
            state["region"],
            state["period"],
            state["description"],
            state["category"],
            state["related_documents"],
            state["emergency_contact_name"],
            state["emergency_contact_phone"],
            state["searched_documents"]
        )
        return {**state, "generation": generation}

    def hallu_checker(self, state: GraphState) -> GraphState:
        hallu_check = self.checker.invoke(
            state["generation"],
            config = {"configurable": {"documents": state["searched_documents"]}}
        )
        return {**state, "hallu_check": hallu_check}

    def check_hallu_complete_form(self, state: GraphState) -> str:
        """
        Grounding 확인 결과를 바탕으로 다음 단계를 결정합니다.
        """
        result = state["hallu_check"]
        score = result.support_score
        # 평균 점수가 0.5를 넘으면 종료, 아니면 재시도
        if score >= 0.5:
            return END
        else:
            return "query_generator"

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
        self.graph.add_node("query_generator", self.query_generator)
        self.graph.add_node("search_document", self.search_document)
        self.graph.add_node("generator", self.generator)
        self.graph.add_node("hallu_checker", self.hallu_checker)
        self.graph.add_node("answer_beautifier", self.answer_beautifier)

        self.graph.add_edge(START, "query_generator")
        self.graph.add_edge("query_generator", "search_document")
        self.graph.add_edge("search_document", "generator")
        self.graph.add_edge("generator", "hallu_checker")
        # self.graph.add_conditional_edges(
        #     "hallu_checker",
        #     self.check_hallu_complete_form,
        #     {
        #         END: END,
        #         "query_generator": "query_generator"
        #     }
        # )
        self.graph.add_edge("hallu_checker", "answer_beautifier")
        self.graph.add_edge("answer_beautifier", END)
        # 컴파일된 그래프 반환
        return self.graph.compile(checkpointer=MemorySaver())
