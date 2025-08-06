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
import os
import warnings
import json
from datetime import datetime
from .vector_store import store_vector_db
from .case_vector_store import store_vector_db as case_vector_db
from .get_weather import get_weather_forecast
import requests



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
doc_vector_store = store_vector_db()
doc_vector_store.load_vector_store("./vector/data/vector_store/doc")

case_climate = case_vector_db()
case_climate.load_vector_store("./vector/data/vector_store/case_climate")

case_festival = case_vector_db()
case_festival.load_vector_store("./vector/data/vector_store/case_festival")



weather_query_generator_system = """
[역할 및 목표]
너는 기상 관련 사고 사례 분석 전문가로서, 주어진 기상 예보를 분석하여 유사한 기상 조건에서 발생했던 과거 사고 사례를 검색하기 위한 최적의 검색어를 생성하는 AI다.

[기상 정보 분석 및 평가 기준]
1.  **기온 및 체감온도**: 현재 월의 평년 기온과 예보된 기온을 비교, 추위/더위 수준을 평가한다. 기온과 체감온도의 차이가 3°C 이상 날 경우 그 원인(높은 습도, 강한 바람 등)과 관련 위험을 언급한다.
2.  **강수량**: 아래의 시간당 강수량 기준에 따라 위험도를 평가한다.
    * **~3mm:** 약한 비 (일상생활에 큰 지장 없음)
    * **3~15mm:** 보통 비 (시야 제한, 도로 미끄러움 시작)
    * **15~30mm:** 강한 비 (배수 불량 지역 침수 가능성, 시야 확보 어려움)
    * **30mm 이상:** 매우 강한 비 (호우특보 수준, 침수 및 교통 통제 위험 급증)
3.  **바람 (풍속)**: 대한민국 내륙 평균 풍속(약 2-3m/s 또는 7-11km/h)과 비교하여 세기를 평가한다. 초속 10m/s(36km/h) 이상 시 강풍으로 인한 시설물 파손 및 보행자 위험을 언급한다.
4.  **UV 지수**: 기상청 UV 지수 단계(0-2 낮음, 3-5 보통, 6-7 높음, 8-10 매우 높음, 11+ 위험)를 기준으로 피부 노출 시 위험성을 평가한다.
5.  **습도**: 쾌적 습도 범위(40-60%)와 비교하여 '건조', '쾌적', '다소 습함', '높음(불쾌지수 상승)' 등으로 상태를 평가하고, 높은 습도와 기온이 결합될 때의 온열질환 위험을 언급한다.
6.  **구름/시정**: 구름 양이 많거나(80% 이상) 강수 시, 시정(가시거리) 악화로 인한 교통사고 위험을 분석한다.


[작업 절차]
1.  주어진 `[24시간 내 기상 예보]`를 위의 `[기상 정보 분석 및 평가 기준]`에 따라 체계적으로 분석한다.
2.  분석 내용을 바탕으로 시간대별 주요 날씨 특징, 위험 요소를 포함하여 자연스러운 문장으로 `weather_summary`를 작성한다. 특히 위험 수준이 '보통'을 초과하는 모든 기상 요소를 구체적으로 명시한다.
3.  분석된 각 위험 요소와 직접적으로 연관된 과거 사고 사례를 찾을 수 있도록, **구체적인 수치를 포함한 검색어와 현상을 설명하는 포괄적인 검색어를 균형 있게 조합하여** `query`를 5개 생성한다.

[현재 시간]
{current_time}

[24시간 내 기상 예보]
{weather_forecast}

[출력 형식]
```json
{{
    "weather_summary": "오늘 날씨는 오전에는 20도로 선선하다가 오후부터 35도까지 기온이 올라갑니다. 이는 8월 대한민국 평년기온인 26.1도보다 매우 높은 수준입니다. 특히 습도가 75%로 높아 체감온도는 38도에 육박하여 온열질환 위험이 매우 큽니다. 오후 3시부터는 강우 확률이 80%로 높아지며, 시간당 강수량이 20mm에 달하는 '강한 비'가 예상되어 시야 확보가 어렵고 저지대 침수 가능성이 있습니다. 풍속은 15km/h로 평소보다 다소 강하며, UV 지수는 8로 '매우 높음' 단계이므로 낮 시간대 외출 시 각별한 주의가 필요합니다.",
    "query": [
        "폭염 온열질환자 발생 사례",
        "여름철 국지성 호우 침수 피해",
        "시간당 20mm 강수량 도로 통제 사례",
        "빗길 교통사고 시야 미확보",
        "강풍 간판 및 시설물 낙하 사고",
        "자외선 지수 매우 높음 야외활동 주의사항"
    ]
}}
```
"""

festival_query_generator_system = """
[역할 및 목표]
너는 주어진 행사 정보에서 핵심 키워드를 조합하여, 과거의 유사 사고 사례를 검색하기 위한 최적의 검색어(query)를 생성하는 '사고 사례 검색어 생성기'다. 너의 유일한 목표는 효과적인 검색어 목록을 만드는 것이다.

[필수 입력 정보]
- 행사명: {place_name}
- 행사장소: {location}
- 행사 유형: {type}
- 행사 기간: {period}
- 예상 참가자 규모: {expected_attendees}명
- 행사 설명: {description}
- 주최측 제공 자료: {related_documents}

[작업 절차]
1. `[필수 입력 정보]`에서 행사 유형, 장소, 규모, 주요 활동과 관련된 핵심 키워드를 추출한다.
2. 아래 `[쿼리 생성 패턴]`에 따라 키워드들을 조합하여, 과거 사고 사례를 찾기 위한 구체적인 검색어 5개를 생성한다.

[쿼리 생성 패턴]
A. **[행사 유형] + [사고 종류]**
   - 예: "음악 페스티벌" + "압사 사고", "지역 축제" + "식중독"
B. **[행사장소] + [시설물] + [사고 종류]**
   - 예: "한강공원" + "임시 무대" + "붕괴", "실내체육관" + "전기 설비" + "화재"
C. **[참가자 규모] + [사고 종류]**
   - 예: "1만명 이상" + "질서유지 실패", "대규모 행사" + "입장 병목"
D. **[행사 활동] + [사고 종류]**
   - 예: "불꽃놀이" + "화상", "마라톤" + "탈수", "EDM 파티" + "소음 민원"
E. **[시기/계절] + [행사 유형] + [사고 종류]**
   - 예: "여름" + "야외 페스티벌" + "온열질환", "겨울" + "얼음 축제" + "익사"

[출력 형식]
```json
{{
    "query": [
        "쿼리 1",
        "쿼리 2",
        "쿼리 3",
        "쿼리 4",
        "쿼리 5",
        "..."
    ]
}}
```
"""



query_generator_system = """
[역할 및 목표]
너는 사용자의 행사 등록 정보를 분석하여, 발생 가능한 모든 안전 위협을 예측하고 관련 정보를 검색하기 위한 '다각적 검색어 생성 AI'다.
너의 임무는 아래 **[사용자 등록 정보]**를 바탕으로, Vector Store에서 가장 정확하고 유용한 안전 문서를 찾아낼 수 있는 구체적이고 다양한 검색어(질문 형태)들을 생성하는 것이다.

[사용자 등록 정보]
- 행사명: {place_name}
- 행사 유형: {type}
- 개최 지역: {location}
- 행사 기간: {period}
- 행사 설명: {description}
- 카테고리: {category}
- 예상 참가자 규모: {expected_attendees}명
- 주최측 제공 자료: {related_documents}

[작업 수행 지침]
1. **정보 분석 및 위험 요소 추론:**
    - **행사 정보:** '{place_name}'({type}, {category})의 특성과 '{description}'에 명시된 주요 활동을 분석하여 잠재적 위험 요소를 예측한다.
    - **개최 지역:** '{location}'의 지리적, 환경적 특성을 고려하여 발생 가능한 안전사고 유형을 추론한다.
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
- 개최 지역: {location}
- 행사 기간: {period}
- 행사 설명: {description}
- 카테고리: {category}
- 주최측 제공 자료: {related_documents}
- 비상 연락 담당자: {emergency_contact_name}
- 비상 연락처: {emergency_contact_phone}
- 예상 참가자 규모: {expected_attendees}
- current_time = {current_time}
- weather_summary = {weather_summary}

[날씨 관련 사례 문서]
{weather_searched_documents}

[행사 관련 사례 문서]
{festival_searched_documents}

[안전 관련 정보]
{safety_searched_documents}


[안내문 작성 가이드라인]
1. **제목:** '{place_name} 안전 안내문'

2. **핵심 안전 수칙:**
   - 행사 기본 정보 요약
   - 필수 안전 수칙 2-3개

3. **상세 안내:**
   - 공통 안전 수칙
   - 행사 특성별 주의사항
   - 시설/장소 관련 안내
   - 날씨 관련 주의사항

4. **비상 연락망:**
   - 주최측: {emergency_contact_name}({emergency_contact_phone})
   - 공공기관: 112, 119

5. **유사 행사 사고 사례:**
   - 검색된 문서에서 관련된 유사 행사 사고 사례가 없을 경우 반드시 "현재 행사와 관련된 유사 사고 사례가 발견되지 않았습니다."라고 명시적으로 표시
   - 관련 사례가 있는 경우에만 [행사 관련 사례 문서]에서 발췌한 주요 사고 사례 나열
   - 관련 사례가 있는 경우에만 각 사례별 예방 대책 및 주의사항 제시

6. **날씨 관련 주의사항:**
   - weather_summary를 활용하여 현재 날씨 특징 언급
   - 검색된 문서에서 관련된 날씨 사고 사례가 없을 경우 반드시 "현재 날씨와 유사한 조건에서의 사고 사례가 발견되지 않았습니다."라고 명시적으로 표시
   - 관련 사례가 있는 경우에만 [날씨 관련 사례 문서] 기반 유사 기상에서의 사고 사례 제시
   - 기상 조건에 따른 구체적 대비 방안 제시

[중요 규칙]
- 정확성: 제공된 문서 기반 작성
- 명확성: 이해하기 쉬운 표현
- 실용성: 구체적 행동 지침
- 구조화: 가독성 높은 형식

[출력 형식]
- Markdown 형식
"""

final_answer_system = \
"""
당신은 텍스트를 깔끔하게 정리하는 AI입니다.
주어진 [generation]는 내용과 citation([숫자])은 정확하지만, 줄바꿈이나 글머리 기호 같은 서식이 모두 사라진 상태입니다.

당신의 임무는 [generation]의 내용과 citation을 유지하면서, 사용자가 읽기 쉽도록 서식을 복원하는 것입니다.
"""


class GraphState(TypedDict):
    generation: Annotated[str, "LLM generated answer"]
    hallu_check: Annotated[dict, "Hallucination check result"]
    festival_query_list: Annotated[List[str], "Festival Query list"]
    weather_query_list: Annotated[List[str], "Weather Query list"]
    safety_query_list: Annotated[List[str], "Safety Query list"]
    # final_answer: Annotated[str, "Final answer"]

    place_name: Annotated[str, "Place name"]
    type: Annotated[str, "Type of place/event"]
    location: Annotated[str, "Location"]
    period: Annotated[str, "Period of event"]
    description: Annotated[str, "Description"]
    category: Annotated[str, "Category of event"]
    related_documents: Annotated[str, "Related documents from user"]
    emergency_contact_name: Annotated[str, "Emergency contact name"]
    emergency_contact_phone: Annotated[str, "Emergency contact phone"]
    festival_searched_documents: Annotated[List[str], "Festival Documents from vector store"]
    weather_searched_documents: Annotated[List[str], "Weather Documents from vector store"]
    safety_searched_documents: Annotated[List[str], "Safety Documents from vector store"]
    current_time: Annotated[str, "Current time"]
    weather_summary: Annotated[str, "Weather summary"]
    expected_attendees: Annotated[str, "Expected attendees"]


class custom_form_chain():
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
        self.weather_api = os.getenv("GEOCODING_API")
########################################################################################################################
# 체인 정의
########################################################################################################################
    def festival_query_generate(self, place_name: str, location: str, type: str, period: str, expected_attendees: str, description: str, related_documents: str) -> str:
        generate_prompt = ChatPromptTemplate.from_template(festival_query_generator_system)
        chain = generate_prompt | self.fast_llm | JsonOutputParser()
        festival_query_list = chain.invoke({
            "place_name": place_name,
            "location": location,
            "type": type,
            "period": period,
            "expected_attendees": expected_attendees,
            "description": description,
            "related_documents": related_documents
        })
        return festival_query_list.get("query")


    def weather_query_generate(self, address: str) -> str:
        generate_prompt = ChatPromptTemplate.from_template(weather_query_generator_system)
        chain = generate_prompt | self.fast_llm | JsonOutputParser()
        weather_query_list = chain.invoke(
            {
                'weather_forecast':get_weather_forecast(address,self.weather_api),
                'current_time':datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        return weather_query_list.get("query"), weather_query_list.get("weather_summary")

    def query_generate(self, palce_name: str, type: str, location: str, period: str, expected_attendees: str, description: str, related_documents: str, category: str) -> str:
        generate_prompt = ChatPromptTemplate.from_template(query_generator_system)
        chain = generate_prompt | self.fast_llm | JsonOutputParser()
        query_list = chain.invoke({
            "place_name": palce_name,
            "type": type,
            "location": location,
            "period": period,
            "expected_attendees": expected_attendees,
            "description": description,
            "related_documents": related_documents,
            "category": category
        })
        return query_list.get("query")


    def generate(self, place_name: str, type: str, location: str, period: str, description: str, category: str, related_documents: str, emergency_contact_name: str, emergency_contact_phone: str, expected_attendees: str, current_time: str, weather_summary: str, weather_searched_documents: List[str], festival_searched_documents: List[str], safety_searched_documents: List[str]) -> str:
        generate_prompt = ChatPromptTemplate.from_template(generate_prompt_system)
        chain = generate_prompt | self.generate_llm | StrOutputParser()
        generation = chain.invoke({
            "place_name": place_name,
            "type": type,
            "location": location,
            "period": period,
            "description": description,
            "category": category,
            "related_documents": related_documents,
            "emergency_contact_name": emergency_contact_name,
            "emergency_contact_phone": emergency_contact_phone,
            "expected_attendees": expected_attendees,
            "current_time": current_time,
            "weather_summary": weather_summary,
            "weather_searched_documents": weather_searched_documents,
            "festival_searched_documents": festival_searched_documents,
            "safety_searched_documents": safety_searched_documents
        })
        return generation
        

    # def clean_up_answer_with_citations(self, generation: str) -> str:
    #     prompt = ChatPromptTemplate.from_messages([("system", final_answer_system), ("user", "[generation]: \n{generation}")])
    #     chain = prompt | self.final_llm | StrOutputParser()
    #     return chain.invoke({"generation": generation})


########################################################################################################################
# 그래프 노드 정의
########################################################################################################################
 
    def festival_query_generator(self, state: GraphState) -> GraphState:
        festival_query_list = self.festival_query_generate(
            state["place_name"],
            state["location"],
            state["type"],
            state["period"],
            state["expected_attendees"],
            state["description"],
            state["related_documents"]
        )
        return {**state, "festival_query_list": festival_query_list}


    def weather_query_generator(self, state: GraphState) -> GraphState:
        weather_query_list, weather_summary = self.weather_query_generate(
            state["location"]
        )
        return {**state, "weather_query_list": weather_query_list, "weather_summary": weather_summary}


    def safety_query_generator(self, state: GraphState) -> GraphState:
        safety_query_list = self.query_generate(
            state["place_name"],
            state["type"],
            state["location"],
            state["period"],
            state["expected_attendees"],
            state["description"],
            state["related_documents"],
            state["category"]
        )
        return {**state, "safety_query_list": safety_query_list}

    def search_safety_document(self, state: GraphState) -> GraphState:
        documents = []
        for query in state["safety_query_list"]:
            searched_documents = doc_vector_store.search(query, k=10)
            reranked_documents = doc_vector_store.document_rerank(query, searched_documents, k=3)
            documents.extend(reranked_documents)
        # 중복 문서 제거
        unique_documents = []
        seen_contents = set()
        for doc in documents:
            if doc.page_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc.page_content)
        return {**state, "safety_searched_documents": unique_documents}

    def search_weather_document(self, state: GraphState) -> GraphState:
        documents = []
        for query in state["weather_query_list"]:
            searched_documents = case_climate.search(query, k=3)
            reranked_documents = case_climate.document_rerank(query, searched_documents, k=1)
            documents.extend(reranked_documents)
        # 중복 문서 제거
        unique_documents = []
        seen_contents = set()
        for doc in documents:
            if doc.page_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc.page_content)
        return {**state, "weather_searched_documents": unique_documents}
    
    def search_festival_document(self, state: GraphState) -> GraphState:
        documents = []
        for query in state["festival_query_list"]:
            searched_documents = case_festival.search(query, k=3)
            reranked_documents = case_festival.document_rerank(query, searched_documents, k=1)
            documents.extend(reranked_documents)

        # 중복 문서 제거
        unique_documents = []
        seen_contents = set()
        for doc in documents:
            if doc.page_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc.page_content)
        return {**state, "festival_searched_documents": unique_documents}



    def generator(self, state: GraphState) -> GraphState:
        # 현재 시간이 없으면 생성
        current_time = state.get('current_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        generation = self.generate(
            state["place_name"],
            state["type"],
            state['location'],
            state['period'],
            state['description'],
            state['category'],
            state['related_documents'],
            state['emergency_contact_name'],
            state['emergency_contact_phone'],
            state['expected_attendees'],
            current_time,
            state.get('weather_summary', ''),
            state.get('weather_searched_documents', []),
            state.get('festival_searched_documents', []),
            state.get('safety_searched_documents', [])
        )
        return {**state, "generation": generation}

    def hallu_checker(self, state: GraphState) -> GraphState:
        # 모든 검색된 문서를 합침
        all_documents = []
        if state.get("safety_searched_documents"):
            all_documents.extend(state["safety_searched_documents"])
        if state.get("weather_searched_documents"):
            all_documents.extend(state["weather_searched_documents"])
        if state.get("festival_searched_documents"):
            all_documents.extend(state["festival_searched_documents"])
            
        hallu_check = self.checker.invoke(
            state["generation"],
            config = {"configurable": {"documents": all_documents}}
        )
        return {**state, "hallu_check": hallu_check}

    # def answer_beautifier(self, state: GraphState) -> GraphState:
    #     final_answer = state["hallu_check"].answer_with_citations
    #     final_answer = self.clean_up_answer_with_citations(final_answer)
    #     return {**state, "final_answer": final_answer}

########################################################################################################################
# 그래프 노드 연결
########################################################################################################################

    def link_nodes(self):
        """
        LangGraph의 노드들을 연결하여 그래프를 구성합니다.
        """
        self.graph.add_node("festival_query_generator", self.festival_query_generator)  
        self.graph.add_node("weather_query_generator", self.weather_query_generator)
        self.graph.add_node("safety_query_generator", self.safety_query_generator)
        self.graph.add_node("search_safety_document", self.search_safety_document)
        self.graph.add_node("search_weather_document", self.search_weather_document)
        self.graph.add_node("search_festival_document", self.search_festival_document)
        self.graph.add_node("generator", self.generator)
        self.graph.add_node("hallu_checker", self.hallu_checker)
        # self.graph.add_node("answer_beautifier", self.answer_beautifier)

        self.graph.add_edge(START,'festival_query_generator')
        self.graph.add_edge('festival_query_generator','search_festival_document')
        self.graph.add_edge('search_festival_document','weather_query_generator')
        self.graph.add_edge('weather_query_generator','search_weather_document')
        self.graph.add_edge('search_weather_document','safety_query_generator')
        self.graph.add_edge('safety_query_generator','search_safety_document')
        self.graph.add_edge('search_safety_document','generator')
        self.graph.add_edge("generator", "hallu_checker")
        self.graph.add_edge("hallu_checker", END)
        # self.graph.add_edge("hallu_checker", "answer_beautifier")
        # self.graph.add_edge("answer_beautifier", END)

        # 컴파일된 그래프 반환
        return self.graph.compile(checkpointer=MemorySaver())
