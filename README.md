# 동적 RAG 시스템 기반 맞춤형 안전 안내문 생성기

이 프로젝트는 사용자가 입력한 행사 정보와 참고 문서를 바탕으로, 발생 가능한 모든 위험 요소를 다각적으로 분석하여 맞춤형 안전 안내문을 생성하는 **동적 RAG(Retrieval-Augmented Generation, 검색 증강 생성) 시스템**입니다.

Google의 Gemini 모델, LangChain, LangGraph 프레임워크를 기반으로 구축되었으며, 단순한 정보 검색을 넘어 **위치 기반 날씨 예측, 과거 사고 사례 분석, 안전 매뉴얼 검색** 등 여러 정보 소스를 동적으로 결합하여 실용적이고 신뢰도 높은 안전 가이드를 제공하는 것을 목표로 합니다.

**※ RAG 시스템 구축에 필요한 데이터는 [Hugging Face Datasets](https://huggingface.co/datasets/bong9513/safety_rag_data)에서 다운로드할 수 있습니다.**

## 핵심 기능 및 아키텍처

본 시스템은 사용자의 요청을 처리하기 위해 정교하게 설계된 워크플로우를 따릅니다. 핵심 아키텍처는 다음과 같습니다.

![System Architecture](https://i.imgur.com/your-architecture-diagram.png)  <!-- TODO: 아키텍처 다이어그램 이미지 링크 추가 -->

### 1. 다중 소스 컨텍스트 분석 (Multi-Source Context Analysis)

안내문 생성의 첫 단계는 사용자가 입력한 정보를 입체적으로 분석하여 검색에 필요한 키워드를 추출하는 것입니다. 시스템은 세 가지 독립적인 정보 소스를 활용합니다.

-   **가. 위치 기반 날씨 정보 분석 (Location-based Weather Analysis)**
    -   **Geocoding**: 사용자가 입력한 행사장 주소(`location`)를 **Google Geocoding API**를 통해 위도와 경도로 변환합니다.
    -   **날씨 예보 조회**: 변환된 좌표를 사용하여 **Open-Meteo API**에서 향후 24시간의 상세한 기상 예보(기온, 강수량, 풍속, UV 지수 등)를 가져옵니다.
    -   **LLM 기반 날씨 위험 분석**: LLM(Gemini Flash)이 기상 예보 데이터를 분석하여, 폭염, 호우, 강풍 등 잠재적 위험 요소를 식별하고, 관련 과거 사고 사례를 검색하기 위한 검색어(예: "여름철 국지성 호우 침수 피해")와 날씨 요약 정보를 생성합니다.

-   **나. 행사 정보 기반 사고 사례 분석 (Event-based Case Analysis)**
    -   LLM이 행사명, 유형, 규모, 설명 등 사용자가 입력한 정보를 분석합니다.
    -   행사의 특성과 관련된 과거 사고 사례를 검색하기 위한 검색어(예: "음악 페스티벌 압사 사고", "야외 행사장 임시 무대 붕괴")를 생성합니다.

-   **다. 사용자 제공 문서 분석 (OCR for User-Provided Documents)**
    -   사용자가 PDF 형태의 참고 자료(행사 타임테이블, 안내도 등)를 제출하면, **Google Cloud Document AI (OCR)**를 통해 텍스트를 추출합니다.
    -   추출된 텍스트는 LLM이 행사 내용을 더 깊이 이해하고, 안전 수칙을 생성하는 데 중요한 추가 컨텍스트로 활용됩니다.

### 2. 다중 벡터 DB 기반 정보 검색 (Multi-VectorStore Retrieval)

시스템은 목적에 따라 특화된 세 개의 독립적인 FAISS 벡터 데이터베이스를 운영하여 검색의 정확성과 효율성을 극대화합니다.

1.  **안전 매뉴얼 DB (`doc_vector_store`)**: 정부 및 공공기관에서 배포한 각종 안전 매뉴얼과 가이드라인이 저장되어 있습니다. 일반적인 안전 수칙과 절차에 대한 정보를 제공합니다.
2.  **기후 관련 사고사례 DB (`case_climate`)**: 기상 조건과 관련된 과거 재난 및 사고 사례 데이터가 저장되어 있습니다. 날씨 위험 분석 단계에서 생성된 검색어로 이곳을 검색합니다.
3.  **행사 관련 사고사례 DB (`case_festival`)**: 국내외 다양한 행사에서 발생했던 사고 사례 데이터가 저장되어 있습니다. 행사 정보 분석 단계에서 생성된 검색어로 이곳을 검색합니다.

각 분석 단계에서 생성된 검색어들은 해당하는 벡터 DB에 전달되어, **하이브리드 검색(의미 기반 + 키워드 기반)과 Vertex AI Reranker**를 통해 가장 관련성 높은 문서를 추출합니다.

### 3. LangGraph 기반 동적 안내문 생성 (Dynamic Generation via LangGraph)

추출된 모든 정보(날씨 요약, 날씨 관련 사고 사례, 행사 관련 사고 사례, 안전 매뉴얼)는 최종적으로 **LangGraph**로 구성된 에이전트에게 전달됩니다.

-   **구조화된 프롬프트**: LLM(Gemini Flash)은 '안전 전문가' 역할을 부여받고, 구조화된 프롬프트 템플릿에 따라 모든 정보를 종합합니다.
-   **체계적인 안내문 생성**: LLM은 환영 인사, 핵심 안전 수칙, 상세 안내, 비상 연락망, 그리고 **검색된 사고 사례와 날씨 관련 주의사항**을 포함한 체계적이고 상세한 맞춤형 안전 안내문을 생성합니다.
-   **신뢰도 검증**: 생성된 내용의 신뢰도를 높이기 위해 Vertex AI의 **Grounding 기능**을 활용하여, 답변이 검색된 문서에 기반했는지 확인하고 인용(Citation)을 추가합니다.

## 기술 스택

-   **언어**: Python 3
-   **핵심 프레임워크**:
    -   `LangChain` & `LangGraph`: RAG 파이프라인 및 복잡한 Agent 로직 구성
    -   `FastAPI`: API 서버 구축
-   **AI & LLM**:
    -   `google-generativeai` & `google-cloud-aiplatform`: Google Gemini & Vertex AI 모델 활용
    -   `langchain-google-vertexai`: LangChain과 Vertex AI 통합
    -   **Google Geocoding API**: 주소 -> 좌표 변환
    -   **Google Cloud Document AI**: PDF 문서 OCR
-   **데이터 처리 및 저장**:
    -   `FAISS`: 벡터 데이터베이스 (유사도 검색)
    -   `Open-Meteo API`: 날씨 데이터 조회
    -   `pdfplumber`: PDF 텍스트 추출
-   **배포**:
    -   `Docker` & `docker-compose`: 컨테이너 기반 배포 및 실행 환경
    -   `uvicorn`: ASGI 서버

## API 사용 가이드

FastAPI를 통해 3개의 주요 API 엔드포인트를 제공합니다.

### 1. `/api/custom_form`

**가장 핵심적인 API**로, 사용자가 입력한 행사 정보를 바탕으로 위에서 설명한 모든 과정을 거쳐 맞춤형 안전 안내문을 생성합니다.

-   **Python 코드 예제**:
    ```python
    import requests

    custom_form_url = "http://127.0.0.1:8000/api/custom_form"
    payload = {
        "place_name": "2025 한강 여름 뮤직 페스티벌",
        "type": "대규모 야외 공연",
        "location": "서울 여의도 한강공원",
        "period": "2025년 8월 8일 ~ 2025년 8월 10일",
        "description": "EDM, 힙합, 록 등 다양한 장르의 국내외 아티스트들이 참여하는 3일간의 음악 축제. 푸드트럭 존과 불꽃놀이가 예정되어 있습니다.",
        "category": "음악/페스티벌",
        "emergency_contact_name": "종합상황실 안전관리팀",
        "emergency_contact_phone": "02-123-4567",
        "expected_attendees": "일일 20,000명",
        # "related_documents": "..." # PDF 파일을 base64 인코딩하여 문자열로 전달
    }

    response = requests.post(custom_form_url, json=payload)
    result = response.json()
    print(result.get('generation'))
    ```

### 2. `/api/chat` 및 `/api/form_chat`

일반적인 질의응답 및 생성된 안내문에 대한 후속 질문을 처리하는 RAG API입니다.

---
*더 상세한 프로젝트 구조, 설치 및 실행 방법은 아래 내용을 참고하세요.*
(기존 README 내용 유지)