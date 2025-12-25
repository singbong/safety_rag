# PDF 기반 질의응답 RAG 시스템

이 프로젝트는 PDF 문서 내용을 기반으로 사용자의 질문에 답변하는 RAG(Retrieval-Augmented Generation, 검색 증강 생성) 시스템입니다. Google Gemini 모델과 LangChain 프레임워크를 활용하여 구현되었습니다.


**※ [RAG] 구축에 필요한 데이터는 https://huggingface.co/datasets/bong9513/safety_rag_data 에서 다운로드 받을 수 있습니다.**
**※ 이 프로젝트는 https://github.com/sumkyun/safety-navigator 와 연결되어 있습니다.**

## 주요 기능

- **문서 처리**: PDF 파일을 텍스트로 변환하고 의미 기반으로 분할(Semantic Chunking)합니다.
- **벡터 임베딩**: 처리된 텍스트를 벡터로 변환하여 FAISS 벡터 스토어에 저장합니다.
- **질의응답**: 저장된 벡터 데이터를 기반으로 사용자의 질문에 가장 관련성 높은 답변을 생성합니다.
- **다양한 인터페이스**: 일반 채팅, 양식(Form) 기반 채팅 등 여러 종류의 대화 체인을 제공합니다.
- **컨테이너 기반**: Docker 및 Docker Compose를 사용하여 프로젝트 환경을 쉽게 구성하고 실행할 수 있습니다.

## 기술 스택

-   **언어**: Python 3
-   **핵심 프레임워크**:
    -   `LangChain` & `LangGraph`: RAG 파이프라인 및 복잡한 Agent 로직 구성
    -   `FastAPI`: API 서버 구축
-   **AI**:
    -   `google-generativeai` & `google-cloud-aiplatform`: Google Gemini 모델 API 활용
    -   `langchain-google-vertexai`: LangChain과 Vertex AI 통합
    -   `google-cloud-vision`: Google Cloud Vision API (OCR 처리)
    -   `FAISS`: 벡터 데이터베이스 (유사도 검색)
    -   `transformers` & `sentence-transformers`: 자연어 처리 및 임베딩
-   **데이터 처리**:
    -   `pdfplumber`: PDF 텍스트 추출
    -   `pandas`, `numpy`: 데이터 조작 및 분석
-   **배포**:
    -   `Docker` & `docker-compose`: 컨테이너 기반 배포 및 실행 환경
    -   `uvicorn`: ASGI 서버

## 프로젝트 구조

```
.
├── .env                    # API 키, 프로젝트 ID 등 환경 변수 설정 파일
├── .git/                   # Git 버전 관리 시스템 디렉토리
├── .gitignore              # Git 추적 제외 목록 파일
├── api_example.ipynb       # API 사용법 예시를 담은 Jupyter Notebook
├── docker-compose.yml      # Docker 다중 컨테이너 실행을 위한 설정 파일
├── Dockerfile              # 애플리케이션 Docker 이미지 빌드 설정 파일
├── fire_app.py             # FastAPI 서버 실행 및 CLI 명령어 처리를 위한 메인 스크립트
├── README.md               # 프로젝트 설명 문서 (현재 파일)
├── requirements.txt        # Python 패키지 의존성 목록
└── vector/                 # 데이터 처리 및 RAG 로직 관련 디렉토리
    ├── data/               # 데이터 저장 디렉토리
    │   ├── chunked_docs/   # 의미 기반으로 분할된 문서 조각(.pkl) 저장 위치
    │   ├── contextual_content_docs/ # 컨텍스트 요약이 추가된 문서 조각(.pkl) 저장 위치
    │   ├── original_full_text/    # PDF에서 추출된 원본 텍스트(.txt) 저장 위치
    │   ├── original_pdf/          # 분석할 원본 PDF 문서 저장 위치
    │   └── vector_store/          # 생성된 FAISS 벡터 DB(.index, .pkl) 저장 위치
    └── definition/         # RAG 파이프라인의 핵심 로직 정의 디렉토리
        ├── __pycache__/           # Python 컴파일 캐시 파일 디렉토리
        ├── arcane-footing-464017-v9-a73a60318d02.json # Google Cloud 서비스 계정 인증 키
        ├── chain_for_chat.py      # 일반 채팅 RAG 체인 정의
        ├── chain_for_form.py      # 양식(Form) 기반 안전 안내문 생성 체인 정의
        ├── chain_for_form_chat.py # 생성된 안내문에 대한 후속 질문 처리 체인 정의
        ├── make_context.py        # 문서 조각에 대한 컨텍스트 요약 생성 스크립트
        ├── make_contextual_content_with_caching.py # 캐싱을 적용하여 컨텍스트 요약을 생성하는 스크립트
        ├── make_docs.py           # 텍스트를 의미 기반으로 분할하는 스크립트
        ├── make_full_text.py      # PDF에서 전체 텍스트를 추출하는 스크립트
        ├── semantic_split_genai.py # GenAI 모델을 사용한 의미 기반 분할 유틸리티
        ├── semantic_split_vertex.py# Vertex AI 모델을 사용한 의미 기반 분할 유틸리티
        ├── custom.py              # 날씨 기반 안전 분석 및 맞춤형 안전 안내문 생성 체인
        └── vector_store.py        # FAISS 벡터 DB 생성, 저장, 로드 및 검색/재순위 로직 관리
```

## 설치 및 실행 방법

### 사전 준비 사항

- [Docker](https://www.docker.com/get-started)와 [Docker Compose](https://docs.docker.com/compose/install/)가 설치되어 있어야 합니다.
- Google Cloud API 키 및 서비스 계정 인증 정보가 필요합니다.

### 실행 절차

1.  **프로젝트 복제**
    ```bash
    git clone https://github.com/singbong/safety_rag.git
    cd safety_rag
    ```

2.  **환경 변수 설정**
    프로젝트 루트 디렉터리에 `.env` 파일을 생성하고 다음과 같이 Google 관련 정보를 추가합니다. `docker-compose.yml` 파일에서 이 `.env` 파일을 참조하여 컨테이너 내에 환경 변수를 설정합니다.

    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    PROJECT_ID="YOUR_PROJECT_ID"
    GOOGLE_APPLICATION_CREDENTIALS="PATH/TO/YOUR/CREDENTIALS.json"
    ```
    *`GOOGLE_APPLICATION_CREDENTIALS`에 지정된 `.json` 인증 키 파일은 프로젝트 내부에 위치해야 합니다. (예: `vector/definition/your-credentials.json`)*

3.  **데이터 준비**
    `vector/data/original_pdf/` 디렉터리에 분석할 PDF 파일들을 추가합니다.
    
    **참고**: 행사 관련 PDF 파일(행사장 안내도, 타임테이블, 셔틀버스 운행 정보 등)을 API로 업로드하면, 시스템이 Google Cloud Vision API를 사용하여 자동으로 OCR 처리하여 텍스트를 추출합니다. 이를 통해 AI가 행사의 구체적인 세부사항을 파악하여 더 정확한 안전 안내문을 생성할 수 있습니다.

4.  **문서 처리 및 벡터 DB 생성**
    아래 스크립트들을 **순서대로** 실행하여 데이터 처리 파이프라인을 구동합니다. 각 단계는 Docker 컨테이너 내에서 실행됩니다.

    -   **1. 텍스트 추출**: `pdfplumber`를 사용하여 PDF 문서의 텍스트를 추출합니다.
        ```bash
        docker-compose run --rm app python vector/definition/make_full_text.py
        ```
    -   **2. 의미 기반 분할 (Chunking)**: 추출된 텍스트를 문서 조각(Chunk)으로 분할합니다.
        ```bash
        docker-compose run --rm app python vector/definition/make_docs.py
        ```
    -   **3. 컨텍스트 보강 (Context Enrichment)**: 각 문서 조각에 요약 컨텍스트를 추가하여 검색 정확도를 높입니다.
        ```bash
        docker-compose run --rm app python vector/definition/make_context.py
        ```
    -   **4. 벡터화 및 저장**: 최종 텍스트를 임베딩 벡터로 변환하고 `FAISS` 벡터 스토어를 생성합니다. 이 단계는 `vector_store.py`의 `create_vector_store` 함수를 직접 호출해야 합니다.
        ```bash
        docker-compose run --rm app python -c "from vector.definition.vector_store import store_vector_db; db = store_vector_db(); db.create_vector_store(save_path='../data/vector_store/faiss_vector_db')"
        ```

5.  **애플리케이션 실행**

    -   **API 서버 실행**:
        아래 명령어로 API 서버를 백그라운드에서 실행합니다.
        ```bash
        docker-compose up --build -d
        ```
        API는 `http://127.0.0.1:8000`에서 접근할 수 있습니다.

    -   **CLI 채팅 실행 (로컬)**:
        *참고: 현재 CLI 채팅 기능은 Docker 환경에서 직접 실행하는 것을 지원하지 않습니다. 로컬 Python 환경에서 실행해야 합니다.*
        ```bash
        # 가상환경 활성화
        # source venv/bin/activate
        python fire_app.py chat
        ```

## 기술 상세 설명

### 문서 처리 및 벡터화

-   **텍스트 추출**: `pdfplumber`를 사용하여 원본 PDF에서 텍스트와 페이지 번호 등 메타데이터를 추출합니다.
-   **OCR 처리**: 사용자가 API로 업로드한 행사 관련 PDF 파일(행사장 안내도, 타임테이블, 셔틀버스 운행 정보 등)을 Google Cloud Vision API를 사용하여 자동으로 OCR 처리하여 텍스트를 추출하고, 이를 AI 분석에 활용합니다.
-   **의미 기반 분할 (Semantic Chunking)**:
    -   `Langchain`의 `SemanticChunker`와 Google의 `gemini-embedding-001` 모델을 사용하여 문서를 의미적 경계에 따라 1차적으로 분할합니다.
    -   분할된 청크가 Gemini 모델의 토큰 제한(2048 토큰)을 초과할 경우, `RecursiveCharacterTextSplitter`와 유사한 방식으로 추가 분할하여 모든 청크가 토큰 제한을 준수하도록 합니다.
-   **컨텍스트 보강 (Contextual Enrichment) - Powered by Anthropic's Technique**:
    -   본 시스템은 검색 정확도를 극대화하기 위해 **Anthropic의 "Contextual Retrieval" 논문에서 제시된 아이디어에 착안**하여 각 문서 조각(Chunk)에 풍부한 컨텍스트를 부여합니다.
    -   `gemini-2.5-flash-lite` 모델이 전체 문서의 맥락을 파악하여, 각 조각의 핵심 내용을 요약하는 **헤더(Header)를 생성**합니다.
    -   생성된 헤더는 원본 문서 조각의 내용과 결합되어 (예: `헤더: [요약 내용]\n\n내용: [원본 문서 조각]`) 하나의 완성된 텍스트로 만들어집니다.
    -   **바로 이 결합된 텍스트가 임베딩**되어 벡터 스토어에 저장됩니다. 이를 통해 단순 키워드 매칭을 넘어선 깊이 있는 의미 기반 검색이 가능해지며, 사용자의 질문 의도에 가장 부합하는 정보를 정확하게 찾아낼 수 있습니다.
    -   이 과정에서 Google GenAI의 Caching API를 활용하여 전체 문서 텍스트를 캐시에 저장함으로써, 반복적인 API 호출 비용과 시간을 절약합니다.
-   **벡터화**:
    -   컨텍스트가 보강된 텍스트(`"컨텍스트 요약 : 원본 청크 내용"`)를 `gemini-embedding-001` 모델을 통해 임베딩 벡터로 변환합니다.
    -   생성된 벡터는 `FAISS` (IndexFlatL2)를 사용하여 벡터 데이터베이스에 저장됩니다.

### 검색 및 재순위 (Retrieval & Reranking)

-   **하이브리드 검색 (Hybrid Search)**:
    -   `Langchain`의 `EnsembleRetriever`를 사용하여 두 가지 검색 방식을 결합합니다.
    -   **의미 검색 (Semantic Search)**: FAISS 벡터 저장소를 사용하여 사용자의 질문과 의미적으로 유사한 문서를 찾습니다. (가중치 70%)
    -   **키워드 검색 (Keyword Search)**: `BM25Retriever`를 사용하여 질문에 포함된 핵심 키워드와 일치하는 문서를 찾습니다. (가중치 30%)
    -   이 두 가지 방식의 결과를 결합하여 초기 검색 결과(150개)를 생성합니다.
-   **재순위 (Reranking)**:
    -   초기 검색된 150개의 문서를 `VertexAIRank` 모델을 사용하여 질문과의 관련성이 높은 순으로 재정렬합니다.
    -   `VertexAIRank`는 Google Cloud Vertex AI에서 제공하는 의미 기반 랭킹 서비스로, `semantic-ranker-default-004` 모델을 사용하여 쿼리와 문서 간의 의미적 유사도를 정밀하게 평가합니다.
    -   최종적으로 가장 관련성이 높은 상위 20개의 문서를 답변 생성에 사용합니다.

## 주요 로직 (Chains)

이 프로젝트는 LangGraph를 사용하여 세 가지 다른 목적의 RAG(검색 증강 생성) 체인을 구현합니다. 각 체인은 특정 시나리오에 맞춰진 노드(Node)들의 그래프로 구성되어 있으며, 환각(Hallucination) 현상을 최소화하기 위해 Vertex AI의 Grounding 기능을 활용합니다.

### 1. `chain_for_chat.py`: 일반 채팅 체인

일반적인 대화형 질의응답을 처리하는 가장 기본적인 RAG 체인입니다.

-   **주요 역할**: 사용자의 질문 의도를 명확히 하고, 관련 문서를 찾아 신뢰도 높은 답변을 생성합니다.
-   **작동 흐름**:
    1.  **질문 재작성 (Re-writer)**: 대화 기록을 참고하여 사용자의 질문에 포함된 대명사(예: '그것')를 구체적인 용어로 바꾸어 명확하게 만듭니다.
    2.  **질문 분해 (Question Decomposer)**: 복잡한 질문을 여러 개의 단순한 하위 질문으로 분해하여 검색 정확도를 높입니다.
    3.  **문서 검색 (Search Document)**: 분해된 질문들을 사용하여 벡터 스토어에서 관련 문서를 검색하고, Reranker를 통해 최종 답변에 사용할 문서의 순위를 재조정합니다.
    4.  **답변 생성 (Generator)**: 검색된 문서를 바탕으로 '안전 지키미 AI' 역할을 수행하며, 질문에 대한 직접적인 답변과 함께 알아두면 좋은 추가 정보를 생성합니다.
    5.  **Grounding 확인 (Hallucination Checker)**: 생성된 답변이 검색된 문서에 근거했는지 확인하고, 인용(Citation)을 추가합니다.
    6.  **답변 정제 (Answer Beautifier)**: 최종 답변을 사용자가 읽기 쉽도록 마크다운 형식으로 정리합니다.

### 2. `chain_for_form.py`: 양식 기반 안전 안내문 생성 체인

사용자가 웹 양식을 통해 입력한 행사 정보를 바탕으로 맞춤형 안전 안내문을 생성하는 체인입니다.

-   **주요 역할**: 행사 정보의 잠재적 위험 요소를 분석하여, 방문객을 위한 상세하고 실용적인 안전 가이드를 자동으로 작성합니다.
-   **작동 흐름**:
    1.  **다각적 검색어 생성 (Query Generator)**: 행사명, 유형, 기간, 장소, OCR 처리된 관련 문서 내용 등을 조합하여 발생 가능한 모든 위험 시나리오(예: "여름철 야외 행사 식중독 예방", "공연장 압사 사고 예방")에 대한 검색어를 생성합니다.
    2.  **문서 검색 (Search Document)**: 생성된 검색어들을 사용하여 안전 관련 문서를 جامع적으로 검색합니다.
    3.  **안내문 생성 (Generator)**: '안전 전문가' 역할을 수행하며, 검색된 문서와 사용자가 입력한 행사 정보를 종합하여 체계적인 구조의 맞춤형 안전 안내문을 생성합니다.
    4.  **Grounding 확인 및 정제**: 생성된 안내문의 신뢰도를 확인하고 최종본을 마크다운 형식으로 다듬습니다.

### 3. `chain_for_form_chat.py`: 후속 질문 처리 체인

`chain_for_form.py`에 의해 생성된 안전 안내문에 대해 사용자가 추가로 질문할 경우, 이를 처리하는 특화된 채팅 체인입니다.

-   **주요 역할**: 기존에 생성된 안내문과 대화의 맥락을 이해하고 후속 질문에 정확하게 답변합니다.
-   **작동 흐름**:
    1.  **질문 재작성 (Re-writer)**: 사용자의 후속 질문(예: "거기서 첫 번째 항목이 왜 중요한가요?")을 이전에 생성된 안내문과 대화 기록을 바탕으로 "화재 발생 시 신속한 대피가 왜 가장 중요한가요?"와 같이 명확한 질문으로 재작성합니다.
    2.  **서브 쿼리 생성 (Query Generator)**: 재작성된 질문을 바탕으로, 답변에 필요한 배경 정보, 예방법, 관련 사례 등을 찾기 위한 추가 검색어들을 생성합니다.
    3.  **문서 검색, 답변 생성, Grounding 및 정제**: `chain_for_chat`과 유사한 과정을 거쳐 사용자의 후속 질문에 대한 상세하고 정확한 답변을 생성합니다.

### 4. `custom.py`: 날씨 기반 안전 분석 및 맞춤형 안전 안내문 생성 체인

`custom.py`는 이 시스템의 핵심 기능 중 하나로, Google Geocoding API와 Open-Meteo 날씨 API를 활용하여 행사 장소의 실시간 날씨 정보를 분석하고, 이를 기반으로 날씨 관련 안전 위험 요소를 예측하는 고급 기능을 제공합니다.

#### **주요 특징**
- **LangGraph 기반 복합 체인**: 여러 단계의 노드들이 순차적으로 연결되어 복잡한 안전 분석을 수행합니다.
- **실시간 날씨 데이터 통합**: 행사 장소의 24시간 날씨 예보를 실시간으로 수집하여 분석에 활용합니다.
- **다중 벡터 스토어 활용**: 안전 문서, 날씨 관련 사고 사례, 행사 관련 사고 사례를 각각 별도의 벡터 스토어에서 검색합니다.
- **Vertex AI Grounding**: 생성된 안전 안내문의 신뢰도를 Vertex AI의 Grounding 기능으로 검증합니다.

<img width="2965" height="4005" alt="KakaoTalk_20250807_215414775_01" src="https://github.com/user-attachments/assets/830fdf44-2ced-40fc-a978-af480921e4da" />

#### **작동 흐름**
    1. **행사 정보 기반 검색어 생성**:
       - `festival_query_generator`: 행사 유형, 장소, 규모, OCR 처리된 관련 문서 내용 등을 분석하여 행사 관련 사고 사례 검색어 생성
       - `weather_query_generator`: 행사 장소의 실시간 날씨 정보를 분석하여 날씨 관련 위험 요소 검색어 생성
       - `safety_query_generator`: 행사 정보와 OCR 처리된 문서 내용을 종합하여 일반적인 안전 위험 요소 검색어 생성

2. **다중 벡터 스토어 검색**:
   - `search_festival_document`: 행사 관련 사고 사례 벡터 스토어에서 유사 사례 검색
   - `search_weather_document`: 날씨 관련 사고 사례 벡터 스토어에서 유사 기상 조건의 사고 사례 검색
   - `search_safety_document`: 일반 안전 문서 벡터 스토어에서 관련 안전 정보 검색

3. **맞춤형 안전 안내문 생성**:
   - `generator`: 검색된 모든 정보를 종합하여 행사 특성과 현재 날씨에 최적화된 안전 안내문 생성
   - `hallu_checker`: Vertex AI Grounding을 통해 생성된 안내문의 신뢰도 검증 및 인용 추가

#### **날씨 데이터 처리 파이프라인**
- **지리 좌표 변환**: Google Geocoding API를 사용하여 행사 장소명을 위도/경도 좌표로 변환
- **24시간 날씨 예보 수집**: Open-Meteo API를 통해 상세한 날씨 정보 수집 (기온, 강수량, 풍속, 습도, UV 지수 등)
- **LLM 기반 날씨 분석**: Gemini 2.5 Flash Lite 모델이 날씨 데이터를 전문적으로 분석하여 위험 요소 추출
- **날씨 기반 검색어 생성**: 분석된 날씨 정보를 바탕으로 유사한 기상 조건의 사고 사례를 찾기 위한 검색어 자동 생성

#### **환경 변수 설정**
```env
GEOCODING_API="YOUR_GOOGLE_GEOCODING_API_KEY"
```

#### **활용 시나리오**
- **여름철 야외 행사**: 폭염, 국지성 호우, 높은 UV 지수 등에 대한 사전 경고 및 대비 방안 제공
- **겨울철 실외 행사**: 한파, 강설, 빙판길 등에 대한 안전 수칙 안내
- **봄/가을 행사**: 일교차, 강풍, 미세먼지 등 계절별 특수 기상 조건에 대한 주의사항 제공
- **실시간 기상 변화 대응**: 행사 당일 급변하는 날씨에 대한 즉각적인 안전 정보 업데이트

이 기능을 통해 행사 주최자는 단순한 행사 정보만으로도 날씨에 특화된 맞춤형 안전 안내문을 자동으로 생성할 수 있으며, 방문객들은 현재 기상 조건에 최적화된 안전 수칙을 받아볼 수 있습니다.

## API 사용 가이드

이 시스템은 FastAPI를 통해 4개의 주요 API 엔드포인트를 제공합니다. 각 API는 `http://<YOUR_SERVER_IP>:<PORT>` 주소로 요청할 수 있습니다.

### 1. `/api/chat`

일반적인 질의응답을 위한 API입니다.

-   **Python 코드 예제**:
    ```python
    import requests

    chat_url = "http://127.0.0.1:8000/api/chat"  # 실제 서버 주소로 변경 필요
    chat_payload = {
        "question": "지진 발생 시 행동 요령 알려줘",
        "session_id": "user123_session_abc"
    }

    try:
        response = requests.post(chat_url, json=chat_payload)
        response.raise_for_status()  # 200번대 응답이 아니면 에러 발생
        chat_result = response.json()
        
        print("--- 최종 답변 ---")
        print(chat_result.get('final_answer'))
        
    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")

    ```

### 2. `/api/generate_form`

사용자가 입력한 양식(Form) 데이터를 기반으로 맞춤형 안전 안내문을 생성합니다.

**참고**: `related_documents` 필드에는 행사 관련 PDF 파일(행사장 안내도, 타임테이블, 셔틀버스 운행 정보 등)을 업로드하면, 시스템이 Google Cloud Vision API를 사용하여 자동으로 OCR 처리하여 텍스트를 추출합니다. 이 정보는 AI가 행사의 구체적인 세부사항을 파악하여 더 정확한 안전 안내문을 생성하는 데 활용됩니다.

-   **Python 코드 예제**:
    ```python
    import requests

    form_url = "http://127.0.0.1:8000/api/generate_form" # 실제 서버 주소로 변경 필요
    form_payload = {
        "place_name": "2025 한강 여름 뮤직 페스티벌",
        "type": "대규모 야외 공연",
        "region": "서울, 여의도 한강공원",
        "period": "2025년 8월 8일 ~ 2025년 8월 10일",
        "description": "뜨거운 여름밤을 식혀줄 대한민국 최고의 뮤직 페스티벌! 다양한 장르의 아티스트들과 함께하는 3일간의 축제. 푸드트럭 존과 체험 이벤트도 준비되어 있습니다.",
        "category": "음악/페스티벌",
        "related_documents": "festival_guide.pdf",  # PDF 파일 업로드
        "emergency_contact_name": "종합상황실 안전관리팀",
        "emergency_contact_phone": "02-123-4567"
    }

    try:
        response = requests.post(form_url, json=form_payload)
        response.raise_for_status()
        form_result = response.json()

        print("--- 생성된 안전 안내문 ---")
        print(form_result.get('final_answer'))
        
        # 후속 질문을 위해 생성된 안내문을 변수에 저장
        generated_form_content = form_result.get('final_answer')

    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
    ```

### 3. `/api/form_chat`

`generate_form`으로 생성된 안전 안내문에 대한 후속 질문을 처리합니다.

-   **Python 코드 예제**:
    ```python
    import requests
    # 'generate_form' API 호출 후 반환된 'final_answer' 값을 사용합니다.
    # 예시에서는 위 코드 블록의 'generated_form_content' 변수를 사용한다고 가정합니다.
    
    # generated_form_content = "..." # 실제로는 이전 API 호출 결과

    if 'generated_form_content' in locals():
        form_chat_url = "http://127.0.0.1:8000/api/form_chat" # 실제 서버 주소로 변경 필요
        form_chat_payload = {
            "generated_form": generated_form_content,
            "query": "온열질환 증상으로는 구체적으로 어떤 것들이 있나요?",
            "session_id": "user123_session_abc"
        }

        try:
            response = requests.post(form_chat_url, json=form_chat_payload)
            response.raise_for_status()
            form_chat_result = response.json()

            print("--- 후속 질문에 대한 답변 ---")
            print(form_chat_result.get('final_answer'))

        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패: {e}")
    else:
        print("먼저 /api/generate_form 을 호출하여 'generated_form_content'를 생성해야 합니다.")

    ```

### 4. `/api/custom`

`custom.py`에 구현된 날씨 기반 안전 분석 기능을 활용하여 행사 정보와 실시간 날씨 데이터를 종합한 맞춤형 안전 안내문을 생성합니다. 이 API는 Google Geocoding API와 Open-Meteo 날씨 API를 자동으로 호출하여 행사 장소의 24시간 날씨 예보를 분석하고, 이를 기반으로 날씨 관련 위험 요소를 예측합니다.

**참고**: `related_documents` 필드에는 행사 관련 PDF 파일(행사장 안내도, 타임테이블, 셔틀버스 운행 정보 등)을 업로드하면, 시스템이 Google Cloud Vision API를 사용하여 자동으로 OCR 처리하여 텍스트를 추출합니다. 이 정보는 AI가 행사의 구체적인 세부사항을 파악하여 더 정확한 안전 안내문을 생성하는 데 활용됩니다.

-   **Python 코드 예제**:
    ```python
    import requests

    custom_url = "http://127.0.0.1:8000/api/custom"  # 실제 서버 주소로 변경 필요
    custom_payload = {
        "place_name": "2025 한강 여름 뮤직 페스티벌",
        "type": "대규모 야외 공연",
        "location": "서울, 여의도 한강공원",
        "period": "2025년 8월 8일 ~ 2025년 8월 10일",
        "description": "뜨거운 여름밤을 식혀줄 대한민국 최고의 뮤직 페스티벌! 다양한 장르의 아티스트들과 함께하는 3일간의 축제. 푸드트럭 존과 체험 이벤트도 준비되어 있습니다.",
        "category": "음악/페스티벌",
        "related_documents": "festival_guide.pdf",  # PDF 파일 업로드
        "emergency_contact_name": "종합상황실 안전관리팀",
        "emergency_contact_phone": "02-123-4567",
        "expected_attendees": "50000"
    }

    try:
        response = requests.post(custom_url, json=custom_payload)
        response.raise_for_status()
        custom_result = response.json()

        print("--- 날씨 기반 맞춤형 안전 안내문 ---")
        print(custom_result.get('generation'))
        
        # Grounding 검증 결과도 확인 가능
        if 'hallu_check' in custom_result:
            print("\n--- Grounding 검증 결과 ---")
            print(f"신뢰도: {custom_result['hallu_check'].get('answer_with_citations', 'N/A')}")

    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
    ```

-   **주요 특징**:
    - **실시간 날씨 분석**: 행사 장소의 24시간 날씨 예보를 자동으로 수집하고 분석
    - **다중 벡터 스토어 검색**: 안전 문서, 날씨 관련 사고 사례, 행사 관련 사고 사례를 각각 별도 검색
    - **Vertex AI Grounding**: 생성된 안내문의 신뢰도를 자동으로 검증
    - **맞춤형 위험 요소 예측**: 행사 특성과 현재 날씨를 종합하여 구체적인 위험 요소 분석

-   **필수 환경 변수**:
    ```env
    GEOCODING_API="YOUR_GOOGLE_GEOCODING_API_KEY"
    ```

-   **응답 형식**:
    ```json
    {
        "generation": "생성된 날씨 기반 맞춤형 안전 안내문",
        "hallu_check": {
            "answer_with_citations": "Grounding 검증이 완료된 안내문 (인용 포함)"
        },
        "weather_summary": "분석된 날씨 요약 정보",
        "festival_query_list": ["생성된 행사 관련 검색어 목록"],
        "weather_query_list": ["생성된 날씨 관련 검색어 목록"],
        "safety_query_list": ["생성된 안전 관련 검색어 목록"]
    }
    ```
