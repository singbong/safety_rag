# Fire RAG (소방 재난 정보 RAG)

소방 및 재난 안전 관련 문서에 대한 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템입니다. LangGraph와 Google Gemini 모델을 기반으로 구축되었습니다.

## 1. 사용 준비 사항

이 시스템을 로컬 환경이나 서버에서 실행하기 위해 필요한 준비 과정입니다.

### 1.1. 필수 설치 요소

- **Docker 및 Docker Compose:** 애플리케이션을 컨테이너 환경에서 실행하기 위해 필요합니다.
- **Python 3.11+:** 데이터 전처리 스크립트 실행을 위해 필요합니다.
- **Google Cloud 계정 및 프로젝트:** Gemini API와 Vertex AI 서비스를 사용하기 위해 필요합니다.

### 1.2. 환경 설정

1.  **저장소 복제(Clone)**
    ```bash
    git clone <저장소_URL>
    cd fire_rag
    ```

2.  **Python 가상환경 및 의존성 설치**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Google Cloud 인증 정보 설정**
    - Google Cloud Platform에서 서비스 계정을 생성하고 JSON 키 파일을 다운로드합니다.
    - 다운로드한 키 파일을 저장하고 그 경로를 .env 파일에 입력합니다. **(주의: 이 파일은 gitignore에 등록되어 있어야 합니다.)**

4.  **.env 파일 생성**
    - `fire_rag` 디렉토리 최상단에 `.env` 파일을 생성하고 아래 내용을 채웁니다.
    ```env
    # .env 파일 예시
    PROJECT_ID="your-gcp-project-id"
    GOOGLE_API_KEY="your-google-api-key"
    GOOGLE_APPLICATION_CREDENTIALS="your-google-application-credentials-path"

    # LangSmith (선택 사항)
    LANGCHAIN_TRACING="true"
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY="your-langsmith-api-key"
    LANGCHAIN_PROJECT="your_project_name"
    ```

### 1.3. 데이터 전처리 및 벡터 DB 생성

1.  **원본 PDF 파일 준비**
    - 답변의 근거 자료가 될 PDF 파일들을 `vector/data/original_pdf/` 디렉토리에 넣습니다.(original_pdf 파일이 없다면 생성하세요)

2.  **데이터 처리 스크립트 실행**
    - `vector/definition/` 디렉토리로 이동하여 아래 스크립트들을 순서대로 실행합니다. 이 과정은 PDF에서 텍스트를 추출하고, 의미 기반으로 문서를 분할한 뒤, 각 청크에 대한 요약 컨텍스트를 생성하고, 최종적으로 벡터 데이터베이스를 구축합니다.
    ```bash
    cd vector/definition/

    # 1. PDF -> 전체 텍스트 추출
    python make_full_text.py

    # 2. 전체 텍스트 -> 의미 기반 청킹
    python make_docs.py

    # 3. 청크 -> 컨텍스트 요약 생성
    python make_context.py

    # 4. 컨텍스트가 추가된 청크 -> 벡터 DB 생성
    python vectore_store.py
    ```
    - 위 과정이 모두 성공적으로 완료되면 `vector/data/vector_store/` 디렉토리에 FAISS 인덱스 파일(`.index`)과 문서 데이터 파일(`.pkl`)이 생성됩니다.

### 1.4. 애플리케이션 실행

- **Docker Compose 사용 (권장)**
  - `fire_rag` 디렉토리 최상단에서 아래 명령어를 실행합니다.
  ```bash
  docker-compose up --build -d
  ```

- **직접 실행**
  - `fire_rag` 디렉토리 최상단에서 Uvicorn을 사용하여 직접 실행할 수 있습니다.
  ```bash
  uvicorn fire_app:app --host 0.0.0.0 --port 8000
  ```

## 2. API 사용 방법

애플리케이션이 실행되면 local host의 8000번 포트에서 두 개의 API 엔드포인트를 사용할 수 있습니다.

### 2.1. `/api/chat` (채팅 API)

대화의 맥락을 기억하는 채팅 형식의 답변을 생성합니다. 세션 유지를 위해 `session_id`가 필수입니다.

-   **URL:** `http://<host>:8000/api/chat`
-   **Method:** `POST`
-   **Headers:** `Content-Type: application/json`
-   **Input Body (JSON):
    ```json
    {
      "question": "질문 내용",
      "session_id": "고유한 세션 ID"
    }
    ```
-   **예시 (cURL):**
    ```bash
    curl -X POST "http://localhost:8000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
      "question": "지진 발생 시 행동 요령 알려줘",
      "session_id": "user123_session_abc"
    }'
    ```
-   **Return 값 예시 (JSON):**
    ```json
    {
      "return_answer": "지진 발생 시에는 먼저 머리를 보호하고 튼튼한 책상이나 테이블 밑으로 들어가 몸을 피해야 합니다. 건물이 흔들리는 동안에는 이동하지 않는 것이 안전하며, 흔들림이 멈추면 전기와 가스를 차단하고 신속하게 건물 밖으로 대피해야 합니다...",
      "hallu_check": {
        "support_score": 0.95,
        "answer_with_citation": "지진 발생 시에는 먼저 머리를 보호하고 튼튼한 책상이나 테이블 밑으로 들어가 몸을 피해야 합니다.[0]",
        "cited_chunks": [
          {
            "chunk_text": "지진 발생 시 대피 요령...",
            "source": {
              "page_content": "...",
              "metadata": {
                "pdf_filename": "2-2(분책) 1장 지진.pdf",
                "page_numbers": [3]
              }
            }
          }
        ],
        "claims": [
          {
            "claim_text": "지진 발생 시에는 먼저 머리를 보호하고...",
            "citation_indices": [0]
          }
        ]
      }
    }
    ```
    - `return_answer`: 최종 생성된 답변 텍스트입니다.
    - `hallu_check`: 답변의 신뢰도를 검증한 결과입니다. `support_score`가 1에 가까울수록 참조한 문서 내용에 기반한 답변임을 의미합니다. `answer_with_citation`은 생성된 답변에서 각 문장을 뒷받침하는 document들을 citation을 추가한 답변입니다.

### 2.2. `/api/generate_form` (양식 생성 API)

대화 맥락 없이, 단일 질문에 대해 양식(form) 형태의 답변을 생성하는 데 사용됩니다. `session_id`가 필요 없습니다.

-   **URL:** `http://<host>:8000/api/generate_form`
-   **Method:** `POST`
-   **Headers:** `Content-Type: application/json`
-   **Input Body (JSON):**
    ```json
    {
      "question": "질문 내용"
    }
    ```
-   **예시 (cURL):**
    ```bash
    curl -X POST "http://localhost:8000/api/generate_form" \
    -H "Content-Type: application/json" \
    -d '{
      "question": "화재 발생 시 초기 진화 방법"
    }'
    ```
-   **Return 은 /api/chat과 동일합니다.**
    ```

## 3. 기술 상세 설명

### 3.1. 문서 처리 및 벡터화

1.  **텍스트 추출:** `pdfplumber`를 사용하여 원본 PDF에서 텍스트와 페이지 번호 등 메타데이터를 추출합니다.
2.  **의미 기반 분할 (Semantic Chunking):**
    -   `Langchain`의 `SemanticChunker`와 Google의 `gemini-embedding-001` 모델을 사용하여 문서를 의미적 경계에 따라 1차적으로 분할합니다.
    -   분할된 청크가 Gemini 모델의 토큰 제한(2048 토큰)을 초과할 경우, 토큰 수를 기준으로 하는 `RecursiveCharacterTextSplitter`와 유사한 방식으로 추가 분할하여 모든 청크가 토큰 제한을 준수하도록 합니다.
3.  **컨텍스트 보강 (Contextual Enrichment):**
    -   각 청크의 검색 정확도를 높이기 위해, `gemini-1.5-flash` 모델을 사용하여 전체 문서의 내용을 참조하여 각 청크에 대한 한 문장 요약 컨텍스트를 생성합니다.
    -   이 과정에서 Google GenAI의 Caching API를 활용하여 전체 문서 텍스트를 캐시에 저장함으로써, 반복적인 API 호출 비용과 시간을 절약합니다.
4.  **벡터화:**
    -   컨텍스트가 보강된 텍스트(`"컨텍스트 요약 : 원본 청크 내용"`)를 `gemini-embedding-001` 모델을 통해 임베딩 벡터로 변환합니다.
    -   생성된 벡터는 `FAISS` (IndexFlatL2)를 사용하여 벡터 데이터베이스에 저장됩니다.

### 3.2. 검색 및 재순위 (Retrieval & Reranking)

1.  **하이브리드 검색 (Hybrid Search):**
    -   `Langchain`의 `EnsembleRetriever`를 사용하여 두 가지 검색 방식을 결합합니다.
    -   **의미 검색 (Semantic Search):** FAISS 벡터 저장소를 사용하여 사용자의 질문과 의미적으로 유사한 문서를 찾습니다. (가중치 70%)
    -   **키워드 검색 (Keyword Search):** `BM25Retriever`를 사용하여 질문에 포함된 핵심 키워드와 일치하는 문서를 찾습니다. (가중치 30%)
    -   이 두 가지 방식의 결과를 결합하여 초기 검색 결과(150개)를 생성합니다.

2.  **재순위 (Reranking):**
    -   초기 검색된 150개의 문서를 `VertexAIRank` 모델을 사용하여 질문과의 관련성이 높은 순으로 재정렬합니다.
        -   `VertexAIRank`는 Google Cloud Vertex AI에서 제공하는 의미 기반 랭킹(semantic reranking) 서비스로, 쿼리(질문)와 각 문서(청크) 간의 의미적 유사도를 정밀하게 평가하여 점수를 부여합니다.
        -   VertexAIRank에서 사용하는 임베딩 및 랭킹 모델은 `semantic-ranker-default-004`입니다. 이 모델은 쿼리와 문서 모두에 대해 1024토큰까지 임베딩을 생성하고, 최신 LLM 기반의 의미적 비교를 통해 쿼리-문서 쌍의 관련성을 평가합니다. (공식 문서 및 Vertex AI 콘솔 기준)
        -   단순 키워드 일치가 아니라, 쿼리와 문서의 의미적 맥락을 반영하여 랭킹을 수행하므로, BM25나 TF-IDF 등 전통적 방식보다 훨씬 더 정밀한 의미 기반 랭킹이 가능합니다.
        -   VertexAIRank는 Vertex AI 플랫폼에서 제공하는 관리형 서비스로, 대규모 데이터셋과 다양한 도메인에 대해 높은 성능과 확장성을 보장합니다.
    -   최종적으로 가장 관련성이 높은 상위 20개의 문서를 답변 생성에 사용합니다.

### 3.3. 답변 생성 및 검증 (Generation & Verification)

이 시스템은 `LangGraph`를 사용하여 다음과 같은 단계적 체인(Chain)을 구성합니다.

1.  **질문 재작성 (Query Rewriting):** `gemini-1.5-flash` 모델이 이전 대화 기록(`chat_history`)을 참고하여 사용자의 현재 질문에 포함된 대명사 등을 명확한 키워드로 치환하여 검색에 더 적합한 질문으로 재작성합니다.
2.  **질문 분해 (Query Decomposition):** 재작성된 질문이 복잡할 경우(예: "A와 B에 대해 알려줘"), `gemini-1.5-flash` 모델이 이를 독립적으로 답변할 수 있는 여러 개의 단순한 질문(예: "A에 대해 알려줘", "B에 대해 알려줘")으로 분해합니다.
3.  **문서 검색 및 재순위:** 분해된 질문들을 사용하여 위에서 설명한 하이브리드 검색 및 재순위 과정을 통해 관련 문서를 찾습니다.
4.  **답변 생성:** 검색된 최종 문서를 근거 자료로 하여, `gemini-2.5-flash` 모델이 사용자의 질문에 대한 최종 답변을 생성합니다.
5.  **환각 검증 (Hallucination Check):**
    -   생성된 답변이 근거 문서에 기반했는지 확인하기 위해 `VertexAICheckGroundingWrapper`를 사용합니다.
    -   답변의 각 문장이 근거 문서에 의해 얼마나 뒷받침되는지를 나타내는 `support_score`를 계산합니다.
    -   이 점수가 0.5 미만일 경우, 답변의 신뢰도가 낮다고 판단하여 그래프는 다시 **질문 재작성** 단계로 돌아가 더 나은 답변을 생성하려고 시도합니다. 이 과정을 통해 답변의 정확성과 신뢰도를 높입니다.
