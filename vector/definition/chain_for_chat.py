import vertexai
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
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
import re

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


re_write_system = """
당신의 역할은 사용자의 query를 키워드 중심으로 재작성하는 것입니다.
절대 답변을 생성하거나, chat_history을 요약·설명하거나, query 외의 어떤 정보도 추가하지 마세요.

- query에 있는 대명사/지시어(이것, 그것, 그 보직 등)는 chat history에서 언급된 명사로 치환하세요.
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
너는 대한민국 소방청의 신뢰받는 '안전 지키미 AI' 챗봇이야.
너의 임무는 사용자의 안전을 최우선으로 생각하며, 주어진 [document]를 바탕으로 [query]에 대해 가장 정확하고 유용한 답변을 제공하는 것이야.

[답변 생성 가이드라인]
    1. 핵심 답변 우선: 먼저, 사용자의 [query]에 대한 가장 직접적이고 명확한 답변을 [document]에서 찾아 제시해야 해.
    2. 추가 정보 탐색 및 제공: 핵심 답변을 완료한 후, 사용자가 궁금해할 만한 추가적인 안전 정보를 [document] 내에서 탐색하여 함께 제공해야 해. 이 정보는 사용자의 안전 의식을 높이고 잠재적 위험을 예방하는 데 도움이 되어야 해.

        **탐색할 정보의 예시:**
            *사전 예방 수칙:** [query]와 관련된 위험을 미리 방지할 수 있는 방법
            *상황 발생 시 행동 요령:** 위급 상황이 발생했을 때의 구체적인 대처법
            *관련 법규 또는 기준:** [query]와 관련된 안전 법규나 기준 정보
            *잘못 알려진 상식:** 사용자가 오해할 수 있는 부분에 대한 정확한 정보

    3. 답변 구조화: 답변은 명확하게 두 부분으로 나누어 제공해줘.
        **[질문에 대한 답변]:** [query]에 대한 직접적인 답변 부분
        **[함께 알아두면 좋은 안전 정보]:** 선제적으로 찾아서 제공하는 추가 정보 부분

    **[중요 규칙]**
        *근거 중심: 모든 답변은 [document]에 명시된 내용을 근거로 해야 해.
        *추측 금지: [document]에 없는 내용은 절대로 추측하거나 임의로 생성해서는 안 돼. 만약 [document]에 관련 내용이 없다면, "제공된 문서에서는 해당 정보(또는 추가 정보)를 찾을 수 없습니다."라고 명확히 밝혀줘.
        *친절한 전문가 어조: '안전 지키미'로서 전문적이면서도, 누구나 이해하기 쉬운 친절하고 명확한 말투를 사용해줘.
"""

final_answer_system = """
당신은 텍스트를 깔끔하게 정리하는 AI입니다.
주어진 [Raw Text]는 내용과 citation([숫자])은 정확하지만, 줄바꿈이나 글머리 기호 같은 서식이 모두 사라진 상태입니다.

당신의 임무는 [Raw Text]의 내용과 citation을 그대로 유지하면서, 사용자가 읽기 쉽도록 서식을 복원하는 것입니다.

**[규칙]**
1.  **내용 보존:** 원본 텍스트의 모든 단어와 citation([숫자])을 빠짐없이 사용해야 합니다.
2.  **서식 복원:** 문맥에 맞게 줄바꿈, 글머리 기호(*), 제목(##) 등의 마크다운 서식을 자연스럽게 추가해주세요.
3.  **내용 수정 금지:** 원본에 없는 내용을 추가하거나, 기존 내용을 변경하지 마세요.
4.  **출력:** 다른 설명 없이, 서식이 복원된 최종 텍스트만 출력합니다.
"""

class GraphState(TypedDict):
    question: Annotated[str, "User question"]
    decomposed_question: Annotated[List[str], "Decomposed question"]
    chat_history: Annotated[str, "Chat history for context"]
    document: Annotated[List[str], "Combined documents"]
    generation: Annotated[str, "LLM generated answer"]
    hallu_check: Annotated[dict, "Hallucination check result"]
    final_answer: Annotated[str, "Final answer with citations"]

class chat_chain():
    def __init__(self):
        self.simple_llm = ChatVertexAI(model_name="gemini-2.5-flash-lite", temperature=0.1, max_output_tokens=512, verbose=True)
        self.generate_llm = ChatVertexAI(model_name="gemini-2.5-flash-lite", temperature=0.4, verbose=True)
        self.final_llm = ChatVertexAI(model_name="gemini-2.5-flash-lite", temperature=0, verbose=True)
        self.graph = StateGraph(GraphState)
        self.checker = VertexAICheckGroundingWrapper(
            project_id=os.getenv("PROJECT_ID"),
            location_id="us-central1",
            grounding_config="default_grounding_config",
            citation_threshold=0.5,
            credentials=credentials,
        )

    def re_write(self, chat_history: str, query: str) -> str:
        rewrite_prompt = ChatPromptTemplate.from_messages([("system", re_write_system), ("user", "chat history: \n {chat_history} \n\n query: \n {query}")])
        chain = rewrite_prompt | self.simple_llm | StrOutputParser() | RunnableLambda(lambda text: text.strip())
        return chain.invoke({"chat_history": chat_history, "query": query})

    def question_decompose(self, query: str) -> List[str]:
        question_decomposer_prompt = ChatPromptTemplate.from_messages([("system", question_decomposer_system), ("user", "query: {query}")])
        chain = question_decomposer_prompt | self.simple_llm | StrOutputParser() | RunnableLambda(lambda text: text.strip())
        result = chain.invoke({"query": query})
        return json.loads(result)

    def generate(self, query: str, document: List[str]) -> str:
        generate_prompt = ChatPromptTemplate.from_messages([("system", generate_prompt_system), ("user", "query: {query} \n document: {document}")])
        chain = generate_prompt | self.generate_llm | StrOutputParser()
        return chain.invoke({"query": query, "document": document})

    def clean_up_answer_with_citations(self, raw_text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([("system", final_answer_system), ("user", "[Raw Text]: {raw_text}")])
        chain = prompt | self.final_llm | StrOutputParser()
        return chain.invoke({"raw_text": raw_text})

    def re_writer(self, state: GraphState) -> GraphState:
        rewritten = self.re_write(state["chat_history"], state["question"])
        return {**state, "question": rewritten}

    def question_decomposer(self, state: GraphState) -> GraphState:
        decomposed = self.question_decompose(state["question"])
        return {**state, "decomposed_question": decomposed}

    def search_document(self, state: GraphState) -> GraphState:
        searched_documents = []
        for decomposed_question in state["decomposed_question"]:
            searched_documents.extend(vector_store.search(decomposed_question, k=50))
        unique_documents = []
        seen_contents = set()
        for doc in searched_documents:
            if doc['document'] not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(doc['document'])
        searched_documents = unique_documents
        searched_documents = vector_store.document_rerank(state["question"], searched_documents, k=25)
        return {**state, "document": searched_documents}

    def generator(self, state: GraphState) -> GraphState:
        generation = self.generate(state["question"], state["document"])
        return {**state, "generation": generation}

    def hallu_checker(self, state: GraphState) -> GraphState:
        hallu_check_result = self.checker.invoke(state["generation"], config={"configurable": {"documents": state["document"]}})
        return {**state, "hallu_check": hallu_check_result}

    def answer_beautifier(self, state: GraphState) -> GraphState:
        # CheckGroundingResponse 객체의 .answer_with_citations 속성에 직접 접근
        raw_text = state["hallu_check"].answer_with_citations
        
        # citation 번호를 1부터 시작하도록 조정
        def replace_citation(match):
            number = int(match.group(1))
            return f"[{number + 1}]"
        adjusted_text = re.sub(r'\[(\d+)\]', replace_citation, raw_text)
        
        cleaned_answer = self.clean_up_answer_with_citations(adjusted_text)
        return {**state, "final_answer": cleaned_answer}

    def check_hallu_complete_chat(self, state: GraphState) -> str:
        # CheckGroundingResponse 객체의 .support_score 속성에 직접 접근
        score = state["hallu_check"].support_score
        if score >= 0.5:
            return "clean"
        else:
            return "re_writer"

    def link_nodes(self):
        self.graph.add_node("re_writer", self.re_writer)
        self.graph.add_node("question_decomposer", self.question_decomposer)
        self.graph.add_node("search_document", self.search_document)
        self.graph.add_node("generator", self.generator)
        self.graph.add_node("hallu_checker", self.hallu_checker)
        self.graph.add_node("answer_beautifier", self.answer_beautifier)

        self.graph.add_edge(START, "re_writer")
        self.graph.add_edge("re_writer", "question_decomposer")
        self.graph.add_edge("question_decomposer", "search_document")
        self.graph.add_edge("search_document", "generator")
        self.graph.add_edge("generator", "hallu_checker")
        
        self.graph.add_conditional_edges(
            "hallu_checker",
            self.check_hallu_complete_chat,
            {
                "clean": "answer_beautifier",
                "re_writer": "re_writer"
            }
        )
        self.graph.add_edge("answer_beautifier", END)

        return self.graph.compile(checkpointer=MemorySaver())