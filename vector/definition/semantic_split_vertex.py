"""
semantic_split_vertex.py - PDF Semantic Chunking with Google Vertex AI Embeddings

주요 기능:
1. PDF 문서를 의미적 경계에서 청크로 분할
2. Google Vertex AI 임베딩 모델의 토큰 제한(2048) 준수
3. 페이지 번호 추적 및 메타데이터 보존

처리 방식:
1. PDF 텍스트 추출 및 페이지 메타데이터 보존
2. SemanticChunker로 의미적 경계에서 1차 분할
3. 토큰 제한 초과 청크를 RecursiveCharacterTextSplitter로 2차 분할
4. Google 토크나이저로 정확한 토큰 수 계산 및 검증

클래스 구조:
- VertexEmbeddings: Google Vertex AI 래퍼, 토큰 제한 처리
- PDFSemanticChunker: PDF 의미적 청킹 메인 클래스

함수별 역할:
[VertexEmbeddings]
- __init__: 프로젝트 ID 설정 및 Vertex AI 클라이언트 초기화
- _count_tokens: Google 토크나이저로 정확한 토큰 수 계산
- _check_and_truncate_text: 토큰 제한 검증 및 이진 탐색으로 텍스트 자르기
- embed_documents: 문서 리스트 임베딩 (토큰 제한 보장)
- embed_query: 단일 쿼리 임베딩 (토큰 제한 보장)

[PDFSemanticChunker]
- __init__: 임베딩 모델 및 분할기 초기화
- extract_text_from_pdf_with_metadata: PDF 텍스트 추출 및 페이지 메타데이터 보존
- find_pages_for_chunk: 청크가 속한 페이지 번호 찾기
- create_semantic_chunks: 메인 청킹 프로세스 (의미적 분할 → 토큰 제한 검증 → 후처리)

사용법:
    chunker = PDFSemanticChunker()
    docs = chunker.create_semantic_chunks("document.pdf")
"""

import pdfplumber
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any, Optional
import re
import google.generativeai as genai
from dotenv import load_dotenv
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account


os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
project_id = os.getenv("PROJECT_ID")
google_api_key = os.getenv("GOOGLE_API_KEY")
credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_file(credential_path)


class VertexEmbeddings(Embeddings):
    """
    Google Vertex AI를 이용한 LangChain Embeddings 래퍼 클래스
    """
    def __init__(self, project_id: Optional[str] = None, region: str = "asia-northeast1", model: str = "gemini-embedding-001"):
        # .env에서 프로젝트 ID 로드
        if project_id is None:
            # 현재 파일 위치 기준으로 google.env 로드
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_id = project_id
        if not project_id:
            raise ValueError("PROJECT_ID 환경변수가 설정되지 않았습니다.")

        self.model = model
        self.project_id = project_id
        self.region = region
        
        # Vertex AI 초기화
        try:
            vertexai.init(project=project_id, location=region, credentials=credentials)
            self.embedding_model = TextEmbeddingModel.from_pretrained(model)
            print("Vertex AI 초기화 성공")
        except Exception as e:
            print(f"Vertex AI 초기화 실패: {e}")
            print("서비스 계정 키가 올바르게 설정되었는지 확인하세요.")
            raise
        
        # 토큰 계산을 위한 genai 클라이언트 (기존 방식 유지)
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.client = genai
        else:
            self.client = None

    def _count_tokens(self, text: str) -> int:
        """
        Google AI의 GenerativeModel을 사용하여 정확한 토큰 수 계산
        """
        try:
            if self.client:
                model = self.client.GenerativeModel(self.model)
                count_response = model.count_tokens(text)
                return count_response.total_tokens
            else:
                # genai 클라이언트가 없으면 추정치 사용
                return int(len(text) / 1.5)
        except Exception as e:
            print(f"토큰 계산 오류, 추정치 사용: {e}")
            # 오류 시 추정치 사용
            return int(len(text) / 1.5)
    
    def _check_and_truncate_text(self, text: str, max_tokens: int = 2000) -> str:
        """
        텍스트의 토큰 수를 정확히 체크하고 필요시 자르기
        """
        actual_tokens = self._count_tokens(text)
        
        if actual_tokens > max_tokens:
            # 이진 탐색으로 적절한 길이 찾기
            left, right = 0, len(text)
            best_text = text[:int(len(text) * max_tokens / actual_tokens)]
            
            while left < right:
                mid = (left + right + 1) // 2
                test_text = text[:mid]
                test_tokens = self._count_tokens(test_text)
                
                if test_tokens <= max_tokens:
                    best_text = test_text
                    left = mid
                else:
                    right = mid - 1
            
            print(f"텍스트가 {actual_tokens}토큰에서 {self._count_tokens(best_text)}토큰으로 자릅니다.")
            return best_text
        
        return text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        여러 문서(문장) 임베딩
        """
        embeddings = []
        for text in texts:
            try:
                # 토큰 길이 체크 및 자르기
                safe_text = self._check_and_truncate_text(text)
                
                # Vertex AI 임베딩 모델 사용
                result = self.embedding_model.get_embeddings([safe_text])
                embeddings.append(result[0].values)
            except Exception as e:
                print(f"임베딩 생성 오류: {e}")
                # 오류 시 빈 벡터 반환
                raise
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리 임베딩
        """
        try:
            # 토큰 길이 체크 및 자르기
            safe_text = self._check_and_truncate_text(text)
            
            # Vertex AI 임베딩 모델 사용
            result = self.embedding_model.get_embeddings([safe_text])
            return result[0].values
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            raise

class TokenBasedTextSplitter:
    """
    Google 토크나이저를 사용한 토큰 기반 텍스트 분할기
    """
    def __init__(self, embeddings_model, max_tokens: int = 2000, overlap_tokens: int = 200):
        self.embeddings_model = embeddings_model
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.separators = ["\n\n", "\n", ".", " "]
    
    def split_text(self, text: str) -> List[str]:
        """
        토큰 기준으로 텍스트를 분할
        """
        if self.embeddings_model._count_tokens(text) <= self.max_tokens:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # 최대 토큰 수만큼 텍스트 추출
            chunk_end = self._find_chunk_end(text, current_pos)
            chunk = text[current_pos:chunk_end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # 다음 시작 위치 계산 (overlap 고려)
            if chunk_end >= len(text):
                break
                
            overlap_start = self._find_overlap_start(text, chunk_end)
            current_pos = overlap_start
        
        return chunks
    
    def _find_chunk_end(self, text: str, start_pos: int) -> int:
        """
        토큰 제한 내에서 적절한 끝 위치 찾기
        """
        # 이진 탐색으로 최대 길이 찾기
        left, right = start_pos + 1, len(text)
        best_end = start_pos + min(3000, len(text) - start_pos)  # 초기 추정치
        
        while left <= right:
            mid = (left + right) // 2
            chunk = text[start_pos:mid]
            tokens = self.embeddings_model._count_tokens(chunk)
            
            if tokens <= self.max_tokens:
                best_end = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # 문장 경계에서 자르기
        return self._adjust_to_separator(text, best_end)
    
    def _adjust_to_separator(self, text: str, pos: int) -> int:
        """
        가장 가까운 구분자 위치로 조정
        """
        if pos >= len(text):
            return len(text)
        
        # 현재 위치에서 뒤로 검색하여 적절한 구분자 찾기
        for sep in self.separators:
            search_start = max(0, pos - 500)  # 500자 범위 내에서 검색
            sep_pos = text.rfind(sep, search_start, pos)
            if sep_pos != -1:
                return sep_pos + len(sep)
        
        return pos
    
    def _find_overlap_start(self, text: str, chunk_end: int) -> int:
        """
        오버랩을 고려한 다음 시작 위치 찾기
        """
        if chunk_end >= len(text):
            return len(text)
        
        # 오버랩 토큰 수만큼 뒤로 이동
        overlap_start = max(0, chunk_end - 500)  # 대략적인 시작점
        
        # 이진 탐색으로 정확한 오버랩 위치 찾기
        left, right = overlap_start, chunk_end
        best_start = chunk_end
        
        while left <= right:
            mid = (left + right) // 2
            overlap_text = text[mid:chunk_end]
            tokens = self.embeddings_model._count_tokens(overlap_text)
            
            if tokens <= self.overlap_tokens:
                best_start = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return best_start


class PDFSemanticChunker:
    """
    PDF 파일을 semantic chunking하면서 페이지 번호와 파일명을 메타데이터로 보존하는 클래스
    """

    def __init__(self, embeddings_model=None, 
                 breakpoint_threshold_type: str = 'percentile',
                 breakpoint_threshold_amount: Optional[float] = 75):
        """
        PDFSemanticChunker 초기화
        """
        # Vertex AI 임베딩 모델 기본값 사용
        if embeddings_model is None:
            embeddings_model = VertexEmbeddings()
        
        self.embeddings_model = embeddings_model
        
        # 토큰 기반 후처리용 분할기
        self.post_splitter = TokenBasedTextSplitter(
            embeddings_model=embeddings_model,
            max_tokens=2000,
            overlap_tokens=50
        )
        
        # SemanticChunker 초기화
        self.text_splitter = SemanticChunker(
            embeddings=embeddings_model,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
    
    def extract_text_from_pdf_with_metadata(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        PDF 파일에서 텍스트를 추출하고 페이지 번호와 함께 메타데이터를 보존
        """
        pdf_filename = os.path.basename(pdf_path)
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages_data.append({
                            'text': page_text,
                            'page_number': page_num,
                            'pdf_filename': pdf_filename,
                            'source': pdf_path
                        })
        except Exception as e:
            print(f"PDF 파일 읽기 오류: {e}")
            return []
        
        return pages_data
    
    def find_pages_for_chunk(self, chunk_text: str, pages_data: List[Dict[str, Any]]) -> List[int]:
        """
        특정 chunk가 어떤 페이지들에 속하는지 찾는 함수
        """
        page_numbers = []
        
        def normalize_text(text):
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s가-힣]', '', text)
            return text.strip().lower()
        
        normalized_chunk = normalize_text(chunk_text)
        chunk_words = normalized_chunk.split()
        
        if len(chunk_words) < 10:
            search_patterns = [normalized_chunk[:100], normalized_chunk[-100:]]
        else:
            search_patterns = [
                ' '.join(chunk_words[:10]),
                ' '.join(chunk_words[-10:])
            ]
        
        for page_data in pages_data:
            normalized_page = normalize_text(page_data['text'])
            
            for pattern in search_patterns:
                if pattern and len(pattern) > 10 and pattern in normalized_page:
                    if page_data['page_number'] not in page_numbers:
                        page_numbers.append(page_data['page_number'])
                    break
        
        return sorted(page_numbers) if page_numbers else [1]
    
    def create_semantic_chunks(self, pdf_path: str) -> List[Document]:
        """
        PDF 파일을 semantic chunking하고 메타데이터를 보존
        """
        # PDF에서 텍스트와 메타데이터 추출
        pages_data = self.extract_text_from_pdf_with_metadata(pdf_path)
        
        if not pages_data:
            print("PDF에서 텍스트를 추출할 수 없습니다.")
            return []
        
        # 모든 페이지의 텍스트를 하나로 합치기
        full_text = '\n\n'.join([page['text'] for page in pages_data])
        
        try:
            # 1단계: 전체 텍스트를 semantic chunking으로 의미적 분할
            semantic_chunks = self.text_splitter.split_text(full_text)
            print(f"Semantic chunking으로 {len(semantic_chunks)}개 청크 생성됨")
            
            # 2단계: 각 semantic chunk가 토큰 제한을 초과하면 RecursiveCharacterTextSplitter로 재분할
            final_chunks = []
            for chunk in semantic_chunks:
                # Google AI의 count_tokens을 사용한 정확한 토큰 수 계산
                actual_tokens = self.embeddings_model._count_tokens(chunk)
                
                if actual_tokens > 2000:
                    print(f"청크가 {actual_tokens}토큰으로 계산되어 재분할합니다 (길이: {len(chunk)}자)")
                    # RecursiveCharacterTextSplitter로 재분할
                    sub_chunks = self.post_splitter.split_text(chunk)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            
            chunks = final_chunks
            print(f"최종 {len(chunks)}개 청크 생성됨")
            
        except Exception as e:
            print(f"Semantic chunking 오류: {e}")
            return []
        
        # 각 chunk에 대해 해당하는 페이지 번호를 찾고 메타데이터 추가
        documents = []
        pdf_filename = os.path.basename(pdf_path)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # 해당 chunk가 속한 페이지들을 찾기
            page_numbers = self.find_pages_for_chunk(chunk, pages_data)
            
            # 실제 토큰 수 계산
            actual_tokens = self.embeddings_model._count_tokens(chunk)
            
            metadata = {
                # 'source': pdf_path,  # source는 저장하지 않음
                'pdf_filename': pdf_filename,
                'page_numbers': page_numbers,
                'chunk_index': i,
                'chunk_type': 'semantic',
                'chunk_size': actual_tokens,  # 토큰 수로 변경
                'char_length': len(chunk),    # 글자 수는 별도 저장
                'total_pages': len(pages_data)
            }
            
            if page_numbers:
                metadata['page'] = page_numbers[0]
            
            document = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(document)
        
        return documents

