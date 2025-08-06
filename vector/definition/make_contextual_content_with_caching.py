from typing import Optional, List, Any, Dict
from langchain.schema import Document
from tqdm import tqdm
import os
import hashlib
import logging
from datetime import datetime
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
from google import genai
from google.genai.types import Content as GenaiContent, CreateCachedContentConfig, Part as GenaiPart
import warnings
warnings.filterwarnings("ignore")
from google.oauth2 import service_account



os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["PROJECT_ID"] = os.getenv("PROJECT_ID")
os.environ["GOOGLE_CLOUD_LOCATION"] = "asia-northeast1"



# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Anthropic's Contextual Enrichment Prompt (Korean)
SYSTEM_PROMPT = """
You are an expert at providing brief context to document chunks. Your task is to provide specific contextual information that explains what this chunk is referring to, including concrete details from the full document.
Always respond in Korean language only.
"""


def load_text_from_file(text_file_path: str) -> str:
    """
    original_full_text 디렉토리에서 텍스트 파일 로드
    
    Args:
        text_file_path: 텍스트 파일 경로
        
    Returns:
        전체 문서 텍스트
    """
    text_file_path = "../data/original_full_text/" + text_file_path + ".txt"


    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"텍스트 파일을 찾을 수 없습니다: {text_file_path}")
    
    with open(text_file_path, 'rb') as f:
        full_text = f.read().decode('utf-8')
        
    logger.info(f"텍스트 파일 로드 완료: {text_file_path} ({len(full_text)} 문자)")
    return full_text


def count_tokens(text: str, model: str = "gemini-1.5-flash") -> int:
    """
    Google AI의 GenerativeModel을 사용하여 정확한 토큰 수 계산
    semantic_split_vertex.py의 _count_tokens 방식을 따름
    
    Args:
        text: 토큰을 계산할 텍스트
        model: 사용할 모델명
        
    Returns:
        토큰 수
    """
    try:
        # genai 모듈이 설정되어 있는지 확인
        if google_api_key:
            import google.generativeai as genai_count
            genai_count.configure(api_key=google_api_key)
            model_instance = genai_count.GenerativeModel(model)
            count_response = model_instance.count_tokens(text)
            return count_response.total_tokens
        else:
            logger.warning("GOOGLE_API_KEY가 설정되지 않아 작업을 중지합니다.")
            raise
    except Exception as e:
        logger.warning(f"토큰 계산 오류 작업 중지")
        raise

def check_document_size(text: str, max_tokens: int = 131000) -> tuple[bool, int]:
    """
    문서 크기를 토큰 단위로 체크하여 캐시 생성 가능 여부 확인
    
    Args:
        text: 체크할 텍스트
        max_tokens: 최대 허용 토큰 수 (Gemini 캐시 제한: 131,072)
        
    Returns:
        (크기_적합_여부, 실제_토큰_수) 튜플
    """
    actual_tokens = count_tokens(text)
    is_valid = actual_tokens <= max_tokens
    
    if not is_valid:
        logger.warning(f"⚠️ 문서가 토큰 제한을 초과합니다: {actual_tokens:,} 토큰 (제한: {max_tokens:,} 토큰)")
        logger.warning(f"   문서 길이: {len(text):,} 글자")
    else:
        logger.info(f"✅ 문서 크기 확인 완료: {actual_tokens:,} 토큰 (제한: {max_tokens:,} 토큰)")
    
    return is_valid, actual_tokens

def generate_cache_key(text: str, prefix: str = "pdf_context") -> str:
    """
    텍스트의 해시값을 기반으로 캐시 키 생성
    
    Args:
        text: 캐싱할 텍스트
        prefix: 캐시 키 접두사
        
    Returns:
        캐시 키
    """
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"{prefix}_{timestamp}_{text_hash}"

def create_cached_content(full_text: str, cache_ttl_hours: int = 1) -> str:
    """
    Google GenAI API를 사용하여 전체 문서 텍스트 캐싱
    
    Args:
        full_text: 캐싱할 전체 문서 텍스트
        cache_ttl_hours: 캐시 TTL (시간)
        
    Returns:
        캐시 이름(ID)
        
    Raises:
        ValueError: 텍스트가 너무 짧거나 토큰 제한을 초과하는 경우
        Exception: 캐시 생성 실패
    """
    # 최소 토큰 요구사항 체크 (대략 1024토큰 = 1536문자)
    if len(full_text) < 1536:
        raise ValueError(f"텍스트가 너무 짧습니다 ({len(full_text)} 문자, 최소 1536 문자 필요)")
    
    # 문서 크기 체크 (토큰 제한 확인)
    is_valid, actual_tokens = check_document_size(full_text, max_tokens=131072)
    if not is_valid:
        raise ValueError(f"문서가 토큰 제한을 초과합니다: {actual_tokens:,} 토큰 (제한: 131,072 토큰). 문서를 분할하여 처리하세요.")
    
    # GenAI 클라이언트 초기화 (Vertex AI 모드)
    client = genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
    )
    
    # 캐시 키 생성
    cache_key = generate_cache_key(full_text)
    
    # 문서 내용을 컨텐츠로 구성
    contents = [
        GenaiContent(
            role="user",
            parts=[GenaiPart(text=f"다음은 전체 문서의 내용입니다:\n\n{full_text}")]
        )
    ]
    
    # 캐시 생성
    content_cache = client.caches.create(
        model="gemini-2.5-flash-lite",
        config=CreateCachedContentConfig(
            contents=contents,
            system_instruction=SYSTEM_PROMPT,
            display_name=cache_key,
            ttl=f"{int(cache_ttl_hours * 3600)}s",
        )
    )
    
    logger.info(f"문서 캐싱 완료: {cache_key}")
    logger.info(f"캐시 이름: {content_cache.name}")
    if hasattr(content_cache, 'usage_metadata') and content_cache.usage_metadata:
        logger.info(f"토큰 수: {content_cache.usage_metadata.total_token_count}")
    logger.info(f"TTL: {cache_ttl_hours}시간")
    
    return content_cache.name

def get_or_create_cache(
    document_name: Optional[str] = None,
    text_file_path: Optional[str] = None,
    full_text: Optional[str] = None,
    cache_ttl_hours: int = 1
) -> tuple[Optional[str], str]:
    """
    기존 캐시를 찾거나 새로운 캐시 생성
    
    Args:
        document_name: 문서명 (original_full_text에서 해당 텍스트 파일 검색)
        text_file_path: 직접 지정된 텍스트 파일 경로
        full_text: 직접 제공된 전체 텍스트
        cache_ttl_hours: 캐시 TTL (시간)
        
    Returns:
        (캐시 이름 또는 None, 전체 텍스트) 튜플
        캐시 이름이 None이면 토큰 제한 초과로 캐시 생성 불가
    """
    # 전체 텍스트 확보 우선순위: full_text > text_file_path > document_name
    if full_text is None:
        if text_file_path is not None:
            full_text = load_text_from_file(text_file_path)
        elif document_name is not None:
            full_text = load_text_from_file(document_name)
        else:
            raise ValueError("document_name, text_file_path 또는 full_text 중 하나는 반드시 제공되어야 합니다")
    
    # 문서 크기 사전 체크
    is_valid, actual_tokens = check_document_size(full_text, max_tokens=131000)
    if not is_valid:
        logger.warning(f"📄 문서가 토큰 제한을 초과하여 캐시를 생성하지 않습니다: {actual_tokens:,} 토큰")
        logger.warning(f"   context_enrichment를 None으로 설정합니다")
        return None, full_text
    
    # 기존 캐시 검색
    cache_key = generate_cache_key(full_text)
    
    # GenAI 클라이언트 초기화
    client = genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
    )
    
    # 기존 캐시 조회
    try:
        cached_contents = client.caches.list()
        
        for cached_content in cached_contents:
            if cached_content.display_name == cache_key:
                logger.info(f"기존 캐시 발견: {cache_key}")
                return cached_content.name, full_text
        
        logger.info(f"기존 캐시 없음. 새 캐시 생성: {cache_key}")
    except Exception as e:
        logger.warning(f"캐시 목록 조회 실패: {e}. 새 캐시 생성")
    
    # 새 캐시 생성
    cache_name = create_cached_content(full_text, cache_ttl_hours)
    return cache_name, full_text

def contextual_content_with_caching_batch(
    docs: List[Document],
    document_name: Optional[str] = None,
    text_file_path: Optional[str] = None,
    full_text: Optional[str] = None,
    model: Optional[GenerativeModel] = "gemini-1.5-flash",
    batch_size: int = 5,
    cache_ttl_hours: int = 1,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    캐싱된 전체 문서를 컨텍스트로 사용한 배치 처리 맥락 정보 생성
    
    Args:
        docs: 문서 청크 리스트
        document_name: 문서명 (original_full_text에서 해당 텍스트 파일 검색)
        text_file_path: 직접 지정된 텍스트 파일 경로
        full_text: 직접 제공된 전체 텍스트
        model: Gemini 모델 (None이면 기본 모델 사용)
        batch_size: 한 번에 처리할 청크 수
        cache_ttl_hours: 캐시 TTL (시간)
        show_progress: 진행률 표시 여부
        
    Returns:
        JSON 형태의 문서 리스트
    """
    if not docs:
        return []
    
    # 문서 정렬
    logger.info(f"정렬 전 문서 개수: {len(docs)}")
    sorted_docs = sorted(docs, key=lambda doc: (
        doc.metadata.get('page', 0), 
        doc.metadata.get('chunk_index', 0)
    ))
    
    # 캐시된 콘텐츠 생성/조회
    cache_name, _ = get_or_create_cache(
        document_name=document_name,
        text_file_path=text_file_path,
        full_text=full_text, 
        cache_ttl_hours=cache_ttl_hours
    )
    
    # 캐시가 None이면 모든 청크의 context_enrichment를 None으로 설정
    if cache_name is None:
        logger.info(f"토큰 제한 초과로 인해 모든 청크의 context_enrichment를 None으로 설정합니다")
        enriched_docs = []
        for i, doc in enumerate(sorted_docs):
            enhanced_doc = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "context_enrichment": None
            }
            enriched_docs.append(enhanced_doc)
        logger.info(f"전체 처리 완료: {len(enriched_docs)}개 청크 (context_enrichment: None)")
        return enriched_docs
    
    # 캐시된 콘텐츠로 모델 초기화
    if model is None:
        vertexai.init(
            location="asia-northeast1"
        )
        model = GenerativeModel.from_cached_content(cache_name)
        logger.info(f"캐싱된 콘텐츠를 사용한 모델 초기화 완료: {cache_name}")
    
    enriched_docs = []
    total_batches = (len(sorted_docs) + batch_size - 1) // batch_size
    
    if show_progress:
        batch_iterator = tqdm(range(total_batches), desc=f"캐싱된 컨텍스트 배치 처리 (배치크기={batch_size})")
    else:
        batch_iterator = range(total_batches)
    
    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sorted_docs))
        batch_docs = sorted_docs[start_idx:end_idx]
        
        try:
            # 배치 프롬프트 구성 (캐싱된 전체 문서를 컨텍스트로 사용)
            batch_prompts = []
            for i, doc in enumerate(batch_docs):
                global_idx = start_idx + i
                
                prompt_part = f"""
                    청크 {global_idx + 1}:
                    <chunk>
                    {doc.page_content}
                    </chunk>
                """
                batch_prompts.append(prompt_part)
            
            # 전체 배치 프롬프트 (캐싱된 문서 컨텍스트 활용)
            full_batch_prompt = f"""
                위의 캐싱된 전체 문서를 참고하여, 각 청크에 대한 간결한 한 문장의 맥락 요약을 제공해주세요. 
                전체 문서의 구체적인 세부사항(이름, 날짜, 장소, 사건)을 포함하여 검색에 유용한 정보를 제공하세요.

                예시: 청크가 "해당 사고 이후 5시간 만에 복구작업이 완료되었다"라면, 
                컨텍스트는 "2020년 3월 1일 강원도에서 발생한 5중 추돌 사고의 복구 과정"이어야 합니다.

                다음 {len(batch_docs)}개의 청크를 처리해주세요:

                {"".join(batch_prompts)}

                정확히 {len(batch_docs)}개의 한국어 문장으로 답변하세요. 각 청크마다 순서대로 하나씩:
                1. [첫 번째 청크 컨텍스트]
                2. [두 번째 청크 컨텍스트]
                ...

                "이 청크는"이나 "이 부분은" 같은 표현은 사용하지 마세요.
            """
            
            # Gemini API 호출 (캐싱된 컨텍스트 활용)
            response = model.generate_content(full_batch_prompt)

            response_text = response.text.strip()
            
            # 응답 파싱
            lines = response_text.split('\n')
            contexts = []
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith(str(len(contexts) + 1) + '.') or line.startswith(f"{len(contexts) + 1}.")):
                    # 번호 제거하고 컨텍스트만 추출
                    context = line.split('.', 1)[1].strip() if '.' in line else line
                    contexts.append(context)
                    if len(contexts) >= len(batch_docs):
                        break
            
            # 응답이 부족한 경우 기본값 채우기
            while len(contexts) < len(batch_docs):
                contexts.append("전체 문서 맥락에서의 관련 내용")
            
            # 결과 구성
            for i, (doc, context) in enumerate(zip(batch_docs, contexts)):
                enhanced_doc = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "context_enrichment": context
                }
                enriched_docs.append(enhanced_doc)
                    
        except Exception as e:
            logger.error(f"배치 {batch_idx + 1} 처리 중 오류 발생: {e}")
            # 해당 배치의 모든 청크를 기본값으로 처리
            for doc in batch_docs:
                enhanced_doc = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "context_enrichment": None
                }
                enriched_docs.append(enhanced_doc)
    
    logger.info(f"전체 처리 완료: {len(enriched_docs)}개 청크")
    return enriched_docs

def contextual_content_with_caching(
    docs: List[Document],
    document_name: Optional[str] = None,
    text_file_path: Optional[str] = None,
    full_text: Optional[str] = None,
    model: Optional[GenerativeModel] = "gemini-2.5-flash-lite",
    cache_ttl_hours: int = 1,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    캐싱된 전체 문서를 컨텍스트로 사용한 순차 처리 맥락 정보 생성
    
    Args:
        docs: 문서 청크 리스트
        document_name: 문서명 (original_full_text에서 해당 텍스트 파일 검색)
        text_file_path: 직접 지정된 텍스트 파일 경로
        full_text: 직접 제공된 전체 텍스트
        model: Gemini 모델 (기본값: gemini-2.5-flash-lite)
        cache_ttl_hours: 캐시 TTL (시간)
        show_progress: 진행률 표시 여부
        
    Returns:
        JSON 형태의 문서 리스트
    """
    if not docs:
        return []
    
    # 문서 정렬
    logger.info(f"정렬 전 문서 개수: {len(docs)}")
    sorted_docs = sorted(docs, key=lambda doc: (
        doc.metadata.get('page', 0), 
        doc.metadata.get('chunk_index', 0)
    ))
    
    # 캐시된 콘텐츠 생성/조회
    cache_name, _ = get_or_create_cache(
        document_name=document_name,
        text_file_path=text_file_path,
        full_text=full_text, 
        cache_ttl_hours=cache_ttl_hours
    )
    
    # 캐시가 None이면 모든 청크의 context_enrichment를 None으로 설정
    if cache_name is None:
        logger.info(f"토큰 제한 초과로 인해 모든 청크의 context_enrichment를 None으로 설정합니다")
        enriched_docs = []
        for i, doc in enumerate(sorted_docs):
            enhanced_doc = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "context_enrichment": None
            }
            enriched_docs.append(enhanced_doc)
        logger.info(f"전체 처리 완료: {len(enriched_docs)}개 청크 (context_enrichment: None)")
        return enriched_docs
    
    # 캐시된 콘텐츠로 모델 초기화
    if model is None:
        vertexai.init(
            project=project_id, 
            location="asia-northeast1",
            credentials=credentials
        )
        model = GenerativeModel.from_cached_content(cache_name)
        logger.info(f"캐싱된 콘텐츠를 사용한 모델 초기화 완료: {cache_name}")
    
    iterator = tqdm(enumerate(sorted_docs), desc="캐싱된 컨텍스트 순차 처리") if show_progress else enumerate(sorted_docs)
    
    enriched_docs = []
    
    for i, doc in iterator:
        try:
            prompt = f"""
            위의 캐싱된 전체 문서를 참고하여, 다음 청크에 대한 간결한 한 문장의 맥락 요약을 제공해주세요.
            전체 문서의 구체적인 세부사항(이름, 날짜, 장소, 사건)을 포함하여 검색에 유용한 정보를 제공하세요.

            <chunk>
            {doc.page_content}
            </chunk>

            예시: 청크가 "해당 사고 이후 5시간 만에 복구작업이 완료되었다"라면, 
            컨텍스트는 "2020년 3월 1일 강원도에서 발생한 5중 추돌 사고의 복구 과정"이어야 합니다.

            오직 한 문장의 간결한 한국어로만 답변하세요. "이 청크는"이나 "이 부분은" 같은 표현은 사용하지 마세요.
            """
            
            response = model.generate_content(prompt)
            contextual_info = response.text.strip()
            
            enhanced_doc = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "context_enrichment": contextual_info
            }
            enriched_docs.append(enhanced_doc)
                    
        except Exception as e:
            logger.error(f"청크 {i} 처리 중 오류 발생: {e}")
            raise
    
    logger.info(f"전체 처리 완료: {len(enriched_docs)}개 청크")
    return enriched_docs

def cleanup_expired_caches(dry_run: bool = True) -> List[str]:
    """
    만료된 캐시들을 정리
    
    Args:
        dry_run: True면 삭제할 캐시만 나열, False면 실제 삭제
        
    Returns:
        처리된 캐시 목록
    """
    # GenAI 클라이언트 초기화
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location="asia-northeast1",
        credentials=credentials
    )
    
    cached_contents = client.caches.list()
    expired_caches = []
    
    for cached_content in cached_contents:
        try:
            # 캐시 만료 시간 확인
            if hasattr(cached_content, 'expire_time') and cached_content.expire_time:
                current_time = datetime.now()
                expire_time = cached_content.expire_time
                
                # timezone aware datetime을 naive로 변환
                if hasattr(expire_time, 'replace') and expire_time.tzinfo is not None:
                    expire_time = expire_time.replace(tzinfo=None)
                
                if current_time > expire_time:
                    expired_caches.append(cached_content.display_name)
                    if not dry_run:
                        client.caches.delete(cached_content.name)
                        logger.info(f"만료된 캐시 삭제: {cached_content.display_name}")
        except Exception as e:
            logger.warning(f"캐시 {cached_content.display_name} 처리 중 오류: {e}")
    
    if dry_run:
        logger.info(f"삭제 예정 캐시 {len(expired_caches)}개: {expired_caches}")
    else:
        logger.info(f"삭제 완료 캐시 {len(expired_caches)}개")
        
    return expired_caches

if __name__ == "__main__":
    pass