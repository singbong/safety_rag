from make_contextual_content_with_caching import contextual_content_with_caching_batch, contextual_content_with_caching, cleanup_expired_caches
import pickle
from tqdm import tqdm
import glob
import os
from pathlib import Path
import logging
import warnings
warnings.filterwarnings("ignore")



# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_chunked_docs_with_caching():
    """
    chunked_docs에서 pkl 파일들을 읽어서 캐싱된 컨텍스트 생성 후 저장
    """
    # chunked_docs 폴더의 pkl 파일들 가져오기
    chunked_docs_paths = glob.glob("../data/chunked_docs/*.pkl")
    print(chunked_docs_paths)
    
    # 각 파일 처리
    for chunked_docs_path in tqdm(chunked_docs_paths):

        file_name = chunked_docs_path.split('/')[-1].split('.')[0]
        output_path = f"../data/contextual_content_docs/{file_name}.pkl"
        # 이미 처리된 파일 건너뛰기
        if os.path.exists(output_path):
            logger.info(f"이미 처리된 파일 건너뛰기: {file_name}")
            continue
        
        try:
            # 청크 문서 로드
            with open(chunked_docs_path, "rb") as f:
                docs = pickle.load(f)
            
            logger.info(f"{file_name} 처리 중 - {len(docs)}개 청크")
            
            # 캐싱된 컨텍스트 생성 (문서명으로 자동 텍스트 파일 검색)
            result = contextual_content_with_caching_batch(
                docs=docs,
                document_name=file_name,  # pkl 파일명과 동일한 텍스트 파일 검색
                cache_ttl_hours=3,  
                show_progress=True,
                batch_size=5
            )
            
            # 결과 저장
            with open(output_path, "wb") as f:
                pickle.dump(result, f)
            
            logger.info(f"저장 완료: {output_path}")
            
        except FileNotFoundError as e:
            logger.warning(f"{file_name}에 해당하는 텍스트 파일을 찾을 수 없습니다: {e}")
            logger.info("해당 파일은 건너뛰고 계속 진행합니다.")
            continue
        
        except ValueError as e:
            logger.warning(f"{file_name} 처리 중 값 오류: {e}")
            logger.info("해당 파일은 건너뛰고 계속 진행합니다.")
            continue
            
        except Exception as e:
            logger.error(f"{file_name} 처리 중 예상치 못한 오류 발생: {e}")
            logger.info("해당 파일은 건너뛰고 계속 진행합니다.")
            continue
    
    logger.info("모든 파일 처리 완료!")

if __name__ == "__main__":
    os.makedirs("../data/contextual_content_docs", exist_ok=True)
    cleanup_expired_caches(dry_run=False)
    process_chunked_docs_with_caching()