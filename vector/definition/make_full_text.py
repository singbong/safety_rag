#!/usr/bin/env python3
"""
PDF Full Text Extraction Script

PDF 파일들에서 전체 텍스트를 추출하여 텍스트 파일로 저장하는 스크립트
original_pdf 폴더의 모든 PDF 파일을 처리하여 original_full_text 폴더에 저장
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber
from tqdm import tqdm
import json
from datetime import datetime
import argparse


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    """PDF 텍스트 추출기 클래스"""
    
    def __init__(self, 
                 input_dir: str = None, 
                 output_dir: str = None,
                 batch_size: int = 5):

        # 기본 경로 설정
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "data"
        
        self.input_dir = Path(input_dir) if input_dir else Path("../data/original_pdf")
        self.output_dir = Path(output_dir) if output_dir else Path("../data/original_full_text")
        self.batch_size = batch_size
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        
        # 처리 통계
        self.stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None,
            "errors": []
        }
        
        logger.info(f"PDF 텍스트 추출기 초기화")
        logger.info(f"입력 디렉토리: {self.input_dir}")
        logger.info(f"출력 디렉토리: {self.output_dir}")
        logger.info(f"배치 크기: {self.batch_size}")
    
    def get_pdf_files(self) -> List[Path]:
        """PDF 파일 목록 반환"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {self.input_dir}")
        
        pdf_files = list(self.input_dir.glob("*.pdf"))
        logger.info(f"발견된 PDF 파일 수: {len(pdf_files)}")
        
        for pdf_file in pdf_files[:5]:  # 처음 5개 파일명 로깅
            logger.info(f"  - {pdf_file.name}")
        
        if len(pdf_files) > 5:
            logger.info(f"  ... 및 {len(pdf_files) - 5}개 추가 파일")
        
        return sorted(pdf_files)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, any]:
        """
        PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            추출 결과 딕셔너리
        """
        result = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "success": False,
            "full_text": "",
            "page_count": 0,
            "character_count": 0,
            "extraction_time": None,
            "error_message": None,
            "metadata": {}
        }
        
        start_time = datetime.now()
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                total_pages = len(pdf.pages)
                
                logger.debug(f"PDF 파일 열기 성공: {pdf_path.name} (총 {total_pages}페이지)")
                
                # 메타데이터 추출
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    result["metadata"] = {
                        "title": pdf.metadata.get('Title', ''),
                        "author": pdf.metadata.get('Author', ''),
                        "subject": pdf.metadata.get('Subject', ''),
                        "creator": pdf.metadata.get('Creator', ''),
                        "producer": pdf.metadata.get('Producer', ''),
                        "creation_date": str(pdf.metadata.get('CreationDate', '')),
                        "modification_date": str(pdf.metadata.get('ModDate', ''))
                    }
                
                # 페이지별 텍스트 추출
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            # 페이지 헤더 추가
                            page_text = f"\n\n--- 페이지 {page_num} ---\n{text.strip()}"
                            pages_text.append(page_text)
                            logger.debug(f"페이지 {page_num} 텍스트 추출 완료 ({len(text)} 문자)")
                        else:
                            logger.warning(f"페이지 {page_num}에서 텍스트를 찾을 수 없습니다")
                            
                    except Exception as page_error:
                        logger.error(f"페이지 {page_num} 처리 중 오류: {page_error}")
                        continue
                
                # 전체 텍스트 결합
                full_text = "\n".join(pages_text)
                
                # 결과 설정
                result.update({
                    "success": True,
                    "full_text": full_text,
                    "page_count": total_pages,
                    "character_count": len(full_text),
                    "extraction_time": (datetime.now() - start_time).total_seconds()
                })
                
                logger.info(f"텍스트 추출 완료: {pdf_path.name} ({total_pages}페이지, {len(full_text):,}문자)")
                
        except Exception as e:
            result.update({
                "success": False,
                "error_message": str(e),
                "extraction_time": (datetime.now() - start_time).total_seconds()
            })
            logger.error(f"PDF 텍스트 추출 실패 - {pdf_path.name}: {e}")
        
        return result
    
    def save_text_file(self, result: Dict[str, any], output_format: str = "txt") -> bool:
        """
        추출된 텍스트를 파일로 저장
        
        Args:
            result: 추출 결과 딕셔너리
            output_format: 출력 형식 ("txt" 또는 "json")
            
        Returns:
            저장 성공 여부
        """
        if not result["success"]:
            return False
        
        try:
            # 파일명 생성 (확장자 제거하고 .txt 추가)
            base_name = Path(result["file_name"]).stem
            
            if output_format == "txt":
                output_file = self.output_dir / f"{base_name}.txt"
                
                # 텍스트 파일 저장
                with open(output_file, 'w', encoding='utf-8') as f:
                    # 헤더 정보 추가
                    f.write(f"# 추출 정보\n")
                    f.write(f"원본 파일: {result['file_name']}\n")
                    f.write(f"페이지 수: {result['page_count']}\n")
                    f.write(f"문자 수: {result['character_count']:,}\n")
                    f.write(f"추출 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"처리 시간: {result['extraction_time']:.2f}초\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # 본문 텍스트
                    f.write(result["full_text"])
                
                logger.info(f"텍스트 파일 저장 완료: {output_file}")
                
            elif output_format == "json":
                output_file = self.output_dir / f"{base_name}.json"
                
                # JSON 파일 저장
                json_data = {
                    "extraction_info": {
                        "file_name": result["file_name"],
                        "page_count": result["page_count"],
                        "character_count": result["character_count"],
                        "extraction_time": result["extraction_time"],
                        "extraction_date": datetime.now().isoformat()
                    },
                    "metadata": result["metadata"],
                    "full_text": result["full_text"]
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"JSON 파일 저장 완료: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"파일 저장 실패 - {result['file_name']}: {e}")
            return False
    
    def process_batch(self, pdf_files: List[Path], output_format: str = "txt") -> None:
        """배치 단위로 PDF 파일들 처리"""
        for i in range(0, len(pdf_files), self.batch_size):
            batch = pdf_files[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(pdf_files) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"배치 {batch_num}/{total_batches} 처리 중 ({len(batch)}개 파일)")
            
            for pdf_file in batch:
                # 이미 처리된 파일 확인
                base_name = pdf_file.stem
                output_file = self.output_dir / f"{base_name}.{output_format}"
                
                if output_file.exists():
                    logger.info(f"이미 존재하는 파일 건너뛰기: {pdf_file.name}")
                    self.stats["skipped"] += 1
                    continue
                
                # PDF 텍스트 추출
                result = self.extract_text_from_pdf(pdf_file)
                
                if result["success"]:
                    # 텍스트 파일 저장
                    if self.save_text_file(result, output_format):
                        self.stats["successful"] += 1
                    else:
                        self.stats["failed"] += 1
                        self.stats["errors"].append({
                            "file": pdf_file.name,
                            "error": "파일 저장 실패"
                        })
                else:
                    self.stats["failed"] += 1
                    self.stats["errors"].append({
                        "file": pdf_file.name,
                        "error": result["error_message"]
                    })
    
    def run(self, output_format: str = "txt", force: bool = False) -> Dict[str, any]:
        """
        전체 처리 실행
        
        Args:
            output_format: 출력 형식 ("txt" 또는 "json")
            force: 기존 파일 덮어쓰기 여부
            
        Returns:
            처리 결과 통계
        """
        self.stats["start_time"] = datetime.now()
        
        try:
            # PDF 파일 목록 가져오기
            pdf_files = self.get_pdf_files()
            self.stats["total_files"] = len(pdf_files)
            
            if not pdf_files:
                logger.warning("처리할 PDF 파일이 없습니다.")
                return self.stats
            
            # 기존 파일 정리 (force 옵션)
            if force:
                logger.info("기존 출력 파일들을 삭제합니다.")
                for existing_file in self.output_dir.glob(f"*.{output_format}"):
                    existing_file.unlink()
            
            # 배치 처리 실행
            logger.info(f"PDF 텍스트 추출 시작 - 총 {len(pdf_files)}개 파일")
            
            with tqdm(total=len(pdf_files), desc="PDF 처리") as pbar:
                for i in range(0, len(pdf_files), self.batch_size):
                    batch = pdf_files[i:i + self.batch_size]
                    
                    for pdf_file in batch:
                        # 이미 처리된 파일 확인
                        base_name = pdf_file.stem
                        output_file = self.output_dir / f"{base_name}.{output_format}"
                        
                        if output_file.exists() and not force:
                            logger.debug(f"이미 존재하는 파일 건너뛰기: {pdf_file.name}")
                            self.stats["skipped"] += 1
                            pbar.update(1)
                            continue
                        
                        # PDF 텍스트 추출
                        result = self.extract_text_from_pdf(pdf_file)
                        
                        if result["success"]:
                            # 텍스트 파일 저장
                            if self.save_text_file(result, output_format):
                                self.stats["successful"] += 1
                            else:
                                self.stats["failed"] += 1
                                self.stats["errors"].append({
                                    "file": pdf_file.name,
                                    "error": "파일 저장 실패"
                                })
                        else:
                            self.stats["failed"] += 1
                            self.stats["errors"].append({
                                "file": pdf_file.name,
                                "error": result["error_message"]
                            })
                        
                        pbar.update(1)
            
        except Exception as e:
            logger.error(f"전체 처리 중 오류 발생: {e}")
            self.stats["errors"].append({
                "file": "전체 처리",
                "error": str(e)
            })
        
        finally:
            self.stats["end_time"] = datetime.now()
            self._print_summary()
        
        return self.stats
    
    def _print_summary(self) -> None:
        """처리 결과 요약 출력"""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        logger.info("=" * 60)
        logger.info("PDF 텍스트 추출 완료")
        logger.info("=" * 60)
        logger.info(f"총 파일 수: {self.stats['total_files']}")
        logger.info(f"성공: {self.stats['successful']}")
        logger.info(f"실패: {self.stats['failed']}")
        logger.info(f"건너뛰기: {self.stats['skipped']}")
        logger.info(f"처리 시간: {duration:.2f}초")
        
        if self.stats["errors"]:
            logger.error(f"오류 발생 ({len(self.stats['errors'])}건):")
            for error in self.stats["errors"]:
                logger.error(f"  - {error['file']}: {error['error']}")
        
        logger.info(f"출력 디렉토리: {self.output_dir}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="PDF 파일들에서 전체 텍스트를 추출하여 텍스트 파일로 저장"
    )
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        help="PDF 입력 디렉토리 (기본값: ../data/original_pdf)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="텍스트 출력 디렉토리 (기본값: ../data/original_full_text)"
    )
    
    parser.add_argument(
        "--format", 
        choices=["txt", "json"], 
        default="txt",
        help="출력 형식 (기본값: txt)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=5,
        help="배치 처리 크기 (기본값: 5)"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="기존 파일 덮어쓰기"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # PDF 텍스트 추출기 초기화
        extractor = PDFTextExtractor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        
        # 처리 실행
        stats = extractor.run(
            output_format=args.format,
            force=args.force
        )
        
        # 종료 코드 설정
        if stats["failed"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()