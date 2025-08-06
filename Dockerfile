# 1. 기본 이미지 설정
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. uv 설치
RUN pip install uv

# 4. 가상환경 생성
RUN python -m venv /app/venv

# 5. 가상환경 활성화
ENV PATH="/app/venv/bin:$PATH"

# 6. 의존성 파일 복사
COPY requirements.txt .

# 7. pip를 사용하여 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 8. 애플리케이션 코드 복사
COPY . .

# 9. 포트 노출
EXPOSE 8000

# 10. 애플리케이션 실행 (가상환경의 uvicorn 사용, 디버그 모드 활성화)
CMD ["uvicorn", "fire_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]