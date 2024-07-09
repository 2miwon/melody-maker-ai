# Python 공식 이미지를 기반으로 설정
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 Python 패키지 설치
RUN pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app

# 애플리케이션 실행
CMD ["reflex", "run"]