# 프로젝트 Context: 제약사 연구소 문서 기반 RAG Agent 아키텍처

본 문서는 신약 개발 연구소의 문서를 기반으로 답변을 생성하는 RAG Agent 프로젝트의 요구사항, 아키텍처 구조, 구현 내용 및 의사결정 사항을 요약한 것입니다.

## 1. 프로젝트 개요 및 요구사항
- **목표**: 제약사 연구소에서 발생하는 다양한 문서(임상시험계획서, 연구노트, 특허 등)를 기반으로 정확한 정보를 추출하고 질문에 답변할 수 있는 RAG(Retrieval-Augmented Generation) 시스템 구축
- **입력 데이터**: PDF 또는 DOCX 포맷의 문서 파일
- **핵심 요구사항**:
  - 문서 내의 표(Table) 형태 데이터와 목차(TOC) 등의 구조적 컨텍스트를 손실 없이 보존하고 인식해야 함.
  - 특정 문서 포맷에 대해 고정된 키-값(Key-Values) 추출 기능이 필요함.
  - 보안과 성능을 고려하여 엔터프라이즈급 데이터베이스 및 파이프라인을 구축해야 함.

## 2. 아키텍처 구조 (GCP 기반)

아키텍처는 Google Cloud Platform(GCP)의 다양한 AI 및 데이터베이스 서비스들을 결합하여 구성되었습니다.

1. **문서 스토리지 (Google Cloud Storage - GCS)**
   - 역할: 원본 문서(PDF, DOCX) 및 처리 파이프라인의 비동기 결과물 임시 저장소
2. **문서 파싱 및 청킹 (Document AI & Vertex AI Search)**
   - **Vertex AI Search (Document Processing API)**: 단순 파싱(Layout-aware chunking) 용도로만 사용. 문서의 목차 구조와 표(Table)를 HTML 또는 마크다운 형태의 구조화된 텍스트로 자동 보존하며 청킹 수행.
   - **Document AI (Custom Extractor)**: 정형화된 문서에서 특정 데이터(Key Values, Tables)를 집중적으로 분석하고 추출하기 위해 맞춤형 모델을 학습하여 사용.
3. **임베딩 및 벡터 데이터베이스 (Vertex AI Embeddings & AlloyDB AI)**
   - **임베딩 모델**: 파싱된 텍스트 청크를 `text-embedding-004` 모델을 활용하여 고차원 벡터로 변환.
   - **저장소**: 서버리스가 아닌 **완전 관리형 클러스터(Fully Managed Cluster)** 형태의 AlloyDB를 사용. `pgvector` 확장을 통해 텍스트 메타데이터 및 벡터 임베딩을 저장.
4. **검색 및 답변 생성 (RAG Agent)**
   - **Retriever (검색)**: AlloyDB AI에서 사용자 질문 벡터와 가장 코사인 유사도가 높은 문서 청크를 검색 (벡터 기반 검색).
   - **Generator (생성)**: 검색된 텍스트 및 표 데이터를 Context로 제공받아 **Vertex AI Gemini 1.5 Pro** 모델을 통해 사용자 질문에 대한 최종 답변 생성.

## 3. 구현 산출물 목록

모든 구현체는 `c:\Users\workspace\DaewonPharm` 경로에 저장되어 있습니다.

- **gcp_setup_guide.md**: 프로젝트 API 활성화, 버킷 생성, Document AI Custom Extractor 학습, AlloyDB 완전 관리형 클러스터 및 Service Account 설정 방법 안내서
- **requirements.txt**: `google-cloud-documentai`, `langchain-google-vertexai`, `asyncpg`, `pgvector` 등 파이프라인 실행을 위한 필수 파이썬 패키지 명세서
- **01_upload_documents.py**: 로컬 환경의 문서들을 GCS 버킷에 비동기/병렬 방식으로 업로드하는 파이프라인 시작점
- **02_document_processing.py**: 업로드된 문서를 Vertex AI Layout-aware chunking API와 Document AI Custom Extractor를 통해 각각의 목적에 맞게 파싱하는 모듈
- **03_embed_and_store.py**: 분리된 청크 데이터를 Vertex AI 임베딩 API로 벡터화한 후, AlloyDB(pgvector)에 일괄 삽입(Insert)하는 스크립트
- **04_rag_agent.py**: 사용자 질문을 벡터화하여 AlloyDB에서 검색하고, 추출된 Context(특히 표의 HTML 포맷)를 바탕으로 Gemini 1.5 Pro가 답변을 제공하는 최종 Agent 스크립트

## 4. 논의 및 의사결정 히스토리
- **경로 이슈**: 초기 윈도우 OS 시스템의 권한 문제(Access Denied)로 인해 사용자 홈 폴더를 활용하였으나, 이후 원본 요구 경로인 `c:\Users\workspace\DaewonPharm`로 정상 이관 및 경로 매핑 완료함.
- **검색 엔진 분리**: Vertex AI Search는 데이터 스토어를 활용한 E2E 검색 엔진이 아닌, 순수 파싱 모듈(Layout-aware chunking)로만 역할을 한정시키고, 벡터 서치 엔진의 주도권은 AlloyDB AI에 두는 아키텍처로 의사결정함.
- **맞춤형 파서**: 문서 분석 시 Key Values, Tables, OCR 중 최적의 방법을 선택하기 위해 Document AI Custom Extractor 학습 단계를 아키텍처 및 가이드에 포함함.
