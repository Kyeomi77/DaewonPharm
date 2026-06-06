# GCP 서비스 설정 가이드: 제약사 연구소 RAG Agent 아키텍처

이 가이드는 신약 개발 문서를 기반으로 한 Agent 아키텍처 구축을 위해 필요한 Google Cloud Platform(GCP) 서비스들을 설정하는 방법입니다.

---

## 1. 프로젝트 및 API 활성화

GCP 콘솔(https://console.cloud.google.com/) 에 접속하여 새 프로젝트를 생성하거나 기존 프로젝트를 선택한 후, 아래 API들을 활성화합니다.
- `Cloud Storage API`
- `Document AI API`
- `Vertex AI API`
- `AlloyDB API`
- `Service Usage API`

> **Tip**: Cloud Shell을 사용하면 다음 명령어로 한 번에 활성화할 수 있습니다.
> ```bash
> gcloud services enable storage.googleapis.com documentai.googleapis.com aiplatform.googleapis.com alloydb.googleapis.com
> ```

---

## 2. Cloud Storage (GCS) 설정

원문 PDF/DOCX 파일을 업로드할 버킷과 Document AI / Vertex AI의 비동기 처리 결과를 저장할 버킷을 생성합니다.

1. Cloud Storage > **버킷(Buckets)** 로 이동하여 **만들기(Create)** 클릭
2. 버킷 이름 지정 (예: `dawwon-pharm-docs-input`, `dawwon-pharm-docs-output`)
3. 위치(Region) 선택: `asia-northeast3` (서울) 권장.
4. 스토리지 클래스: Standard 선택
5. 액세스 제어: 균일한 버킷 수준 액세스 (Uniform) 체크 후 생성.

---

## 3. Document AI: Custom Extractor 설정 및 학습

연구소 내 특정 포맷(예: 임상시험 보고서, 특허 문서 등)에서 필요한 정보(Key-Values, Tables)를 고정적으로 추출해야 하는 경우 **Custom Extractor**를 학습시킵니다.

1. GCP 콘솔에서 **Document AI** 메뉴로 이동합니다.
2. **프로세서 갤러리(Processor Gallery)** 에서 **Custom Extractor**를 선택하고 **프로세서 만들기**를 클릭합니다.
3. 프로세서 이름을 입력하고 위치(US 또는 EU)를 선택합니다. (현재 Custom Extractor는 US/EU 리전에서만 생성 가능하므로 가급적 US 리전을 선택하세요.)
4. 프로세서가 생성되면 **학습(Train)** 탭으로 이동합니다.
5. **데이터 세트 연결**: GCS 버킷(예: `dawwon-pharm-docai-dataset`)을 지정하여 학습용 PDF 파일들을 업로드합니다.
6. **스키마 정의(Schema Definition)**: 추출하고자 하는 라벨(예: `experiment_date`, `results_table` 등)을 정의합니다.
7. **라벨링(Labeling)**: 업로드된 문서에서 정의한 스키마에 맞게 텍스트 영역 및 테이블 영역을 드래그하여 라벨링을 진행합니다.
8. **모델 학습**: 라벨링이 충분히 완료되면(최소 수십 장 권장) **새 모델 학습(Train new model)** 버튼을 눌러 모델을 학습시키고, 학습 완료 후 모델을 **배포(Deploy)** 합니다.
9. 배포 후 제공되는 `Processor ID`를 복사해 둡니다.

---

## 4. Vertex AI Search: Layout-aware Chunking 파싱 설정

Vertex AI Search를 단순 파싱 용도로 사용하여 문서 내의 구조(목차, 섹션, 표 등)를 유지하며 청킹(Chunking)을 수행합니다.

이 기능은 `google-cloud-discoveryengine` 라이브러리의 파싱 API를 호출하여 사용할 수 있습니다. 파이프라인(스크립트) 상에서 API 코드로 직접 파싱 로직을 구현하므로, Vertex AI API가 활성화되어 있으면 사용할 수 있습니다. 벡터 데이터베이스는 AlloyDB를 사용하므로 Vertex AI Search의 데이터 스토어는 검색 용도로 구축하지 않고 오직 **문서 파싱(Document Extraction) API**로만 활용합니다.

---

## 5. AlloyDB AI 완전 관리형 클러스터 및 pgvector 설정

임베딩된 벡터 데이터와 청크 텍스트를 저장하고 벡터 검색(RAG)을 수행하기 위한 완전 관리형 AlloyDB 클러스터를 구축합니다.

### 5.1 클러스터 생성
1. GCP 콘솔에서 **AlloyDB** 메뉴로 이동하여 **클러스터 만들기(Create cluster)** 클릭.
2. **고가용성 완전 관리형(Highly available fully managed)** 선택.
3. 클러스터 ID(예: `dawwon-rag-cluster`), 비밀번호, 리전(`asia-northeast3` 등) 입력.
4. 네트워크: 사용 중인 기본 VPC 네트워크 선택 (Private Services Access 설정 필요).
5. **기본 인스턴스 구성**: 인스턴스 ID(예: `dawwon-rag-primary`), 머신 유형(예: 4 vCPU, 32GB RAM 등 워크로드에 맞게 선택).
6. **데이터베이스 플래그 추가**: (중요) 머신러닝 연동 및 벡터 관련 설정을 위해 `google_ml_integration.enable_model_endpoint = on` 등의 플래그 확인 (pgvector는 기본 제공됨).
7. 설정 완료 후 **클러스터 만들기** 클릭. (수 분 소요)

### 5.2 데이터베이스 및 확장(Extension) 활성화
1. 인스턴스가 생성되면 AlloyDB Studio(또는 psql 클라이언트)로 접속합니다.
2. 새 데이터베이스 생성:
   ```sql
   CREATE DATABASE pharm_rag;
   ```
3. `pharm_rag` 데이터베이스로 접속 후 확장 기능 활성화:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS google_ml_integration;
   ```
4. 테이블 생성:
   ```sql
   CREATE TABLE document_chunks (
       id bigserial PRIMARY KEY,
       document_name text,
       page_number integer,
       chunk_type text, -- 'text', 'table', 'toc' 등
       content text,
       embedding vector(768) -- text-embedding-004 모델은 768차원 사용
   );
   ```

---

## 6. 서비스 계정(Service Account) 생성 및 키 발급

Python 코드에서 GCP 리소스에 접근하기 위한 서비스 계정을 설정합니다.

1. **IAM 및 관리자 > 서비스 계정**으로 이동.
2. **서비스 계정 만들기** 클릭. (예: `pharm-rag-agent`)
3. 다음 역할을 부여합니다.
   - `저장소 개체 관리자` (Storage Object Admin)
   - `Document AI API 사용자` (Document AI API User)
   - `Vertex AI 사용자` (Vertex AI User)
   - `AlloyDB 클라이언트` (AlloyDB Client)
4. 서비스 계정 생성 후 해당 계정을 클릭 > **키(Keys)** 탭 > **키 추가** > **새 키 만들기** (JSON) 선택.
5. 다운로드된 JSON 파일을 로컬 환경에 안전하게 저장하고, 환경 변수로 등록합니다.
   ```bash
   # Windows (PowerShell)
   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your-service-account-file.json"
   ```
