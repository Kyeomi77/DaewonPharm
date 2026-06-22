```markdown
# 엔터프라이즈급 RAG 에이전트 배포 및 Gemini 연동 가이드

AlloyDB의 행 수준 보안(RLS)과 하이브리드 검색(RRF)을 결합한 RAG 에이전트(`04_rag_agent.py`)를 Vertex AI Agent Engine에 배포하고, Gemini Enterprise(Google Workspace)에서 사용하기 위한 단계별 가이드입니다.

## ⚠️ 원본 문서의 주요 오류 및 수정 사항

이전 버전(AI 생성본)에는 다음과 같은 오류가 포함되어 있어 수정하였습니다.

**오류 1. 존재하지 않는 CLI 명령어**

- **원본:** `adk deploy agent_engine` — 이 명령어는 Google Cloud에 존재하지 않습니다.
- **수정:** Vertex AI Agent Engine 배포는 `vertexai` Python SDK의 `reasoning_engines.ReasoningEngine.create()` 를 사용합니다.

**오류 2. 존재하지 않는 패키지명**

- **원본:** `google-adk`, `google-adk[extensions]` — PyPI에 존재하지 않는 가상의 패키지입니다.
- **수정:** `google-cloud-aiplatform>=1.64.0`을 사용합니다. (2025년 6월 기준 최신 안정 버전)

**오류 3. 허구화된 UI/개념**

- **원본:** 'SPIFFE 기반 Agent Identity', 'Agent Gallery' 등 — 현재 Google Cloud 콘솔 및 Gemini for Workspace에 존재하지 않는 UI/개념입니다.
- **수정:** 실제 콘솔 경로(Vertex AI > Extensions, Google Workspace Admin Console)에 맞게 수정하였습니다.

**오류 4. 패키지 버전 부정확**

- **원본:** `google-cloud-aiplatform>=1.64.0` — 이 버전은 `reasoning_engines` 모듈을 포함하지만 ADK(Agent Development Kit) 통합은 1.71.0 이상에서 안정화되었습니다.
- **수정:** `google-cloud-aiplatform[reasoningengine,langchain]>=1.71.0`으로 업데이트합니다.

**오류 5. deploy.py 스크립트의 API 오류**

- **원본:** `reasoning_engines.create(...)` — 올바른 API가 아닙니다.
- **수정:** `reasoning_engines.ReasoningEngine.create(...)`가 정확한 메서드입니다.

---

## 1. VS Code를 활용한 로컬 개발 및 테스트 환경 구축

클라우드에 배포하기 전, VS Code에서 코드가 정상적으로 동작하는지 로컬에서 테스트해야 합니다.

### Step 1: 작업 디렉토리 및 파일 준비

1.  VS Code를 열고 작업할 폴더를 엽니다.
2.  폴더 내에 `04_rag_agent.py` 파일을 저장합니다.
3.  **의존성 파일(`requirements.txt`) 생성**: 동일한 폴더에 `requirements.txt`를 생성하고 아래 내용을 입력합니다.

    ```text
    # requirements.txt
    google-cloud-aiplatform[reasoningengine,langchain]>=1.71.0
    asyncpg>=0.29.0
    pgvector>=0.3.5
    langchain-google-vertexai>=2.0.0
    google-cloud-secret-manager>=2.16.0
    ```

### Step 2: 가상 환경(Virtual Environment) 생성 및 패키지 설치

4.  VS Code 상단 메뉴에서 **Terminal > New Terminal**을 엽니다.
5.  아래 명령어를 통해 Python 가상 환경을 생성하고 활성화합니다.

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

6.  VS Code에서 가상 환경을 인터프리터로 설정합니다. (`Ctrl/Cmd + Shift + P` → `Python: Select Interpreter` → 생성한 `venv` 폴더의 Python 선택)
7.  필요한 라이브러리를 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```
    # DECIMER 버전
    pip install rdkit
    pip install pillow
    pip install requests
    pip install decimer --no-deps
    # MolScribe 버전
    pip install MolScribe --no-deps

### Step 3: Google Cloud 인증 및 로컬 실행

8.  Google Cloud CLI가 설치되어 있어야 합니다. 터미널에서 아래 명령어로 인증을 수행합니다.

    ```bash
    gcloud init
    gcloud auth application-default login
    gcloud config set project uk-adc-core-geminienterprise
    ```

    > **⚠️ 보안 경고:** `04_rag_agent.py` 내에 하드코딩된 AlloyDB 비밀번호는 반드시 코드에서 제거하고, Secret Manager 또는 환경 변수(`os.environ.get('DB_PASSWORD')`)로 주입받도록 수정하세요.

9.  VS Code에서 `04_rag_agent.py` 파일을 열고, 우측 상단의 ▶ 버튼 또는 터미널에서 아래 명령어로 실행하여 오류가 없는지 확인합니다.

    ```bash
    python 04_rag_agent.py
    ```

---

## 2. Vertex AI Agent Engine 배포 절차

로컬 테스트가 완료되면, 코드를 Vertex AI의 관리형 런타임인 **Agent Engine (Reasoning Engine)**으로 배포합니다. CLI 명령어는 지원되지 않으며, **Python SDK**를 통해 배포해야 합니다.

### 배포 전 사전 준비

- **GCS 스테이징 버킷 생성:**
  ```bash
  gsutil mb -l asia-northeast3 gs://daewonpharm-bucket-run-2026
  ```
- **Vertex AI API 활성화:**
  ```bash
  gcloud services enable aiplatform.googleapis.com
  ```
- **필요 IAM 역할:** Vertex AI User, Storage Object Admin

### 배포 스크립트 작성 (deploy.py)

`04_rag_agent.py`가 있는 폴더에 `deploy.py`를 새로 만들고 아래 코드를 작성합니다.

```python
import vertexai
from vertexai.preview import reasoning_engines

# GCP 프로젝트 설정
PROJECT_ID = "uk-adc-core-geminienterprise"
LOCATION = "asia-northeast3"
STAGING_BUCKET = "gs://daewonpharm-bucket-run-2026"  # 미리 생성한 GCS 버킷

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET
)

# 04_rag_agent.py 내의 메인 에이전트 클래스를 임포트
# from rag_agent import PharmaRAGAgent

# 에이전트 배포 --- reasoning_engines.create()가 아닌
# ReasoningEngine.create()가 올바른 API입니다.
engine = reasoning_engines.ReasoningEngine.create(
    # PharmaRAGAgent(),  # 실제 에이전트 인스턴스 전달
    requirements=[
        "google-cloud-aiplatform[reasoningengine,langchain]>=1.71.0",
        "asyncpg>=0.29.0",
        "pgvector>=0.3.5",
        "langchain-google-vertexai>=2.0.0",
        "cryptography==41.0.7",
        "google-cloud-secret-manager>=2.16.0",
    ],
    display_name="pharma_rag_compliance_agent",
    description="AlloyDB RLS + Hybrid RRF Search 기반 의약품 규정 준수 RAG 에이전트"
)

print(f"배포 완료!")
print(f"Resource Name: {engine.resource_name}")
```

### 배포 실행

```bash
python deploy.py
```

배포가 완료되면 아래와 같은 형태의 리소스 이름이 출력됩니다. 이 값을 메모해 둡니다.

```text
projects/uk-adc-core-geminienterprise/locations/asia-northeast3/reasoningEngines/1234567890
```

> *※ 배포에는 5~15분이 소요될 수 있으며, Vertex AI > Agent Engine 콘솔에서 진행 상태를 확인할 수 있습니다.*

---

## 3. Gemini Enterprise (Google Workspace) 연동

배포된 백엔드 에이전트를 사내 Gemini 앱에서 사용하려면, **Vertex AI Extensions**를 생성하여 연동하는 것이 표준적인 방법입니다.

### Step 1: Vertex AI Extension 생성

10. Google Cloud 콘솔에서 **Vertex AI > Extensions** 메뉴로 이동합니다.
11. **\[Create Extension\]**을 클릭하고, 방금 배포한 Agent Engine(Reasoning Engine)의 리소스 경로를 지정합니다.
12. Extension의 API 스키마(OpenAPI 형식)를 정의하여 에이전트의 입출력 인터페이스를 선언합니다.

> *※ 이 Extension이 사용자 질문을 수신하여 Reasoning Engine으로 라우팅하는 역할을 담당합니다.*

### Step 2: 네트워크 및 보안 권한 부여

13. 에이전트가 사용할 서비스 계정(Service Account)을 생성하거나 기존 것을 지정합니다.
14. 이 서비스 계정에 **Cloud AlloyDB Client** 역할을 부여하고, VPC 방화벽 규칙을 통해 AlloyDB 인스턴스 IP(`34.50.58.39`)에 대한 접근을 허용합니다.
15. **(권장) Secret Manager 연동**: AlloyDB 비밀번호를 Secret Manager에 저장하고 서비스 계정에 **Secret Manager Secret Accessor** 역할을 부여합니다. 코드에서는 다음과 같이 비밀번호를 조회합니다.

    ```python
    from google.cloud import secretmanager

    def get_db_password() -> str:
        client = secretmanager.SecretManagerServiceClient()
        name = "projects/uk-adc-core-geminienterprise/secrets/alloydb-password/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    ```

### Step 3: Google Workspace Admin 승인

16. Google Workspace Admin Console(`admin.google.com`)에 접속합니다.
17. **앱 > Google Workspace > Gemini** 설정에서 서드파티 확장 프로그램(Extensions) 항목을 찾아 방금 생성한 Vertex AI Extension을 활성화합니다.
18. 연구원 그룹 등 특정 조직 단위(OU)에 대해 해당 에이전트 노출을 승인(Approve) 처리합니다.

> *※ 승인 처리 후 최대 24시간 내에 대상 사용자에게 에이전트가 표시됩니다.*

---

## 4. Gemini Enterprise Apps에서 에이전트 사용 방법

승인이 완료되면 사내 연구원들은 Gemini 채팅 UI에서 `@` 를 통해 에이전트를 호출할 수 있습니다.

### 사용 예시

```text
@pharma_rag_compliance_agent
신약 파이프라인 DW-2026의 임상 1상 결과 요약 및 통계 표 해석해줘.
```

### 내부 작동 메커니즘

19. **의도 파악 및 라우팅**: Gemini가 `@` 를 통해 지정된 에이전트(Extension)로 태스크를 라우팅합니다.
20. **컨텍스트 주입**: 현재 로그인한 Workspace 사용자 계정 정보를 기반으로 사용자 식별 정보를 에이전트로 전달합니다. AlloyDB의 RLS(행 수준 보안)를 통해 사용자 권한에 맞는 데이터만 조회됩니다.
21. **도구 실행 및 데이터 검색**: Agent Engine에서 실행 중인 에이전트가 AlloyDB에 RLS가 적용된 쿼리를 전송하여 권한 범위 내의 문서와 통계 데이터를 가져옵니다.
22. **답변 생성**: 추출된 데이터를 기반으로 그라운딩(Grounding) 답변을 생성하여 Gemini UI에 마크다운 형태로 출력합니다.

---

## 5. 주요 오류 해결 (Troubleshooting)

**오류:** `ModuleNotFoundError: No module named 'vertexai.preview.reasoning_engines'`

- **원인:** `google-cloud-aiplatform` 버전이 낮습니다.
- **해결:** `pip install -U google-cloud-aiplatform[reasoningengine]>=1.71.0` 으로 업그레이드합니다.

**오류:** `Permission denied on AlloyDB`

- 서비스 계정에 **Cloud AlloyDB Client** 역할이 부여되어 있는지 확인합니다.
- VPC 방화벽 규칙에서 AlloyDB 포트(5432)가 허용되어 있는지 확인합니다.

**오류:** `Staging bucket access denied`

- 서비스 계정에 **Storage Object Admin** 역할이 부여되어 있는지 확인합니다.
- 버킷과 프로젝트가 동일한 리전(`asia-northeast3`)에 있는지 확인합니다.

---

> *문서 최종 수정일: 2026년 6월 | 검토자: Claude (Anthropic) | 기반 환경: Google Cloud / Vertex AI*
```