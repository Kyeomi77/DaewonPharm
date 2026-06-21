import vertexai
from vertexai.preview import reasoning_engines

# GCP 프로젝트 설정
PROJECT_ID = "uk-adc-core-geminienterprise"
LOCATION   = "asia-northeast3"
STAGING_BUCKET = "gs://daewonpharm-bucket-in-2026"  # 미리 생성한 GCS 버킷

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET
)

# 04_rag_agent.py 내의 메인 에이전트 클래스를 임포트
# from rag_agent import PharmaRAGAgent

# 에이전트 배포 — reasoning_engines.create()가 아닌
# ReasoningEngine.create()가 올바른 API입니다.
engine = reasoning_engines.ReasoningEngine.create(
    # PharmaRAGAgent(),  # 실제 에이전트 인스턴스 전달
    requirements=[
        "google-cloud-aiplatform[reasoningengine,langchain]>=1.71.0",
        "asyncpg>=0.29.0",
        "pgvector>=0.3.5",
        "langchain-google-vertexai>=2.0.0",
        "cryptography==41.0.7"
        "google-cloud-secret-manager>=2.16.0",
    ],
    display_name="pharma_rag_compliance_agent",
    description="AlloyDB RLS + Hybrid RRF Search 기반 의약품 규정 준수 RAG 에이전트"
)

print(f"배포 완료!")
print(f"Resource Name: {engine.resource_name}")
print(f"Display Name: {engine.display_name}")
print(f"Description: {engine.description}")