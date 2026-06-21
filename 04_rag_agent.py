import os
import json
import asyncio
from typing import List, Dict, Any
import asyncpg
from pgvector.asyncpg import register_vector
from google.adk import Agent  # Google Agent Platform ADK 라이브러리
from langchain_google_vertexai import VertexAIEmbeddings

# --- 설정 변수 ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "uk-adc-core-geminienterprise")
LOCATION = "asia-northeast3"

# AlloyDB / PostgreSQL 연결 정보 (보안을 위해 환경변수 오버라이드 지원)
DB_HOST = os.environ.get("DB_HOST", "34.50.58.39")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "DawornPharm2026!!")
DB_NAME = os.environ.get("DB_NAME", "postgres")

# 엔터프라이즈 임베딩 모델 선언 (1536 차원 수렴)
embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project=PROJECT_ID,
    location=LOCATION,
    dimensions=768
)

# 전역 커넥션 풀 변수 (초기 지연 로딩용)
db_pool = None

async def get_db_pool():
    """AlloyDB 연결 풀을 안전하게 싱글톤으로 관리합니다."""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME,
            min_size=2, max_size=10
        )
        async with db_pool.acquire() as conn:
            await register_vector(conn)
    return db_pool


# ---------------------------------------------------------------------------
# 1. 커스텀 도구 정의 (Agent Platform이 호출할 핵심 RAG 엔진)
# ---------------------------------------------------------------------------
#@tool
async def search_pharma_knowledge_base(user_query: str, user_clearance: int, actor_name: str) -> str:
    """
    대원제약 내부 연구 문서 및 실험 기밀 데이터를 정밀 검색하는 도구입니다. 
    행 수준 보안(RLS) 필터와 하이브리드 RRF 유사도 알고리즘이 내장되어 있습니다.

    Args:
        user_query: 연구원이 입력한 자연어 형태의 검색 질의어
        user_clearance: 현재 질문을 던진 연구원의 검증된 보안 등급 (1~4)
        actor_name: 시스템에 로그인된 연구원의 실제 이름 및 직급
    """
    print(f"\n[Tool] '{actor_name}'(보안등급: {user_clearance})의 기밀 DB 검색 요청 수신.")
    
    # 1. 자연어 질문 임베딩 변환
    query_vector = await embedding_model.aembed_query(user_query)
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # 단일 트랜잭션 시작 (RLS 및 감사 로그의 무결성 보장)
        async with conn.transaction():
            
            # [A] PostgreSQL 세션 변수에 사용자 보안 등급 주입 (RLS 가동)
            await conn.execute("SELECT set_config('app.user_clearance', $1, true);", str(user_clearance))

            # [B] 스키마 지침에 맞춘 하이브리드 검색 (RRF) 쿼리 실행
            rrf_query = """
            WITH vec AS (
              SELECT chunk_id, parent_id, doc_id, content,
                     ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rnk
              FROM chunks ORDER BY embedding <=> $1 LIMIT 50
            ),
            kw AS (
              SELECT chunk_id, parent_id, doc_id, content,
                     ROW_NUMBER() OVER (
                       ORDER BY ts_rank(content_tsv, plainto_tsquery('simple', $2)) DESC
                     ) AS rnk
              FROM chunks WHERE content_tsv @@ plainto_tsquery('simple', $2) LIMIT 50
            )
            SELECT 
                COALESCE(vec.parent_id, kw.parent_id) AS parent_id,
                COALESCE(vec.doc_id, kw.doc_id) AS doc_id
            FROM vec FULL OUTER JOIN kw ON vec.chunk_id = kw.chunk_id
            ORDER BY (1.0 / (60 + COALESCE(vec.rnk, 1000))) + (1.0 / (60 + COALESCE(kw.rnk,  1000))) DESC
            LIMIT 3;
            """
            rows = await conn.fetch(rrf_query, query_vector, user_query)
            if not rows:
                return "죄송합니다. 요청하신 질문에 권한이 있거나 일치하는 연구 문서 데이터를 찾을 수 없습니다."

            parent_ids = list(set([r['parent_id'] for r in rows if r['parent_id']]))
            referenced_docs = list(set([str(r['doc_id']) for r in rows]))

            # [C] Parent Chunks 단원/섹션 원문 컨텍스트 및 표 데이터 확장 조회
            context_pieces = []
            if parent_ids:
                # 상위 문맥 조회
                p_rows = await conn.fetch(
                    "SELECT section_title, content FROM parent_chunks WHERE parent_id = ANY($1);", 
                    parent_ids
                )
                for pr in p_rows:
                    context_pieces.append(f"[상위 단원 문맥 소스: {pr['section_title']}]\n{pr['content']}")
                
                # 구조화 마크다운 표 조회
                t_rows = await conn.fetch(
                    "SELECT markdown_repr FROM doc_tables WHERE parent_id = ANY($1);", 
                    parent_ids
                )
                for tr in t_rows:
                    if tr['markdown_repr']:
                        context_pieces.append(f"[관련 구조화 표 자료]\n{tr['markdown_repr']}")

            # [D] GxP 컴플라이언스 감사 로그 기록 (ALCOA+ 준수)
            audit_query = """
            INSERT INTO audit_log (entity, entity_id, action, actor, reason, diff)
            VALUES ($1, $2, $3, $4, $5, $6);
            """
            diff_data = json.dumps({"action_details": {"triggered_by_query": user_query, "via": "ADK_Agent_Platform"}})
            
            for doc_id in referenced_docs:
                await conn.execute(
                    audit_query, "documents", doc_id, "SELECT", actor_name, 
                    f"Gemini ADK Agent Grounding Search. Clearance level verified: {user_clearance}", 
                    diff_data
                )

            return "\n\n".join(context_pieces)


# ---------------------------------------------------------------------------
# 2. ADK 네이티브 에이전트 오케스트레이션 정의
# ---------------------------------------------------------------------------
pharma_agent = Agent(
    name="pharma_rag_compliance_agent",
    model="gemini-2.5-flash",  # 표 해석 및 인프라 기밀 추론에 특화된 모델 기동
    instruction="""
    당신은 대한민국 제약사 연구소의 수석 AI 연구원입니다.
    연구원의 질문이 들어오면 반드시 'search_pharma_knowledge_base' 도구를 사용하여 내부 지식 소스를 조회하십시오.
    
    [핵심 연동 규칙]
    1. 사용자의 대화 세션이나 입력 컨텍스트에서 질문자의 이름(actor_name)과 보안 등급(user_clearance)을 확인하여 도구의 매개변수로 정확하게 넘겨주어야 합니다.
    2. 도구가 반환한 정보(상위 문맥 및 표 데이터)만을 엄격한 근거로 삼아 답변을 작성하십시오.
    3. 안전성 및 실험 수치에 대해서는 임의로 수정하거나 조작(거짓말)해서는 안 됩니다 (ALCOA+ 지침 준수).
    """,
    tools=[search_pharma_knowledge_base]  # 우리가 작성한 RLS 기반 하이브리드 검색 툴 바인딩
)


# ---------------------------------------------------------------------------
# 3. 추가 및 고도화된 테스트 코드 (로컬 검증 및 시뮬레이터용)
# ---------------------------------------------------------------------------
async def run_local_test_simulation():
    """배포 전, 로컬 환경에서 ADK 도구와 DB 트랜잭션의 정상 작동 여부를 시뮬레이션합니다."""
    print("=" * 60)
    print("[테스트 시작] 대원제약 RAG 에이전트 ADK 로컬 기능 검증")
    print("=" * 60)
    
    test_query = "신약 파이프라인 DW-2026의 임상 1상 결과 요약 및 통계 표 해석해줘"
    
    # 시나리오 A: 일반 연구원 (보안 등급 1) -> 기밀문서 차단 테스트
    print("\n▶ 시나리오 A: 홍길동 연구원 (보안 레벨 1) 시뮬레이션")
    try:
        context_low = await search_pharma_knowledge_base(
            user_query=test_query,
            user_clearance=1,
            actor_name="홍길동 연구원"
        )
        print(f"[추출된 내부 컨텍스트 결과]:\n{context_low[:300]}...")
    except Exception as e:
        print(f"❌ 시나리오 A 실행 중 에러 발생: {e}")

    print("-" * 60)

    # 시나리오 B: 핵심 책임 연구원 (보안 등급 3) -> 기밀 문서 및 감사 로그 적재 테스트
    print("\n▶ 시나리오 B: 김부장 책임연구원 (보안 레벨 3) 시뮬레이션")
    try:
        context_high = await search_pharma_knowledge_base(
            user_query=test_query,
            user_clearance=3,
            actor_name="김부장 책임연구원"
        )
        print(f"[추출된 내부 컨텍스트 결과]:\n{context_high[:500]}...")
    except Exception as e:
        print(f"❌ 시나리오 B 실행 중 에러 발생: {e}")

    # 글로벌 커넥션 풀 종료
    global db_pool
    if db_pool:
        await db_pool.close()
        print("\n[테스트 종료] 데이터베이스 커넥션 풀이 안전하게 닫혔습니다.")

if __name__ == "__main__":
    # 로컬에서 스크립트를 직접 실행(`python 04_rag_agent.py`)할 경우 테스트 시뮬레이터가 작동합니다.
    asyncio.run(run_local_test_simulation())