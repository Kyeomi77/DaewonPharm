import os
import json
import asyncio
from typing import List, Dict, Any
import asyncpg
from pgvector.asyncpg import register_vector
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 설정 변수
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "uk-adc-core-geminienterprise")
LOCATION = "asia-northeast3"

# AlloyDB 연결 정보
DB_HOST = "34.50.58.39"
DB_USER = "postgres"
DB_PASS = "DawornPharm2026!!"
DB_NAME = "postgres"

class PharmaRAGAgent:
    def __init__(self, db_pool, embedding_model, llm):
        self.pool = db_pool
        self.embedding_model = embedding_model
        self.llm = llm

    async def _retrieve_context(self, conn, query_text: str, query_vector: List[float], user_clearance: int, limit: int = 3) -> Dict[str, Any]:
        """
        [보안 및 검색 레이어]
        동일 트랜잭션 내 세션에 보안 등급(RLS)을 주입하고, 하이브리드 검색(RRF)을 통해 
        최적의 하위 청크를 찾은 후 상위 문맥(Parent) 및 표(Table) 데이터를 확장 조회합니다.
        """
        # 1. PostgreSQL 세션 변수에 사용자 보안 등급 주입 (RLS 가동)
        # set_config(..., true)를 통해 현재 트랜잭션(Local) 동안만 유효하도록 격리 제어
        await conn.execute("SELECT set_config('app.user_clearance', $1, true);", str(user_clearance))

        # 2. 스키마 표준 지침에 맞춘 하이브리드 검색 (RRF) 쿼리 실행
        rrf_query = """
        WITH vec AS (
          SELECT chunk_id, parent_id, doc_id, content,
                 ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rnk
          FROM chunks
          ORDER BY embedding <=> $1 LIMIT 50
        ),
        kw AS (
          SELECT chunk_id, parent_id, doc_id, content,
                 ROW_NUMBER() OVER (
                   ORDER BY ts_rank(content_tsv, plainto_tsquery('simple', $2)) DESC
                 ) AS rnk
          FROM chunks
          WHERE content_tsv @@ plainto_tsquery('simple', $2)
          LIMIT 50
        )
        SELECT 
            COALESCE(vec.chunk_id, kw.chunk_id) AS chunk_id,
            COALESCE(vec.parent_id, kw.parent_id) AS parent_id,
            COALESCE(vec.doc_id, kw.doc_id) AS doc_id,
            COALESCE(vec.content, kw.content) AS chunk_content,
            (1.0 / (60 + COALESCE(vec.rnk, 1000))) +
            (1.0 / (60 + COALESCE(kw.rnk,  1000))) AS rrf_score
        FROM vec FULL OUTER JOIN kw ON vec.chunk_id = kw.chunk_id
        ORDER BY rrf_score DESC
        LIMIT $3;
        """
        
        chunk_rows = await conn.fetch(rrf_query, query_vector, query_text, limit)
        if not chunk_rows:
            return {"parent_contexts": [], "table_contexts": [], "referenced_docs": []}

        parent_ids = list(set([row['parent_id'] for row in chunk_rows if row['parent_id']]))
        referenced_docs = list(set([str(row['doc_id']) for row in chunk_rows]))

        # 3. Parent Chunks 단원/섹션 원문 컨텍스트 확장 조회
        parent_contexts = []
        if parent_ids:
            parent_query = """
            SELECT section_title, content FROM parent_chunks 
            WHERE parent_id = ANY($1);
            """
            parent_rows = await conn.fetch(parent_query, parent_ids)
            for p_row in parent_rows:
                parent_contexts.append(f"Header: {p_row['section_title']}\nContent: {p_row['content']}")

        # 4. 관련 구조화 표(doc_tables) 마크다운 데이터 함께 결합
        table_contexts = []
        if parent_ids:
            table_query = """
            SELECT markdown_repr FROM doc_tables 
            WHERE parent_id = ANY($1);
            """
            table_rows = await conn.fetch(table_query, parent_ids)
            for t_row in table_rows:
                if t_row['markdown_repr']:
                    table_contexts.append(t_row['markdown_repr'])

        return {
            "parent_contexts": parent_contexts,
            "table_contexts": table_contexts,
            "referenced_docs": referenced_docs
        }

    async def _write_audit_log(self, conn, doc_id: str, actor: str, query: str, reason: str):
        """
        [GxP 컴플라이언스 레이어]
        ALCOA+ 지침에 따라 어떤 연구원이 어떤 문서를 참조하여 조회했는지 불변 로그를 기록합니다.
        """
        audit_query = """
        INSERT INTO audit_log (entity, entity_id, action, actor, reason, diff)
        VALUES ($1, $2, $3, $4, $5, $6);
        """
        diff_data = json.dumps({"action_details": {"triggered_by_query": query}})
        await conn.execute(audit_query, "documents", doc_id, "SELECT", actor, reason, diff_data)

    async def answer_question(self, user_query: str, user_clearance: int, actor_name: str) -> str:
        """
        [오케스트레이션 레이어]
        자연어 질문을 받아 전체 RAG 및 컴플라이언스 파이프라인을 실행합니다.
        """
        print(f"\n[Agent] '{actor_name}'(보안등급: {user_clearance})님의 질문 분석 중...")
        
        # 1. 질문 임베딩 변환 (1536 차원 수렴)
        query_vector = await self.embedding_model.aembed_query(user_query)

        # 단일 연결 및 단일 트랜잭션 시작 (RLS 및 Audit Log의 일관성 보장)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                
                # 2. RLS 기반 하이브리드 컨텍스트 추출
                context_data = await self._retrieve_context(conn, user_query, query_vector, user_clearance)
                
                if not context_data["parent_contexts"]:
                    return "죄송합니다. 요청하신 질문에 권한이 있거나 일치하는 연구 문서 데이터를 찾을 수 없습니다."

                # 3. Gemini 프롬프트 구성 및 생성 조율
                context_str = "\n\n".join(context_data["parent_contexts"])
                table_str = "\n\n".join(context_data["table_contexts"])
                
                prompt = ChatPromptTemplate.from_template("""
                당신은 대한민국 제약사 연구소의 수석 AI 연구원입니다. 
                제공된 신약 개발 및 실험 문서의 [상위 문맥 소스]와 [구조화된 표 자료]를 바탕으로 정밀하고 신뢰할 수 있는 답변을 작성하십시오.
                안전성 및 실험 수치에 대해서는 임의로 조작하거나 거짓을 작성해서는 안 됩니다 (ALCOA+ 지침 준수).

                [상위 문맥 소스]
                {context}

                [관련 표 자료 (Markdown)]
                {tables}

                [연구원 질문]
                {question}

                [전문적인 연구 답변]
                """)
                
                chain = prompt | self.llm | StrOutputParser()
                response_text = await chain.ainvoke({
                    "context": context_str,
                    "tables": table_str,
                    "question": user_query
                })

                # 4. 참조된 모든 최상위 문서에 대해 감사 로그 자동 생성 (Insert-Only)
                for doc_id in context_data["referenced_docs"]:
                    await self._write_audit_log(
                        conn=conn,
                        doc_id=doc_id,
                        actor=actor_name,
                        query=user_query,
                        reason=f"Gemini RAG Agent Q&A Generation. Clearance level verified: {user_clearance}"
                    )
                
                return response_text

# --- 실행 메인 함수 (테스트용 예시) ---
async def main():
    # 1. AlloyDB 연결 풀 구성 및 벡터 핸들러 등록
    pool = await asyncpg.create_pool(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME,
        min_size=1, max_size=5
    )
    async with pool.acquire() as conn:
        await register_vector(conn)

    # 2. 엔터프라이즈 컴포넌트 모델 정의
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=PROJECT_ID,
        location="asia-northeast3",
        output_dimensionality=1536
    )
    
    # 추론 및 표 구조 해석 능력이 뛰어난 Gemini 1.5 Pro 탑재
    gemini_llm = ChatVertexAI(
        model_name="gemini-1.5-pro",
        project=PROJECT_ID,
        location="asia-northeast3",
        temperature=0.2
    )

    # 3. 에이전트 인스턴스화
    agent = PharmaRAGAgent(db_pool=pool, embedding_model=embeddings, llm=gemini_llm)

    # 4. 테스트 시나리오 가동
    # 시나리오 A: 일반 연구원 (보안 등급 1) -> 기밀(레벨2 이상) 문서는 RLS에 의해 자동 차단됨
    answer_low = await agent.answer_question(
        user_query="신약 파이프라인 DW-2026의 임상 1상 결과 요약 및 통계 표 해석해줘",
        user_clearance=1,
        actor_name="홍길동 연구원"
    )
    print(f"\n[홍길동 연구원 답변 결과]:\n{answer_low}")

    print("-" * 50)

    # 시나리오 B: 핵심 책임 연구원 (보안 등급 3) -> RLS 통과하여 상세 기밀 문서까지 결합해 답변 생성
    answer_high = await agent.answer_question(
        user_query="신약 파이프라인 DW-2026의 임상 1상 결과 요약 및 통계 표 해석해줘",
        user_clearance=3,
        actor_name="김부장 책임연구원"
    )
    print(f"\n[김부장 책임연구원 답변 결과]:\n{answer_high}")

    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())