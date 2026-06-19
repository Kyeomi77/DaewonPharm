import os
import asyncio
import asyncpg
from pgvector.asyncpg import register_vector
from langchain_google_vertexai import VertexAIEmbeddings

# 설정 변수
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "uk-adc-core-geminienterprise")
LOCATION = "asia-northeast3" # Embeddings API Location

# AlloyDB 연결 정보
DB_HOST = "127.0.0.1" # AlloyDB Auth Proxy 또는 내부 IP
DB_USER = "postgres"
DB_PASS = "your-password"
DB_NAME = "pharm_rag"

# 임베딩 모델 설정
embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004", 
    project=PROJECT_ID,
    location=LOCATION
)

async def init_db_pool():
    """AlloyDB 연결 풀을 초기화합니다."""
    pool = await asyncpg.create_pool(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        min_size=1,
        max_size=10
    )
    # pgvector 타입을 사용하기 위해 풀에 등록
    async with pool.acquire() as conn:
        await register_vector(conn)
    return pool

async def store_chunks_in_alloydb(pool, document_name: str, chunks: list):
    """
    텍스트 청크들을 벡터화하여 AlloyDB(pgvector)에 저장합니다.
    """
    print(f"[{document_name}] 총 {len(chunks)}개의 청크 임베딩 생성 중...")
    
    # 청크 내용 리스트
    texts = [chunk["content"] for chunk in chunks]
    
    # Vertex AI를 통해 임베딩 일괄 생성
    embeddings = embedding_model.embed_documents(texts)
    
    # DB에 삽입할 데이터 준비
    records = []
    for i, chunk in enumerate(chunks):
        records.append((
            document_name,
            chunk.get("page_number", 1),
            chunk.get("type", "text"),
            chunk["content"],
            embeddings[i]
        ))
    
    # AlloyDB에 Insert
    query = """
    INSERT INTO document_chunks (document_name, page_number, chunk_type, content, embedding)
    VALUES ($1, $2, $3, $4, $5)
    """
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(query, records)
            
    print(f"[{document_name}] AlloyDB 저장 완료.")

async def main():
    pool = await init_db_pool()
    
    try:
        # 가상의 파싱된 데이터 (02_document_processing.py 에서 얻은 데이터라 가정)
        sample_document_name = "sample_report.pdf"
        parsed_chunks = [
            {"type": "text", "page_number": 1, "content": "임상 시험 목적: 신약 A의 효능 평가."},
            {"type": "table", "page_number": 2, "content": "<table><tr><th>부작용</th><th>비율</th></tr><tr><td>두통</td><td>5%</td></tr></table>"}
        ]
        
        await store_chunks_in_alloydb(pool, sample_document_name, parsed_chunks)
        
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
