import os
import asyncio
import asyncpg
from pgvector.asyncpg import register_vector
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 설정 변수
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = "asia-northeast3"

# AlloyDB 연결 정보
DB_HOST = "127.0.0.1" 
DB_USER = "postgres"
DB_PASS = "your-password"
DB_NAME = "pharm_rag"

# LLM 및 임베딩 모델 초기화
llm = ChatVertexAI(
    model_name="gemini-1.5-pro",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.0
)

embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project=PROJECT_ID,
    location=LOCATION
)

async def search_similar_documents(pool, query: str, top_k: int = 5):
    """사용자 쿼리를 임베딩하여 AlloyDB에서 가장 유사한 문서 청크를 검색합니다."""
    query_vector = embedding_model.embed_query(query)
    
    # 코사인 유사도(<=>) 기반 검색 (HNSW 인덱스가 있다면 빠르게 동작)
    sql = """
    SELECT document_name, page_number, chunk_type, content, 
           1 - (embedding <=> $1) AS similarity
    FROM document_chunks
    ORDER BY embedding <=> $1
    LIMIT $2
    """
    
    async with pool.acquire() as conn:
        await register_vector(conn)
        rows = await conn.fetch(sql, query_vector, top_k)
        
    return rows

def format_context(rows) -> str:
    """검색된 데이터(테이블 HTML, 텍스트)를 프롬프트 Context로 포맷팅합니다."""
    context_str = ""
    for row in rows:
        context_str += f"\n[문서명: {row['document_name']}, 페이지: {row['page_number']}, 타입: {row['chunk_type']}]\n"
        context_str += f"{row['content']}\n"
        context_str += "-" * 50
    return context_str

async def ask_agent(pool, query: str):
    """검색 결과를 바탕으로 Gemini를 통해 답변을 생성합니다."""
    print(f"\n질문: {query}")
    print("AlloyDB 벡터 검색 중...")
    
    similar_rows = await search_similar_documents(pool, query, top_k=3)
    context = format_context(similar_rows)
    
    print("검색된 Context:\n", context)
    
    prompt_template = PromptTemplate.from_template("""
    당신은 제약사 연구소의 신약 개발 문서 분석을 돕는 전문 AI 어시스턴트입니다.
    아래 제공된 [검색된 문서 내용(Context)]만을 사용하여 사용자의 [질문]에 답변하세요.
    문서 내용에 표(Table)가 포함된 경우 HTML 포맷을 참고하여 데이터를 정확히 분석해야 합니다.
    문서에 답이 없다면 "제공된 문서에서 답변을 찾을 수 없습니다."라고 답변하세요.

    [검색된 문서 내용(Context)]
    {context}

    [질문]
    {query}
    
    답변:
    """)
    
    # LangChain 파이프라인
    chain = prompt_template | llm | StrOutputParser()
    
    print("Gemini 응답 생성 중...\n")
    response = chain.invoke({"context": context, "query": query})
    print("================== [Agent 답변] ==================")
    print(response)
    print("==================================================")

async def main():
    pool = await asyncpg.create_pool(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        min_size=1,
        max_size=10
    )
    
    try:
        user_query = "신약 A의 임상 시험에서 두통 부작용 발생률은 몇 퍼센트인가요?"
        await ask_agent(pool, user_query)
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
