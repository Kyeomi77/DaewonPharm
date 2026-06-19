import os
import re
import asyncio
import mimetypes
from typing import List, Dict, Any
from google.cloud import storage
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import documentai_v1 as documentai
import asyncpg
from pgvector.asyncpg import register_vector
from langchain_google_vertexai import VertexAIEmbeddings

# --- 설정 변수 ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "uk-adc-core-geminienterprise")
LOCATION = "global"  # Vertex AI Search API Location
INPUT_GCS_BUCKET = "daewonpharm-bucket-in-2026" # 파싱할 문서가 있는 버킷
OUTPUT_GCS_BUCKET = "daewonpharm-bucket-out-2026" # 처리 완료된 문서가 이동될 버킷
ERROR_GCS_BUCKET = "daewonpharm-bucket-err-2026"  # 처리 중 오류가 발생한 문서가 이동될 버킷

# Custom Extractor 설정 (옵션)
DOCAI_LOCATION = "us"
DOCAI_PROCESSOR_ID = "6373054663fffe8a"

# AlloyDB 연결 정보
DB_HOST = "127.0.0.1"
DB_USER = "postgres"
DB_PASS = "your-password" # 실제 비밀번호로 변경하세요.
DB_NAME = "pharm_rag"

# 임베딩 모델 설정
embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004", 
    project=PROJECT_ID,
    location="asia-northeast3" # 임베딩 모델 위치는 서울 리전
)

# --- 데이터베이스 함수 (03_embed_and_store.py 에서 가져옴) ---

async def init_db_pool():
    """AlloyDB 연결 풀을 초기화합니다."""
    pool = await asyncpg.create_pool(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME,
        min_size=1, max_size=10
    )
    async with pool.acquire() as conn:
        await register_vector(conn)
    return pool

async def store_chunks_in_alloydb(pool, document_name: str, chunks: list):
    """텍스트 청크들을 벡터화하여 AlloyDB(pgvector)에 저장합니다."""
    if not chunks:
        print(f"[{document_name}] 저장할 청크가 없습니다.")
        return

    print(f"[{document_name}] 총 {len(chunks)}개의 청크 임베딩 생성 및 저장 중...")
    
    texts = [chunk["content"] for chunk in chunks]
    embeddings = await embedding_model.aembed_documents(texts)
    
    records = [
        (document_name, chunk.get("page_number", 1), chunk.get("type", "text"), chunk["content"], embedding)
        for chunk, embedding in zip(chunks, embeddings)
    ]
    
    query = "INSERT INTO document_chunks (document_name, page_number, chunk_type, content, embedding) VALUES ($1, $2, $3, $4, $5)"
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(query, records)
            
    print(f"[{document_name}] AlloyDB 저장 완료.")

# --- 문서 파싱 함수 ---

def process_with_vertex_layout_parser(gcs_uri: str, mime_type: str) -> List[Dict[str, Any]]:
    """Vertex AI Search의 Layout-aware 파싱 API를 사용하여 문서를 구조화된 청크로 분리합니다."""
    client_options = {"api_endpoint": "discoveryengine.googleapis.com"}
    client = discoveryengine.DocumentServiceClient(client_options=client_options)
    
    parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    request = discoveryengine.ProcessDocumentRequest(
        parent=parent,
        gcs_source=discoveryengine.GcsSource(gcs_uri=gcs_uri, mime_type=mime_type),
        process_options=discoveryengine.ProcessOptions(
            ocr_config=discoveryengine.OcrConfig(
                enable_native_pdf_parsing=True,
                enable_image_quality_scores=True,
            )
        ),
    )
    
    print(f"Vertex AI Search (Layout-aware)로 문서 파싱 중: {gcs_uri}")
    response = client.process_document(request=request)
    
    chunks = []
    doc = response.document

    for page_index, page in enumerate(doc.pages):
        page_number = page_index + 1
        
        # 1. 테이블(Table) 처리
        for table in page.tables:
            html_table = "<table>"
            for row in table.header_rows:
                html_table += "<tr>" + "".join([f"<th>{cell.layout.text_anchor.content}</th>" for cell in row.cells]) + "</tr>"
            for row in table.body_rows:
                html_table += "<tr>" + "".join([f"<td>{cell.layout.text_anchor.content}</td>" for cell in row.cells]) + "</tr>"
            html_table += "</table>"
            chunks.append({"type": "table", "page_number": page_number, "content": html_table})

        # 2. 텍스트 블록(Paragraph, List 등) 처리
        for block in page.blocks:
            text_content = block.layout.text_anchor.content
            is_in_table = any(
                text_content in cell.layout.text_anchor.content
                for table in page.tables
                for row in table.header_rows + table.body_rows
                for cell in row.cells
            )
            if not is_in_table and text_content.strip():
                chunks.append({"type": "text", "page_number": page_number, "content": text_content})

    return chunks

def process_with_docai_custom_extractor(gcs_uri: str, mime_type: str) -> Dict[str, Any]:
    """Document AI Custom Extractor를 사용하여 정형화된 특정 필드를 추출합니다."""
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)
    
    gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, gcs_document=gcs_document)
    
    print(f"Document AI Custom Extractor로 문서 파싱 중: {gcs_uri}")
    result = client.process_document(request=request)
    document = result.document
    
    extracted_data = {entity.type_: entity.mention_text for entity in document.entities}
    return extracted_data

def move_blob(storage_client: storage.Client, source_blob: storage.Blob, dest_bucket_name: str):
    """GCS Blob을 다른 버킷으로 이동(복사 후 삭제)합니다."""
    source_bucket = source_blob.bucket
    dest_bucket = storage_client.bucket(dest_bucket_name)
    
    # Blob 복사
    source_bucket.copy_blob(source_blob, dest_bucket, source_blob.name)
    # 원본 Blob 삭제
    source_blob.delete()
    print(f"파일을 gs://{source_bucket.name}/{source_blob.name} -> gs://{dest_bucket.name}/{source_blob.name} (으)로 이동했습니다.")

async def process_and_store_document(pool, storage_client: storage.Client, blob: storage.Blob):
    """단일 문서를 파싱하고 DB에 저장하며, 결과에 따라 파일을 이동시키는 비동기 태스크입니다."""
    gcs_uri = f"gs://{blob.bucket.name}/{blob.name}"
    mime_type, _ = mimetypes.guess_type(gcs_uri)

    if not mime_type:
        print(f"MIME 타입을 알 수 없어 건너<binary data, 1 bytes>니다: {blob.name}")
        return

    try:
        # 여기서는 모든 문서에 Layout Parser를 적용합니다.
        # 파일 이름 패턴 등에 따라 Custom Extractor를 선택적으로 호출하는 로직을 추가할 수 있습니다.
        # 예: if "report" in blob.name: ...
        chunks = process_with_vertex_layout_parser(gcs_uri, mime_type)
        await store_chunks_in_alloydb(pool, blob.name, chunks)

        # 성공 시 OUTPUT 버킷으로 이동
        print(f"[{blob.name}] 처리 성공.")
        move_blob(storage_client, blob, OUTPUT_GCS_BUCKET)

    except Exception as e:
        print(f"문서 처리 중 오류 발생 [{blob.name}]: {e}")
        # 실패 시 ERROR 버킷으로 이동
        print(f"[{blob.name}] 처리 실패. 오류 버킷으로 이동합니다.")
        move_blob(storage_client, blob, ERROR_GCS_BUCKET)

async def main():
    """GCS 버킷의 모든 문서를 처리하여 AlloyDB에 저장합니다."""
    storage_client = storage.Client(project=PROJECT_ID)
    db_pool = None
    try:
        # 버킷 존재 여부 확인
        for bucket_name in [INPUT_GCS_BUCKET, OUTPUT_GCS_BUCKET, ERROR_GCS_BUCKET]:
            if not storage_client.bucket(bucket_name).exists():
                print(f"오류: GCS 버킷 '{bucket_name}'을 찾을 수 없습니다. 'gcp_setup_guide.md'를 참고하여 버킷을 생성해주세요.")
                return

        blobs = storage_client.list_blobs(INPUT_GCS_BUCKET)
        
        # 처리할 문서가 있는지 확인
        doc_list = [blob for blob in blobs if not blob.name.endswith('/')]
        if not doc_list:
            print(f"'{INPUT_GCS_BUCKET}' 버킷에 처리할 문서가 없습니다.")
            return

        print(f"총 {len(doc_list)}개의 문서를 처리합니다.")
        db_pool = await init_db_pool()
        
        tasks = [process_and_store_document(db_pool, storage_client, blob) for blob in doc_list]
        await asyncio.gather(*tasks)

        print("\n모든 문서 처리가 완료되었습니다.")

    except Exception as e:
        print(f"파이프라인 실행 중 오류 발생: {e}")
    finally:
        if db_pool:
            await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
