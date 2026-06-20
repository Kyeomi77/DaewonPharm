import os
import re
import asyncio
import mimetypes
import hashlib
import json
from typing import List, Dict, Any
from google.cloud import storage
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import documentai_v1 as documentai
import asyncpg
from pgvector.asyncpg import register_vector
from langchain_google_vertexai import VertexAIEmbeddings

# --- 설정 변수 ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "uk-adc-core-geminienterprise")
LOCATION = "global"
INPUT_GCS_BUCKET = "daewonpharm-bucket-in-2026"
OUTPUT_GCS_BUCKET = "daewonpharm-bucket-out-2026"
ERROR_GCS_BUCKET = "daewonpharm-bucket-err-2026"

# AlloyDB 연결 정보
DB_HOST = "34.50.58.39"
DB_USER = "postgres"
DB_PASS = "DawornPharm2026!!"
DB_NAME = "postgres"

# 임베딩 모델 설정 (1536 차원 고정)
embedding_model = VertexAIEmbeddings(
    model_name="text-embedding-004", 
    project=PROJECT_ID,
    location="asia-northeast3",
    output_dimensionality=1536
)

# --- 가상의 화학 구조식 인식(OSR) 모듈 인터페이스 ---
def analyze_chemical_structure(image_uri: str) -> Dict[str, Any]:
    """
    DECIMER / MolScribe 등 화학식 광학 인식 엔진 연동 스텁(Stub)
    실제 운영 환경에서는 해당 도커 컨테이너 API 또는 라이브러리를 호출합니다.
    """
    # 예시 구조 데이터 반환 (실제 모델 연동 필요)
    return {
        "smiles": "CC(=O)NC1=CC=C(O)C=C1", # 아세트아미노펜 예시
        "inchi": "InChI=1S/C8H9NO2/c1-5(10)9-6-2-4-7(11)3-1-6/h1-4,11H,1H3,(H,9,10)",
        "inchikey": "RZVAJGRAIKMMMZ-UHFFFAOYSA-N",
        "osr_confidence": 0.98
    }

# --- 데이터베이스 함수 ---

async def init_db_pool():
    """AlloyDB 연결 풀을 초기화합니다."""
    pool = await asyncpg.create_pool(
        host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME,
        min_size=1, max_size=10
    )
    async with pool.acquire() as conn:
        await register_vector(conn)
    return pool

async def store_hierarchical_document(pool, blob: storage.Blob, parsed_data: Dict[str, Any]):
    """
    Parent-Child 계층 구조, 표, 이미지 및 GxP 감사 로그를 
    하나의 안전한 데이터베이스 트랜잭션 내에서 처리합니다.
    """
    # 1. 파일 검증용 해시 생성 (ALCOA+ 준수)
    file_bytes = blob.download_as_bytes()
    checksum_sha256 = hashlib.sha256(file_bytes).hexdigest()
    
    ext = os.path.splitext(blob.name)[1].lower()
    doc_format = {'pdf':'pdf', '.docx':'word', '.xlsx':'excel', '.jpg':'jpg'}.get(ext, 'pdf')
    source_uri = f"gs://{blob.bucket.name}/{blob.name}"

    async with pool.acquire() as conn:
        async with conn.transaction():
            # [A] 활성 임베딩 모델 레지스트리 확인
            model_id = await conn.fetchval(
                "SELECT model_id FROM embedding_models WHERE is_active = TRUE AND dimensions = 1536 LIMIT 1"
            )
            if not model_id:
                model_id = await conn.fetchval(
                    "INSERT INTO embedding_models (name, dimensions, modality, is_active) VALUES ($1, $2, $3, $4) RETURNING model_id",
                    "text-embedding-004", 1536, "text", True
                )

            # [B] 최상위 문서 메타데이터 적재
            doc_id = await conn.fetchval(
                """
                INSERT INTO documents (title, doc_format, source_uri, checksum_sha256, security_level, author)
                VALUES ($1, $2, $3, $4, $5, $6) RETURNING doc_id
                """,
                blob.name, doc_format, source_uri, checksum_sha256, 2, "Automated Pipeline"
            )

            # [C] Parent Chunks (섹션/단원) 적재 및 매핑 맵 생성
            parent_map = {} # 파싱 순서 인덱스 -> DB parent_id 매핑
            for p_idx, p_chunk in enumerate(parsed_data["parents"]):
                parent_id = await conn.fetchval(
                    """
                    INSERT INTO parent_chunks (doc_id, section_title, page_start, page_end, content)
                    VALUES ($1, $2, $3, $4, $5) RETURNING parent_id
                    """,
                    doc_id, p_chunk["title"], p_chunk["page_start"], p_chunk["page_end"], p_chunk["content"]
                )
                parent_map[p_idx] = parent_id

            # [D] 하위 텍스트 청크(chunks) 임베딩 및 적재 (Parent 연결)
            if parsed_data["chunks"]:
                texts = [c["content"] for c in parsed_data["chunks"]]
                embeddings = await embedding_model.aembed_documents(texts)
                
                chunk_records = []
                for c, emb in zip(parsed_data["chunks"], embeddings):
                    p_id = parent_map.get(c["parent_index"]) # 매핑된 상위 문맥 ID 추출
                    chunk_records.append((doc_id, p_id, c["page_num"], 'text', c["content"], model_id, emb))
                
                await conn.executemany(
                    """
                    INSERT INTO chunks (doc_id, parent_id, page_num, chunk_kind, content, model_id, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    chunk_records
                )

            # [E] 하위 표 데이터(doc_tables) 임베딩 및 적재
            if parsed_data["tables"]:
                table_texts = [t["content"] for t in parsed_data["tables"]]
                table_embeddings = await embedding_model.aembed_documents(table_texts)
                
                table_records = []
                for t, emb in zip(parsed_data["tables"], table_embeddings):
                    p_id = parent_map.get(t["parent_index"])
                    table_records.append((doc_id, p_id, t["page_num"], t["content"], model_id, emb))
                
                await conn.executemany(
                    """
                    INSERT INTO doc_tables (doc_id, parent_id, page_num, markdown_repr, model_id, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    table_records
                )

            # [F] 하위 이미지/화학식(figures) 임베딩 및 적재
            if parsed_data["figures"]:
                fig_records = []
                for f in parsed_data["figures"]:
                    p_id = parent_map.get(f["parent_index"])
                    
                    # 캡션 기반 멀티모달 프록시 임베딩 생성
                    emb = await embedding_model.aembed_query(f"{f['caption']} {f['ocr_text']}")
                    
                    # 화학식 구조식일 경우 OSR 데이터 추출 인터페이스 가동
                    osr_data = {}
                    if f["figure_kind"] == "chemical_structure":
                        osr_data = analyze_chemical_structure(f["image_uri"])
                    
                    fig_records.append((
                        doc_id, p_id, f["page_num"], f["figure_kind"], f["caption"], f["ocr_text"],
                        osr_data.get("smiles"), osr_data.get("inchi"), osr_data.get("inchikey"),
                        osr_data.get("osr_confidence"), f["image_uri"], model_id, emb
                    ))
                
                await conn.executemany(
                    """
                    INSERT INTO figures (
                        doc_id, parent_id, page_num, figure_kind, caption, ocr_text, 
                        smiles, inchi, inchikey, osr_confidence, image_uri, model_id, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    fig_records
                )

            # [G] GxP 감사 추적 로그 기록 (ALCOA+ 만족)
            await conn.execute(
                """
                INSERT INTO audit_log (entity, entity_id, action, actor, reason, diff)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                "documents", doc_id, "INSERT", "SYSTEM_PIPELINE", 
                "Automated document processing and hierarchical chunking ingestion.",
                json.dumps({"after": {"title": blob.name, "checksum": checksum_sha256}})
            )

    print(f"[{blob.name}] 계층 구조화 파싱 데이터 완전 적재 및 감사 로그 갱신 성공.")

# --- 문서 계층 분석 및 파싱 함수 ---

def process_hierarchical_document(gcs_uri: str, mime_type: str) -> Dict[str, Any]:
    """
    Vertex AI Search Layout Parser 결과를 바탕으로 
    Section(Parent) - Paragraph/Table/Figure(Child) 관계를 정밀 구조화합니다.
    """
    client_options = {"api_endpoint": "discoveryengine.googleapis.com"}
    client = discoveryengine.DocumentServiceClient(client_options=client_options)
    
    parent_path = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    request = discoveryengine.ProcessDocumentRequest(
        parent=parent_path,
        gcs_source=discoveryengine.GcsSource(gcs_uri=gcs_uri, mime_type=mime_type),
        process_options=discoveryengine.ProcessOptions(
            ocr_config=discoveryengine.OcrConfig(enable_native_pdf_parsing=True)
        ),
    )
    
    print(f"계층 분석 파싱 기동: {gcs_uri}")
    response = client.process_document(request=request)
    doc = response.document
    
    # 정규화 데이터 컨테이너
    parsed_result = {"parents": [], "chunks": [], "tables": [], "figures": []}
    
    # 1. 1차 패스: 대단원/섹션(Parent Chunks) 식별 및 생성 (휴리스틱 분석)
    # 실제로는 block.layout의 스타일이나 폰트 크기 메타데이터를 사용하여 단원을 구별합니다.
    current_parent_idx = 0
    
    # 문서가 비어있을 경우를 대비해 기본 부모 컨텍스트 초기화
    parsed_result["parents"].append({
        "title": "요약 및 기본 서문", "page_start": 1, "page_end": len(doc.pages),
        "content": f"문서 {gcs_uri}의 통합 본문 컨텍스트입니다."
    })

    # 2. 2차 패스: 페이지 요소를 탐색하며 부모 단원에 매핑
    for page_idx, page in enumerate(doc.pages):
        page_num = page_idx + 1
        
        # [부모 단원 갱신 감지 캡처 예시]
        # 만약 특정 텍스트 블록이 "제 1장", "연구 배경" 등 제목 규격을 만족하면 부모 노드를 신규 추가
        for block in page.blocks:
            text = block.layout.text_anchor.content.strip()
            if re.match(r'^(제\s?\d+\s?장|목차|참고문헌|\d+\.\s?\uB300\uB2E8\uC6D0)', text):
                parsed_result["parents"].append({
                    "title": text[:50], "page_start": page_num, "page_end": len(doc.pages),
                    "content": f"단원 제목: {text}"
                })
                current_parent_idx = len(parsed_result["parents"]) - 1

        # [하위 테이블 처리]
        for table in page.tables:
            html_table = "<table>"
            for row in table.header_rows:
                html_table += "<tr>" + "".join([f"<th>{c.layout.text_anchor.content}</th>" for c in row.cells]) + "</tr>"
            for row in table.body_rows:
                html_table += "<tr>" + "".join([f"<td>{c.layout.text_anchor.content}</td>" for c in row.cells]) + "</tr>"
            html_table += "</table>"
            
            parsed_result["tables"].append({
                "parent_index": current_parent_idx, "page_num": page_num, "content": html_table
            })

        # [하위 텍스트 청크 처리]
        for block in page.blocks:
            text_content = block.layout.text_anchor.content
            if text_content.strip():
                parsed_result["chunks"].append({
                    "parent_index": current_parent_idx, "page_num": page_num, "content": text_content
                })

        # [하위 시각 자료/이미지(Figures) 처리 및 분류]
        # Layout Parser가 탐지한 도표 영역이나 이미지 요소를 탐색합니다.
        # 아래는 스키마 적재 흐름을 보여주기 위한 정형화 추출 레이어입니다.
        if hasattr(page, 'visual_elements'):
            for element in page.visual_elements:
                # 파일 확장자나 이름 패턴, 혹은 부서에 따라 Kind 분기
                kind = "chemical_structure" if "pharma" in gcs_uri.lower() else "chart"
                parsed_result["figures"].append({
                    "parent_index": current_parent_idx,
                    "page_num": page_num,
                    "figure_kind": kind,
                    "caption": "자동 추출된 연구 이미지 시각 데이터 캡션 설명",
                    "ocr_text": "이미지 내부 인지 텍스트 데이터 블록",
                    "image_uri": f"{gcs_uri}_fig_{page_num}.png"
                })

    return parsed_result

# --- 오케스트레이션 메인 흐름 ---

def move_blob(storage_client: storage.Client, source_blob: storage.Blob, dest_bucket_name: str):
    source_bucket = source_blob.bucket
    dest_bucket = storage_client.bucket(dest_bucket_name)
    source_bucket.copy_blob(source_blob, dest_bucket, source_blob.name)
    source_blob.delete()

async def process_and_store_document(pool, storage_client: storage.Client, blob: storage.Blob):
    gcs_uri = f"gs://{blob.bucket.name}/{blob.name}"
    mime_type, _ = mimetypes.guess_type(gcs_uri)
    if not mime_type: return

    try:
        # 1. 문서 고도화 계층 분석 파싱
        parsed_data = process_hierarchical_document(gcs_uri, mime_type)
        
        # 2. RAG 전용 스키마에 동기화 저장
        await store_hierarchical_document(pool, blob, parsed_data)

        print(f"[{blob.name}] 전공정 파이프라인 처리 완결.")
        move_blob(storage_client, blob, OUTPUT_GCS_BUCKET)
    except Exception as e:
        print(f"문서 실패 로그 [{blob.name}]: {e}")
        move_blob(storage_client, blob, ERROR_GCS_BUCKET)

async def main():
    storage_client = storage.Client(project=PROJECT_ID)
    db_pool = None
    try:
        for bucket_name in [INPUT_GCS_BUCKET, OUTPUT_GCS_BUCKET, ERROR_GCS_BUCKET]:
            if not storage_client.bucket(bucket_name).exists():
                print(f"버킷 누락 오류: {bucket_name}")
                return

        blobs = storage_client.list_blobs(INPUT_GCS_BUCKET)
        doc_list = [b for b in blobs if not b.name.endswith('/')]
        if not doc_list: return

        db_pool = await init_db_pool()
        tasks = [process_and_store_document(db_pool, storage_client, blob) for blob in doc_list]
        await asyncio.gather(*tasks)
        print("\n[성공] 제약 엔터프라이즈 통합 데이터 적재 완료.")
    finally:
        if db_pool: await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())