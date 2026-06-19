import os
import re
from typing import List, Dict, Any
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import documentai_v1 as documentai

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "uk-adc-core-geminienterprise")
LOCATION = "global"  # Vertex AI Search API Location
OUTPUT_GCS_BUCKET = "dawwon-pharm-docs-output" # 파싱 결과(이미지 등)를 저장할 버킷

# Custom Extractor 설정 (옵션: 특정 문서에 한해 사용)
DOCAI_LOCATION = "us"
DOCAI_PROCESSOR_ID = "your-custom-extractor-processor-id"

def process_with_vertex_layout_parser(gcs_uri: str, mime_type: str):
    """
    Vertex AI Search의 Layout-aware 파싱 API를 사용하여 문서를 구조화된 청크로 분리합니다.
    (문서의 목차(TOC), 표(Table), 텍스트(Text) 정보를 유지하며 파싱)
    """
    # API 클라이언트 초기화
    client_options = {"api_endpoint": f"discoveryengine.googleapis.com"}
    client = discoveryengine.DocumentServiceClient(client_options=client_options)
    
    # API 요청 구성
    parent = client.project_path(project=PROJECT_ID)
    request = discoveryengine.ProcessDocumentRequest(
        parent=parent,
        document=discoveryengine.Document(
            gcs_uri=gcs_uri,
            mime_type=mime_type,
        ),
        # OCR 및 이미지 추출 활성화
        process_options=discoveryengine.ProcessOptions(
            ocr_config=discoveryengine.OcrConfig(
                enable_native_pdf_parsing=True,
                enable_image_quality_scores=True,
            )
        ),
    )
    
    print(f"Vertex AI Search (Layout-aware)로 문서 파싱 중: {gcs_uri}")
    response = client.process_document(request=request)
    
    # API 응답을 기반으로 청크 생성
    chunks = []
    doc = response.document

    for page_index, page in enumerate(doc.pages):
        page_number = page_index + 1
        
        # 1. 테이블(Table) 처리
        for table in page.tables:
            # 테이블을 HTML 형식으로 변환
            html_table = "<table>"
            for row in table.header_rows:
                html_table += "<tr>" + "".join([f"<th>{cell.layout.text}</th>" for cell in row.cells]) + "</tr>"
            for row in table.body_rows:
                html_table += "<tr>" + "".join([f"<td>{cell.layout.text}</td>" for cell in row.cells]) + "</tr>"
            html_table += "</table>"
            chunks.append({"type": "table", "page_number": page_number, "content": html_table})

        # 2. 텍스트 블록(Paragraph, List 등) 처리
        for block in page.blocks:
            # 테이블에 속하지 않은 텍스트만 추출
            is_in_table = any(block.layout.text in table_cell.layout.text for table in page.tables for table_row in table.body_rows for table_cell in table_row.cells)
            if not is_in_table:
                chunks.append({"type": "text", "page_number": page_number, "content": block.layout.text})

        # 3. 이미지 처리 (화학식, 차트 등 포함)
        for image in page.images:
            # 멀티모달 LLM이 직접 참조할 수 있도록 GCS URI를 포함한 마크다운 형식으로 저장
            image_content = f"![이미지: 페이지 {page_number}]({image.uri})"
            chunks.append({"type": "image", "page_number": page_number, "content": image_content})

    return chunks

def process_with_docai_custom_extractor(gcs_uri: str, mime_type: str):
    """
    Document AI Custom Extractor를 사용하여 정형화된 특정 필드를 추출합니다.
    """
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)
    
    gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type=mime_type)
    request = documentai.ProcessRequest(
        name=name,
        gcs_document=gcs_document
    )
    
    print(f"Document AI Custom Extractor로 문서 파싱 중: {gcs_uri}")
    result = client.process_document(request=request)
    document = result.document
    
    extracted_data = {}
    for entity in document.entities:
        extracted_data[entity.type_] = entity.mention_text
        
    return extracted_data

if __name__ == "__main__":
    # 실행 예시
    sample_gcs_uri = "gs://dawwon-pharm-docs-input/sample_report.pdf"
    
    # 1. 단순 구조 파싱(표, 목차 보존)이 필요한 경우 - Vertex AI Layout 파싱
    layout_chunks = process_with_vertex_layout_parser(sample_gcs_uri, "application/pdf")
    print("Layout-aware Chunks:", layout_chunks)
    
    # 2. 특정 필드 추출이 필요한 경우 - Document AI Custom Extractor
    # extracted_fields = process_with_docai_custom_extractor(sample_gcs_uri, "application/pdf")
    # print("Extracted Fields:", extracted_fields)
