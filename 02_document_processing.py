import os
import re
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import documentai_v1 as documentai

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = "global" # Vertex AI Search API Location

# Custom Extractor 설정 (옵션: 특정 문서에 한해 사용)
DOCAI_LOCATION = "us"
DOCAI_PROCESSOR_ID = "your-custom-extractor-processor-id"

def process_with_vertex_layout_parser(gcs_uri: str, mime_type: str):
    """
    Vertex AI Search의 Layout-aware 파싱 API를 사용하여 문서를 구조화된 청크로 분리합니다.
    (문서의 목차(TOC), 표(Table), 텍스트(Text) 정보를 유지하며 파싱)
    """
    client = discoveryengine.DocumentServiceClient()

    # Document Extraction (Layout-aware)을 위한 요청 생성
    # 참고: 해당 기능은 alpha 버전 API(v1alpha)에서 지원하는 Layout 추출 기능을 가정합니다.
    # 사용자의 GCP 환경에 따라 데이터 스토어를 통한 파싱 또는 API 직접 호출을 선택합니다.
    # 아래는 API를 직접 호출하는 방식의 의사코드(pseudo-code) 패턴입니다.
    
    request = discoveryengine.ProcessDocumentRequest(
        parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
        gcs_document=discoveryengine.GcsDocument(
            gcs_uri=gcs_uri,
            mime_type=mime_type
        )
    )
    
    print(f"Vertex AI Search (Layout-aware)로 문서 파싱 중: {gcs_uri}")
    # response = client.process_document(request=request)
    
    # 가상의 반환 데이터 (실제 응답은 Layout 구조를 포함한 JSON 또는 Document 객체)
    chunks = []
    # 예시: 응답으로부터 Layout 구조에 따른 Chunk 추출
    # for layout_item in response.document.layout:
    #     if layout_item.type == 'TABLE':
    #         chunks.append({"type": "table", "content": layout_item.html_content})
    #     elif layout_item.type == 'TOC':
    #         chunks.append({"type": "toc", "content": layout_item.text_content})
    #     else:
    #         chunks.append({"type": "text", "content": layout_item.text_content})
            
    # 더미 반환
    chunks = [
        {"type": "text", "content": "임상 시험 목적: 신약 A의 효능 평가."},
        {"type": "table", "content": "<table><tr><th>부작용</th><th>비율</th></tr><tr><td>두통</td><td>5%</td></tr></table>"}
    ]
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
