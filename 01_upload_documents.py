import os
import asyncio
from google.cloud import storage

# 설정 변수
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "test000")
BUCKET_NAME = "dawwon-pharm-docs-input"  # 입력 문서를 업로드할 버킷 이름
LOCAL_DOC_DIR = "./sample_docs"          # 로컬 문서가 있는 디렉토리

async def upload_file_async(bucket, local_path, destination_blob_name):
    """비동기 방식으로 파일을 GCS 버킷에 업로드합니다."""
    loop = asyncio.get_event_loop()
    blob = bucket.blob(destination_blob_name)
    
    print(f"[{destination_blob_name}] 업로드 시작...")
    # 스토리지 IO 작업은 동기식이므로 run_in_executor를 사용하여 블로킹을 방지
    await loop.run_in_executor(None, blob.upload_from_filename, local_path)
    print(f"[{destination_blob_name}] 업로드 완료.")

async def main():
    if not os.path.exists(LOCAL_DOC_DIR):
        print(f"로컬 디렉토리를 찾을 수 없습니다: {LOCAL_DOC_DIR}")
        os.makedirs(LOCAL_DOC_DIR, exist_ok=True)
        print("샘플 문서를 해당 디렉토리에 넣어주세요.")
        return

    # GCS 클라이언트 초기화
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)

    # 버킷 존재 여부 확인
    if not bucket.exists():
        print(f"버킷 {BUCKET_NAME}이 존재하지 않습니다. 먼저 버킷을 생성하세요.")
        return

    upload_tasks = []
    
    # 디렉토리 내의 PDF 및 DOCX 파일 탐색
    for root, _, files in os.walk(LOCAL_DOC_DIR):
        for file in files:
            if file.lower().endswith((".pdf", ".docx")):
                local_path = os.path.join(root, file)
                # 하위 디렉토리 구조 유지
                destination_blob_name = os.path.relpath(local_path, LOCAL_DOC_DIR).replace("\\", "/")
                
                # 업로드 태스크 추가
                task = asyncio.create_task(upload_file_async(bucket, local_path, destination_blob_name))
                upload_tasks.append(task)
    
    if not upload_tasks:
        print("업로드할 PDF 또는 DOCX 파일을 찾을 수 없습니다.")
        return

    # 병렬로 모든 파일 업로드 실행
    await asyncio.gather(*upload_tasks)
    print(f"총 {len(upload_tasks)}개의 파일 업로드가 완료되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())
