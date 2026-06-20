-- =============================================================================
-- 대한민국 제약사 연구소 PGVector 스키마
-- 대상 DB: PostgreSQL + PGVector
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 0. 확장 활성화
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;      -- PGVector (벡터 저장·검색)
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- 트라이그램 (키워드 보조 검색)
-- CREATE EXTENSION IF NOT EXISTS rdkit;    -- 화학 구조 검색 (선택, RDKit-Postgres)


-- ---------------------------------------------------------------------------
-- 1. 공통 ENUM
-- ---------------------------------------------------------------------------
CREATE TYPE doc_format    AS ENUM ('pdf', 'word', 'excel', 'jpg');
CREATE TYPE chunk_kind    AS ENUM ('text', 'handwritten_note', 'caption');
CREATE TYPE figure_kind   AS ENUM ('chart', 'chemical_structure', 'raw_image');
CREATE TYPE review_status AS ENUM ('auto', 'pending_review', 'verified', 'rejected');


-- ---------------------------------------------------------------------------
-- 2. embedding_models — 임베딩 모델 레지스트리
--    · 테이블마다 차원을 하드코딩하지 않고 모델·버전을 중앙 관리
--    · 모델 교체/재임베딩 시 추적성 확보
-- ---------------------------------------------------------------------------
CREATE TABLE embedding_models (
    model_id    SMALLSERIAL  PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,          -- 예: text-embedding-3-small
    dimensions  INTEGER      NOT NULL,          -- 1536, 3072 ...
    modality    VARCHAR(20)  NOT NULL,          -- text | multimodal
    is_active   BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

COMMENT ON TABLE  embedding_models              IS '임베딩 모델 레지스트리 — 차원·버전 추적';
COMMENT ON COLUMN embedding_models.modality     IS 'text | multimodal';


-- ---------------------------------------------------------------------------
-- 3. documents — 최상위 문서 메타데이터
--    · 버전 체인(supersedes), 체크섬, 보안등급, 소프트삭제 포함
--    · GxP: 물리 삭제 금지 → deleted_at 소프트삭제
-- ---------------------------------------------------------------------------
CREATE TABLE documents (
    doc_id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    title            VARCHAR(255) NOT NULL,
    doc_format       doc_format,
    department       VARCHAR(100),              -- 연구소 부서
    project_code     VARCHAR(100),              -- 신약/프로젝트 코드
    author           VARCHAR(100),
    version          INTEGER      NOT NULL DEFAULT 1,
    is_latest        BOOLEAN      NOT NULL DEFAULT TRUE,
    supersedes       UUID         REFERENCES documents(doc_id),  -- 이전 버전 참조
    source_uri       TEXT         NOT NULL,     -- 불변 스토리지 경로 (S3 등)
    checksum_sha256  CHAR(64)     NOT NULL,     -- 원본 파일 무결성 (ALCOA+)
    security_level   SMALLINT     NOT NULL DEFAULT 1,  -- 1 일반 ~ 4 기밀
    created_at       TIMESTAMPTZ,               -- 원본 문서 작성일
    ingested_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    deleted_at       TIMESTAMPTZ,              -- NULL = 유효, NOT NULL = 소프트삭제
    metadata         JSONB
);

CREATE INDEX idx_documents_project  ON documents (project_code)    WHERE deleted_at IS NULL;
CREATE INDEX idx_documents_dept     ON documents (department)      WHERE deleted_at IS NULL;
CREATE INDEX idx_documents_latest   ON documents (is_latest)       WHERE deleted_at IS NULL;

COMMENT ON TABLE  documents                    IS '원본 문서 메타데이터 — 버전·무결성·보안등급 관리';
COMMENT ON COLUMN documents.supersedes         IS '이전 버전 doc_id (개정 체인)';
COMMENT ON COLUMN documents.checksum_sha256    IS 'SHA-256 해시 — 원본 무결성 검증 (ALCOA+)';
COMMENT ON COLUMN documents.security_level     IS '1=일반, 2=내부, 3=제한, 4=기밀';
COMMENT ON COLUMN documents.deleted_at         IS 'GxP: 물리 삭제 금지, 소프트삭제만 허용';


-- ---------------------------------------------------------------------------
-- 4. parent_chunks — 상위 문맥 단위 (섹션/챕터)
--    · 임베딩 없음 — LLM 컨텍스트 보강용
--    · 검색은 child chunks 에서, 응답 생성 시 parent 로 문맥 확장
-- ---------------------------------------------------------------------------
CREATE TABLE parent_chunks (
    parent_id     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id        UUID         NOT NULL REFERENCES documents(doc_id),
    section_title VARCHAR(255),
    page_start    INTEGER,
    page_end      INTEGER,
    content       TEXT         NOT NULL,        -- 섹션 전체 원문 (컨텍스트용)
    metadata      JSONB
);

CREATE INDEX idx_parent_chunks_doc ON parent_chunks (doc_id);

COMMENT ON TABLE parent_chunks IS 'Parent-Child 청킹의 상위 단위 — 임베딩 없이 컨텍스트만 보관';


-- ---------------------------------------------------------------------------
-- 5. chunks — 텍스트 / 연구노트 / 수기 데이터 청크 (검색 대상)
--    · tsvector 생성 컬럼으로 FTS 하이브리드 검색 지원
--    · ocr_confidence / review_status 로 수기 데이터 품질 추적 (GxP)
--    · embedding 차원은 model_id 참조 기준 (기본 1536, 교체 시 마이그레이션)
-- ---------------------------------------------------------------------------
CREATE TABLE chunks (
    chunk_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID         NOT NULL REFERENCES documents(doc_id),
    parent_id       UUID         REFERENCES parent_chunks(parent_id),
    page_num        INTEGER,
    chunk_kind      chunk_kind   NOT NULL DEFAULT 'text',
    content         TEXT         NOT NULL,
    bbox            JSONB,                      -- 원본 좌표 {page,x0,y0,x1,y1}
    ocr_confidence  REAL,                       -- OCR/수기 신뢰도 (NULL=네이티브 텍스트)
    review_status   review_status NOT NULL DEFAULT 'auto',
    model_id        SMALLINT     REFERENCES embedding_models(model_id),
    embedding       vector(1536),               -- ※ 모델 변경 시 차원 조정
    -- 하이브리드 검색용 tsvector (자동 생성)
    content_tsv     tsvector     GENERATED ALWAYS AS
                        (to_tsvector('simple', content)) STORED,
    metadata        JSONB
);

-- 벡터 검색 (HNSW, 코사인 유사도)
CREATE INDEX idx_chunks_embedding   ON chunks USING hnsw (embedding vector_cosine_ops);
-- 전문 검색 (FTS)
CREATE INDEX idx_chunks_tsv         ON chunks USING gin  (content_tsv);
-- 메타 사전 필터
CREATE INDEX idx_chunks_doc_kind    ON chunks (doc_id, chunk_kind);
CREATE INDEX idx_chunks_review      ON chunks (review_status) WHERE review_status = 'pending_review';

COMMENT ON TABLE  chunks                   IS '텍스트·연구노트·수기 청크 — 검색 기본 단위';
COMMENT ON COLUMN chunks.ocr_confidence    IS 'Document AI OCR 신뢰도 (0~1), NULL=네이티브 텍스트';
COMMENT ON COLUMN chunks.content_tsv       IS 'FTS용 자동 생성 컬럼 — 한국어 정밀도 필요 시 pg_bigm/mecab-ko 사전 적용';

-- ---------------------------------------------------------------------------
-- [참고] 3072차원 모델(text-embedding-3-large 등) 사용 시 HNSW 인덱스
--   PGVector의 vector 타입은 HNSW/IVFFlat에서 2,000차원 상한 존재.
--   halfvec 캐스팅으로 최대 4,000차원까지 인덱싱 가능 (메모리 ~50% 절감).
--
--   CREATE INDEX idx_chunks_embedding_3072
--     ON chunks USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);
--
--   쿼리 시:
--   ORDER BY embedding::halfvec(3072) <=> $1::halfvec(3072)
-- ---------------------------------------------------------------------------


-- ---------------------------------------------------------------------------
-- 6. doc_tables — 표 데이터 (Excel·PDF 내 표)
-- ---------------------------------------------------------------------------
CREATE TABLE doc_tables (
    table_id      UUID     PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id        UUID     NOT NULL REFERENCES documents(doc_id),
    parent_id     UUID     REFERENCES parent_chunks(parent_id),
    page_num      INTEGER,
    table_caption TEXT,
    table_content JSONB,                        -- 구조화 행/열 데이터
    markdown_repr TEXT,                         -- LLM 컨텍스트 및 임베딩 소스
    model_id      SMALLINT REFERENCES embedding_models(model_id),
    embedding     vector(1536),                 -- markdown_repr 임베딩
    content_tsv   tsvector GENERATED ALWAYS AS
                      (to_tsvector('simple', coalesce(markdown_repr, ''))) STORED,
    metadata      JSONB
);

CREATE INDEX idx_doc_tables_embedding ON doc_tables USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_doc_tables_tsv       ON doc_tables USING gin  (content_tsv);
CREATE INDEX idx_doc_tables_doc       ON doc_tables (doc_id);

COMMENT ON TABLE  doc_tables               IS '표 데이터 — JSONB(구조) + 마크다운(임베딩·검색) 이중 저장';
COMMENT ON COLUMN doc_tables.markdown_repr IS '임베딩 및 LLM 컨텍스트 전달용 마크다운 표현';


-- ---------------------------------------------------------------------------
-- 7. figures — 차트 / 화학식 / 이미지
--    · 화학식: OSR(DECIMER/MolScribe/OSRA)로 SMILES·InChI·InChIKey 추출
--      → InChIKey 정확 매칭, 핑거프린트 기반 구조 유사도 검색 가능
--    · 차트/이미지: Vision LLM 캡션 + OCR 텍스트
-- ---------------------------------------------------------------------------
CREATE TABLE figures (
    figure_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id         UUID         NOT NULL REFERENCES documents(doc_id),
    parent_id      UUID         REFERENCES parent_chunks(parent_id),
    page_num       INTEGER,
    figure_kind    figure_kind  NOT NULL,
    caption        TEXT,                        -- Vision LLM 생성 설명
    ocr_text       TEXT,                        -- 이미지 내 추출 텍스트
    -- 화학식 전용 (figure_kind = 'chemical_structure')
    smiles         TEXT,                        -- 정규화 SMILES
    inchi          TEXT,                        -- InChI 문자열
    inchikey       CHAR(27),                    -- 27자 해시 — 정확 구조 매칭 키
    osr_confidence REAL,                        -- 광학 구조 인식 신뢰도
    review_status  review_status NOT NULL DEFAULT 'auto',
    image_uri      TEXT         NOT NULL,       -- 원본 이미지 스토리지 경로
    model_id       SMALLINT     REFERENCES embedding_models(model_id),
    embedding      vector(1536),                -- caption + ocr_text 멀티모달 임베딩
    metadata       JSONB
);

CREATE INDEX idx_figures_embedding  ON figures USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_figures_inchikey   ON figures (inchikey);      -- 동일 구조 즉시 조회
CREATE INDEX idx_figures_kind       ON figures (figure_kind);
CREATE INDEX idx_figures_doc        ON figures (doc_id);

COMMENT ON TABLE  figures             IS '차트·화학식·이미지 — 화학식은 InChIKey 정확 매칭·구조 검색 지원';
COMMENT ON COLUMN figures.inchikey    IS 'OSR 추출 InChIKey 27자 — 동일 구조 정확 매칭용';
COMMENT ON COLUMN figures.smiles      IS '정규화 SMILES — 핑거프린트 기반 유사도 검색 원본';


-- ---------------------------------------------------------------------------
-- 8. audit_log — GxP 감사 추적 (불변 로그)
--    · ALCOA+: 누가(actor)·언제(occurred_at)·무엇을(entity/entity_id)·이유(reason)
--    · INSERT ONLY — UPDATE/DELETE 금지 (트리거로 강제 권장)
-- ---------------------------------------------------------------------------
CREATE TABLE audit_log (
    audit_id    BIGSERIAL    PRIMARY KEY,
    entity      VARCHAR(40)  NOT NULL,          -- documents | chunks | figures | doc_tables
    entity_id   UUID         NOT NULL,
    action      VARCHAR(20)  NOT NULL,          -- INSERT | UPDATE | SOFT_DELETE
    actor       VARCHAR(100) NOT NULL,          -- 전자서명 주체 (사용자/시스템)
    reason      TEXT,                           -- 변경 사유
    diff        JSONB,                          -- 변경 전/후 스냅샷
    occurred_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_entity ON audit_log (entity, entity_id);
CREATE INDEX idx_audit_actor  ON audit_log (actor);
CREATE INDEX idx_audit_time   ON audit_log (occurred_at DESC);

COMMENT ON TABLE  audit_log            IS 'GxP 불변 감사 로그 — ALCOA+ 준수, INSERT ONLY';
COMMENT ON COLUMN audit_log.diff       IS '변경 전/후 JSON 스냅샷 {"before": {...}, "after": {...}}';


-- ---------------------------------------------------------------------------
-- 9. 행 수준 보안 (RLS) — 신약 기밀 접근통제
--    · app.user_clearance 세션 변수에 현재 사용자 보안등급 주입
--    · 예: SET LOCAL app.user_clearance = '2';
-- ---------------------------------------------------------------------------
ALTER TABLE documents  ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks     ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_tables ENABLE ROW LEVEL SECURITY;
ALTER TABLE figures    ENABLE ROW LEVEL SECURITY;

CREATE POLICY policy_documents  ON documents
    USING (security_level <= current_setting('app.user_clearance', TRUE)::SMALLINT);

CREATE POLICY policy_chunks     ON chunks
    USING (doc_id IN (
        SELECT doc_id FROM documents
        WHERE security_level <= current_setting('app.user_clearance', TRUE)::SMALLINT
    ));

CREATE POLICY policy_doc_tables ON doc_tables
    USING (doc_id IN (
        SELECT doc_id FROM documents
        WHERE security_level <= current_setting('app.user_clearance', TRUE)::SMALLINT
    ));

CREATE POLICY policy_figures    ON figures
    USING (doc_id IN (
        SELECT doc_id FROM documents
        WHERE security_level <= current_setting('app.user_clearance', TRUE)::SMALLINT
    ));


-- ---------------------------------------------------------------------------
-- 10. 하이브리드 검색 뷰 — 청크 벡터+FTS 결합 (RRF)
--     · $1 = 쿼리 임베딩 (vector), $2 = 키워드 (text)
--     · 실제 사용 시 prepared statement 또는 함수로 래핑 권장
-- ---------------------------------------------------------------------------
-- 참고용 RRF 쿼리 (뷰는 파라미터 미지원으로 함수 형태 권장)
--
-- WITH vec AS (
--   SELECT chunk_id,
--          ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rnk
--   FROM chunks
--   ORDER BY embedding <=> $1 LIMIT 50
-- ),
-- kw AS (
--   SELECT chunk_id,
--          ROW_NUMBER() OVER (
--            ORDER BY ts_rank(content_tsv, plainto_tsquery('simple', $2)) DESC
--          ) AS rnk
--   FROM chunks
--   WHERE content_tsv @@ plainto_tsquery('simple', $2)
--   LIMIT 50
-- )
-- SELECT COALESCE(vec.chunk_id, kw.chunk_id) AS chunk_id,
--        (1.0 / (60 + COALESCE(vec.rnk, 1000))) +
--        (1.0 / (60 + COALESCE(kw.rnk,  1000))) AS rrf_score
-- FROM vec FULL OUTER JOIN kw USING (chunk_id)
-- ORDER BY rrf_score DESC
-- LIMIT 10;


-- ---------------------------------------------------------------------------
-- 완료
-- ---------------------------------------------------------------------------
