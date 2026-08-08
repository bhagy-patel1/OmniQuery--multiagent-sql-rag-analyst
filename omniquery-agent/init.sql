-- Enable pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Structured Data: AR/VR Sales & Software Licenses
CREATE TABLE product_sales (
    id SERIAL PRIMARY KEY,
    region VARCHAR(50),
    product_line VARCHAR(100),
    revenue NUMERIC(12, 2),
    units_sold INT,
    fiscal_quarter VARCHAR(10)
);

INSERT INTO product_sales (region, product_line, revenue, units_sold, fiscal_quarter) VALUES 
('North America', 'AR Interior Designer Pro (License)', 850000.00, 1200, 'Q1-2026'),
('Europe', 'AR Interior Designer Pro (License)', 420000.00, 600, 'Q1-2026'),
('North America', 'Computer Vision API Tracker', 1250000.00, 310, 'Q1-2026'),
('Asia-Pacific', 'Computer Vision API Tracker', 980000.00, 245, 'Q1-2026');

-- 2. Unstructured Data: Hybrid Search Document Store
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_name VARCHAR(255),
    chunk_content TEXT,
    embedding vector(1024),          -- UPDATED: 1024 for Cohere embed-english-v3.0
    fts_tokens tsvector              -- Sparse vector for BM25 keyword matching
);

-- 3. Create Indexes for fast retrieval
CREATE INDEX idx_dense_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_sparse_fts ON document_chunks USING gin (fts_tokens);