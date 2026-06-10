-- Relay durable schema (DESIGN.md §5.4).
--
-- Postgres is the source of truth for *analytics*: "p99 by model over the last
-- hour", "average batch size", "cache-hit rate by routing policy" — exactly the
-- queries that feed the benchmark report and the Pareto frontier. Redis handles
-- the ephemeral coordination state; these tables are the durable record.

CREATE TABLE IF NOT EXISTS api_keys (
  key_hash    TEXT PRIMARY KEY,
  tenant_id   TEXT NOT NULL,
  rps_limit   INT  NOT NULL DEFAULT 10,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS requests (
  id            UUID PRIMARY KEY,
  tenant_id     TEXT NOT NULL,
  model         TEXT NOT NULL,
  params        JSONB NOT NULL,
  status        TEXT NOT NULL,          -- queued|running|done|error
  batch_id      UUID,
  worker_id     TEXT,
  prefix_hash   TEXT,                   -- for cache-locality analysis
  cache_hit     BOOLEAN,                -- did the worker reuse a prefix?
  queue_wait_ms INT,
  inference_ms  INT,
  total_ms      INT,
  created_at    TIMESTAMPTZ DEFAULT now(),
  completed_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS requests_model_created_idx ON requests (model, created_at);
CREATE INDEX IF NOT EXISTS requests_status_idx        ON requests (status);
CREATE INDEX IF NOT EXISTS requests_prefix_hash_idx   ON requests (prefix_hash);

CREATE TABLE IF NOT EXISTS batches (
  id           UUID PRIMARY KEY,
  model        TEXT NOT NULL,
  size         INT  NOT NULL,
  worker_id    TEXT NOT NULL,
  inference_ms INT,
  created_at   TIMESTAMPTZ DEFAULT now()
);

-- Dev-only seed key so `curl` works out of the box against docker-compose.
-- key_hash = sha256("dev-key-please-change"). NEVER ship a real deployment
-- with this row present.
INSERT INTO api_keys (key_hash, tenant_id, rps_limit)
VALUES (
  '0b1e7c2f9d3a4b5c6d7e8f9012345678abcdef0123456789abcdef0123456789',
  'dev', 1000
)
ON CONFLICT (key_hash) DO NOTHING;
