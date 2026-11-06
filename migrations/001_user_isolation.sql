-- Multi-tenant document isolation.
-- Existing documents remain ownerless and are intentionally hidden from all users.
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

DROP INDEX IF EXISTS documents_content_hash_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS documents_user_content_hash_uidx
    ON documents (user_id, content_hash)
    WHERE user_id IS NOT NULL AND content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS documents_user_id_idx ON documents(user_id);
