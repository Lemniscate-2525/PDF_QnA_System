-- Reviewed multi-tenant migration for PDF Q&A v2.
-- Existing rows retain user_id = NULL and remain invisible to all accounts.
ALTER TABLE public.documents
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS documents_user_id_idx
    ON public.documents(user_id);

-- Replace any former global deduplication with per-account deduplication.
DROP INDEX IF EXISTS public.documents_content_hash_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS documents_user_content_hash_uidx
    ON public.documents(user_id, content_hash)
    WHERE user_id IS NOT NULL AND content_hash IS NOT NULL;

-- The app accesses these tables only through its trusted server-side Postgres
-- connection. No anon/authenticated PostgREST policies are granted.
-- This blocks direct client API access; the trusted DB owner connection bypasses RLS.
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.embeddings ENABLE ROW LEVEL SECURITY;

-- Private staging bucket. Only the server-side worker uses the service-role key.
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('pdf-uploads', 'pdf-uploads', false, 20971520, ARRAY['application/pdf'])
ON CONFLICT (id) DO UPDATE
SET public = false,
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;
