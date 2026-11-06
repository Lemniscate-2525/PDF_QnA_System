# Productionization v2 deployment

This branch changes the app from a single shared PDF workspace into an authenticated, user-isolated application with Redis-backed controls and background ingestion.

## Services

| Service | Responsibility |
| --- | --- |
| Streamlit Community Cloud | Authenticated UI and question-answering workflow |
| Supabase Auth | Email/password identity and session validation |
| Supabase PostgreSQL | Documents, ownership metadata, chunks and pgvector |
| Supabase Storage | Private temporary PDF objects waiting for ingestion |
| Redis | Atomic sliding-window limits, shared answer cache and RQ queue |
| RQ worker | PDF extraction, OCR, chunking, embedding and persistence |
| OpenRouter | LLM inference |

## Required setup

### 1. Supabase Auth

In Supabase, enable Email authentication. Configure email confirmation according to the account workflow you want. Add the app's deployed URL to the allowed redirect/site URL settings if required by your project.

Set these server-side secrets:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

The service-role key is used only by server-side application and worker code for private Storage operations. Never expose it in browser code or commit it.

### 2. Supabase Storage

Create a **private** bucket named `pdf-uploads` (or set `UPLOAD_BUCKET` to another name). The app writes objects under `<user_id>/<random_id>.pdf`. The worker deletes the source object after successful processing. Failed jobs retain the object so retries can run.

### 3. Database migration

The app's additive initialization adds `documents.user_id`, replaces the old global content-hash uniqueness rule with a per-user uniqueness rule, and creates the owner index. The equivalent SQL is in `migrations/001_user_isolation.sql`.

Existing rows have a NULL owner and are deliberately invisible to authenticated users. Do not bulk-assign them to an account unless you have verified who owns each document. Back up the database before deploying schema changes.

The application connects with its server-side PostgreSQL credentials. All document reads, duplicate checks, writes and vector retrievals include the authenticated user ID. Do not add an unscoped document query to the UI.

### 4. Redis

Provision a Redis instance with TLS where supported and configure `REDIS_URL`.

The sliding-window limiter is a Lua script using Redis server time and a sorted set, so the check-and-increment operation is atomic across app instances. Current defaults:

- 5 PDF uploads per user per hour
- 30 questions per user per minute

The answer cache key includes the user ID, document ID, query and retrieval depth. This prevents cross-user cache collisions. Cached answers expire after 180 seconds.

Redis is required for the intended production behavior. Without `REDIS_URL`, the limiter and cache helpers operate in local-development fallback mode; do not deploy publicly in that mode.

### 5. Streamlit secrets

Set `APP_ENV = "production"` and configure the existing database and OpenRouter values plus:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
UPLOAD_BUCKET = "pdf-uploads"
REDIS_URL = "rediss://default:password@host:port"
```

Keep all values in Streamlit Secrets or the worker's environment, never in Git.

### 6. Deploy the worker separately

Streamlit Community Cloud runs the UI, not a persistent RQ worker. Deploy this repository as a separate background worker service using `Dockerfile.worker`, with the start command:

```bash
python worker_entrypoint.py
```

Give the worker the same database connection settings, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `UPLOAD_BUCKET`, `REDIS_URL`, and OCR settings. The worker needs enough memory and CPU for Sentence Transformers and OCR. Run one worker initially; scale only after measuring queue wait time and memory use.

The worker uses RQ retries. The worker entrypoint enables the scheduler for delayed retries.

## Security behavior

- Unauthenticated sessions cannot reach the PDF workspace.
- Supabase Auth identity is checked/refreshed on app reruns.
- Documents are listed only where `documents.user_id` matches the authenticated user.
- Similarity search joins documents and checks the same owner ID.
- Duplicate detection is scoped to the owner.
- PDF excerpts are explicitly untrusted in the generation prompt.
- Uploads are limited to 20 MB and 250 pages.
- Questions are limited to 2,000 characters.
- Upload and question limits are shared through Redis.
- LLM calls have a 30-second timeout, bounded retries, a process-local circuit breaker and a 1,200-token output cap.

## Verification before public rollout

1. Create two separate Supabase Auth users.
2. Upload a distinct PDF as User A.
3. Sign in as User B and verify the PDF is absent.
4. Attempt to query User A's document ID as User B; the database retrieval must return no chunks.
5. Upload the same PDF as both users; each user should get a separate owned document.
6. Restart the Streamlit process and verify the user must authenticate again.
7. Confirm Redis rate limits apply across multiple app sessions.
8. Submit a large/scanned PDF and verify the UI returns a job ID while the worker processes it.
9. Force a worker failure and verify the source object remains available for retry.
10. Inspect logs for accidental secrets, prompts, raw document text or full query content.

This is an engineering implementation, not a claim of independent security certification. Run the tests and the two-user isolation checks before merging or exposing it to real private documents.
