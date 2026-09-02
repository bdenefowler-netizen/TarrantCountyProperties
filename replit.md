# Tarrant County Property Research

Streamlit website that enriches Tarrant County property workbooks with public-record research and exports the results as Excel.

## Run & Operate

- `streamlit run app.py --server.port 5000` — run the property research website
- `pnpm --filter @workspace/api-server run dev` — run the shared API server
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `app.py` — Streamlit application and property research engine
- `pyproject.toml` / `uv.lock` — Python dependencies and lockfile
- `attached_assets/` — original uploaded source file

## Architecture decisions

- The uploaded Streamlit application is kept as the primary website entry point.
- TAD enrichment uses official property PDFs after resolving an account number; it does not depend on an undocumented search endpoint.
- PubRecord remains a manual-review link only, respecting its stated search limits.
- Uploaded workbooks are processed in the running session and can be downloaded as enriched Excel.

## Product

- Upload an `.xlsx` property or foreclosure workbook.
- Optionally upload a public TAD CSV/XLSX address-to-account index.
- Validate recognized headers, enrich rows, review match confidence and missing data, and export the completed workbook.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

_Populate as you build — sharp edges, "always run X before Y" rules._

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
