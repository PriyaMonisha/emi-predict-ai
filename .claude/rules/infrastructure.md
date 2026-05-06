---
paths:
  - "docker/**"
  - "docker-compose*.yml"
  - "Dockerfile*"
  - "airflow/dags/**"
  - ".env.example"
---

# Infrastructure Rules — EMI Predict AI

## Docker Non-Negotiables
- PYTHONPATH=/app in EVERY Dockerfile — this is a locked project decision
- Multi-stage builds always: builder stage then runtime stage
- Non-root user in all containers: create and use USER appuser
- Health check defined on every service in docker-compose
- .dockerignore must exclude: data/raw/, mlruns/, *.pkl, *.joblib, __pycache__, .git

## Phase 2 Compatibility (Design for AWS ECS Now — Zero Rewrites Later)
- No localhost references in any config file (use Docker service names)
- All secrets via environment variables — never hardcoded anywhere
- Services must be stateless (all state lives in Redis or PostgreSQL)
- All logs to stdout — not to files (ECS log driver picks up stdout)
- Port mappings must match future ECS task definition patterns

## Docker Compose Service Standards
Every service must have:
  - image or build specification
  - environment variables from .env file
  - health check with interval, timeout, retries
  - restart: unless-stopped
  - networks assignment
  - meaningful container_name

## Airflow DAG Standards
- schedule: '0 2 * * *' — never use @daily shorthand
- max_active_runs: 1
- catchup: False — always, no exceptions
- default_args retries: 3
- default_args retry_delay: timedelta(minutes=5)
- All paths from Variable.get('variable_name') — never hardcoded
- Task IDs: descriptive snake_case

## Redis Standards
- Key pattern: emi:features:{hashed_customer_id}
- TTL: 86400 on every SET operation — no exceptions
- Serialization: JSON only — never pickle
- Never store plaintext PII — always hash customer identifiers