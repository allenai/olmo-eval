Alembic Database Migrations
===========================

Two databases on the same PostgreSQL instance:

- **olmo_eval**: Evaluation results, experiments, task metrics
- **olmo_eval_metrics**: Inference performance telemetry

## Running Migrations

```bash
# Both databases (with AWS Secrets Manager support)
scripts/internal/db-migrate upgrade head

# Individual databases
scripts/internal/db-migrate upgrade head results
scripts/internal/db-migrate upgrade head metrics

# Or via make
make db-upgrade
```

## Creating Migrations

```bash
# Results database
alembic -c alembic/results.ini revision --autogenerate -m "Add column"

# Metrics database
alembic -c alembic/metrics.ini revision --autogenerate -m "Add column"
```

## Configuration

Uses the same environment variables as the CLI:

```bash
export OLMO_EVAL_DB_HOST="your-db.rds.amazonaws.com"
export OLMO_EVAL_DB_PORT="5432"
export OLMO_EVAL_DB_USER="postgres"
export OLMO_EVAL_DB_SECRET_ARN="arn:aws:secretsmanager:..."  # or OLMO_EVAL_DB_PASSWORD
```

## Directory Structure

```
alembic/
├── results.ini
├── metrics.ini
├── results/
│   ├── env.py
│   └── versions/
└── metrics/
    ├── env.py
    └── versions/
```

## Models

- Results: `src/olmo_eval/storage/backends/postgres/models.py` (Base)
- Metrics: `src/olmo_eval/storage/backends/postgres/metrics_models.py` (MetricsBase)
