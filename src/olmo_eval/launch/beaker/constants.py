"""Constants for Beaker launch configuration."""

# Beaker secret names for --store
OLMO_EVAL_DB_ARN_SECRET_NAME = "olmo_eval_DB_SECRET_ARN"
OLMO_EVAL_PGHOST_SECRET_NAME = "olmo_eval_PGHOST"

# Default database connection parameters for --store
STORE_DEFAULTS = {
    "PGPORT": "5432",
    "PGDATABASE": "olmo_eval",
    "PGUSER": "postgres",
}

# Default S3 storage parameters
DEFAULT_S3_BUCKET = "ai2-llm"
DEFAULT_S3_PREFIX = "olmo-eval"

# Infrastructure environment variables for Beaker jobs
# These configure olmo-eval's InfrastructureConfig when running in Beaker
BEAKER_INFRA_ENV_VARS = {
    "OLMO_CONTAINER_RUNTIME": "podman",
    "SWEREX_REGISTRY": "docker.io/olmo-eval",
    "OLMO_PASTA_HOST_IP": "169.254.1.2",
    "OLMO_RESULT_DIR": "/results",
    "OLMO_S3_BUCKET": DEFAULT_S3_BUCKET,
    "OLMO_S3_PREFIX": DEFAULT_S3_PREFIX,
}
