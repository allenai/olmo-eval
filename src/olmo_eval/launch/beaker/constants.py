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
