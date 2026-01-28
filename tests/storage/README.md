# Storage Tests

Tests for the storage backend modules.

## Test Organization

### Unit Tests (No Database Required)

- **test_base.py** - Tests for dataclasses (EvalResult, TaskResult)
- **test_models.py** - Tests for SQLAlchemy ORM models
- **test_session.py** - Tests for session management (mocked)

### Integration Tests (Require Docker/Database)

Located in `tests/integration/`:

- **test_storage.py** - Backward compatibility tests for PostgresBackend and S3Backend
- **test_storage_postgres_new.py** - Tests for new features (instance predictions, pairwise queries)
- **test_repository.py** - Tests for repository layer (ExperimentRepository, InstancePredictionRepository)

## Running Tests

### Unit Tests Only
```bash
pytest tests/storage/
```

### Integration Tests
```bash
# Start Docker services first
cd tests/integration
docker compose up -d

# Run integration tests
pytest tests/integration/test_storage*.py tests/integration/test_repository.py --integration

# Cleanup
docker compose down -v
```

### All Tests
```bash
# Start services
cd tests/integration && docker compose up -d && cd ../..

# Run all
pytest tests/storage/ tests/integration/test_storage*.py tests/integration/test_repository.py --integration

# Cleanup
cd tests/integration && docker compose down -v
```

## Test Coverage

### PostgresBackend
- ✅ Save/retrieve experiments
- ✅ Save experiments with instance predictions
- ✅ Query by model, task, time range, workspace
- ✅ Query with `latest=True` to get most recent result
- ✅ Delete with cascade
- ✅ Pagination

### Repository Layer
- ✅ ExperimentRepository CRUD operations
- ✅ InstancePredictionRepository save/retrieve
- ✅ Pairwise instance queries
- ✅ Best experiments query
- ✅ Pagination

### Query Helpers
- ✅ Dashboard summaries
- ✅ Model results
- ✅ Pairwise metric differences
- ✅ Last-best models

### Session Management
- ✅ Engine creation
- ✅ Session lifecycle
- ✅ Transaction management
- ✅ Connection pooling configuration
- ✅ Password from environment

## CI Integration

Integration tests are designed to work in CI with:
- Docker Compose services (local)
- GitHub Actions services block (CI)

Set `CI_SERVICES_AVAILABLE=1` in CI to skip Docker Compose management.
