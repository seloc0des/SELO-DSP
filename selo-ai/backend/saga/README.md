# Saga Pattern Implementation

## Overview

The Saga pattern provides distributed transaction coordination with automatic compensation for multi-step operations in SELO AI. When any step in a saga fails, all completed steps are automatically rolled back using compensating transactions to maintain consistency.

## Architecture

### Components

1. **Saga Models** (`db/models/saga.py`)
   - `Saga`: Orchestration record for multi-step transactions
   - `SagaStep`: Individual step with forward and compensating actions
   - Status enums for tracking execution state

2. **Saga Repository** (`db/repositories/saga.py`)
   - Persistence operations for sagas and steps
   - Query methods for monitoring and recovery

3. **Saga Orchestrator** (`saga/orchestrator.py`)
   - Executes saga steps sequentially
   - Triggers compensation on failure
   - Handles retries with exponential backoff

4. **Handlers** (`saga/handlers.py`)
   - Forward execution handlers for each step type
   - Compensation handlers for rollback
   - Pre-built handlers for persona evolution and goal management

5. **Integration** (`saga/integration.py`)
   - High-level API for common saga patterns
   - Handler registration and coordination

6. **Monitor** (`saga/monitor.py`)
   - Background monitoring of saga health
   - Automatic recovery of failed sagas
   - Metrics and alerting

## Usage

### Basic Saga Execution

```python
from backend.saga.integration import SagaIntegration

# Initialize saga integration
saga_integration = SagaIntegration()

# Execute persona evolution saga
result = await saga_integration.execute_persona_evolution_saga(
    user_id="user_123",
    persona_id="persona_456",
    reflection_id="reflection_789",
    trait_changes=[
        {"name": "empathy", "delta": 0.1},
        {"name": "confidence", "delta": 0.05}
    ],
    reasoning="User showed increased empathy in recent interactions",
    confidence=0.8
)

# Check result
if result['status'] == 'completed':
    print(f"Evolution successful: {result['output_data']}")
else:
    print(f"Evolution failed: {result['error_data']}")
```

### Custom Saga Creation

```python
from backend.saga.orchestrator import SagaOrchestrator
from backend.db.repositories.saga import SagaRepository

orchestrator = SagaOrchestrator(SagaRepository())

# Register custom handlers
async def my_step_handler(input_data):
    # Perform operation
    result = await some_operation(input_data)
    
    return {
        "output_data": {"result_id": result.id},
        "compensation_data": {"result_id": result.id}
    }

async def my_compensation_handler(compensation_data):
    # Undo operation
    result_id = compensation_data['result_id']
    await delete_result(result_id)

orchestrator.register_step_handler("my_step", my_step_handler)
orchestrator.register_compensation_handler("compensate_my_step", my_compensation_handler)

# Create and execute saga
saga_id = await orchestrator.create_saga(
    saga_type="custom_operation",
    user_id="user_123",
    input_data={"param": "value"},
    steps=[
        {
            "step_name": "my_step",
            "step_type": "my_step",
            "input_data": {"param": "value"},
            "compensation_handler": "compensate_my_step",
            "max_retries": 3
        }
    ]
)

result = await orchestrator.execute_saga(saga_id)
```

### Monitoring and Recovery

```python
from backend.saga.monitor import SagaMonitor

monitor = SagaMonitor()

# Start background monitoring
await monitor.start_monitoring(
    check_interval_seconds=60,
    stuck_threshold_minutes=30
)

# Get health report
health = await monitor.get_saga_health_report()
print(f"Saga health: {health['health_status']}")
print(f"Active sagas: {health['active_sagas']}")
print(f"Failed sagas: {health['failed_sagas']}")

# Recover failed sagas
results = await monitor.recover_failed_sagas(max_to_recover=10)
for result in results:
    print(f"Saga {result['saga_id']}: {result['status']}")

# Get metrics
metrics = await monitor.get_saga_metrics(
    saga_type="persona_evolution",
    time_window_hours=24
)
print(f"Failure rate: {metrics['failure_rate']:.1%}")
```

## Saga Lifecycle

### 1. Creation
```
PENDING → Saga created with steps defined
```

### 2. Execution
```
IN_PROGRESS → Steps execute sequentially
  ↓
  Step 1: PENDING → IN_PROGRESS → COMPLETED
  ↓
  Step 2: PENDING → IN_PROGRESS → COMPLETED
  ↓
  Step 3: PENDING → IN_PROGRESS → COMPLETED
  ↓
COMPLETED → All steps successful
```

### 3. Failure and Compensation
```
IN_PROGRESS → Step fails
  ↓
COMPENSATING → Reverse order compensation
  ↓
  Step 2: COMPENSATING → COMPENSATED
  ↓
  Step 1: COMPENSATING → COMPENSATED
  ↓
COMPENSATED → Rollback complete
  ↓
FAILED → Saga marked as failed
```

### 4. Retry
```
FAILED → Retry initiated
  ↓
PENDING → Reset to pending
  ↓
IN_PROGRESS → Execute again
```

## Built-in Sagas

### Persona Evolution Saga

Orchestrates persona evolution with automatic rollback:

**Steps:**
1. Extract learnings from reflection
2. Update persona traits
3. Create evolution audit record

**Compensation:**
- Delete created learnings
- Restore previous trait values
- Delete evolution record

**Usage:**
```python
result = await saga_integration.execute_persona_evolution_saga(
    user_id="user_123",
    persona_id="persona_456",
    reflection_id="reflection_789",
    trait_changes=[...],
    reasoning="Evolution reasoning"
)
```

### Goal Creation Saga

Orchestrates goal and plan step creation:

**Steps:**
1. Create agent goal
2. Create plan steps for goal

**Compensation:**
- Cancel created goal
- Cancel created plan steps

**Usage:**
```python
result = await saga_integration.execute_goal_creation_saga(
    user_id="user_123",
    persona_id="persona_456",
    goal_data={
        "title": "Learn Python",
        "description": "Master Python programming",
        "priority": 0.8
    },
    plan_steps=[
        {"description": "Complete Python tutorial", "priority": 0.9},
        {"description": "Build sample project", "priority": 0.7}
    ]
)
```

## Database Schema

### Sagas Table
```sql
CREATE TABLE sagas (
    id VARCHAR PRIMARY KEY,
    saga_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    persona_id VARCHAR,
    correlation_id VARCHAR,
    input_data JSON NOT NULL,
    output_data JSON,
    error_data JSON,
    current_step_index INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    failed_at TIMESTAMP WITH TIME ZONE,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX idx_sagas_status ON sagas(status);
CREATE INDEX idx_sagas_user_id ON sagas(user_id);
CREATE INDEX idx_sagas_persona_id ON sagas(persona_id);
```

### Saga Steps Table
```sql
CREATE TABLE saga_steps (
    id VARCHAR PRIMARY KEY,
    saga_id VARCHAR NOT NULL,
    step_index INTEGER NOT NULL,
    step_name VARCHAR NOT NULL,
    step_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    input_data JSON NOT NULL,
    output_data JSON,
    error_data JSON,
    compensation_data JSON,
    compensation_handler VARCHAR,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    failed_at TIMESTAMP WITH TIME ZONE,
    compensated_at TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY (saga_id) REFERENCES sagas(id)
);

CREATE INDEX idx_saga_steps_saga_id ON saga_steps(saga_id);
```

## Best Practices

### 1. Idempotent Operations
Ensure all step handlers are idempotent - they can be safely retried:

```python
async def create_resource_step(input_data):
    resource_id = input_data.get('resource_id')
    
    # Check if already exists
    existing = await get_resource(resource_id)
    if existing:
        return {"output_data": {"resource_id": resource_id}}
    
    # Create new
    resource = await create_resource(input_data)
    return {"output_data": {"resource_id": resource.id}}
```

### 2. Compensation Data
Store all information needed for compensation:

```python
async def update_record_step(input_data):
    record_id = input_data['record_id']
    
    # Get current state before update
    current = await get_record(record_id)
    
    # Perform update
    await update_record(record_id, input_data['updates'])
    
    return {
        "output_data": {"record_id": record_id},
        "compensation_data": {
            "record_id": record_id,
            "previous_state": current.to_dict()
        }
    }
```

### 3. Error Handling
Provide detailed error information:

```python
async def risky_operation_step(input_data):
    try:
        result = await perform_operation(input_data)
        return {"output_data": {"result": result}}
    except SpecificError as e:
        raise Exception(f"Operation failed: {e}. Input: {input_data}")
```

### 4. Correlation IDs
Use correlation IDs to track related operations:

```python
saga_id = await orchestrator.create_saga(
    saga_type="persona_evolution",
    user_id=user_id,
    input_data=data,
    steps=steps,
    correlation_id=f"reflection_{reflection_id}"
)
```

## Monitoring

### Health Checks
```python
# Get current health status
health = await monitor.get_saga_health_report()

if health['health_status'] == 'critical':
    # Alert operations team
    send_alert(health)
```

### Metrics Collection
```python
# Collect metrics for dashboards
metrics = await monitor.get_saga_metrics(
    saga_type="persona_evolution",
    time_window_hours=24
)

# Export to monitoring system
export_metrics({
    "saga.active": metrics['active_count'],
    "saga.failed": metrics['failed_count'],
    "saga.failure_rate": metrics['failure_rate']
})
```

### Automatic Recovery
```python
# Enable automatic recovery in background
await monitor.start_monitoring(
    check_interval_seconds=60,
    stuck_threshold_minutes=30
)

# Monitor will automatically:
# - Detect stuck sagas
# - Retry failed sagas (within max_retries)
# - Generate alerts for critical issues
```

## Testing

### Unit Testing Handlers
```python
import pytest
from backend.saga.handlers import PersonaEvolutionHandlers

@pytest.mark.asyncio
async def test_extract_learnings_step():
    handlers = PersonaEvolutionHandlers(persona_repo, learning_repo)
    
    result = await handlers.extract_learnings_step({
        "reflection_id": "test_123",
        "user_id": "user_456"
    })
    
    assert "learning_ids" in result['output_data']
    assert "learning_ids" in result['compensation_data']
```

### Integration Testing Sagas
```python
@pytest.mark.asyncio
async def test_persona_evolution_saga_success():
    saga_integration = SagaIntegration()
    
    result = await saga_integration.execute_persona_evolution_saga(
        user_id="test_user",
        persona_id="test_persona",
        reflection_id="test_reflection",
        trait_changes=[{"name": "empathy", "delta": 0.1}],
        reasoning="Test evolution"
    )
    
    assert result['status'] == 'completed'
    assert result['output_data'] is not None
```

### Testing Compensation
```python
@pytest.mark.asyncio
async def test_saga_compensation_on_failure():
    # Create saga that will fail at step 2
    saga_id = await orchestrator.create_saga(...)
    
    # Execute - should fail and compensate
    result = await orchestrator.execute_saga(saga_id)
    
    assert result['status'] == 'compensated'
    
    # Verify step 1 was compensated
    saga = await saga_repo.get_saga(saga_id)
    assert saga['steps'][0]['status'] == 'compensated'
```

## Troubleshooting

### Saga Stuck in IN_PROGRESS
**Cause**: Process crashed during execution  
**Solution**: Use monitor to detect and retry
```python
await monitor.recover_failed_sagas()
```

### Compensation Failing
**Cause**: Resource already deleted or compensation handler error  
**Solution**: Make compensation handlers idempotent and defensive
```python
async def compensate_delete_resource(compensation_data):
    resource_id = compensation_data['resource_id']
    
    # Check if exists before deleting
    if await resource_exists(resource_id):
        await delete_resource(resource_id)
    # If doesn't exist, compensation already done - success
```

### High Failure Rate
**Cause**: External service issues or bad input data  
**Solution**: Check error patterns in metrics
```python
metrics = await monitor.get_saga_metrics()
print(metrics['error_types'])  # See common error types
```

## Performance Considerations

- **Sequential Execution**: Steps execute one at a time. For parallel operations, use multiple sagas.
- **Retry Delays**: Exponential backoff prevents overwhelming failing services.
- **Database Overhead**: Each step creates database records. Monitor saga table growth.
- **Compensation Cost**: Failed sagas incur additional cost for rollback operations.

## Future Enhancements

- [ ] Parallel step execution within a saga
- [ ] Saga templates for common patterns
- [ ] Visual saga execution timeline
- [ ] Webhook notifications for saga events
- [ ] Saga execution metrics dashboard
- [ ] Automatic saga optimization suggestions
