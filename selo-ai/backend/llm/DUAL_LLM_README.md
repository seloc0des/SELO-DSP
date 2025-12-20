# SELO AI Dual LLM Setup

SELO AI supports using two different LLMs for optimal performance and clarity of roles:
- **Conversational LLM:** Handles user-facing, chatty, or empathetic dialogue (e.g. Humanish-Llama3)
- **Analytical LLM:** Handles backend reasoning, code, and knowledge extraction (e.g. CodeLlama, GPT-4)

## Configuration

Set your models via environment variables:

```
SELO_CONVERSATIONAL_LLM=humanish-llama3:latest
SELO_ANALYTICAL_LLM=codellama:latest
```
If not set, the defaults above are used.

## LLM Routing by Subsystem

| Subsystem/Component      | LLM Used         | Notes                                  |
|-------------------------|------------------|----------------------------------------|
| Chat/Conversation       | Conversational   | All user-facing chat, empathy, persona |
| Persona Prompt Gen      | Conversational   | Persona prompt generation              |
| Persona Evolution       | Analytical       | Analytical persona learning            |
| SDL Integration         | Analytical       | All SDL/knowledge extraction           |
| Reflection (API/SIO)    | Analytical       | Reflection generation/listing          |
| Vector Store Embeddings | Analytical       | Semantic search, memory                |

## Dependency Injection (DI) Usage

All LLM controllers are provided via DI in both FastAPI and Socket.IO layers. Avoid direct instantiation—always use the provided DI functions.

### FastAPI Example
```python
from api.dependencies import get_conversational_llm_controller, get_analytical_llm_controller

conversational_llm = await get_conversational_llm_controller()
analytical_llm = await get_analytical_llm_controller()
```

### Socket.IO Namespace Example
```python
from socketio.namespaces.reflection import ReflectionNamespace
from api.dependencies import get_reflection_processor

reflection_processor = await get_reflection_processor()
namespace = ReflectionNamespace(reflection_processor=reflection_processor)
```

## Extending or Customizing LLM Routing
- For new tasks/components, use the DI functions to select the appropriate LLM.
- For advanced use cases (e.g., dynamic routing), implement a router or manager that inspects the task and chooses the LLM at runtime.
- Add logging to monitor which LLM is used for which task.

## Backward Compatibility
- `get_llm_controller()` returns the analytical LLM by default for legacy code. Update older code to use the explicit DI functions when possible.

## Future Enhancements
- **Dynamic LLM Routing:** Route requests based on intent or content type.
- **LLM Usage Logging:** Add analytics/logging to monitor LLM usage and performance.
- **Central LLM Router:** Implement a centralized service for LLM selection and fallback.

## Quick Reference
- Use `get_conversational_llm_controller` for chat, persona prompts, and user-facing tasks.
- Use `get_analytical_llm_controller` for SDL, reflection, persona evolution, and backend analytics.
- Always inject via DI, never instantiate LLMController directly.
