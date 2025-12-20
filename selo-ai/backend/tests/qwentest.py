import asyncio
import json
import time
from typing import Dict, Any, List
import httpx
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODELS_TO_TEST = [
    "qwen2.5:1.5b",
    "qwen2.5:3b"
]

# Define our expected schema
class ReflectionOutput(BaseModel):
    content: str
    themes: List[str]
    insights: List[str]
    actions: List[str]
    emotional_state: str
    metadata: Dict[str, Any]
    trait_changes: Dict[str, str]

async def test_model_generation(
    model_name: str, 
    prompt: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Test a single model with the given prompt"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 2048,
        }
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                # Try to parse as JSON
                try:
                    content = json.loads(result["response"])
                    # Validate against our schema
                    ReflectionOutput(**content)
                    return {
                        "success": True,
                        "model": model_name,
                        "response": content,
                        "error": None,
                        "latency": result.get("total_duration", 0) / 1e9,
                        "tokens": len(result.get("response", "").split())
                    }
                except (json.JSONDecodeError, ValidationError) as e:
                    return {
                        "success": False,
                        "model": model_name,
                        "response": result.get("response"),
                        "error": str(e),
                        "latency": result.get("total_duration", 0) / 1e9,
                        "tokens": len(result.get("response", "").split())
                    }
                    
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "model": model_name,
                    "response": None,
                    "error": str(e),
                    "latency": 0,
                    "tokens": 0
                }
            await asyncio.sleep(1)  # Backoff before retry

def format_error(error: str) -> str:
    """Format error message to fit in table"""
    if len(error) > 50:
        return error[:47] + "..."
    return error

async def run_tests():
    console = Console()
    
    test_prompts = [
        """Generate a reflection on artificial intelligence ethics. 
        Respond with a JSON object containing these exact fields:
        - "content": A string with the main reflection text (100-200 words)
        - "themes": List of 3 key themes
        - "insights": List of 2 insights
        - "actions": List of 2 suggested actions
        - "emotional_state": A string describing the emotional tone
        - "metadata": A JSON object with any additional metadata
        - "trait_changes": A JSON object with any trait changes
        
        Example:
        {
            "content": "Reflecting on AI ethics...",
            "themes": ["Accountability", "Bias", "Transparency"],
            "insights": ["AI systems can amplify existing biases", "Ethical frameworks vary by culture"],
            "actions": ["Implement bias detection", "Create diverse training datasets"],
            "emotional_state": "Contemplative",
            "metadata": {"source": "test", "version": "1.0"},
            "trait_changes": {"openness": "+0.2", "conscientiousness": "+0.1"}
        }"""
    ]
    
    results = []
    
    for prompt in test_prompts:
        console.print(f"\n[bold]Testing models with prompt:[/bold]")
        console.print(f"[dim]{prompt[:150]}...[/dim]")
        
        tasks = [test_model_generation(model, prompt) for model in MODELS_TO_TEST]
        model_results = await asyncio.gather(*tasks)
        results.extend(model_results)
        
        # Display results in a table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", width=15)
        table.add_column("Success", width=8)
        table.add_column("Tokens", width=8)
        table.add_column("Latency (s)", width=12)
        table.add_column("Error", width=40)
        
        for result in model_results:
            table.add_row(
                result["model"],
                "[green]✅[/green]" if result["success"] else "[red]❌[/red]",
                str(result["tokens"]),
                f"{result['latency']:.2f}",
                format_error(result["error"] or "None")
            )
        
        console.print(table)
        
        # Save detailed results
        timestamp = int(time.time())
        with open(f"llm_test_results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_tests())