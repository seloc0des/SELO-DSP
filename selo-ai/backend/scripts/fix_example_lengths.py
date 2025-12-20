"""
Fix overly long reflection examples that are causing prompt size issues.

This script truncates example content to ~180 words to prevent memory/context issues.
"""

import asyncio
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.repositories.example import ExampleRepository

def truncate_content(content: str, max_words: int = 180) -> str:
    """Truncate content to approximately max_words while preserving sentence boundaries."""
    words = content.split()
    if len(words) <= max_words:
        return content
    
    # Find a good sentence boundary near the target length
    truncated = ' '.join(words[:max_words])
    
    # Try to end at a sentence
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclaim = truncated.rfind('!')
    
    last_sentence_end = max(last_period, last_question, last_exclaim)
    
    if last_sentence_end > len(truncated) * 0.8:  # If we found a sentence end in the last 20%
        return truncated[:last_sentence_end + 1]
    else:
        # Just truncate at word boundary
        return truncated + '...'

async def fix_example_lengths():
    """Fix overly long examples in the database."""
    
    repo = ExampleRepository()
    
    print("Fetching all examples...")
    examples = await repo.get_all_examples()
    
    print(f"Found {len(examples)} examples")
    
    fixed_count = 0
    for example in examples:
        try:
            full_json = example.get('full_json', {})
            if isinstance(full_json, str):
                full_json = json.loads(full_json)
            
            content = full_json.get('content', '')
            word_count = len(content.split())
            
            if word_count > 200:
                print(f"\n📝 Example '{example['scenario']}': {word_count} words -> truncating...")
                
                # Truncate content
                new_content = truncate_content(content, max_words=180)
                new_word_count = len(new_content.split())
                
                # Update full_json
                full_json['content'] = new_content
                
                # Update in database
                await repo.update_example(
                    example_id=str(example['id']),
                    updates={'full_json': full_json}
                )
                
                print(f"   ✅ Truncated to {new_word_count} words")
                fixed_count += 1
            else:
                print(f"✓ Example '{example['scenario']}': {word_count} words (OK)")
                
        except Exception as e:
            print(f"❌ Failed to process example {example.get('scenario', 'unknown')}: {e}")
    
    print(f"\n🎉 Fixed {fixed_count} examples!")
    print(f"All examples now ≤200 words for optimal prompt size")

if __name__ == "__main__":
    asyncio.run(fix_example_lengths())
