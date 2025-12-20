"""
Manifesto Loader Utility

Loads the persona attribute manifesto from JSON and provides access to initial weighted attributes.
"""
import json
import os
from typing import List, Dict, Any

MANIFESTO_PATH = os.path.join(os.path.dirname(__file__), "manifesto.json")

def load_manifesto() -> Dict[str, Any]:
    with open(MANIFESTO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_initial_attributes() -> List[Dict[str, Any]]:
    manifesto = load_manifesto()
    return manifesto.get("attributes", [])

def get_locked_attributes() -> List[str]:
    manifesto = load_manifesto()
    return manifesto.get("locked", [])
