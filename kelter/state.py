"""State rendering. TypeSafe's `state` may be a string or any JSON value."""
from __future__ import annotations

import json
from typing import Any


def render_state(state: Any) -> str:
    """Strings pass through. Other JSON values become indented JSON with keys in
    their given order, so offsets into the rendered text are deterministic."""
    if isinstance(state, str):
        return state
    return json.dumps(state, indent=1, ensure_ascii=False)
