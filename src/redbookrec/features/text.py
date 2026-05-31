from __future__ import annotations

import re


def tokenize_text(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[\w]+|[\u4e00-\u9fff]", text)
    return tokens[:512]
