import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Basic text normalization for LegalIR.

    The purpose is consistency, not text reduction.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Trim only surrounding whitespace
    return text.strip()