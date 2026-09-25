import re
import unicodedata


def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_nfc(text)).strip()
