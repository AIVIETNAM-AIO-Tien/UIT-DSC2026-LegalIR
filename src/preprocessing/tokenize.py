from pyvi import ViTokenizer

from src.types import RawText, Token


def tokenize_vietnamese(text: RawText) -> list[Token]:
    """
    Tokenize Vietnamese text using PyVi.

    Parameters
    ----------
    text : str
        Input text. The text is expected to have been
        normalized before calling this function.

    Returns
    -------
    list[str]
        List of Vietnamese tokens.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokenized_text = ViTokenizer.tokenize(text)

    return tokenized_text.split()

def decode_vietnamese(tokens: list[Token]) -> RawText:
    return RawText(" ".join(tokens))
