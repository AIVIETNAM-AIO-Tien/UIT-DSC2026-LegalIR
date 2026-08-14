from pyvi import ViTokenizer


def tokenize_vietnamese(text: str) -> list[str]:
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

def decode_vietnamese(tokens: list[str]) -> str:
    return " ".join(tokens)