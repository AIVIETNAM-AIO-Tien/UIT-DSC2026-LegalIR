from pathlib import Path
from typing import Callable, NewType


RawText = NewType('RawText', str)
Token = NewType('Token', str)
type TokenizerFunc = Callable[[RawText], list[Token]]
type DecoderFunc = Callable[[list[Token]], RawText]
type CorpusTokens = list[list[Token]]

type DirPath = str | Path
