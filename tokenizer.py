from tokenizers import Tokenizer


TOKENIZER_FILE = "tokenizer.json"


def load():

    return Tokenizer.from_file(
        TOKENIZER_FILE
    )


def encode(text):

    tokenizer = load()

    return tokenizer.encode(
        text
    ).ids


def decode(tokens):

    tokenizer = load()

    return tokenizer.decode(
        tokens
    )
