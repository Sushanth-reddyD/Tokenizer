from unigram.tokenizer import pretokenize


# ---------- pretokenize ----------

def test_pretokenize_simple_sentence():
    assert pretokenize("Hello world") == ["Hello", " world"]


def test_pretokenize_lossless():
    text = "Hello, world!  How are you?\n"
    assert "".join(pretokenize(text)) == text


def test_pretokenize_contractions():
    chunks = pretokenize("don't won't I'm")
    assert "don" in chunks
    assert "'t" in chunks
    assert "'m" in chunks


def test_pretokenize_empty():
    assert pretokenize("") == []
