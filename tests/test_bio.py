from ptc.data.bio import B, I, O, content_offsets, decode_bio, encode_bio, make_windows, roundtrip


def ws_offsets(text):
    """Whitespace tokenizer with SentencePiece-style offsets (token includes leading space)."""
    offs, i = [], 0
    for word in text.split(" "):
        start = i - 1 if i > 0 else 0
        offs.append((start, i + len(word)))
        i += len(word) + 1
    return content_offsets(text, offs)


TEXT = "The corrupt elite want you afraid of the future ."


def test_content_offsets_strip_leading_space():
    offs = ws_offsets(TEXT)
    assert TEXT[offs[1][0]:offs[1][1]] == "corrupt"


def test_aligned_spans_roundtrip_exactly():
    offs = ws_offsets(TEXT)
    gold = [(4, 17), (27, 33)]  # "corrupt elite", "afraid"
    assert [TEXT[s:e] for s, e in gold] == ["corrupt elite", "afraid"]
    rt = roundtrip("a", offs, gold)
    assert rt.exact == 2 and not rt.bugs and not rt.misaligned
    assert encode_bio(offs, gold)[:4] == [O, B, I, O]


def test_mid_token_span_is_reported_not_a_bug():
    offs = ws_offsets(TEXT)
    rt = roundtrip("a", offs, [(6, 17)])  # starts inside "corrupt"
    assert rt.misaligned == [(6, 17)] and not rt.bugs
    assert rt.decoded == [(4, 17)]


def test_adjacent_spans_stay_separate():
    offs = ws_offsets(TEXT)
    gold = [(4, 11), (12, 17)]
    assert decode_bio(offs, encode_bio(offs, gold)) == gold


def test_empty_tokens_do_not_split_spans():
    offs = [(0, 3), (4, 4), (4, 9)]  # middle token is a lone whitespace piece
    assert encode_bio(offs, [(0, 9)]) == [B, I, I]
    assert decode_bio(offs, [B, O, I]) == [(0, 9)]


def test_windows_cover_everything_with_overlap():
    wins = make_windows(1000, 256, 64)
    assert wins[0] == (0, 254) and wins[-1][1] == 1000
    for (a, b), (c, d) in zip(wins, wins[1:]):
        assert c < b and b - c == 64


def test_sentencepiece_style_tokenizer_roundtrip():
    """Metaspace/Unigram (DeBERTa-v3 style '▁' pieces) must round-trip token-aligned spans."""
    import random

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    from ptc.data.tokenization import encode_article
    from transformers import PreTrainedTokenizerFast

    rng = random.Random(0)
    words = "the corrupt elite want you afraid of radical clowns never surrender now".split()
    texts = [" ".join(rng.choice(words) for _ in range(40)) + "\n\nAnd  “quoted”  text — é." for _ in range(50)]
    tk = Tokenizer(models.Unigram())
    tk.pre_tokenizer = pre_tokenizers.Metaspace()
    tk.train_from_iterator(texts, trainers.UnigramTrainer(vocab_size=60, special_tokens=["[UNK]"], unk_token="[UNK]"))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]")
    for i, text in enumerate(texts[:10]):
        enc = encode_article(tok, str(i), text)
        # gold = every whole word "corrupt elite" / "radical clowns" occurrence
        gold = []
        for phrase in ("corrupt elite", "radical clowns", "“quoted”"):
            start = text.find(phrase)
            while start != -1:
                gold.append((start, start + len(phrase)))
                start = text.find(phrase, start + 1)
        from ptc.data.spans import merge_spans
        rt = roundtrip(str(i), enc.offsets, merge_spans(gold))
        assert not rt.bugs, rt.bugs
        for s, e in enc.offsets:
            assert text[s:e] == text[s:e].strip()
