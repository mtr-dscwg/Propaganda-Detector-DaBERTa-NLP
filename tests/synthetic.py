"""Synthetic corpus in the exact PTC release layout + a tiny local backbone.

Lets the whole pipeline run end-to-end offline in a couple of minutes on CPU.
"""
from __future__ import annotations

import random
from pathlib import Path

PHRASES = {
    "Loaded_Language": ["disgusting traitors", "vile scheme", "shameful betrayal"],
    "Name_Calling,Labeling": ["radical clowns", "globalist puppets"],
    "Slogans": ["make the nation strong again", "never surrender"],
    "Doubt": ["can we really trust them", "who believes these numbers"],
    "Flag-Waving": ["for the glory of our homeland"],
    "Appeal_to_fear-prejudice": ["they will destroy your family"],
}
NEUTRAL = ("the council met on tuesday to review the budget and several members asked about "
           "roads schools water funding reports and the timeline for next year while residents "
           "listened and officials answered questions about planning permits").split()


def _sentence(rng: random.Random) -> str:
    return " ".join(rng.choice(NEUTRAL) for _ in range(rng.randint(6, 14))).capitalize() + "."


def make_article(rng: random.Random) -> tuple[str, list[tuple[str, int, int]]]:
    text, spans = "", []
    for p in range(rng.randint(3, 6)):
        para = []
        for _ in range(rng.randint(2, 4)):
            sent = _sentence(rng)
            if rng.random() < 0.5:
                tech = rng.choice(list(PHRASES))
                phrase = rng.choice(PHRASES[tech])
                offset = len(text) + sum(len(s) + 1 for s in para) + len(sent) + 1
                sent = f"{sent} {phrase.capitalize()}!"
                spans.append((tech, offset, offset + len(phrase)))
            para.append(sent)
        text += " ".join(para) + "\n\n"
    return text, spans


def write_corpus(root: Path, seed: int = 0, sizes=(("train", 40), ("dev", 12), ("test", 5))) -> None:
    rng = random.Random(seed)
    aid = 700000000
    for split, n in sizes:
        (root / f"{split}-articles").mkdir(parents=True, exist_ok=True)
        if split != "test":
            (root / f"{split}-labels-task2-technique-classification").mkdir(parents=True, exist_ok=True)
        for _ in range(n):
            aid += rng.randint(1, 9999)
            text, spans = make_article(rng)
            (root / f"{split}-articles" / f"article{aid}.txt").write_bytes(text.encode("utf-8"))
            if split != "test":
                lines = "".join(f"{aid}\t{t}\t{s}\t{e}\n" for t, s, e in spans)
                (root / f"{split}-labels-task2-technique-classification" /
                 f"article{aid}.task2-TC.labels").write_text(lines, encoding="utf-8")


def make_tiny_backbone(out: Path, corpus_root: Path) -> None:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    texts = [p.read_text(encoding="utf-8") for p in corpus_root.rglob("*.txt")]
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tk.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=400, special_tokens=specials))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]")
    tok.save_pretrained(out)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=512)
    BertModel(cfg).save_pretrained(out)
