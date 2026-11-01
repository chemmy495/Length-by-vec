#!/usr/bin/env python3
"""fastText (word2vec 系 CBOW) の日本語ベクトルから、ブラウザ用の軽量データを作るスクリプト.

使い方:
  curl -sS https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz \
    | gunzip -c | head -n 100001 > ja100k.vec
  python3 scripts/build_vectors.py ja100k.vec site/data --max-words 50000

出力:
  words.json   単語リスト (頻度順)
  vectors.bin  ヘッダなし。単語ごとに [float32 scale][int8 x dim] を並べたもの
  meta.json    次元数・単語数など
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

# ひらがな・カタカナ・漢字・英字を少なくとも1文字含む語だけ残す
HAS_LETTER = re.compile(r"[A-Za-zぁ-ゖァ-ヺー一-鿿㐀-䶿々〆ヵヶ]")
BAD = re.compile(r"[\s<>\"\\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vec")
    ap.add_argument("outdir")
    ap.add_argument("--max-words", type=int, default=50000)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    words, vecs, seen = [], [], set()
    with open(args.vec, encoding="utf-8", errors="ignore") as f:
        n_total, dim = map(int, f.readline().split())
        for line in f:
            parts = line.rstrip("\n").rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            w = parts[0]
            if w in seen or not HAS_LETTER.search(w) or BAD.search(w):
                continue
            seen.add(w)
            words.append(w)
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
            if len(words) >= args.max_words:
                break

    m = np.vstack(vecs)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12  # 単位ベクトル化
    scale = np.abs(m).max(axis=1) / 127.0
    q = np.round(m / scale[:, None]).astype(np.int8)

    with open(out / "vectors.bin", "wb") as f:
        for s, row in zip(scale, q):
            f.write(struct.pack("<f", float(s)))
            f.write(row.tobytes())

    with open(out / "words.json", "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, separators=(",", ":"))

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "dim": dim,
            "count": len(words),
            "source": "fastText cc.ja.300 (Common Crawl + Wikipedia, CBOW)",
            "quantization": "per-vector int8, unit-normalized",
        }, f, ensure_ascii=False, indent=2)

    deq = q.astype(np.float32) * scale[:, None]
    err = np.abs((deq * m).sum(1) - 1).max()
    print(f"words={len(words)} dim={dim} max self-cos error={err:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
