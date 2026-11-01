# ことばの距離 (Length-by-vec)

Word2Vec 系の日本語単語ベクトルを使って、言葉と言葉の「意味の距離」を測る静的サイトです。
GitHub Pages でそのまま動き、計算はすべてブラウザ内で行います（サーバー不要）。

## できること

- **2語の距離**：コサイン類似度・距離（1−cos）・角度・近さ順位を表示
- **まとめて比較**：基準の言葉に対して複数の言葉を近い順に並べ、2次元の意味マップを描画
- **近い言葉**：ある言葉に近い言葉 TOP30
- **言葉の足し算**：`王 − 男 ＋ 女 ≒ 女王` のようなベクトル演算
- URL に入力内容が残るので、結果をそのまま共有できます

## モデル

[fastText 日本語ベクトル cc.ja.300](https://fasttext.cc/docs/en/crawl-vectors.html)（Common Crawl + Wikipedia、CBOW）の
頻出上位 5 万語を、単位ベクトル化して int8 に量子化（約 15MB）したものを `site/data/` に置いています。
ライセンスは CC BY-SA 3.0 です。

語彙数を変えたい場合：

```sh
curl -sS https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz \
  | gunzip -c | head -n 100001 > ja100k.vec
python3 scripts/build_vectors.py ja100k.vec site/data --max-words 50000
```

## ローカルで動かす

```sh
cd site && python3 -m http.server 8000
# → http://localhost:8000
```

## GitHub Pages で公開する

1. リポジトリの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** にする
2. `main` に push すると `.github/workflows/pages.yml` が `site/` を公開します
3. 公開 URL: `https://<ユーザー名>.github.io/length-by-vec/`
