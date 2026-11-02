"""Build + execute notebooks/NLP_Review_Sentiment.ipynb (reads the outputs written by main.py)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
 md("# Sentiment of product reviews with an LSTM - and the effect of review length\n"
    "This notebook walks through the 7 required steps. The expensive training is done by `python main.py` "
    "(about 40 min on CPU); here we re-run the cheap steps live (data, cleaning, tokenisation, model definition) "
    "and **display the saved results** (`results/`, `figures/`) so every number matches the report."),
 code("import sys, json, os\nos.chdir(r'%s'); sys.path.insert(0, '.')\nimport pandas as pd, numpy as np\n"
      "from IPython.display import Image, display, Markdown\npd.set_option('display.width', 200, 'display.max_colwidth', 120)\n"
      "from src.config import Config\nfrom src import data, preprocess as pp\ncfg = Config(); R, F = cfg.results, cfg.figures" % ROOT),
 md("## Step 1 - Load a labelled review dataset\nAmazon Polarity (Hugging Face): 1-2 star = negative, 4-5 star = positive, no neutral reviews. "
    "We draw a class-balanced sample and make a stratified 70/15/15 split."),
 code("df = data.split_data(data.load_reviews(cfg), cfg)\ndata.summarize(df, cfg)"),
 md("## Step 2 - Clean and tokenize\nLower-casing, HTML/URL removal, contractions expanded (`didn't` -> `did not`), "
    "negations kept, `!`/`?` kept as tokens."),
 code("for t in df.text.iloc[:3]:\n    print('RAW   :', t[:200]); print('TOKENS:', pp.tokenize(t)[:30]); print()\n"
      "print(pp.tokenize(\"I don't like it, it isn't great and I can't recommend it!!!\"))"),
 code("print(json.load(open(R/'preprocessing_info.json')))\ndisplay(Image(F/'length_distribution.png'))"),
 md("## Step 3 - Embeddings\nAn `nn.Embedding` maps each token id to a trainable 100-d vector. We compare random-initialised vs GloVe-initialised."),
 code("import torch\nfrom src.models import RecurrentClassifier, count_params\ninfo = json.load(open(R/'preprocessing_info.json'))\n"
      "m = RecurrentClassifier(info['vocab_size'], cfg.emb_dim, cfg.hidden, 'lstm', cfg.dropout)\nprint(m)\nprint('trainable parameters:', count_params(m))\n"
      "x = torch.randint(2, 1000, (3, info['max_len'])); l = torch.tensor([10, 50, info['max_len']])\nprint('embedding output', m.embedding(x).shape, '| logits', m(x, l).shape)"),
 md("## Step 4 - LSTM training (curves from the saved training run)"),
 code("main = (R/'main_model.txt').read_text(); print('main LSTM (best validation accuracy):', main)\n"
      "display(pd.read_csv(R/f'history_{main}.csv').round(4)); display(Image(F/f'training_curves_{main}.png'))"),
 md("## Step 5 - Evaluation on the held-out test set"),
 code("display(pd.read_csv(R/'metrics_overall.csv').round(4))\nprint((R/f'classification_report_{main}.txt').read_text())\ndisplay(Image(F/'confusion_matrices_all.png'))"),
 md("## Step 6 - Comparison with baselines"),
 code("display(pd.read_csv(R/'metrics_overall.csv').round(4)); display(Image(F/'model_comparison.png'))"),
 md("## Step 7 - Short vs long reviews"),
 code("display(pd.read_csv(R/'metrics_by_length_group.csv').round(4)); display(Image(F/'accuracy_by_length_group.png')); display(Image(F/'accuracy_vs_length.png'))"),
 code("for f in ['stats_accuracy_ci_by_group','stats_short_vs_long_gap','stats_main_vs_baseline_by_group','stats_advantage_change','error_cue_rates']:\n"
      "    print(f); display(pd.read_csv(R/f'{f}.csv').round(4))"),
 code("display(Markdown((R/'error_examples.md').read_text(encoding='utf8')))"),
]
nb = nbf.v4.new_notebook(cells=cells)
out = ROOT / "notebooks"; out.mkdir(exist_ok=True)
ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
nbf.write(nb, out / "NLP_Review_Sentiment.ipynb")
print("notebook written")
