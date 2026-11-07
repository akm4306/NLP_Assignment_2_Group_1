"""builds the colab notebook from the files in src/, so the notebook always has the same code.
the notebook writes these files itself, so nothing has to be uploaded on colab."""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["src/common.py", "src/task1_ngram.py", "src/task2_awd_lstm.py", "src/task3_hybrid.py",
         "src/make_plots.py", "run_all.sh"]

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Project 1: N-gram Smoothing, AWD-LSTM Reproduction and an Interpolated Hybrid Language Model\n"
       "**NLP CS60075 Assignment II · Dataset: WikiText-2 · Bonus benchmark: BLiMP**\n\n"
       "This notebook reproduces all three tasks end to end:\n\n"
       "1. **Task 1:** a 4-gram language model with Add-k, Good-Turing and interpolated Kneser-Ney smoothing.\n"
       "2. **Task 2:** a reproduction of the AWD-LSTM (Merity et al., 2017).\n"
       "3. **Task 3:** a linear interpolation of the Kneser-Ney and AWD-LSTM models.\n\n"
       "Every model is evaluated by perplexity on the WikiText-2 test split and by accuracy on 10 BLiMP paradigms.\n\n"
       "### How to run\n"
       "Select a GPU runtime (*Runtime → Change runtime type → T4 GPU*), then choose *Runtime → Run all*. "
       "The notebook is self-contained: it writes the project source files, installs the dependencies and "
       "downloads WikiText-2 and BLiMP from the Hugging Face Hub."),
    md("## 0. Configuration\n"
       "* `QUICK = True` runs a short test of the whole pipeline. Its numbers are not meaningful.\n"
       "* `QUICK = False` runs the full experiment, training the AWD-LSTM for 50 epochs.\n"
       "* `USE_DRIVE = True` saves checkpoints to Google Drive, so an interrupted run resumes from the last "
       "completed epoch."),
    code("QUICK = False\n"
         "EPOCHS = 50\n"
         "USE_DRIVE = False"),
    md("## 1. Environment setup"),
    code("!nvidia-smi -L\n"
         "!pip -q install datasets matplotlib"),
    code("import os\n"
         "os.makedirs('/content/project/src', exist_ok=True)\n"
         "os.chdir('/content/project')\n"
         "if USE_DRIVE:\n"
         "    from google.colab import drive\n"
         "    drive.mount('/content/drive')\n"
         "    os.environ['CKPT_DIR'] = '/content/drive/MyDrive/nlp_p1_checkpoints'\n"
         "os.environ['EPOCHS'] = str(EPOCHS)\n"
         "RES = 'results_quick' if QUICK else 'results'\n"
         "if QUICK:  # keep quick-test outputs separate from the real results\n"
         "    os.environ.setdefault('CKPT_DIR', 'checkpoints_quick')\n"
         "    os.environ.update(RESULTS_DIR=RES, CACHE_DIR='cache_quick')"),
    md("## 2. Source code\n"
       "The project modules: shared data pipeline and metrics, one module per task, and the figure script."),
]
for f in FILES:
    with open(os.path.join(ROOT, f)) as fh:
        cells.append(code(f"%%writefile {f}\n" + fh.read()))
cells += [
    md("## 3. Task 1: Baseline, a smoothed n-gram language model\n"
       "This step builds 1- to 4-gram count tables from the training split and evaluates five smoothing variants: "
       "Add-k (k = 1, 0.5, 0.1), Good-Turing (Katz back-off) and interpolated Kneser-Ney (d = 0.75). Kneser-Ney's "
       "λ is tuned on the validation split. It reports test perplexity for every variant, and BLiMP accuracy for "
       "the best variant by validation perplexity."),
    code("!python -W ignore src/task1_ngram.py 2>&1 | grep -v 'Generating\\|HF Hub' | tail -n 60"),
    md("## 4. Task 2: Paper reproduction, the AWD-LSTM\n"
       "The model is a 3-layer LSTM with embedding size 400, hidden size 1150 and tied weights. It uses "
       "DropConnect (0.5), embedding dropout (0.1) and output dropout (0.4), and trains with NT-ASGD "
       "(learning rate 30, batch size 80, BPTT length 70, gradient clipping 0.25) for 50 epochs.\n\n"
       "### 4.1 Training\n"
       "*Note: the outputs saved in this notebook come from a run that loaded the already trained checkpoint "
       "(50 epochs, 4.15 h on a GTX 1650), so this cell only shows `resumed from epoch 50`. The full per-epoch "
       "training log is in `logs/task2_train.log` and `results/task2_training_log.json`. Running the notebook "
       "from scratch trains the model again.*"),
    code("# quick mode trains 1 short epoch, full mode trains EPOCHS full epochs\n"
         "n_epochs = 1 if QUICK else EPOCHS\n"
         "extra = '--max_batches 30' if QUICK else ''\n"
         "!python -W ignore src/task2_awd_lstm.py train --epochs $n_epochs $extra 2>&1 | grep --line-buffered -v 'ms/batch'"),
    md("### 4.2 Evaluation\nThis step reports test perplexity and BLiMP accuracy for the checkpoint with the "
       "best validation perplexity."),
    code("!python -W ignore src/task2_awd_lstm.py eval 2>&1 | grep -v 'Generating\\|HF Hub'"),
    md("## 5. Task 3: Combine and improve, an interpolated hybrid language model\n"
       "P_hybrid = λ · P_KN + (1 − λ) · P_AWD, with λ grid-searched over {0.0, 0.1, …, 1.0} on the validation split. "
       "This step builds the final Task 1 vs Task 2 vs Task 3 comparison."),
    code("!python -W ignore src/task3_hybrid.py 2>&1 | grep -v 'Generating\\|HF Hub'\n"
         "!python src/make_plots.py"),
    md("## 6. Results"),
    code("from IPython.display import Markdown, Image, display\n"
         "for t in ('task1', 'task2', 'task3'):\n"
         "    display(Markdown(open(f'{RES}/{t}_results.md').read()))\n"
         "for f in sorted(os.listdir(f'{RES}/figures')):\n"
         "    display(Image(f'{RES}/figures/{f}'))"),
]
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = os.path.join(ROOT, "26AI60R06_NLP_Assignment_2.ipynb")
nbf.write(nb, out)
print("wrote", out)
