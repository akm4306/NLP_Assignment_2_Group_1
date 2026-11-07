"""task 2 - reproducing the AWD-LSTM paper (Merity, Keskar & Socher, 2017)

model:          3 layer lstm, embedding 400, hidden 1150, embedding and softmax weights are tied
regularisation: dropconnect 0.5 on the hidden-to-hidden weights, embedding dropout 0.1,
                output dropout 0.4. the rest is the paper's wikitext-2 setting: input dropout 0.65,
                dropout between layers 0.2, AR (alpha 2), TAR (beta 1), weight decay 1.2e-6,
                and random bptt length
optimiser:      NT-ASGD with lr 30, n = 5, checked once per epoch, clip 0.25, batch 80, bptt 70
change:         we train ~50 epochs instead of 500-750 like the paper (not enough time/compute)

how to run:
    python src/task2_awd_lstm.py train  [--epochs 50] [--nhid 1150]
    python src/task2_awd_lstm.py eval            # test ppl + blimp, also saves scores for task 3
"""
import argparse
import math
import os
import sys
import time

import numpy as np

# helps avoid memory fragmentation on a small gpu
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE_DIR, RESULTS_DIR, blimp_accuracy, format_blimp_table,  # noqa: E402
                    get_blimp, get_data, ppl_from_logprobs, save_json, BLIMP_CONTEXT)

CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(os.path.dirname(RESULTS_DIR), "checkpoints"))
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------- model parts ----------------
class LockedDropout(nn.Module):
    """variational dropout: same dropout mask used at every time step"""

    def forward(self, x, p):
        if not self.training or p == 0:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - p) / (1 - p)
        return x * mask


def embedded_dropout(embed, words, p):
    """embedding dropout: drops whole words (rows of the embedding matrix)"""
    if embed.training and p > 0:
        mask = embed.weight.new_empty(embed.weight.size(0), 1).bernoulli_(1 - p) / (1 - p)
        weight = embed.weight * mask
    else:
        weight = embed.weight
    return F.embedding(words, weight)


class WeightDropLSTM(nn.Module):
    """one lstm layer with dropconnect on the hidden-to-hidden matrix W_hh

    a new mask for W_hh is sampled once per batch and used for all time steps,
    same as the paper. we pass the dropped weight straight to the cudnn lstm
    kernel, so it is just as fast as a normal nn.LSTM.
    """

    def __init__(self, ninp, nhid, wdrop):
        super().__init__()
        self.ninp, self.nhid, self.wdrop = ninp, nhid, wdrop
        # same init as pytorch's nn.LSTM
        k = 1.0 / math.sqrt(nhid)
        self.w_ih = nn.Parameter(torch.empty(4 * nhid, ninp).uniform_(-k, k))
        self.w_hh = nn.Parameter(torch.empty(4 * nhid, nhid).uniform_(-k, k))
        self.b_ih = nn.Parameter(torch.empty(4 * nhid).uniform_(-k, k))
        self.b_hh = nn.Parameter(torch.empty(4 * nhid).uniform_(-k, k))

    def forward(self, x, hidden):
        # dropconnect: dropout on the weights, not on the activations
        w_hh = F.dropout(self.w_hh, p=self.wdrop, training=self.training)
        out, h, c = torch._VF.lstm(x, hidden, [self.w_ih, w_hh, self.b_ih, self.b_hh],
                                   True, 1, 0.0, self.training, False, False)
        return out, (h, c)


class AWDLSTM(nn.Module):
    def __init__(self, ntoken, ninp=400, nhid=1150, nlayers=3, dropout=0.4, dropouth=0.2,
                 dropouti=0.65, dropoute=0.1, wdrop=0.5):
        super().__init__()
        self.ntoken, self.ninp, self.nhid, self.nlayers = ntoken, ninp, nhid, nlayers
        self.dropout, self.dropouth, self.dropouti, self.dropoute = dropout, dropouth, dropouti, dropoute
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        # last layer outputs 400 (= embedding size) so we can tie the weights
        self.rnns = nn.ModuleList([
            WeightDropLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, wdrop)
            for l in range(nlayers)])
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight          # weight tying
        self.encoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()

    def init_hidden(self, bsz):
        w = next(self.parameters())
        return [(w.new_zeros(1, bsz, r.nhid), w.new_zeros(1, bsz, r.nhid)) for r in self.rnns]

    def forward(self, inp, hidden):
        # embedding -> dropout -> 3 lstm layers (dropout between them) -> output dropout
        emb = embedded_dropout(self.encoder, inp, self.dropoute if self.training else 0)
        x = self.lockdrop(emb, self.dropouti)
        new_hidden, raw_outputs, outputs = [], [], []
        for l, rnn in enumerate(self.rnns):
            x, h = rnn(x, hidden[l])
            new_hidden.append(h)
            raw_outputs.append(x)
            if l != self.nlayers - 1:
                x = self.lockdrop(x, self.dropouth)
                outputs.append(x)
        out = self.lockdrop(x, self.dropout)               # output dropout
        outputs.append(out)
        # we return hidden states, not logits. the caller applies the decoder
        # (or chunked_ce during training)
        return out, new_hidden, raw_outputs, outputs


class ChunkedSoftmaxCE(torch.autograd.Function):
    """cross entropy of softmax(h W^T + b), done in chunks of rows to save memory

    gives exactly the same result as F.cross_entropy(F.linear(h, W, b), y), but
    the full (T*B x 30001) logit matrix never has to be in memory, which is how
    the 1150 model fits on a 4 GB gpu. we work out the gradients in the forward
    pass chunk by chunk so nothing has to be recomputed in backward.
    """

    @staticmethod
    def forward(ctx, h, W, b, y, chunk):
        n = h.size(0)
        gh, gW, gb = torch.empty_like(h), torch.zeros_like(W), torch.zeros_like(b)
        total = h.new_zeros(())
        for s in range(0, n, chunk):
            hc, yc = h[s:s + chunk], y[s:s + chunk]
            logits = torch.addmm(b, hc, W.t())
            lse = torch.logsumexp(logits, dim=1)
            total += (lse - logits.gather(1, yc[:, None]).squeeze(1)).sum()
            # gradient wrt logits is softmax - one_hot(target)
            g = torch.softmax(logits, dim=1)
            g[torch.arange(len(yc), device=y.device), yc] -= 1
            gh[s:s + chunk] = g @ W
            gW.addmm_(g.t(), hc)
            gb += g.sum(0)
            del logits, g
        ctx.save_for_backward(gh, gW, gb)
        ctx.n = n
        return total / n

    @staticmethod
    def backward(ctx, grad_out):
        # gradients were already computed in forward, just scale them
        gh, gW, gb = ctx.saved_tensors
        s = grad_out / ctx.n
        return gh * s, gW * s, gb * s, None, None


def chunked_ce(h, W, b, y, chunk=2048):
    return ChunkedSoftmaxCE.apply(h, W, b, y, chunk)


def repackage(h):
    # cut the graph so backprop stops at the start of the batch
    return [(a.detach(), b.detach()) for a, b in h]


def batchify(ids, bsz, device):
    n = len(ids) // bsz
    # split the stream into bsz columns, shape (T, B)
    data = torch.from_numpy(ids[: n * bsz]).view(bsz, n).t().contiguous()
    return data.to(device)


# ---------------- evaluation ----------------
@torch.no_grad()
def evaluate_loss(model, data, bptt=70):
    """average loss over the validation stream, used after every epoch"""
    model.eval()
    hidden = model.init_hidden(data.size(1))
    total, count = 0.0, 0
    for i in range(0, data.size(0) - 1, bptt):
        seq = min(bptt, data.size(0) - 1 - i)
        x, y = data[i:i + seq], data[i + 1:i + 1 + seq]
        out, hidden, _, _ = model(x, hidden)
        logits = model.decoder(out)
        total += F.cross_entropy(logits.view(-1, model.ntoken), y.reshape(-1), reduction="sum").item()
        count += y.numel()
    return total / count


@torch.no_grad()
def token_logprobs(model, ids, device, bptt=70):
    """log P(w_t | everything before) for every token except the first, batch size 1

    these are the same positions that task 1 scores, so task 3 can mix them word by word
    """
    model.eval()
    data = torch.from_numpy(ids).to(device).view(-1, 1)
    hidden = model.init_hidden(1)
    chunks = []
    for i in range(0, data.size(0) - 1, bptt):
        seq = min(bptt, data.size(0) - 1 - i)
        x, y = data[i:i + seq], data[i + 1:i + 1 + seq]
        h, hidden, _, _ = model(x, hidden)
        lp = F.log_softmax(model.decoder(h), dim=-1).gather(-1, y.unsqueeze(-1)).view(-1)
        chunks.append(lp.cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


@torch.no_grad()
def sentence_token_logprobs(model, sentences, device, bsz=100):
    """log prob of every word of every blimp sentence, each starting after BLIMP_CONTEXT

    returns one array per sentence (one value per word)
    """
    model.eval()
    res = [None] * len(sentences)
    # sort by length so each batch needs little padding
    order = np.argsort([len(s) for s in sentences])
    for b in range(0, len(order), bsz):
        idx = order[b:b + bsz]
        seqs = [np.concatenate([CTX_IDS, sentences[i]]) for i in idx]
        L = max(len(s) for s in seqs)
        x = torch.zeros(L, len(seqs), dtype=torch.long)
        for j, s in enumerate(seqs):
            x[: len(s), j] = torch.from_numpy(s)
        x = x.to(device)
        out, _, _, _ = model(x[:-1], model.init_hidden(len(seqs)))
        lp = F.log_softmax(model.decoder(out), dim=-1).gather(-1, x[1:].unsqueeze(-1)).squeeze(-1).cpu().numpy()
        # keep only the scores of the sentence words (skip context and padding)
        for j, i in enumerate(idx):
            n_ctx = len(CTX_IDS)
            res[i] = lp[n_ctx - 1: len(seqs[j]) - 1, j].astype(np.float64)
    return res


CTX_IDS = None  # set in evaluate() once we have the vocab


# ---------------- training ----------------
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab, ids = get_data()
    train_data = batchify(ids["train"], args.batch_size, device)
    val_data = batchify(ids["validation"], 10, device)

    model = AWDLSTM(len(vocab), args.emsize, args.nhid, 3, args.dropout, args.dropouth,
                    args.dropouti, args.dropoute, args.wdrop).to(device)
    params = list(model.parameters())
    print(f"params: {sum(p.numel() for p in params)/1e6:.1f}M | vocab {len(vocab)} | device {device}", flush=True)
    # starts as plain sgd, the averaging part of nt-asgd is done by hand below
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)

    state = {"epoch": 0, "val_history": [], "asgd": False, "asgd_epoch": None,
             "avg": None, "n_avg": 0, "best_val": float("inf"), "log": []}
    last_path = os.path.join(CKPT_DIR, "awd_last.pt")
    best_path = os.path.join(CKPT_DIR, "awd_best.pt")
    # resume from the last checkpoint if there is one
    if os.path.exists(last_path) and not args.restart:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        state = ck["state"]
        torch.set_rng_state(ck["rng"].cpu())
        np.random.set_state(ck["np_rng"])
        print(f"resumed from epoch {state['epoch']}", flush=True)

    def swap_in_average():
        """put the averaged asgd weights into the model and return the normal ones"""
        raw = [p.data.clone() for p in params]
        for p, a in zip(params, state["avg"]):
            p.data.copy_(a)
        return raw

    for epoch in range(state["epoch"] + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        hidden = model.init_hidden(args.batch_size)
        total_loss, n_batches, i = 0.0, 0, 0
        while i < train_data.size(0) - 2:
            # random bptt length, like section 4.1 of the paper
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            seq_len = min(seq_len, args.bptt + 20, train_data.size(0) - 1 - i)
            for g in optimizer.param_groups:               # scale lr with the sequence length
                g["lr"] = args.lr * seq_len / args.bptt
            x, y = train_data[i:i + seq_len], train_data[i + 1:i + 1 + seq_len]
            hidden = repackage(hidden)
            optimizer.zero_grad(set_to_none=True)
            out, hidden, raw_outs, outs = model(x, hidden)
            raw_loss = chunked_ce(out.reshape(-1, model.ninp), model.decoder.weight,
                                  model.decoder.bias, y.reshape(-1))
            loss = raw_loss
            loss = loss + args.alpha * outs[-1].pow(2).mean()                              # AR: keep activations small
            loss = loss + args.beta * (raw_outs[-1][1:] - raw_outs[-1][:-1]).pow(2).mean()  # TAR: keep them smooth over time
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()
            if state["asgd"]:                                # asgd: keep a running average of the weights
                state["n_avg"] += 1
                k = 1.0 / state["n_avg"]
                for a, p in zip(state["avg"], params):
                    a.add_(p.data - a, alpha=k)
            total_loss += raw_loss.item()
            n_batches += 1
            i += seq_len
            if n_batches % args.log_interval == 0:
                cur = total_loss / n_batches
                print(f"  ep {epoch} batch {n_batches} | train loss {cur:.3f} ppl {math.exp(cur):.1f}"
                      f" | {(time.time()-t0)/n_batches*1000:.0f} ms/batch", flush=True)
            if args.max_batches and n_batches >= args.max_batches:
                break

        # validation, using the averaged weights once asgd has started
        if state["asgd"]:
            raw = swap_in_average()
        val_loss = evaluate_loss(model, val_data, args.bptt)
        train_ppl = math.exp(total_loss / n_batches)
        entry = {"epoch": epoch, "train_ppl": train_ppl, "val_ppl": math.exp(val_loss),
                 "asgd": state["asgd"], "time_s": time.time() - t0}
        state["log"].append(entry)
        print(f"| end of epoch {epoch} | {entry['time_s']:.0f}s | train ppl {train_ppl:.2f} "
              f"| val ppl {entry['val_ppl']:.2f} | {'ASGD' if state['asgd'] else 'SGD'}", flush=True)
        if val_loss < state["best_val"]:
            state["best_val"] = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)
            print("  saved best model", flush=True)
        if state["asgd"]:
            for p, r in zip(params, raw):
                p.data.copy_(r)

        # nt-asgd: switch to averaging if val loss is worse than the best from more
        # than n epochs ago (algorithm 1 in the paper, checked once per epoch)
        if (not state["asgd"] and len(state["val_history"]) > args.nonmono
                and val_loss > min(state["val_history"][:-args.nonmono])):
            print(f"  >>> non-monotone trigger: switching to ASGD at epoch {epoch}", flush=True)
            state["asgd"], state["asgd_epoch"] = True, epoch
            state["avg"] = [p.data.clone() for p in params]
            state["n_avg"] = 1
        state["val_history"].append(val_loss)
        state["epoch"] = epoch
        # save a checkpoint every epoch so training can be resumed
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state, "rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state()}, last_path)
        save_json({"args": vars(args), "log": state["log"], "asgd_epoch": state["asgd_epoch"]},
                  "task2_training_log.json")


# ---------------- final evaluation ----------------
def evaluate(args):
    global CTX_IDS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab, ids = get_data()
    CTX_IDS = vocab.encode(BLIMP_CONTEXT)
    # load the checkpoint with the best validation perplexity
    ck = torch.load(os.path.join(CKPT_DIR, "awd_best.pt"), map_location=device, weights_only=False)
    a = ck["args"]
    model = AWDLSTM(len(vocab), a["emsize"], a["nhid"], 3, a["dropout"], a["dropouth"],
                    a["dropouti"], a["dropoute"], a["wdrop"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"loaded best checkpoint from epoch {ck['epoch']}", flush=True)

    # perplexity on validation and test, also save per-word scores for task 3
    res = {"best_epoch": ck["epoch"], "nhid": a["nhid"]}
    for split in ("validation", "test"):
        lp = token_logprobs(model, ids[split], device)
        np.save(os.path.join(CACHE_DIR, f"task2_{split}_logprobs.npy"), lp)
        res[f"{split}_ppl"] = ppl_from_logprobs(lp)
        print(f"{split} PPL: {res[f'{split}_ppl']:.2f}", flush=True)

    # blimp: does the model give the good sentence a higher (length normalised) score
    blimp = get_blimp(vocab)
    good = sentence_token_logprobs(model, [e["good"] for e in blimp], device)
    bad = sentence_token_logprobs(model, [e["bad"] for e in blimp], device)
    np.save(os.path.join(CACHE_DIR, "task2_blimp_good.npy"), np.array(good, dtype=object), allow_pickle=True)
    np.save(os.path.join(CACHE_DIR, "task2_blimp_bad.npy"), np.array(bad, dtype=object), allow_pickle=True)
    acc = blimp_accuracy(blimp, [g.mean() for g in good], [b.mean() for b in bad])
    res["blimp"] = acc
    save_json(res, "task2_results.json")

    # write the results table
    md = ["# Task 2 - AWD-LSTM results", "",
          f"Best checkpoint: epoch {ck['epoch']} (hidden size {a['nhid']}).", "",
          "| Split | Perplexity |", "|---|---|",
          f"| Validation | {res['validation_ppl']:.2f} |", f"| **Test** | **{res['test_ppl']:.2f}** |", "",
          "## BLiMP accuracy (%)", "", format_blimp_table({"AWD-LSTM": acc})]
    with open(os.path.join(RESULTS_DIR, "task2_results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--emsize", type=int, default=400)
    ap.add_argument("--nhid", type=int, default=1150)
    ap.add_argument("--lr", type=float, default=30.0)
    ap.add_argument("--clip", type=float, default=0.25)
    ap.add_argument("--batch_size", type=int, default=80)
    ap.add_argument("--bptt", type=int, default=70)
    ap.add_argument("--dropout", type=float, default=0.4)     # output dropout
    ap.add_argument("--dropouth", type=float, default=0.2)    # between lstm layers (wikitext-2 value)
    ap.add_argument("--dropouti", type=float, default=0.65)   # on the embedding output
    ap.add_argument("--dropoute", type=float, default=0.1)    # embedding dropout (whole words)
    ap.add_argument("--wdrop", type=float, default=0.5)       # dropconnect on W_hh
    ap.add_argument("--alpha", type=float, default=2.0)       # AR weight
    ap.add_argument("--beta", type=float, default=1.0)        # TAR weight
    ap.add_argument("--wdecay", type=float, default=1.2e-6)
    ap.add_argument("--nonmono", type=int, default=5)         # n in nt-asgd
    ap.add_argument("--seed", type=int, default=1882)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--max_batches", type=int, default=0, help="only run this many batches per epoch (0 = full epoch), for quick tests")
    ap.add_argument("--restart", action="store_true", help="ignore existing checkpoint")
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
