import torch
import itertools
from torch import nn
from torch.nn import functional as F

# import project‑level helpers
from model.util import Linear, make_subsequent_mask


class Generator(nn.Module):
    """Projection + log‑softmax helper used by the decoder."""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.proj = Linear(d_model, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log‑probabilities over the vocabulary."""
        return F.log_softmax(self.proj(x), dim=-1)

    # during search we sometimes need the raw scores
    def greedyscore(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Beam:
    """A minimal beam data‑structure holding partial hypotheses."""

    def __init__(self, beam_size: int):
        self.beam_size = beam_size
        self.candidates: list[list[int]] = []
        self.scores: list[float] = []

    def step(self, log_prob: torch.Tensor, prev_beam: "Beam", is_done_fn):
        """Advance the beam one time‑step.

        Args
        ----
        log_prob: (B, V) tensor with *negative* log‑probabilities for the next token.
        prev_beam: beam at the previous step.
        is_done_fn: lambda that returns True when a hypothesis is finished.
        """
        # Expand previous scores to match current vocab dim
        prev_score = log_prob.new_tensor(prev_beam.scores)  # (B,)
        total_score = log_prob + prev_score.unsqueeze(-1)    # broadcast to (B, V)

        # Choose the best beam_size hypotheses over the flattened (B*V) space
        best_score, best_ix = total_score.view(-1).topk(self.beam_size, largest=False)
        beam_ix = torch.div(best_ix, log_prob.size(1), rounding_mode="floor")
        token_ix = best_ix - beam_ix * log_prob.size(1)

        done, remain = [], []
        for s, b, t in zip(best_score.tolist(), beam_ix.tolist(), token_ix.tolist()):
            candidate = prev_beam.candidates[b] + [t]
            if is_done_fn(candidate):
                done.append((candidate, s))
            else:
                remain.append(b)
                self.candidates.append(candidate)
                self.scores.append(s)
        return done, remain


# -----------------------------------------------------------------------------
# Search helpers
# -----------------------------------------------------------------------------

def greedy(
    args,
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    initid: int,
    eosid: int,
    *objs,
):
    """Greedy decoding that also works without multimodal inputs."""

    # Encode source (optionally multimodal)
    encodings = model.encode(src, src_mask, *objs) if objs else model.encode(src, src_mask)

    # In case encode() returns (hx, ho)
    if isinstance(encodings, tuple):
        encodings = encodings[0]

    B, T, H = encodings.size()
    max_len = int(T * args.length_ratio)

    outs = src.new_full((B, 1), initid)
    finished = torch.zeros(B, dtype=torch.bool, device=src.device)

    for _ in range(max_len):
        subseq_mask = make_subsequent_mask(outs.size(1)).to(src.device)
        dec_out = model.decode(encodings, src_mask, outs, subseq_mask)
        next_token = dec_out[:, -1].log_softmax(-1).argmax(-1)
        outs = torch.cat([outs, next_token.unsqueeze(1)], dim=1)
        finished |= next_token.eq(eosid)
        if finished.all():
            break
    # strip the initial <init>
    return outs[:, 1:]


def beam_search(
    args,
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    initid: int,
    eosid: int,
    *objs,
):
    """Beam search that tolerates the absence of object features."""

    # ------------- Encode ----------------------------------------------------
    enc_ret = model.encode(src, src_mask, *objs) if objs else model.encode(src, src_mask)
    if isinstance(enc_ret, tuple):
        encoding, encoding_obj = enc_ret
    else:
        encoding, encoding_obj = enc_ret, None

    obj_mask = objs[-2] if len(objs) >= 2 else None

    W = args.beam_size
    alpha = args.alpha
    _, T, H = encoding.size()
    max_len = int(T * args.length_ratio)
    min_len = T // 2

    prev_beam = Beam(W)
    prev_beam.candidates = [[initid]]
    prev_beam.scores = [0.0]

    done_hyp, valid_size = [], W
    is_done = lambda x: x[-1] == eosid

    # Pre‑compute positional encodings once (helps a bit)
    allpos = model.addposition(encoding.new_zeros(1, max_len, H))
    hiddens = encoding.new_zeros(1, max_len + 1, args.n_layers + 1, H)

    for t in range(max_len):
        inp_tokens = src.new_tensor([cand[-1] for cand in prev_beam.candidates])
        hiddens[:, t, 0] = model.tgt_embed[0](inp_tokens) + allpos[:, t]

        for l in range(args.n_layers):
            layer = model.decoder.layers[l]
            hiddens[:, t, l + 1] = layer.search(
                hiddens[:, t : t + 1, l], hiddens[:, : t + 1, l], encoding, src_mask
            ).view(-1, H)

        log_prob = model.generate(hiddens[:, t, -1])  # (B, V)
        if t < min_len:
            log_prob[:, eosid] = -float("inf")
        if t == max_len - 1:
            eos_p = log_prob[:, eosid].clone()
            log_prob.fill_(-float("inf"))
            log_prob[:, eosid] = eos_p

        next_beam = Beam(valid_size)
        step_done, remain_idx = next_beam.step(-log_prob, prev_beam, is_done)
        done_hyp.extend(step_done)
        valid_size -= len(step_done)
        if valid_size == 0:
            break

        remain_idx = src.new_tensor(remain_idx)
        encoding = encoding.index_select(0, remain_idx)
        src_mask = src_mask.index_select(0, remain_idx)
        if encoding_obj is not None:
            encoding_obj = encoding_obj.index_select(0, remain_idx)
        if obj_mask is not None:
            obj_mask = obj_mask.index_select(0, remain_idx)
        hiddens = hiddens.index_select(0, remain_idx)
        prev_beam = next_beam

    # Length‑penalised sorting ------------------------------------------------
    hyps, scores = zip(*done_hyp)
    lp = torch.tensor([(5 + len(h)) / 6 for h in hyps], device=encoding.device) ** alpha
    ranked = torch.argsort(torch.tensor(scores, device=encoding.device) / lp)

    best_hyp = hyps[ranked[0]]
    if eosid in best_hyp:
        best_hyp = best_hyp[1 : best_hyp.index(eosid)]
    else:
        best_hyp = best_hyp[1:]

    return src.new_tensor([best_hyp])
