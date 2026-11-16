"""ProsodyNet (training side, torch) and the contour-MLP baseline."""
from __future__ import annotations

import torch
from torch import nn

from pronunciada.prosody.features import N_FEAT, N_IN


class ProsodyNet(nn.Module):
    """conv(5) -> dilated conv(5, residual) -> BiGRU -> pooling over the target syllable
    (attention + mean) -> MLP.  Input [B, L, N_IN]; output [B, n_out] logits."""

    def __init__(self, n_in: int = N_IN, n_out: int = 5, ch: int = 96, hid: int = 64,
                 n_lang: int = 0, n_extra: int = 0, arch: str = "gru", stride: int = 1):
        super().__init__()
        self.arch = arch
        self.stride = stride
        self.conv1 = nn.Conv1d(n_in, ch, 5, padding=2, stride=stride)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=4, dilation=2)
        if arch == "gru":
            self.gru = nn.GRU(ch, hid, batch_first=True, bidirectional=True)
        else:   # "conv": dilated residual conv stack instead of the GRU (parallel, faster on CPU)
            self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in (4, 8, 16, 1)])
            self.proj = nn.Conv1d(ch, 2 * hid, 1)
        self.att = nn.Linear(2 * hid, 1)
        self.n_lang = n_lang
        self.lang = nn.Embedding(n_lang, 8) if n_lang else None
        self.head = nn.Sequential(nn.Linear(4 * hid + (8 if n_lang else 0) + n_extra, 64), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(64, n_out))

    def forward(self, x, lang=None, extra=None):
        return self.head(self.embed(x, lang, extra))

    def embed(self, x, lang=None, extra=None):
        """Pooled per-syllable representation (input of the head)."""
        tgt = x[..., N_FEAT]                                  # target-syllable mask [B, L]
        if self.stride > 1:                                   # mask at the strided rate
            tgt = torch.nn.functional.max_pool1d(tgt.unsqueeze(1), self.stride, self.stride,
                                                 ceil_mode=True).squeeze(1)
        h = torch.relu(self.conv1(x.transpose(1, 2)))
        h = torch.relu(self.conv2(h)) + h
        if self.arch == "gru":
            h, _ = self.gru(h.transpose(1, 2))                # [B, L, 2H]
        else:
            for blk in self.blocks:
                h = torch.relu(blk(h)) + h
            h = self.proj(h).transpose(1, 2)
        a = self.att(h).squeeze(-1) - (1.0 - tgt) * 1e4
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        att = (w * h).sum(1)
        mean = (tgt.unsqueeze(-1) * h).sum(1) / tgt.sum(1, keepdim=True).clamp(min=1.0)
        z = [att, mean]
        if self.lang is not None:
            z.append(self.lang(lang))
        if extra is not None:
            z.append(extra)
        return torch.cat(z, 1)


class SeqToneNet(nn.Module):
    """Utterance-level tone model: the ProsodyNet syllable encoder, then a BiGRU across the
    syllables of the utterance, then a 5-way head on [local embedding, sequence state].
    forward(x [B, S, L, C], n [B]) -> logits [B, S, 5]; forward_utt(x [S, L, C]) for one utterance
    (the exported form)."""

    def __init__(self, ch=64, hid=48, stride=2, seq_in=96, seq_hid=64, n_out=5, n_in=N_IN):
        super().__init__()
        self.enc = ProsodyNet(n_in=n_in, ch=ch, hid=hid, stride=stride)
        d = 4 * hid
        self.proj = nn.Linear(d, seq_in)
        self.seq = nn.GRU(seq_in, seq_hid, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(d + 2 * seq_hid, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, n_out))

    def forward_flat(self, x_flat, index, B, S):
        """x_flat [N, L, C] = all real syllables; index [N] = b * S + s."""
        e = self.enc.embed(x_flat)                                  # [N, D]
        D = e.shape[1]
        grid = torch.zeros(B * S, D, device=e.device, dtype=e.dtype)
        grid = grid.index_copy(0, index, e).view(B, S, D)
        return grid

    def forward(self, x_flat, index, lengths, B, S):
        grid = self.forward_flat(x_flat, index, B, S)
        packed = nn.utils.rnn.pack_padded_sequence(torch.relu(self.proj(grid)), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        h, _ = self.seq(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=S)
        z = torch.cat([grid, h], -1).view(B * S, -1)[index]
        return self.out(z)                                          # [N, 5]

    def forward_utt(self, x):
        e = self.enc.embed(x)                                       # [S, D]
        h, _ = self.seq(torch.relu(self.proj(e)).unsqueeze(0))
        return self.out(torch.cat([e, h[0]], -1))


class ContourMLP(nn.Module):
    def __init__(self, n_in: int = 42, n_out: int = 5, hid: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, hid),
                                 nn.ReLU(), nn.Linear(hid, n_out))

    def forward(self, x):
        return self.net(x)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
