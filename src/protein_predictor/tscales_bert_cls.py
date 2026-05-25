# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — T-scales + BERT/CLS
#
# Pipeline:
#   Protein sequence
#       → T-scales matrix  (seq_len × 5)
#       → Linear projection  (seq_len × d_model)
#       → BERT encoder  (Transformer)
#       → CLS token  (d_model,)
#       → concat with ChemBERTa SMILES + experimental features
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

# ── Load T-scales table relative to this file (always works regardless of cwd)
_TABLE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "T_scales_table.csv"

tsc_df = pd.read_csv(_TABLE_PATH)
tsc_df.columns = tsc_df.columns.str.strip()
tsc_df["symbol"] = tsc_df["symbol"].str.strip()

T_SCALE = {}
for _, row in tsc_df.iterrows():
    aa = row["symbol"]
    T_SCALE[aa] = [float(row[f"T_{i}"]) for i in range(1, 6)]


def sequence_to_tscales(seq):
    """Returns (seq_len, 5) matrix — one row per residue."""
    vecs = [T_SCALE.get(aa, [0.0, 0.0, 0.0, 0.0, 0.0]) for aa in seq]
    return np.array(vecs, dtype=np.float32)


# ── BERT encoder ──────────────────────────────────────────────────────────────

class TScalesBERTEncoder(nn.Module):
    """
    Lightweight BERT-style encoder whose input is a T-scales matrix.

    Instead of a token-embedding lookup table, a linear layer projects the
    5 T-scale values per residue into d_model dimensions.
    A learnable [CLS] token is prepended; its final hidden state is returned
    as the sequence-level vector.
    """

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(5, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embedding = nn.Embedding(max_len + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        positions = torch.arange(L + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x[:, 0, :]


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_tscales(sequences, max_len: int = 1024, device: str = "cpu"):
    matrices = [sequence_to_tscales(seq[:max_len]) for seq in sequences]
    lengths  = [m.shape[0] for m in matrices]
    pad_len  = max(lengths)
    x    = np.zeros((len(sequences), pad_len, 5), dtype=np.float32)
    mask = np.ones( (len(sequences), pad_len),    dtype=bool)
    for i, (mat, ln) in enumerate(zip(matrices, lengths)):
        x[i, :ln, :] = mat
        mask[i, :ln]  = False
    return (torch.tensor(x, device=device),
            torch.tensor(mask, device=device))


# ── Encode series ─────────────────────────────────────────────────────────────

def encode_tscales_cls(sequences,
                       encoder: TScalesBERTEncoder,
                       batch_size: int = 16,
                       device: str = "cpu") -> np.ndarray:
    """Returns np.ndarray of shape (N, d_model) — one CLS vector per sequence."""
    encoder.eval()
    encoder.to(device)
    all_embs = []
    seqs = list(sequences)
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size]
            x, mask = collate_tscales(batch, device=device)
            cls_out = encoder(x, mask)
            all_embs.append(cls_out.cpu().numpy())
    return np.vstack(all_embs)


if __name__ == "__main__":
    test_seqs = pd.Series([
        "MKTAYIAKQRQISFVKSHFSRQ",
        "ACDEFGHIKLMNPQRSTVWY",
    ])
    encoder = TScalesBERTEncoder(d_model=256, nhead=8, num_layers=4)
    out = encode_tscales_cls(test_seqs, encoder)
    print(f"Input sequences : {len(test_seqs)}")
    print(f"CLS output shape: {out.shape}")
    print("Sanity check passed ✓")
