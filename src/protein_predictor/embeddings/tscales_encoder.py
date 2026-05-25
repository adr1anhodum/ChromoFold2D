"""
embeddings/tscales_encoder.py
------------------------------
Wraps TScalesBERTEncoder — now a proper package module, no longer a
root-level file. Import path: protein_predictor.tscales_bert_cls
"""

import numpy as np


class TScalesEncoder:
    """
    Wraps TScalesBERTEncoder to match the same interface as ESMEncoder.

    Parameters
    ----------
    d_model    : transformer model dimension (default 256)
    nhead      : number of attention heads   (default 8)
    num_layers : number of transformer layers (default 4)
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        # Import from the package — no longer relies on root-level file
        from protein_predictor.tscales_bert_cls import TScalesBERTEncoder
        self.encoder = TScalesBERTEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    def encode_series(self, sequences) -> list:
        """
        Encode a pandas Series (or list) of amino-acid sequences.
        Returns a list of 1-D float32 arrays of shape (d_model,).
        """
        from protein_predictor.tscales_bert_cls import encode_tscales_cls
        embeddings = encode_tscales_cls(sequences, self.encoder)
        return [e.astype(np.float32) for e in embeddings]
