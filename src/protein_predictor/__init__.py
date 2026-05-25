"""
protein_predictor
=================
Predict fluorescent-protein emission wavelength and brightness
from ESM-2 or T-scales + ChemBERTa embeddings.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
