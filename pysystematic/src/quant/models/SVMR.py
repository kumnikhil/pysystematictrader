import torch
import polars as pl 
from dataclasses import dataclass

@dataclass
class SVMRConfig:
    """Parameters for SVMR model."""
    num_features: int = 10
    num_classes: int = 2
    kernel: str = 'linear'
    C: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-3

# class SVMR(object):
