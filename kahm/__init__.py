"""Kernel Affine Hull Machines.

The estimator API accepts samples as rows, as in scikit-learn. Legacy modules
retain their original column-oriented interfaces.
"""

from .classifier import KAHMClassifier

__version__ = "0.1.0"
__all__ = ["KAHMClassifier", "__version__"]
