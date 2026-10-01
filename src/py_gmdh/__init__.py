"""Group Method of Data Handling (GMDH) estimators with a scikit-learn API."""

from .aic import AICGMDH, AICGMDHClassifier
from .cfp import CFPGMDH, CFPGMDHClassifier
from .combi import CombiGMDH, CombiGMDHClassifier
from .cv import CVGMDH, CVGMDHClassifier
from .fourier import FourierGMDH, FourierGMDHClassifier
from .gmdh import GMDH, GMDHClassifier
from .hierarchical import HierarchicalGMDH, HierarchicalGMDHClassifier
from .lookback import LookbackGMDH, LookbackGMDHClassifier
from .mars import MARSGMDH, MARSGMDHClassifier
from .radial import RadialGMDH, RadialGMDHClassifier
from .sigmoid import SigmoidGMDH, SigmoidGMDHClassifier
from .ufp import UFPGMDH, UFPGMDHClassifier

__version__ = "0.0.1"

__all__ = [
    "GMDH",
    "CombiGMDH",
    "HierarchicalGMDH",
    "AICGMDH",
    "CVGMDH",
    "UFPGMDH",
    "CFPGMDH",
    "FourierGMDH",
    "RadialGMDH",
    "SigmoidGMDH",
    "MARSGMDH",
    "LookbackGMDH",
    "GMDHClassifier",
    "CombiGMDHClassifier",
    "HierarchicalGMDHClassifier",
    "AICGMDHClassifier",
    "CVGMDHClassifier",
    "UFPGMDHClassifier",
    "CFPGMDHClassifier",
    "FourierGMDHClassifier",
    "RadialGMDHClassifier",
    "SigmoidGMDHClassifier",
    "MARSGMDHClassifier",
    "LookbackGMDHClassifier",
]
