"""
Prior-guided acquisition function for PiBO (wraps BoTorch PriorGuidedAcquisitionFunction).
"""

from typing import ClassVar

import torch
from attrs import define, field

from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.acqfs import ExpectedImprovement
from botorch.acquisition.prior_guided import PriorGuidedAcquisitionFunction


@define(frozen=True)
class PriorGuidedAcquisition(AcquisitionFunction):
    """Prior-guided acquisition function; integrates with BayBE and BoTorch."""

    abbreviation: ClassVar[str] = "PGA"

    acq: AcquisitionFunction = field(default=ExpectedImprovement())
    prior_module: torch.nn.Module = field(default=None)
    log: bool = field(default=False)
    prior_exponent: float = field(default=1.0)
    X_pending: torch.Tensor = field(default=None)

    def to_botorch(self, surrogate, searchspace, train_x, train_y):
        base_acqf = self.acq.to_botorch(surrogate, searchspace, train_x, train_y)
        prior_guided_acqf = PriorGuidedAcquisitionFunction(
            acq_function=base_acqf,
            prior_module=self.prior_module,
            log=self.log,
            prior_exponent=self.prior_exponent,
            X_pending=self.X_pending,
        )
        return prior_guided_acqf
