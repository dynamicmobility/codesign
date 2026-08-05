import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
import jax.numpy as jnp

@dataclass
class TurboState:
    """Turbo state used to track the recent history of the trust region."""
    dim: int
    length: float = 0.8
    failure_counter: int = 0
    success_counter: int = 0
    best_value: float = -float("inf")
    restart_triggered: bool = False

class TurboOptimizer:
    def __init__(
        self,
        dim: int,
        fun: callable,
        max_cholesky_size: float = float("inf"),
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        success_tolerance: int = 10,
        batch_size: int = 4,
    ):
        self.fun = fun
        self.dim = dim
        self.max_cholesky_size = max_cholesky_size
        self.length_min = length_min
        self.length_max = length_max
        self.success_tolerance = success_tolerance
        self.batch_size = batch_size
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

    def update_state(self, state: TurboState, Y_next: torch.Tensor) -> TurboState:
        """Update the state of the trust region based on the new function values."""
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == self.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, self.length_max)
            state.success_counter = 0
        elif state.failure_counter == self.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < self.length_min:
            state.restart_triggered = True
        return state

    def generate_batch(
        self,
        state: TurboState,
        model: SingleTaskGP,  # GP model
        X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
        Y: torch.Tensor,  # Function values
        batch_size: int,
        n_candidates: Optional[int] = None,  # Number of candidates for Thompson sampling
        num_restarts: int = 10,
        raw_samples: int = 512,
        acqf: str = "ts",  # "ei" or "ts"
    ) -> torch.Tensor:
        """Generate a new batch of points."""
        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
        dtype = torch.double
        device = torch.device("cpu")

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            ei = qExpectedImprovement(model, Y.max())
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next

    def eval_objective(self, x: torch.Tensor) -> torch.Tensor:
        """This is a helper function we use to unnormalize and evalaute a point."""
        return self.fun(x)

    def optimize(self, initial_guess: torch.Tensor, num_restarts: int = 10, raw_samples: int = 512, n_candidates: int = 5000) -> None:

        NUM_RESTARTS = num_restarts
        RAW_SAMPLES = raw_samples
        N_CANDIDATES = min(n_candidates, max(2000, 200 * self.dim))

        X_turbo = torch.tensor(initial_guess)
        Y_turbo = self.eval_objective(X_turbo)
        print(X_turbo.shape)
        print(Y_turbo.shape)

        state = TurboState(self.dim, best_value=torch.max(Y_turbo).item())
        while not state.restart_triggered:  # Run until TuRBO converges
            # Fit a GP model
            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0))
            )
            model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                # Fit the model
                fit_gpytorch_mll(mll)

                # Create a batch
                X_next = self.generate_batch(
                    state=state,
                    model=model,
                    X=X_turbo,
                    Y=train_Y,
                    batch_size=self.batch_size,
                    n_candidates=N_CANDIDATES,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    acqf="ts",
                )

            Y_next = self.eval_objective(X_next)

            # Update state
            state = self.update_state(state=state, Y_next=Y_next)

            # Append data
            X_turbo = torch.cat((X_turbo, X_next), dim=0)
            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

            # Print current status
            print(f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")

        return X_next, X_turbo, Y_next, Y_turbo

        
