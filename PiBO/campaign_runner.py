"""
PiBO campaign runner: builds search space, campaign, and runs the optimization loop
with beta decay and yield lookup (faithful to the original implementation).
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch # <-- NEW IMPORT

from baybe import Campaign
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective
from baybe.parameters import CategoricalParameter
from baybe.searchspace import SearchSpace
from baybe.recommenders import BotorchRecommender, TwoPhaseMetaRecommender

from .prior_module import PriorModule
from .acquisition import PriorGuidedAcquisition


class PriorInitialRecommender:
    """Recommends a random point from the prior set each time."""

    def __init__(self, prior_list: List[Tuple[int, ...]], param_names: List[str]):
        self.prior_list = prior_list
        self.param_names = param_names

    def recommend(self, searchspace, batch_size=1, **kwargs):
        # random.choice will respect the random state set inside run()
        chosen = random.choice(self.prior_list)
        recommendation = pd.DataFrame(
            {name: [str(chosen[j])] for j, name in enumerate(self.param_names)}
        )
        return recommendation


class PiBOCampaignRunner:
    """
    Encapsulates the Prior-guided Bayesian Optimization campaign: search space,
    campaign, and optimization loop with beta decay and early stopping.
    """

    def __init__(
        self,
        param_names: List[str],
        param_values: List[List[str]],
        df: pd.DataFrame,
        yield_col: str = "yield",
    ):
        self.param_names = list(param_names)
        self.param_values = list(param_values)
        self.yield_col = yield_col
        self.df_numeric = df.copy()
        self.df_str = self.df_numeric.astype(str)
        self.param_sizes = [len(v) for v in self.param_values]

    def run(
        self,
        prior_set: List[Tuple[int, ...]],
        beta: float,
        target_yield: float,
        max_iterations: int = 200,
        results_dir: Optional[Path] = None,
        set_num: Optional[int] = None,
        prior_set_path: Optional[str] = None,
        percentile: Optional[float] = None,
        actual_percentile: Optional[float] = None,
        n_priors: Optional[int] = None,
        disable_early_stop: bool = False,
        random_seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the PiBO optimization loop.
        """
        
        # ---------------------------------------------------------
        # APPLY RANDOM SEED AT THE LOWEST LEVEL
        # ---------------------------------------------------------
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        prior_set = list(prior_set)
        prior_set_set = set(prior_set)

        target = NumericalTarget(name="Yield", mode="MAX")
        objective = SingleTargetObjective(target=target)

        parameters = [
            CategoricalParameter(name=name, values=values, encoding="OHE")
            for name, values in zip(self.param_names, self.param_values)
        ]

        searchspace = SearchSpace.from_product(parameters)
        prior_module = PriorModule(prior_set=prior_set, param_values=self.param_values)
        acquisition = PriorGuidedAcquisition(
            prior_module=prior_module,
            prior_exponent=beta,
        )
        initial_recommender = PriorInitialRecommender(
            prior_list=prior_set,
            param_names=self.param_names,
        )
        recommender = TwoPhaseMetaRecommender(
            initial_recommender=initial_recommender,
            recommender=BotorchRecommender(acquisition_function=acquisition),
        )
        campaign = Campaign(searchspace, objective, recommender)

        iterations_data = []
        max_seen = 0.0
        iterations_to_target = max_iterations - 1
        results_list = []

        merge_cols = self.param_names + [self.yield_col]
        df_merge = self.df_str[self.param_names + [self.yield_col]]

        for iteration in range(max_iterations):
            recommendation = campaign.recommend(batch_size=1)

            # Decay Beta
            new_acquisition = PriorGuidedAcquisition(
                prior_module=prior_module,
                prior_exponent=(beta / (iteration + 1)),
            )
            recommender.recommender.acquisition_function = new_acquisition

            # Lookup
            search = recommendation[self.param_names].astype(str)
            result = pd.merge(
                search,
                df_merge,
                on=self.param_names,
                how="left",
            )

            row = result.iloc[0]
            param_ints = tuple(int(row[name]) for name in self.param_names)
            yield_val_str = row[self.yield_col]
            if pd.notna(yield_val_str) and str(yield_val_str).lower() != "nan":
                yield_val = float(yield_val_str)
            else:
                yield_val = 0.0

            max_seen = max(max_seen, yield_val)
            recommendation["Yield"] = [yield_val]

            in_prior = param_ints in prior_set_set
            iterations_data.append(
                {
                    "iteration": iteration,
                    **{name: param_ints[j] for j, name in enumerate(self.param_names)},
                    self.yield_col: yield_val,
                    "max_seen": max_seen,
                    "in_prior": in_prior,
                }
            )

            campaign.add_measurements(recommendation)

            if not disable_early_stop and max_seen >= target_yield:
                iterations_to_target = iteration
                results_list.append(iteration)
                break

            if iteration == max_iterations - 1:
                results_list.append(max_iterations - 1)

        iterations_df = pd.DataFrame(iterations_data)
        summary = {
            "beta": beta,
            "iterations_to_target": iterations_to_target,
            "target_yield": target_yield,
            "max_yield_seen": max_seen,
            "total_iterations": len(iterations_data),
            "random_seed_used": random_seed
        }
        if set_num is not None:
            summary["set_num"] = set_num
        if prior_set_path is not None:
            summary["prior_set_path"] = prior_set_path
        if percentile is not None:
            summary["percentile"] = percentile
        if actual_percentile is not None:
            summary["actual_percentile"] = actual_percentile
        if n_priors is not None:
            summary["n_priors"] = n_priors

        if results_dir is not None:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            iterations_df.to_csv(results_dir / "iterations.csv", index=False)
            with open(results_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            np.save(results_dir / "results.npy", np.array(results_list))

        return iterations_df, summary