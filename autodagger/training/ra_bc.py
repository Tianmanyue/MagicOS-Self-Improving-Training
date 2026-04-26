"""Reward-Aware Behavior Cloning (RA-BC) sample weighting.

Implements the RA-BC weighting scheme from the SARM paper: each
timestep is assigned a weight proportional to the *progress gain*
over a temporal window of size ``delta``.  Transitions where the
agent makes positive progress receive higher weight during training,
while stuck or regressing transitions are down-weighted.

Usage in the AutoDAgger pipeline
--------------------------------
1. A progress model (TOP-Reward or SARM) produces per-timestep scores
   for every demonstration trajectory.
2. ``RABCWeighter.compute_weights`` converts those scores into training
   weights.
3. ``apply_weights_to_dataset`` writes the weights into the HDF5 dataset
   so that Robomimic's weighted-sampling dataloader can consume them.
"""

from __future__ import annotations

import logging
import shutil
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class RABCWeighter:
    """Compute per-timestep RA-BC weights from progress model scores.

    Parameters
    ----------
    delta : int
        Temporal offset for computing progress differences.  The weight
        at timestep *t* is derived from ``progress[t + delta] - progress[t]``.
    normalize : bool
        If ``True``, normalize the resulting weights to the range [0, 1].
    """

    def __init__(self, delta: int = 10, normalize: bool = True) -> None:
        if delta < 1:
            raise ValueError(f"delta must be >= 1, got {delta}")
        self.delta = delta
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Core weight computation
    # ------------------------------------------------------------------

    def compute_weights(self, progress_scores: np.ndarray) -> np.ndarray:
        """Compute per-timestep RA-BC weights for a single trajectory.

        Parameters
        ----------
        progress_scores : np.ndarray
            1-D array of progress model outputs, one per timestep.

        Returns
        -------
        np.ndarray
            1-D weight array of the same length as *progress_scores*.
        """
        progress_scores = np.asarray(progress_scores, dtype=np.float64)
        T = len(progress_scores)

        if T == 0:
            return np.array([], dtype=np.float64)

        weights = np.zeros(T, dtype=np.float64)

        for t in range(T):
            t_future = min(t + self.delta, T - 1)
            weights[t] = progress_scores[t_future] - progress_scores[t]

        # Clip negative progress (regression) to zero.
        np.clip(weights, 0.0, None, out=weights)

        # Normalize to [0, 1].
        if self.normalize:
            w_max = weights.max()
            if w_max > 0.0:
                weights /= w_max

        return weights

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_weights_batch(
        self,
        progress_by_trajectory: dict[str, list[float]],
    ) -> dict[str, np.ndarray]:
        """Compute RA-BC weights for multiple trajectories.

        Parameters
        ----------
        progress_by_trajectory : dict[str, list[float]]
            Mapping from trajectory / demo name to a list of per-timestep
            progress scores.

        Returns
        -------
        dict[str, np.ndarray]
            Same keys, with values replaced by weight arrays.
        """
        results: dict[str, np.ndarray] = {}
        for name, scores in progress_by_trajectory.items():
            arr = np.asarray(scores, dtype=np.float64)
            results[name] = self.compute_weights(arr)
        return results

    # ------------------------------------------------------------------
    # Dataset integration
    # ------------------------------------------------------------------

    def apply_weights_to_dataset(
        self,
        hdf5_path: str,
        progress_scores_by_demo: dict[str, np.ndarray],
        output_path: str,
    ) -> None:
        """Write per-timestep weights into an HDF5 dataset.

        For each demonstration in *progress_scores_by_demo*, compute
        RA-BC weights and store them under ``data/<demo>/weights`` in
        the output HDF5 file.

        Parameters
        ----------
        hdf5_path : str
            Path to the source HDF5 dataset.
        progress_scores_by_demo : dict[str, np.ndarray]
            Mapping from demo name (e.g. ``"demo_0"``) to an array of
            progress scores.  The keys must match groups under ``data/``
            in the HDF5 file.
        output_path : str
            Where to write the augmented HDF5 file.  If equal to
            *hdf5_path*, the file is modified in-place.
        """
        in_place = output_path == hdf5_path

        if not in_place:
            logger.info("Copying %s -> %s", hdf5_path, output_path)
            shutil.copy2(hdf5_path, output_path)

        weights_by_demo = self.compute_weights_batch(
            {k: v.tolist() if isinstance(v, np.ndarray) else v
             for k, v in progress_scores_by_demo.items()}
        )

        with h5py.File(output_path, "a") as f:
            data_grp = f.get("data")
            if data_grp is None:
                raise KeyError(
                    f"HDF5 file {output_path} has no 'data' group. "
                    "Expected Robomimic-format dataset."
                )

            demos_written = 0
            for demo_name, weights in weights_by_demo.items():
                demo_grp = data_grp.get(demo_name)
                if demo_grp is None:
                    logger.warning(
                        "Demo '%s' not found in HDF5; skipping.", demo_name
                    )
                    continue

                # Delete existing weights dataset if present.
                if "weights" in demo_grp:
                    del demo_grp["weights"]

                demo_grp.create_dataset(
                    "weights",
                    data=weights.astype(np.float32),
                    compression="gzip",
                )
                demos_written += 1

            logger.info(
                "Wrote RA-BC weights for %d / %d demos to %s",
                demos_written,
                len(weights_by_demo),
                output_path,
            )

            if demos_written == 0:
                logger.warning(
                    "No weights were written. Check that demo names in "
                    "progress_scores_by_demo match those in the HDF5 file."
                )
