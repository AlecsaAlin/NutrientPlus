"""
AHP (Analytic Hierarchy Process) Weight Computer for NutrientPlus
=================================================================
Derives personalised GA fitness weights from user preferences using
Saaty's pairwise comparison method.

Instead of asking 6 pairwise questions (which would hurt mobile UX),
we construct the 4x4 comparison matrix from two user inputs:
  1. goal_type             -- already collected during onboarding
  2. exploration_preference -- new onboarding question

The four criteria being weighted:
  R = Relevance        (two-tower model ranking score)
  N = Nutritional diversity
  C = Category diversity
  V = Novelty          (items not in user history)

Approach
--------
Each (goal_type, exploration_preference) pair maps to a *target priority
vector*.  We convert it to a Saaty-scale pairwise comparison matrix using
  a[i][j] = (w_i / w_j) ^ alpha
where alpha > 1 stretches the ratios onto the 1-9 Saaty scale.  The
geometric mean method then recovers weights, and the consistency ratio
is validated (CR < 0.10).

Reference: Saaty, T.L. (1980). The Analytic Hierarchy Process.
"""

import numpy as np
from typing import Dict


class AHPWeightComputer:
    """Derives personalised GA fitness weights via the Analytic Hierarchy Process."""

    CRITERIA = ['relevance', 'nutrition', 'category', 'novelty']

    _RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}

    _PROFILES: Dict[tuple, list] = {
        # lose_weight: nutrition is important, relevance varies by exploration
        ('lose_weight', 'conservative'):     [0.45, 0.35, 0.10, 0.10],
        ('lose_weight', 'balanced'):         [0.33, 0.32, 0.17, 0.18],
        ('lose_weight', 'adventurous'):      [0.18, 0.27, 0.22, 0.33],

        # gain_muscle: nutrition is highest priority
        ('gain_muscle', 'conservative'):     [0.35, 0.42, 0.10, 0.13],
        ('gain_muscle', 'balanced'):         [0.27, 0.37, 0.16, 0.20],
        ('gain_muscle', 'adventurous'):      [0.15, 0.28, 0.22, 0.35],

        # maintain_weight: relevance-heavy, nutrition secondary
        ('maintain_weight', 'conservative'): [0.52, 0.25, 0.12, 0.11],
        ('maintain_weight', 'balanced'):     [0.38, 0.22, 0.20, 0.20],
        ('maintain_weight', 'adventurous'):  [0.20, 0.17, 0.27, 0.36],

        # eat_healthier: nutrition dominates
        ('eat_healthier', 'conservative'):   [0.28, 0.48, 0.10, 0.14],
        ('eat_healthier', 'balanced'):       [0.22, 0.40, 0.16, 0.22],
        ('eat_healthier', 'adventurous'):    [0.13, 0.28, 0.24, 0.35],
    }

    _ALPHA = 1.5

    def compute_weights(
        self,
        goal_type: str = 'maintain_weight',
        exploration_preference: str = 'balanced',
    ) -> Dict[str, float]:
        """
        Compute personalised fitness weights.

        Parameters
        ----------
        goal_type : str
            One of: lose_weight, gain_muscle, maintain_weight, eat_healthier
        exploration_preference : str
            One of: conservative, balanced, adventurous

        Returns
        -------
        dict with keys w_relevance, w_nutrition, w_category, w_novelty
        (guaranteed to sum to 1.0)
        """
        matrix = self._build_comparison_matrix(goal_type, exploration_preference)
        weights = self._geometric_mean_weights(matrix)
        cr = self._consistency_ratio(matrix, weights)

        if cr > 0.10:
            return self.default_weights()

        return {
            f'w_{name}': round(float(w), 4)
            for name, w in zip(self.CRITERIA, weights)
        }

    def compute_weights_with_cr(
        self,
        goal_type: str = 'maintain_weight',
        exploration_preference: str = 'balanced',
    ) -> tuple:
        """Like compute_weights but also returns the consistency ratio."""
        matrix = self._build_comparison_matrix(goal_type, exploration_preference)
        weights = self._geometric_mean_weights(matrix)
        cr = self._consistency_ratio(matrix, weights)
        w_dict = {
            f'w_{name}': round(float(w), 4)
            for name, w in zip(self.CRITERIA, weights)
        }
        if cr > 0.10:
            w_dict = self.default_weights()
        return w_dict, cr

    def _build_comparison_matrix(
        self, goal_type: str, exploration_preference: str,
    ) -> np.ndarray:
        """
        Build the 4x4 Saaty pairwise comparison matrix from a target
        weight profile.

        Matrix layout (indices):  0=R, 1=N, 2=C, 3=V

        a[i][j] = (w_i / w_j) ^ alpha

        This guarantees a nearly-consistent matrix (low CR) because
        the ratios are derived from a coherent priority vector.  The
        alpha exponent stretches the ratios to utilise more of the
        Saaty 1-9 scale, and the geometric mean method recovers
        weights very close to the original profile.
        """
        key = (goal_type, exploration_preference)
        profile = self._PROFILES.get(key)

        if profile is None:
            profile = self._PROFILES[('maintain_weight', 'balanced')]

        w = np.array(profile, dtype=np.float64)
        w = w / w.sum()  

        n = len(w)
        A = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    ratio = w[i] / w[j]
                    val = ratio ** self._ALPHA
                    A[i, j] = np.clip(val, 1 / 9, 9)

        return A

    @staticmethod
    def _geometric_mean_weights(matrix: np.ndarray) -> np.ndarray:
        """
        Extract priority weights using the row geometric mean method.

        For each row, compute the geometric mean of all elements, then
        normalise so the weights sum to 1.0.  This is the standard AHP
        approximation of the principal eigenvector.
        """
        n = matrix.shape[0]
        geo_means = np.prod(matrix, axis=1) ** (1.0 / n)
        weights = geo_means / geo_means.sum()
        return weights


    def _consistency_ratio(
        self, matrix: np.ndarray, weights: np.ndarray,
    ) -> float:
        """
        Compute the Consistency Ratio (CR).

        CR = CI / RI
        CI = (lambda_max - n) / (n - 1)

        A CR < 0.10 is generally considered acceptable.
        """
        n = matrix.shape[0]
        if n <= 2:
            return 0.0  

        Aw = matrix @ weights
        lambdas = Aw / weights
        lambda_max = float(np.mean(lambdas))

        ci = (lambda_max - n) / (n - 1)
        ri = self._RI.get(n, 0.90)
        if ri == 0:
            return 0.0

        return ci / ri

    @staticmethod
    def default_weights() -> Dict[str, float]:
        """The original hardcoded weights -- used as fallback."""
        return {
            'w_relevance': 0.50,
            'w_nutrition': 0.25,
            'w_category':  0.15,
            'w_novelty':   0.10,
        }

if __name__ == '__main__':
    ahp = AHPWeightComputer()

    goals = ['lose_weight', 'gain_muscle', 'maintain_weight', 'eat_healthier']
    prefs = ['conservative', 'balanced', 'adventurous']

    print('=' * 72)
    print('AHP Weight Computation -- All 12 Combinations')
    print('=' * 72)

    all_pass = True
    for goal in goals:
        for pref in prefs:
            w, cr = ahp.compute_weights_with_cr(goal, pref)
            total = sum(w.values())
            ok = abs(total - 1.0) < 1e-3 and cr < 0.10
            if not ok:
                all_pass = False
            status = 'OK' if ok else 'FAIL'
            print(f'  {goal:18s} + {pref:14s} -> '
                  f'R={w["w_relevance"]:.3f}  N={w["w_nutrition"]:.3f}  '
                  f'C={w["w_category"]:.3f}  V={w["w_novelty"]:.3f}  '
                  f'Sum={total:.4f}  CR={cr:.4f}  {status}')

    print(f'\n{"OK -- All 12 passed" if all_pass else "FAIL -- Some failed"}')
