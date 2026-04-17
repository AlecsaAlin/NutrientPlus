"""
NSGA-III Re-Ranker for NutrientPlus
=====================================
True multi-objective reranker using the Non-dominated Sorting Genetic
Algorithm III (Deb & Jain, 2014).

Unlike the weighted-sum GA variants, NSGA-III optimises all 4 objectives
simultaneously — relevance, nutritional diversity, category diversity, and
novelty — without collapsing them into a scalar. A Pareto front is maintained
across generations. At the end, the AHP-derived weights are used only to pick
one final solution from the front.

No external dependencies — pure Python / NumPy.
"""

import copy
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ga_reranker import CandidateItem

@dataclass
class NSGA3Config:
    """Hyper-parameters for the NSGA-III reranker."""

    population_size: int   = 40    # >= num reference directions (35 for p=4, M=4)
    num_generations: int   = 30
    top_k:           int   = 10
    candidate_pool_size: int = 50

    mutation_rate:   float = 0.25
    crossover_rate:  float = 0.70

    num_divisions:   int   = 4     # → C(7,3) = 35 reference directions for M=4

    w_relevance: float = 0.4537
    w_nutrition: float = 0.1999
    w_category:  float = 0.1732
    w_novelty:   float = 0.1732

    calorie_bands: List[float] = field(
        default_factory=lambda: [0, 200, 400, 600, 800, float('inf')]
    )
    protein_bands: List[float] = field(
        default_factory=lambda: [0, 10, 20, 35, float('inf')]
    )

    def __post_init__(self):
        total = self.w_relevance + self.w_nutrition + self.w_category + self.w_novelty
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")


class ObjectiveEvaluator:
    """
    Evaluates a chromosome → np.ndarray of shape (4,):
        [relevance_score, nutritional_diversity, category_diversity, novelty_score]
    All values in [0, 1].  Higher is better for all objectives.
    """

    def __init__(self, config: NSGA3Config):
        self.cfg = config

    def evaluate(self, chromosome: List[CandidateItem]) -> np.ndarray:
        if not chromosome:
            return np.zeros(4)
        return np.array([
            self._relevance(chromosome),
            self._nutrition(chromosome),
            self._category(chromosome),
            self._novelty(chromosome),
        ], dtype=float)

    def weighted_sum(self, objectives: np.ndarray) -> float:
        """Collapse objectives → scalar for final selection only."""
        w = np.array([
            self.cfg.w_relevance, self.cfg.w_nutrition,
            self.cfg.w_category,  self.cfg.w_novelty,
        ])
        return float(np.clip(np.dot(w, objectives), 0.0, 1.0))


    def _relevance(self, ch: List[CandidateItem]) -> float:
        scores = [item.relevance_score for item in ch]
        dcg  = sum(s / math.log2(r + 2) for r, s in enumerate(scores))
        idcg = sum(s / math.log2(r + 2) for r, s in enumerate(sorted(scores, reverse=True)))
        return dcg / idcg if idcg > 0 else 0.0

    def _nutrition(self, ch: List[CandidateItem]) -> float:
        cal_hits, prot_hits = set(), set()
        for item in ch:
            for i in range(len(self.cfg.calorie_bands) - 1):
                if self.cfg.calorie_bands[i] <= item.calories < self.cfg.calorie_bands[i + 1]:
                    cal_hits.add(i); break
            for i in range(len(self.cfg.protein_bands) - 1):
                if self.cfg.protein_bands[i] <= item.protein < self.cfg.protein_bands[i + 1]:
                    prot_hits.add(i); break
        cal_cov  = len(cal_hits)  / (len(self.cfg.calorie_bands)  - 1)
        prot_cov = len(prot_hits) / (len(self.cfg.protein_bands) - 1)
        return (cal_cov + prot_cov) / 2.0

    def _category(self, ch: List[CandidateItem]) -> float:
        return min(len(set(item.category_id for item in ch)) / len(ch), 1.0)

    def _novelty(self, ch: List[CandidateItem]) -> float:
        return sum(1 for item in ch if not item.in_user_history) / len(ch)


def _generate_reference_directions(num_objectives: int = 4, num_divisions: int = 6) -> np.ndarray:
    """
    Das-Dennis structured reference directions on an (M-1)-simplex.
    Returns array of shape (N_ref, M) where each row sums to 1.
    For M=4, p=6: C(9,3) = 84 directions.
    """
    def _recurse(m, left, prefix, result):
        if m == 1:
            result.append(prefix + [left / num_divisions])
            return
        for i in range(left + 1):
            _recurse(m - 1, left - i, prefix + [i / num_divisions], result)

    result = []
    _recurse(num_objectives, num_divisions, [], result)
    return np.array(result, dtype=float)


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b (a >= b on all objectives, a > b on at least one)."""
    return bool(np.all(a >= b) and np.any(a > b))


def _fast_non_dominated_sort(obj_matrix: np.ndarray) -> List[List[int]]:
    """
    Standard Deb O(M*N^2) non-dominated sort.
    Returns list of fronts; each front is a list of indices into obj_matrix.
    """
    n = len(obj_matrix)
    dominated_by   = [0] * n       # how many individuals dominate i
    dominates_list = [[] for _ in range(n)]  # who i dominates

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(obj_matrix[i], obj_matrix[j]):
                dominates_list[i].append(j)
                dominated_by[j] += 1
            elif _dominates(obj_matrix[j], obj_matrix[i]):
                dominates_list[j].append(i)
                dominated_by[i] += 1

    fronts   = []
    current  = [i for i in range(n) if dominated_by[i] == 0]
    while current:
        fronts.append(current)
        next_front = []
        for i in current:
            for j in dominates_list[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)
        current = next_front

    return fronts


def _normalize_objectives(
    obj_matrix: np.ndarray,
    selected_indices: List[int],
    last_front_indices: List[int],
) -> np.ndarray:
    """
    NSGA-III normalization:
    1. Ideal point = max per objective over (selected + last front).
    2. Translate so ideal → origin (negate since we maximise).
    3. Find extreme points via Achievement Scalarizing Function.
    4. Compute intercepts; normalize by intercept.
    Falls back to range normalization on degenerate cases.
    """
    all_idx   = selected_indices + last_front_indices
    sub       = obj_matrix[all_idx]
    M         = obj_matrix.shape[1]

    ideal     = sub.max(axis=0)
    translated = ideal - obj_matrix  

    extreme = []
    eps = 1e-6
    for m in range(M):
        w = np.full(M, 1e6)
        w[m] = 1.0
        asf = np.max(translated[all_idx] * w, axis=1)
        extreme.append(all_idx[int(np.argmin(asf))])

    extreme_pts = translated[extreme]  

    try:
        intercepts = np.linalg.solve(extreme_pts + eps, np.ones(M))
        intercepts = 1.0 / (intercepts + eps)
        if np.any(intercepts <= eps):
            raise np.linalg.LinAlgError("Non-positive intercept")
    except np.linalg.LinAlgError:
        lo = sub.min(axis=0)
        hi = sub.max(axis=0)
        intercepts = np.where(hi - lo > eps, hi - lo, 1.0)

    normalized = translated / (intercepts + eps)
    return np.clip(normalized, 0.0, None)


def _associate_to_reference_points(
    normalized: np.ndarray,
    indices: List[int],
    ref_dirs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each individual in `indices`, find the closest reference direction
    (by perpendicular distance to the reference line through origin).
    Returns:
        assoc  — (len(indices),) int array: reference index for each individual
        dists  — (len(indices),) float array: perpendicular distance
    """
    pts   = normalized[indices]               
    norms = np.linalg.norm(ref_dirs, axis=1, keepdims=True) + 1e-12
    unit  = ref_dirs / norms                  

    # Project each point onto each reference line: scalar = dot(p, u)
    # perpendicular distance = ||p||^2 - scalar^2
    dots  = pts @ unit.T                      
    p_sq  = np.sum(pts ** 2, axis=1, keepdims=True)  
    perp  = np.sqrt(np.clip(p_sq - dots ** 2, 0.0, None))  

    assoc = np.argmin(perp, axis=1)           
    dists = perp[np.arange(len(indices)), assoc]
    return assoc, dists


def _niching_select(
    niche_counts: np.ndarray,
    last_front_assoc: np.ndarray,
    last_front_dists: np.ndarray,
    last_front_indices: List[int],
    num_needed: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Iteratively select `num_needed` members from the last front using niching:
    - Pick the reference point with the lowest niche count among those that
      have at least one associated member in the last front.
    - If niche count is 0 (no members yet selected for this ref), pick the
      associated last-front member with the smallest perpendicular distance.
    - Otherwise pick a random associated last-front member.
    """
    selected  = []
    remaining = list(range(len(last_front_indices)))  

    for _ in range(num_needed):
        if not remaining:
            break


        active_refs = set(last_front_assoc[remaining])

        min_count  = min(niche_counts[r] for r in active_refs)
        min_refs   = [r for r in active_refs if niche_counts[r] == min_count]
        chosen_ref = rng.choice(min_refs)

        members = [i for i in remaining if last_front_assoc[i] == chosen_ref]

        if niche_counts[chosen_ref] == 0:
            pick = min(members, key=lambda i: last_front_dists[i])
        else:
            pick = int(rng.choice(members))

        selected.append(last_front_indices[pick])
        niche_counts[chosen_ref] += 1
        remaining.remove(pick)

    return selected


class NSGA3Reranker:
    """
    NSGA-III reranker for food recommendation lists.

    Maintains a Pareto front across 4 objectives. At the end, selects a single
    solution from the front using the AHP-derived weights stored in the config.
    """

    def __init__(self, config: Optional[NSGA3Config] = None, seed: int = 42):
        self.cfg       = config or NSGA3Config()
        self.evaluator = ObjectiveEvaluator(self.cfg)
        self.ref_dirs  = _generate_reference_directions(
            num_objectives=4, num_divisions=self.cfg.num_divisions
        )
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)

    def evolve(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        k = min(self.cfg.top_k, len(candidates))
        if k == 0:
            return []

        candidates = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
        pop        = self._init_population(candidates, k)
        obj        = np.array([self.evaluator.evaluate(c) for c in pop]) 

        for _ in range(self.cfg.num_generations):
            offspring     = self._create_offspring(pop, obj, candidates, k)
            off_obj       = np.array([self.evaluator.evaluate(c) for c in offspring])

            combined      = pop + offspring
            combined_obj  = np.vstack([obj, off_obj])

            pop, obj = self._select_next_generation(combined, combined_obj)

        return self._select_final(pop, obj)
    
    def _select_next_generation(
        self,
        population: List[List[CandidateItem]],
        obj_matrix: np.ndarray,
    ) -> Tuple[List[List[CandidateItem]], np.ndarray]:
        N      = self.cfg.population_size
        fronts = _fast_non_dominated_sort(obj_matrix)

        selected_idx = []
        for front in fronts:
            if len(selected_idx) + len(front) <= N:
                selected_idx.extend(front)
                if len(selected_idx) == N:
                    break
            else:
                num_needed = N - len(selected_idx)
                normalized = _normalize_objectives(obj_matrix, selected_idx, front)

                all_so_far = selected_idx + front
                assoc_all, dists_all = _associate_to_reference_points(
                    normalized, all_so_far, self.ref_dirs
                )

                niche_counts = np.zeros(len(self.ref_dirs), dtype=int)
                for i, idx in enumerate(selected_idx):
                    pos_in_all = all_so_far.index(idx)
                    niche_counts[assoc_all[pos_in_all]] += 1

                # Association for last-front members only
                last_front_positions = list(range(len(selected_idx), len(all_so_far)))
                last_front_assoc = assoc_all[last_front_positions]
                last_front_dists = dists_all[last_front_positions]

                picked = _niching_select(
                    niche_counts, last_front_assoc, last_front_dists,
                    front, num_needed, self.rng,
                )
                selected_idx.extend(picked)
                break

        new_pop = [population[i] for i in selected_idx]
        new_obj = obj_matrix[selected_idx]
        return new_pop, new_obj

    def _select_final(
        self,
        population: List[List[CandidateItem]],
        obj_matrix: np.ndarray,
    ) -> List[CandidateItem]:
        """
        Pick the Pareto-front member closest to the ideal point [1,1,1,1].
        This is the standard multi-objective selection approach — it balances
        all objectives equally without biasing toward the AHP weighted sum.
        """
        fronts   = _fast_non_dominated_sort(obj_matrix)
        front0   = fronts[0]
        ideal    = np.ones(obj_matrix.shape[1])
        dists    = [np.linalg.norm(obj_matrix[i] - ideal) for i in front0]
        best_idx = front0[int(np.argmin(dists))]
        return population[best_idx]


    def _create_offspring(
        self,
        population: List[List[CandidateItem]],
        obj_matrix: np.ndarray,
        candidates: List[CandidateItem],
        k: int,
    ) -> List[List[CandidateItem]]:
        """Binary tournament + crossover + mutation → N offspring."""
        offspring = []
        while len(offspring) < self.cfg.population_size:
            pa = self._tournament_select(population, obj_matrix)
            pb = self._tournament_select(population, obj_matrix)

            if random.random() < self.cfg.crossover_rate:
                child = self._crossover(pa, pb, candidates, k)
            else:
                child = copy.deepcopy(pa)

            if random.random() < self.cfg.mutation_rate:
                child = self._mutate(child, candidates)

            offspring.append(child)
        return offspring


    def _init_population(
        self, candidates: List[CandidateItem], k: int
    ) -> List[List[CandidateItem]]:
        pop = [copy.deepcopy(candidates[:k])]
        for _ in range(self.cfg.population_size - 1):
            pop.append(copy.deepcopy(random.sample(candidates, min(k, len(candidates)))))
        return pop

    def _tournament_select(
        self,
        population: List[List[CandidateItem]],
        obj_matrix: np.ndarray,
        tournament_size: int = 2,
    ) -> List[CandidateItem]:
        """Binary tournament via Pareto dominance; break ties randomly."""
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        if len(indices) == 1:
            return copy.deepcopy(population[indices[0]])
        a, b = indices[0], indices[1]
        if _dominates(obj_matrix[a], obj_matrix[b]):
            winner = a
        elif _dominates(obj_matrix[b], obj_matrix[a]):
            winner = b
        else:
            winner = random.choice([a, b])
        return copy.deepcopy(population[winner])

    def _crossover(
        self,
        parent_a: List[CandidateItem],
        parent_b: List[CandidateItem],
        candidates: List[CandidateItem],
        k: int,
    ) -> List[CandidateItem]:
        """Order-preserving single-segment crossover — identical to existing GAs."""
        if k < 2:
            return copy.deepcopy(parent_a)
        start = random.randint(0, k - 2)
        end   = random.randint(start + 1, k - 1)
        child = list(parent_a[start:end + 1])
        ids   = {item.item_id for item in child}
        for item in parent_b:
            if item.item_id not in ids:
                child.append(item)
                ids.add(item.item_id)
            if len(child) == k:
                break
        if len(child) < k:
            for item in candidates:
                if item.item_id not in ids:
                    child.append(item)
                    ids.add(item.item_id)
                if len(child) == k:
                    break
        return child[:k]

    def _mutate(
        self,
        chromosome: List[CandidateItem],
        candidates: List[CandidateItem],
    ) -> List[CandidateItem]:
        """Random mutation: swap two positions or replace one item."""
        chromosome = copy.deepcopy(chromosome)
        if random.choice(['swap', 'replace']) == 'swap' and len(chromosome) >= 2:
            i, j = random.sample(range(len(chromosome)), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        else:
            ids     = {item.item_id for item in chromosome}
            outside = [c for c in candidates if c.item_id not in ids]
            if outside:
                chromosome[random.randint(0, len(chromosome) - 1)] = random.choice(outside)
        return chromosome
