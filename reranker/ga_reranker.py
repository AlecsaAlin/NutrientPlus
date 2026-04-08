"""
GA Re-Ranker for NutrientPlus
==============================
Hybrid Pipeline:
  Two-Tower Model  →  Top-K Candidate Retrieval
        ↓
  SA-GA Re-ranker  →  Evolves final ranked list
        ↓
  Final Recommendations (diverse, nutritionally balanced)

The GA operates AFTER the two-tower model scores items.
It optimises a multi-objective fitness function:
  1. Relevance           – two-tower ranking score (higher = better)
  2. Nutritional diversity – spread across calorie / macro bands
  3. Category diversity   – spread across CategoryID values
  4. Novelty              – penalise items already in user history

Simulated Annealing (SA) extension (inspired by Shenoy et al., 2004 –
"Mining Top-k Ranked Webpages using Simulated Annealing and Genetic
Algorithms"):
  - A temperature T is maintained across generations and cooled via an
    annealing schedule (T_{k+1} = rho * T_k).
  - When a mutated child has LOWER fitness than its parent, it is still
    accepted with probability exp(-(f_parent - f_child) / T).  This
    mirrors SA's Metropolis acceptance criterion and prevents the
    population converging prematurely to a local optimum.
  - At high T (early generations) the algorithm explores broadly;
    at low T (late generations) it exploits the best solutions found.
"""

import math
import numpy as np
import random
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
import copy


@dataclass
class GAConfig:
    """All hyper-parameters for the SA-GA algorithm."""
    population_size: int = 50
    num_generations: int = 40
    top_k: int = 10
    candidate_pool_size: int = 50

    mutation_rate: float = 0.25
    crossover_rate: float = 0.70
    elite_fraction: float = 0.10

    w_relevance: float = 0.50
    w_nutrition: float = 0.25
    w_category: float = 0.15
    w_novelty: float = 0.10


    sa_initial_temperature: float = 1.0
    sa_cooling_rate: float = 0.92
    sa_min_temperature: float = 1e-4

    # --- PSO-guided mutation parameters ---
    # Cognitive: probability of replacing toward personal best 
    pso_cognitive: float = 0.4
    # Social: probability of replacing toward global best 
    pso_social: float = 0.3

    pso_inertia: float = 0.9

    calorie_bands: List[float] = field(
        default_factory=lambda: [0, 200, 400, 600, 800, float('inf')]
    )
    protein_bands: List[float] = field(
        default_factory=lambda: [0, 10, 20, 35, float('inf')]
    )

    def __post_init__(self):
        total = self.w_relevance + self.w_nutrition + self.w_category + self.w_novelty
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Fitness weights must sum to 1.0, got {total}")
        if not (0.0 < self.sa_cooling_rate < 1.0):
            raise ValueError("sa_cooling_rate must be strictly between 0 and 1.")


@dataclass
class CandidateItem:
    """A single recipe candidate with its ML score and nutritional attributes."""
    item_id: int
    relevance_score: float
    calories: float = 0.0
    protein: float = 0.0
    fat: float = 0.0
    carbs: float = 0.0
    sugar: float = 0.0
    fiber: float = 0.0
    category_id: int = 0
    in_user_history: bool = False


class FitnessEvaluator:
    """Evaluates a chromosome (ranked list of CandidateItems) → scalar in [0, 1]."""

    def __init__(self, config: GAConfig):
        self.cfg = config

    def evaluate(self, chromosome: List[CandidateItem]) -> float:
        if not chromosome:
            return 0.0
        fitness = (
            self.cfg.w_relevance * self._relevance_score(chromosome) +
            self.cfg.w_nutrition * self._nutrition_diversity(chromosome) +
            self.cfg.w_category  * self._category_diversity(chromosome) +
            self.cfg.w_novelty   * self._novelty_score(chromosome)
        )
        return float(np.clip(fitness, 0.0, 1.0))

    def _relevance_score(self, chromosome: List[CandidateItem]) -> float:
        scores = [item.relevance_score for item in chromosome]
        dcg  = sum(s / np.log2(r + 2) for r, s in enumerate(scores))
        idcg = sum(s / np.log2(r + 2) for r, s in enumerate(sorted(scores, reverse=True)))
        return dcg / idcg if idcg > 0 else 0.0

    def _nutrition_diversity(self, chromosome: List[CandidateItem]) -> float:
        cal_hits, prot_hits = set(), set()
        for item in chromosome:
            for i in range(len(self.cfg.calorie_bands) - 1):
                if self.cfg.calorie_bands[i] <= item.calories < self.cfg.calorie_bands[i + 1]:
                    cal_hits.add(i); break
            for i in range(len(self.cfg.protein_bands) - 1):
                if self.cfg.protein_bands[i] <= item.protein < self.cfg.protein_bands[i + 1]:
                    prot_hits.add(i); break
        cal_cov  = len(cal_hits)  / (len(self.cfg.calorie_bands) - 1)
        prot_cov = len(prot_hits) / (len(self.cfg.protein_bands) - 1)
        return (cal_cov + prot_cov) / 2.0

    def _category_diversity(self, chromosome: List[CandidateItem]) -> float:
        return min(len(set(item.category_id for item in chromosome)) / len(chromosome), 1.0)

    def _novelty_score(self, chromosome: List[CandidateItem]) -> float:
        return sum(1 for item in chromosome if not item.in_user_history) / len(chromosome)

def _sa_accept(fitness_child: float, fitness_parent: float, temperature: float) -> bool:
    """
    Accept a child chromosome using the Metropolis criterion.
    Better children are always accepted.  Worse children are accepted with
    probability exp(-(f_parent - f_child) / T).
    """
    if fitness_child >= fitness_parent:
        return True
    if temperature <= 0.0:
        return False
    delta = fitness_parent - fitness_child
    return random.random() < math.exp(-delta / temperature)

class GeneticReranker:
    """
    Evolves a population of ranked recipe lists to maximise the multi-objective
    fitness function, with Simulated Annealing acceptance to escape local optima.
    """

    def __init__(self, config: Optional[GAConfig] = None, seed: int = 42):
        self.cfg       = config or GAConfig()
        self.evaluator = FitnessEvaluator(self.cfg)
        self.seed      = seed
        np.random.seed(seed)

    def evolve(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        k = min(self.cfg.top_k, len(candidates))
        if k == 0:
            return []

        candidates     = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
        population     = self._initialise_population(candidates, k)
        fitness_scores = [self.evaluator.evaluate(c) for c in population]
        best_idx       = int(np.argmax(fitness_scores))
        best_chromosome = copy.deepcopy(population[best_idx])
        best_fitness    = fitness_scores[best_idx]
        elite_n         = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))
        temperature     = self.cfg.sa_initial_temperature

        pbest         = [copy.deepcopy(c) for c in population]
        pbest_fitness = list(fitness_scores)
        gbest         = copy.deepcopy(best_chromosome)
        inertia       = self.cfg.pso_inertia

        for _generation in range(self.cfg.num_generations):
            paired = sorted(
                zip(fitness_scores, population, pbest, pbest_fitness),
                key=lambda x: x[0], reverse=True,
            )
            fitness_scores = [p[0] for p in paired]
            population     = [p[1] for p in paired]
            pbest          = [p[2] for p in paired]
            pbest_fitness  = [p[3] for p in paired]

            if fitness_scores[0] > best_fitness:
                best_fitness    = fitness_scores[0]
                best_chromosome = copy.deepcopy(population[0])
                gbest           = copy.deepcopy(population[0])

            next_gen           = list(population[:elite_n])
            next_gen_fitness   = list(fitness_scores[:elite_n])
            next_pbest         = list(pbest[:elite_n])
            next_pbest_fitness = list(pbest_fitness[:elite_n])

            while len(next_gen) < self.cfg.population_size:
                pa = self._tournament_select(population, fitness_scores)
                pb = self._tournament_select(population, fitness_scores)

                if random.random() < self.cfg.crossover_rate:
                    child = self._crossover(pa, pb, candidates)
                else:
                    child = copy.deepcopy(pa)

                child_fitness = self.evaluator.evaluate(child)

                idx_in_pop = len(next_gen)
                if random.random() < self.cfg.mutation_rate:
                    pb_ref = (pbest[idx_in_pop]
                              if idx_in_pop < len(pbest) else None)
                    mutated = self._pso_mutate(
                        child, candidates, pb_ref, gbest, inertia,
                    )
                    mutated_fitness = self.evaluator.evaluate(mutated)
                    if _sa_accept(mutated_fitness, child_fitness, temperature):
                        child         = mutated
                        child_fitness = mutated_fitness

                next_gen.append(child)
                next_gen_fitness.append(child_fitness)

                if idx_in_pop < len(pbest_fitness) and child_fitness > pbest_fitness[idx_in_pop]:
                    next_pbest.append(copy.deepcopy(child))
                    next_pbest_fitness.append(child_fitness)
                elif idx_in_pop < len(pbest):
                    next_pbest.append(pbest[idx_in_pop])
                    next_pbest_fitness.append(pbest_fitness[idx_in_pop])
                else:
                    next_pbest.append(copy.deepcopy(child))
                    next_pbest_fitness.append(child_fitness)

            population     = next_gen
            fitness_scores = next_gen_fitness
            pbest          = next_pbest
            pbest_fitness  = next_pbest_fitness

            temperature = max(
                self.cfg.sa_min_temperature,
                temperature * self.cfg.sa_cooling_rate,
            )
            inertia = max(0.1, inertia * self.cfg.sa_cooling_rate)

        return best_chromosome

    def _initialise_population(self, candidates: List[CandidateItem], k: int) -> List[List[CandidateItem]]:
        population = [copy.deepcopy(candidates[:k])]
        pool_size  = len(candidates)
        for _ in range(self.cfg.population_size - 1):
            population.append(copy.deepcopy(random.sample(candidates, min(k, pool_size))))
        return population

    def _tournament_select(
        self,
        population: List[List[CandidateItem]],
        fitness_scores: List[float],
        tournament_size: int = 3,
    ) -> List[CandidateItem]:
        indices  = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return copy.deepcopy(population[best_idx])

    def _crossover(
        self,
        parent_a: List[CandidateItem],
        parent_b: List[CandidateItem],
        candidates: List[CandidateItem],
    ) -> List[CandidateItem]:
        """Order-preserving single-segment crossover."""
        k = len(parent_a)
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

        if len(child) < k:
            for item in parent_a:
                if len(child) == k:
                    break
                child.append(copy.deepcopy(item))
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

    def _pso_mutate(
        self,
        chromosome: List[CandidateItem],
        candidates: List[CandidateItem],
        pbest: List[CandidateItem],
        gbest: List[CandidateItem],
        inertia: float,
    ) -> List[CandidateItem]:
        """
        PSO-guided mutation: bias item replacements toward known good
        solutions instead of pure randomness.

        With decaying probability (controlled by inertia):
          - Cognitive pull: inject an item from this chromosome's personal
            best (pbest) that isn't in the current chromosome.
          - Social pull: inject an item from the global best (gbest)
            that isn't in the current chromosome.
          - Otherwise: fall back to random mutation.

        As inertia decays (alongside SA temperature), the PSO influence
        weakens and the algorithm transitions from directed exploration
        to random fine-tuning — mirroring the PSO inertia schedule.
        """
        chromosome = copy.deepcopy(chromosome)
        r = random.random()
        current_ids = {item.item_id for item in chromosome}

        p_cognitive = inertia * self.cfg.pso_cognitive
        p_social = inertia * self.cfg.pso_social

        if r < p_cognitive and pbest:
            pbest_only = [it for it in pbest if it.item_id not in current_ids]
            if pbest_only:
                inject = random.choice(pbest_only)
                idx = random.randint(0, len(chromosome) - 1)
                chromosome[idx] = copy.deepcopy(inject)
                return chromosome

        if r < p_cognitive + p_social and gbest:
            gbest_only = [it for it in gbest if it.item_id not in current_ids]
            if gbest_only:
                inject = random.choice(gbest_only)
                idx = random.randint(0, len(chromosome) - 1)
                chromosome[idx] = copy.deepcopy(inject)
                return chromosome

        return self._mutate(chromosome, candidates)



def _safe_read_col(row, col: str, available_cols: list, default: float = 0.0) -> float:
    if col not in available_cols or col not in row.index:
        return default
    try:
        val = row[col]
        if not pd.api.types.is_scalar(val):
            try:
                val = val.iloc[0] if hasattr(val, 'iloc') else val.flat[0]
            except (IndexError, StopIteration):
                return default
        fval = float(val)
        return default if fval != fval else fval
    except (TypeError, ValueError, IndexError):
        return default


def build_candidates_from_model_output(
    item_ids: list,
    relevance_scores: list,
    recipes_df,
    user_history_ids: set,
    food_id_map: dict,
    available_nutrition_cols: list,
    max_values: dict,
) -> List[CandidateItem]:
    """Convert raw two-tower model outputs into CandidateItem objects for the SA-GA."""
    reverse_food_map = {v: k for k, v in food_id_map.items()}

    if 'FoodId' in recipes_df.columns:
        recipes_indexed = (
            recipes_df
            .drop_duplicates(subset='FoodId', keep='first')
            .set_index('FoodId')
        )
    else:
        recipes_indexed = recipes_df

    candidates = []
    for iid, score in zip(item_ids, relevance_scores):
        original_food_id = reverse_food_map.get(iid)
        cal = prot = fat = carbs = sugar = fiber = 0.0
        cat_id = 0

        if original_food_id is not None and original_food_id in recipes_indexed.index:
            row    = recipes_indexed.loc[original_food_id]
            cal    = _safe_read_col(row, 'Calories',            available_nutrition_cols)
            prot   = _safe_read_col(row, 'ProteinContent',      available_nutrition_cols)
            fat    = _safe_read_col(row, 'FatContent',          available_nutrition_cols)
            carbs  = _safe_read_col(row, 'CarbohydrateContent', available_nutrition_cols)
            sugar  = _safe_read_col(row, 'SugarContent',        available_nutrition_cols)
            fiber  = _safe_read_col(row, 'FiberContent',        available_nutrition_cols)
            cat_id = int(_safe_read_col(row, 'CategoryID',      available_nutrition_cols, default=0.0))

        candidates.append(CandidateItem(
            item_id         = iid,
            relevance_score = float(score),
            calories        = cal,
            protein         = prot,
            fat             = fat,
            carbs           = carbs,
            sugar           = sugar,
            fiber           = fiber,
            category_id     = cat_id,
            in_user_history = (iid in user_history_ids),
        ))

    return candidates
