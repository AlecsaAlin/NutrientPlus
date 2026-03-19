import numpy as np
import random
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
import copy


@dataclass
class GAConfig:
    """All hyper-parameters for the genetic algorithm."""
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
    """Evaluates a chromosome (ranked list of CandidateItems) and returns a scalar fitness in [0, 1]."""

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
        dcg    = sum(s / np.log2(r + 1) for r, s in enumerate(scores, start=1))
        idcg   = sum(s / np.log2(r + 1) for r, s in enumerate(sorted(scores, reverse=True), start=1))
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


class GeneticReranker:
    """Evolves a population of ranked recipe lists to maximise the multi-objective fitness function."""

    def __init__(self, config: Optional[GAConfig] = None):
        self.cfg       = config or GAConfig()
        self.evaluator = FitnessEvaluator(self.cfg)

    def evolve(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        k = min(self.cfg.top_k, len(candidates))
        if k == 0:
            return []

        candidates      = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
        population      = self._initialise_population(candidates, k)
        fitness_scores  = [self.evaluator.evaluate(c) for c in population]
        best_chromosome = copy.deepcopy(population[int(np.argmax(fitness_scores))])
        best_fitness    = max(fitness_scores)
        elite_n         = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))

        for _ in range(self.cfg.num_generations):
            paired = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
            fitness_scores, population = zip(*paired)
            fitness_scores = list(fitness_scores)
            population     = list(population)

            if fitness_scores[0] > best_fitness:
                best_fitness    = fitness_scores[0]
                best_chromosome = copy.deepcopy(population[0])

            next_gen = list(population[:elite_n])
            while len(next_gen) < self.cfg.population_size:
                pa = self._tournament_select(population, fitness_scores)
                pb = self._tournament_select(population, fitness_scores)
                child = self._crossover(pa, pb, candidates) if random.random() < self.cfg.crossover_rate else copy.deepcopy(pa)
                if random.random() < self.cfg.mutation_rate:
                    child = self._mutate(child, candidates)
                next_gen.append(child)

            population     = next_gen
            fitness_scores = [self.evaluator.evaluate(c) for c in population]

        return best_chromosome

    def _initialise_population(self, candidates, k):
        population = [copy.deepcopy(candidates[:k])]
        pool_size  = len(candidates)
        for _ in range(self.cfg.population_size - 1):
            population.append(copy.deepcopy(random.sample(candidates, min(k, pool_size))))
        return population

    def _tournament_select(self, population, fitness_scores, tournament_size=3):
        indices  = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return copy.deepcopy(population[best_idx])

    def _crossover(self, parent_a, parent_b, candidates):
        k = len(parent_a)
        if k < 2:
            return copy.deepcopy(parent_a)
        start = random.randint(0, k - 2)
        end   = random.randint(start + 1, k - 1)
        child = list(parent_a[start:end + 1])
        ids   = {item.item_id for item in child}
        for item in parent_b:
            if item.item_id not in ids:
                child.append(item); ids.add(item.item_id)
            if len(child) == k:
                break
        if len(child) < k:
            for item in candidates:
                if item.item_id not in ids:
                    child.append(item); ids.add(item.item_id)
                if len(child) == k:
                    break
        return child[:k]

    def _mutate(self, chromosome, candidates):
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
    """Convert raw two-tower model outputs into CandidateItem objects ready for the GA."""
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
