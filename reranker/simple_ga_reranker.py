"""
Simple GA Re-Ranker for NutrientPlus
=====================================
A baseline Genetic Algorithm reranker WITHOUT:
  - Simulated Annealing acceptance (strictly elitist — only keep improvements)
  - PSO-guided mutation (purely random swap / replace)
  - AHP weight derivation (uses fixed default weights)

Used as a fair baseline to measure how much the SA + PSO + AHP extensions
actually contribute on top of a vanilla GA.
"""

import random
import numpy as np
import pandas as pd
import copy
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimpleGAConfig:
    """Hyper-parameters for the simple (vanilla) GA — no SA, no PSO."""
    population_size: int = 50
    num_generations: int = 40
    top_k: int = 10
    candidate_pool_size: int = 50

    mutation_rate: float = 0.25
    crossover_rate: float = 0.70
    elite_fraction: float = 0.10

    w_relevance: float = 0.50
    w_nutrition: float = 0.25
    w_category:  float = 0.15
    w_novelty:   float = 0.10

    calorie_bands: List[float] = field(
        default_factory=lambda: [0, 200, 400, 600, 800, float('inf')]
    )
    protein_bands: List[float] = field(
        default_factory=lambda: [0, 10, 20, 35, float('inf')]
    )

    def __post_init__(self):
        total = self.w_relevance + self.w_nutrition + self.w_category + self.w_novelty
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Fitness weights must sum to 1.0, got {total:.6f}")


@dataclass
class CandidateItem:
    """Mirrors ga_reranker.CandidateItem so both rerankers share the same objects."""
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


class SimpleFitnessEvaluator:
    """Same multi-objective fitness as the SA-GA evaluator."""

    def __init__(self, config: SimpleGAConfig):
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
        dcg    = sum(s / np.log2(r + 2) for r, s in enumerate(scores))
        idcg   = sum(s / np.log2(r + 2) for r, s in enumerate(sorted(scores, reverse=True)))
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


class SimpleGeneticReranker:
    """
    Vanilla GA reranker — strictly elitist, purely random mutation.

    Differences from GeneticReranker (sa_ga + pso):
      - No SA: worse children are always discarded (greedy hill-climbing only)
      - No PSO: mutation is always random swap-or-replace, never guided by
                personal-best or global-best memories
      - No AHP: fitness weights are the fixed defaults in SimpleGAConfig
    """

    def __init__(self, config: Optional[SimpleGAConfig] = None, seed: int = 42):
        self.cfg       = config or SimpleGAConfig()
        self.evaluator = SimpleFitnessEvaluator(self.cfg)
        random.seed(seed)
        np.random.seed(seed)

    def evolve(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        k = min(self.cfg.top_k, len(candidates))
        if k == 0:
            return []

        candidates  = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
        population  = self._init_population(candidates, k)
        fitness     = [self.evaluator.evaluate(c) for c in population]
        best_idx    = int(np.argmax(fitness))
        best_chrom  = copy.deepcopy(population[best_idx])
        best_fit    = fitness[best_idx]
        elite_n     = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))

        for _ in range(self.cfg.num_generations):
            paired     = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)
            fitness    = [p[0] for p in paired]
            population = [p[1] for p in paired]

            if fitness[0] > best_fit:
                best_fit   = fitness[0]
                best_chrom = copy.deepcopy(population[0])

            next_gen     = list(population[:elite_n])
            next_fitness = list(fitness[:elite_n])

            while len(next_gen) < self.cfg.population_size:
                pa = self._tournament_select(population, fitness)
                pb = self._tournament_select(population, fitness)

                if random.random() < self.cfg.crossover_rate:
                    child = self._crossover(pa, pb, candidates)
                else:
                    child = copy.deepcopy(pa)

                child_fit = self.evaluator.evaluate(child)

                if random.random() < self.cfg.mutation_rate:
                    mutated     = self._mutate(child, candidates)
                    mutated_fit = self.evaluator.evaluate(mutated)
                    if mutated_fit > child_fit:          
                        child     = mutated
                        child_fit = mutated_fit

                next_gen.append(child)
                next_fitness.append(child_fit)

            population = next_gen
            fitness    = next_fitness

        return best_chrom


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
        fitness: List[float],
        tournament_size: int = 3,
    ) -> List[CandidateItem]:
        indices  = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitness[i])
        return copy.deepcopy(population[best_idx])

    def _crossover(
        self,
        parent_a: List[CandidateItem],
        parent_b: List[CandidateItem],
        candidates: List[CandidateItem],
    ) -> List[CandidateItem]:
        """Order-preserving single-segment crossover — identical to SA-GA."""
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
        return child[:k]

    def _mutate(
        self,
        chromosome: List[CandidateItem],
        candidates: List[CandidateItem],
    ) -> List[CandidateItem]:
        """Random mutation: swap two positions or replace one item with an outsider."""
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
