"""
Black Widow Optimization (BWO) Re-Ranker for NutrientPlus
==========================================================
Metaheuristic reranker inspired by black widow spider mating behaviour.
Based on: Hayyolalam & Pourhaji Kazem (2020),
          "Black Widow Optimization Algorithm: A novel meta-heuristic approach
           for solving engineering optimization problems",
          Engineering Applications of Artificial Intelligence, 87, 103249.

Key operators vs Simple GA
──────────────────────────
• Procreation   — each selected pair produces multiple offspring (not just one)
• Sexual cannibalism — the lower-fitness parent is discarded after mating
• Sibling cannibalism — only the top CR% of offspring per pair survive
• Matriphagy    — an offspring replaces its mother if it is fitter

No external dependencies — pure Python / NumPy.
"""

import copy
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from ga_reranker import CandidateItem


@dataclass
class BWOConfig:
    """Hyper-parameters for the Black Widow Optimization reranker."""

    population_size: int   = 50
    num_generations: int   = 40
    top_k:           int   = 10
    candidate_pool_size: int = 50

    procreating_percentage: float = 0.60   # PP — fraction of population selected as parents
    cannibalism_rate:       float = 0.40   # CR — fraction of offspring kept after sibling cannibalism
    mutation_rate:          float = 0.20   # PM — fraction of population mutated each generation

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
        if abs(total - 1.0) > 1e-2:
            raise ValueError(f"Fitness weights must sum to 1.0, got {total:.6f}")
        self.w_relevance /= total
        self.w_nutrition /= total
        self.w_category  /= total
        self.w_novelty   /= total



class BWOFitnessEvaluator:
    def __init__(self, config: BWOConfig):
        self.cfg = config

    def evaluate(self, chromosome: List[CandidateItem]) -> float:
        if not chromosome:
            return 0.0
        return float(np.clip(
            self.cfg.w_relevance * self._relevance(chromosome) +
            self.cfg.w_nutrition * self._nutrition(chromosome) +
            self.cfg.w_category  * self._category(chromosome) +
            self.cfg.w_novelty   * self._novelty(chromosome),
            0.0, 1.0,
        ))

    def _relevance(self, ch: List[CandidateItem]) -> float:
        scores = [item.relevance_score for item in ch]
        dcg    = sum(s / np.log2(r + 2) for r, s in enumerate(scores))
        idcg   = sum(s / np.log2(r + 2) for r, s in enumerate(sorted(scores, reverse=True)))
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
        return (len(cal_hits)  / (len(self.cfg.calorie_bands)  - 1) +
                len(prot_hits) / (len(self.cfg.protein_bands) - 1)) / 2.0

    def _category(self, ch: List[CandidateItem]) -> float:
        return min(len(set(item.category_id for item in ch)) / len(ch), 1.0)

    def _novelty(self, ch: List[CandidateItem]) -> float:
        return sum(1 for item in ch if not item.in_user_history) / len(ch)


class BWOReranker:
    """
    Black Widow Optimization reranker.

    Each solution ("widow") is a ranked list of top_k CandidateItems.
    The four BWO operators are adapted to permutation-based solutions:

    1. Procreation  — selected parent pairs produce ceil(top_k/2) offspring
                      each via order-preserving crossover with randomised alpha
    2. Sexual cannibalism — the lower-fitness parent of each pair is removed
    3. Sibling cannibalism — per pair, only the top CR fraction of offspring survive
    4. Matriphagy   — if the best offspring beats the mother, it replaces her
    5. Mutation     — PM fraction of population: random swap or replace
    """

    def __init__(self, config: Optional[BWOConfig] = None, seed: int = 42):
        self.cfg       = config or BWOConfig()
        self.evaluator = BWOFitnessEvaluator(self.cfg)
        random.seed(seed)
        np.random.seed(seed)

    def evolve(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        k = min(self.cfg.top_k, len(candidates))
        if k == 0:
            return []

        candidates = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
        population = self._init_population(candidates, k)
        fitness    = [self.evaluator.evaluate(c) for c in population]

        best_idx   = int(np.argmax(fitness))
        best_chrom = copy.deepcopy(population[best_idx])
        best_fit   = fitness[best_idx]

        for _ in range(self.cfg.num_generations):
            population, fitness = self._generation(population, fitness, candidates, k)

            gen_best = int(np.argmax(fitness))
            if fitness[gen_best] > best_fit:
                best_fit   = fitness[gen_best]
                best_chrom = copy.deepcopy(population[gen_best])

        return best_chrom

    def _generation(
        self,
        population: List[List[CandidateItem]],
        fitness: List[float],
        candidates: List[CandidateItem],
        k: int,
    ):
        n_parents = max(2, int(self.cfg.procreating_percentage * len(population)))
        if n_parents % 2 != 0:
            n_parents -= 1  # ensure pairs

        # Sort by fitness descending; best widows selected as parents
        paired     = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)
        fitness    = [p[0] for p in paired]
        population = [p[1] for p in paired]

        new_population = []
        new_fitness    = []

        for i in range(0, n_parents, 2):
            mother, mother_fit = population[i],     fitness[i]
            father, father_fit = population[i + 1], fitness[i + 1]

            # Procreation — produce multiple offspring
            n_offspring = max(2, k // 2)
            offspring   = [
                self._crossover(mother, father, candidates, k)
                for _ in range(n_offspring)
            ]
            off_fitness = [self.evaluator.evaluate(c) for c in offspring]

            # Sibling cannibalism — keep top CR fraction of offspring
            n_keep = max(1, int(self.cfg.cannibalism_rate * n_offspring))
            off_pairs  = sorted(zip(off_fitness, offspring), key=lambda x: x[0], reverse=True)
            survivors      = [p[1] for p in off_pairs[:n_keep]]
            surv_fitness   = [p[0] for p in off_pairs[:n_keep]]

            # Sexual cannibalism — discard the lower-fitness parent
            if mother_fit >= father_fit:
                kept_parent     = mother
                kept_parent_fit = mother_fit
            else:
                kept_parent     = father
                kept_parent_fit = father_fit

            # Matriphagy — best offspring replaces mother if fitter
            best_off_fit = surv_fitness[0]
            if best_off_fit > kept_parent_fit:
                new_population.append(survivors[0])
                new_fitness.append(best_off_fit)
            else:
                new_population.append(kept_parent)
                new_fitness.append(kept_parent_fit)

            # Add remaining survivors
            for c, f in zip(survivors[1:], surv_fitness[1:]):
                new_population.append(c)
                new_fitness.append(f)

        # Carry over non-parent widows unchanged
        for i in range(n_parents, len(population)):
            new_population.append(population[i])
            new_fitness.append(fitness[i])

        diversity         = float(np.std(fitness)) if fitness else 0.0
        effective_rate    = self.cfg.mutation_rate * (2.0 if diversity < 0.005 else 1.0)
        n_mutate          = max(1, int(effective_rate * len(new_population)))
        mutate_indices    = random.sample(range(len(new_population)), min(n_mutate, len(new_population)))
        for idx in mutate_indices:
            mutated     = self._mutate(new_population[idx], candidates)
            mutated_fit = self.evaluator.evaluate(mutated)
            if mutated_fit > new_fitness[idx]:
                new_population[idx] = mutated
                new_fitness[idx]    = mutated_fit

        if len(new_population) > self.cfg.population_size:
            pairs = sorted(
                zip(new_fitness, new_population), key=lambda x: x[0], reverse=True
            )[:self.cfg.population_size]
            new_fitness    = [p[0] for p in pairs]
            new_population = [p[1] for p in pairs]
        elif len(new_population) < self.cfg.population_size:
            while len(new_population) < self.cfg.population_size:
                new_population.append(
                    copy.deepcopy(random.sample(candidates, min(k, len(candidates))))
                )
                new_fitness.append(self.evaluator.evaluate(new_population[-1]))

        return new_population, new_fitness

    def _init_population(
        self, candidates: List[CandidateItem], k: int
    ) -> List[List[CandidateItem]]:
        pop = [copy.deepcopy(candidates[:k])]
        for _ in range(self.cfg.population_size - 1):
            pop.append(copy.deepcopy(random.sample(candidates, min(k, len(candidates)))))
        return pop

    def _crossover(
        self,
        parent_a: List[CandidateItem],
        parent_b: List[CandidateItem],
        candidates: List[CandidateItem],
        k: int,
    ) -> List[CandidateItem]:
        """
        BWO procreation crossover: randomised alpha selects a contiguous
        segment from parent_a; remainder filled from parent_b then candidates.
        Re-randomised per offspring call to generate diverse children.
        """
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
        """BWO mutation: randomly exchange two elements (swap) or replace one."""
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
