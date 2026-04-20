"""
MOPSO Re-Ranker for NutrientPlus
==================================
Multi-Objective Particle Swarm Optimization reranker.

Based on: Coello Coello, Pulido & Lechuga (2004),
          "Handling Multiple Objectives with Particle Swarm Optimization",
          IEEE Transactions on Evolutionary Computation, 8(3), 256-279.

--------------------------------------------------------------------
Key features of MOPSO:

1. External Pareto archive
   All non-dominated solutions found by ANY particle across ALL generations
   are stored in a shared archive.  When the archive is full, the most
   crowded solutions are pruned to maintain diversity.

2. Leader selection from archive
   Each particle is guided by a *leader* drawn from the archive using a
   binary tournament that prefers archive members with a high crowding
   distance (i.e. solutions in under-explored regions of the Pareto front).
   This is the key diversity mechanism — different particles can follow
   different leaders, spreading the swarm across the full front.

3. Personal best by Pareto dominance
   A particle's personal best is updated whenever the new position
   dominates the old pbest.  When neither dominates the other, the pbest
   is replaced with 50 % probability (avoids locking in a single region).

4. Final selection
   The archive (not just the final population) is the result.  The returned
   solution is the archive member closest to the ideal point [1,1,1,1]
   using AHP-weighted Euclidean distance — consistent with the fixed
   _select_final in nsga3_reranker.py.

Discrete adaptation (permutation-based positions)
--------------------------------------------------
For permutation-based solutions
the velocity is adapted as a per-position probability in [0, 1]:

    vel[j]  ← clip( w_inertia·vel[j]
                    + c1·r1·Δpbest[j]
                    + c2·r2·Δleader[j], 0, 1 )

where Δpbest[j] = 1 if position j differs from pbest, else 0 (and
similarly for the leader).  A high vel[j] means position j is strongly
attracted toward the pbest/leader item — with probability vel[j] that
position is updated by adopting the suggested item (swap if already
present, otherwise replace).

"""

import copy
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ga_reranker import CandidateItem


@dataclass
class MOPSOConfig:
    """Hyper-parameters for the MOPSO reranker."""

    population_size: int   = 60
    num_generations: int   = 20
    top_k:           int   = 10
    candidate_pool_size: int = 80

    archive_size:    int   = 100   
    w_inertia:       float = 0.72 
    c1:              float = 1.50  
    c2:              float = 1.50  
    mutation_rate:   float = 0.10  

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
        if abs(total - 1.0) > 1e-2:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
        self.w_relevance /= total
        self.w_nutrition /= total
        self.w_category  /= total
        self.w_novelty   /= total


class MOPSOObjectiveEvaluator:
    """
    Evaluates a chromosome → np.ndarray of shape (4,):
        [relevance, nutritional_diversity, category_diversity, novelty]
    All values in [0, 1].  Higher is better for every objective.
    """

    def __init__(self, config: MOPSOConfig):
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

    def _relevance(self, ch: List[CandidateItem]) -> float:
        scores = [item.relevance_score for item in ch]
        dcg  = sum(s / np.log2(r + 2) for r, s in enumerate(scores))
        idcg = sum(s / np.log2(r + 2) for r, s in enumerate(sorted(scores, reverse=True)))
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


def _mopso_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if solution a dominates solution b."""
    return bool(np.all(a >= b) and np.any(a > b))


def _mopso_crowding_distances(obj_matrix: np.ndarray) -> np.ndarray:
    """
    Crowding distance for a set of solutions (same algorithm as NSGA-III).
    Returns a 1-D array; boundary solutions receive np.inf.
    """
    n, M  = obj_matrix.shape
    dist  = np.zeros(n)
    for m in range(M):
        order  = np.argsort(obj_matrix[:, m])
        lo, hi = obj_matrix[order[0], m], obj_matrix[order[-1], m]
        span   = hi - lo
        dist[order[0]]  = np.inf
        dist[order[-1]] = np.inf
        if span < 1e-12:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (
                obj_matrix[order[i + 1], m] - obj_matrix[order[i - 1], m]
            ) / span
    return dist


class MOPSOReranker:
    """
    Multi-Objective PSO reranker for food recommendation lists.

    Each particle is a ranked list of top_k CandidateItems.
    The swarm is guided by personal bests and leaders drawn from
    the shared Pareto archive.  At the end, the archive member
    closest to [1,1,1,1] under AHP-weighted distance is returned.
    """

    def __init__(self, config: Optional[MOPSOConfig] = None, seed: int = 42):
        self.cfg       = config or MOPSOConfig()
        self.evaluator = MOPSOObjectiveEvaluator(self.cfg)
        random.seed(seed)
        np.random.seed(seed)

    def evolve(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        k = min(self.cfg.top_k, len(candidates))
        if k == 0:
            return []

        candidates = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)

        particles  = self._init_particles(candidates, k)
        velocities = [np.full(k, 0.5) for _ in particles] 

        objectives = [self.evaluator.evaluate(p) for p in particles]

        pbest      = [copy.deepcopy(p) for p in particles]
        pbest_obj  = [o.copy() for o in objectives]

        self.archive     = []   
        self.archive_obj = []   
        for p, o in zip(particles, objectives):
            self._update_archive(p, o)

        for gen in range(self.cfg.num_generations):
            for i in range(len(particles)):
                leader, _ = self._select_leader()

                velocities[i] = self._update_velocity(
                    velocities[i], particles[i], pbest[i], leader, k,
                )

                new_particle = self._update_position(
                    particles[i], velocities[i], pbest[i], leader, candidates, k,
                )

                if random.random() < self.cfg.mutation_rate:
                    new_particle = self._mutate(new_particle, candidates)

                new_obj = self.evaluator.evaluate(new_particle)

                if _mopso_dominates(new_obj, pbest_obj[i]):
                    pbest[i]     = copy.deepcopy(new_particle)
                    pbest_obj[i] = new_obj.copy()
                elif not _mopso_dominates(pbest_obj[i], new_obj):
                    if random.random() < 0.5:
                        pbest[i]     = copy.deepcopy(new_particle)
                        pbest_obj[i] = new_obj.copy()

                particles[i]  = new_particle
                objectives[i] = new_obj
                self._update_archive(new_particle, new_obj)

        return self._select_final()


    def _update_archive(
        self,
        chromosome: List[CandidateItem],
        obj: np.ndarray,
    ) -> None:
        """
        Add a solution to the archive if it is not dominated by any existing
        member.  Remove archive members that are dominated by the new solution.
        If the archive exceeds its capacity, prune the most crowded member.
        """
        for arc_obj in self.archive_obj:
            if _mopso_dominates(arc_obj, obj):
                return

        keep = [
            (c, o) for c, o in zip(self.archive, self.archive_obj)
            if not _mopso_dominates(obj, o)
        ]
        self.archive     = [x[0] for x in keep]
        self.archive_obj = [x[1] for x in keep]

        self.archive.append(copy.deepcopy(chromosome))
        self.archive_obj.append(obj.copy())

        if len(self.archive) > self.cfg.archive_size:
            arc_mat = np.array(self.archive_obj)
            dists   = _mopso_crowding_distances(arc_mat)
            finite  = [(d, i) for i, d in enumerate(dists) if np.isfinite(d)]
            if finite:
                victim = min(finite, key=lambda x: x[0])[1]
                self.archive.pop(victim)
                self.archive_obj.pop(victim)

    def _select_leader(self) -> Tuple[List[CandidateItem], np.ndarray]:
        """
        Select a leader from the archive using a binary tournament that
        prefers archive members with higher crowding distance (less crowded
        regions of the Pareto front → promotes swarm diversity).
        """
        if len(self.archive) == 1:
            return self.archive[0], self.archive_obj[0]

        arc_mat = np.array(self.archive_obj)
        dists   = _mopso_crowding_distances(arc_mat)

        a, b = random.sample(range(len(self.archive)), 2)
        winner = a if dists[a] >= dists[b] else b
        return self.archive[winner], self.archive_obj[winner]


    def _update_velocity(
        self,
        velocity: np.ndarray,
        particle: List[CandidateItem],
        pbest:    List[CandidateItem],
        leader:   List[CandidateItem],
        k: int,
    ) -> np.ndarray:
        """
        Discrete velocity update.

        For each position j the velocity measures how much the particle
        wants to change that position.  The pull toward pbest and leader
        is 1 where they differ from the current particle, 0 where they agree.
        """
        r1 = np.random.random(k)
        r2 = np.random.random(k)

        pbest_diff  = np.array([
            0.0 if particle[j].item_id == pbest[j].item_id  else 1.0
            for j in range(k)
        ])
        leader_diff = np.array([
            0.0 if particle[j].item_id == leader[j].item_id else 1.0
            for j in range(k)
        ])

        new_vel = (self.cfg.w_inertia * velocity
                   + self.cfg.c1 * r1 * pbest_diff
                   + self.cfg.c2 * r2 * leader_diff)
        return np.clip(new_vel, 0.0, 1.0)

    def _update_position(
        self,
        particle:  List[CandidateItem],
        velocity:  np.ndarray,
        pbest:     List[CandidateItem],
        leader:    List[CandidateItem],
        candidates: List[CandidateItem],
        k: int,
    ) -> List[CandidateItem]:
        """
        Discrete position update.

        For each position j, with probability vel[j]:
          - With 50 % chance adopt pbest's item at position j
          - With 50 % chance adopt leader's item at position j
        If the suggested item is already in the particle at another position,
        the two positions are swapped (preserving the permutation property).
        """
        new_particle = list(copy.deepcopy(particle))
        id_to_pos    = {item.item_id: j for j, item in enumerate(new_particle)}

        for j in range(k):
            if random.random() >= velocity[j]:
                continue

            source      = pbest if random.random() < 0.5 else leader
            target_item = source[j]

            if target_item.item_id == new_particle[j].item_id:
                continue  

            if target_item.item_id in id_to_pos:
                pos_t = id_to_pos[target_item.item_id]
                new_particle[j], new_particle[pos_t] = \
                    new_particle[pos_t], new_particle[j]
                id_to_pos[new_particle[j].item_id]   = j
                id_to_pos[new_particle[pos_t].item_id] = pos_t
            else:
                old_item        = new_particle[j]
                new_particle[j] = copy.deepcopy(target_item)
                del id_to_pos[old_item.item_id]
                id_to_pos[target_item.item_id] = j

        return new_particle

    def _select_final(self) -> List[CandidateItem]:
        """
        Return the archive member closest to the ideal point [1,1,1,1]
        using AHP-weighted Euclidean distance — consistent with NSGA-III's
        _select_final so both multi-objective algorithms are compared fairly.
        """
        if not self.archive:
            return []

        arc_mat = np.array(self.archive_obj)
        ideal   = np.ones(arc_mat.shape[1])
        w       = np.array([
            self.cfg.w_relevance, self.cfg.w_nutrition,
            self.cfg.w_category,  self.cfg.w_novelty,
        ])
        dists    = [np.linalg.norm((arc_mat[i] - ideal) * w)
                    for i in range(len(self.archive))]
        best_idx = int(np.argmin(dists))
        return self.archive[best_idx]


    def _init_particles(
        self, candidates: List[CandidateItem], k: int
    ) -> List[List[CandidateItem]]:
        pop = [copy.deepcopy(candidates[:k])]
        for _ in range(self.cfg.population_size - 1):
            pop.append(copy.deepcopy(random.sample(candidates, min(k, len(candidates)))))
        return pop

    def _mutate(
        self,
        chromosome: List[CandidateItem],
        candidates: List[CandidateItem],
    ) -> List[CandidateItem]:
        """Random swap or replace — same as other rerankers for consistency."""
        chromosome = copy.deepcopy(chromosome)
        if random.choice(['swap', 'replace']) == 'swap' and len(chromosome) >= 2:
            i, j = random.sample(range(len(chromosome)), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        else:
            ids     = {item.item_id for item in chromosome}
            outside = [c for c in candidates if c.item_id not in ids]
            if outside:
                chromosome[random.randint(0, len(chromosome) - 1)] = \
                    copy.deepcopy(random.choice(outside))
        return chromosome
