"""
Pareto Laboratory Module: Multi-Objective Molecular Optimization.
Implements non-dominated sorting, Pareto frontier calculation,
trade-off visualization coordinates, and candidate ranking.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class ParetoLaboratory:
    def __init__(self):
        pass

    @staticmethod
    def dominates(cand_a: np.ndarray, cand_b: np.ndarray, maximize_mask: np.ndarray) -> bool:
        """
        Check if cand_a Pareto-dominates cand_b.
        maximize_mask: boolean array, True if objective is to be maximized, False if minimized.
        """
        # Convert all to maximization by multiplying minimization objectives by -1
        a_adj = np.where(maximize_mask, cand_a, -cand_a)
        b_adj = np.where(maximize_mask, cand_b, -cand_b)

        greater_or_equal = np.all(a_adj >= b_adj - 1e-6)
        strictly_greater = np.any(a_adj > b_adj + 1e-6)
        return bool(greater_or_equal and strictly_greater)

    def compute_pareto_front(
        self,
        candidates: List[Dict[str, Any]],
        objectives: List[str],
        maximize: List[bool]
    ) -> List[Dict[str, Any]]:
        """
        Perform non-dominated sorting on candidates.
        Adds 'is_pareto_optimal', 'pareto_rank', and 'composite_score' to each candidate.
        """
        n = len(candidates)
        if n == 0:
            return []

        # Extract objective matrix
        obj_matrix = np.zeros((n, len(objectives)), dtype=np.float64)
        for i, c in enumerate(candidates):
            for j, obj_name in enumerate(objectives):
                obj_matrix[i, j] = float(c.get(obj_name, 0.0))

        maximize_mask = np.array(maximize, dtype=bool)

        # Non-dominated sorting
        domination_counts = np.zeros(n, dtype=int)
        dominated_sets = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates(obj_matrix[i], obj_matrix[j], maximize_mask):
                    dominated_sets[i].append(j)
                    domination_counts[j] += 1
                elif self.dominates(obj_matrix[j], obj_matrix[i], maximize_mask):
                    dominated_sets[j].append(i)
                    domination_counts[i] += 1

        # Ranks assignment
        current_front = [i for i in range(n) if domination_counts[i] == 0]
        rank = 1
        pareto_ranks = np.zeros(n, dtype=int)

        while current_front:
            next_front = []
            for i in current_front:
                pareto_ranks[i] = rank
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            rank += 1
            current_front = next_front

        # Normalized composite score (0 to 100)
        norm_matrix = np.zeros_like(obj_matrix)
        for j in range(len(objectives)):
            col = obj_matrix[:, j]
            c_min, c_max = np.min(col), np.max(col)
            spread = max(c_max - c_min, 1e-6)
            if maximize[j]:
                norm_matrix[:, j] = (col - c_min) / spread
            else:
                norm_matrix[:, j] = (c_max - col) / spread
        
        composite_scores = np.mean(norm_matrix, axis=1) * 100.0

        results = []
        for i, c in enumerate(candidates):
            c_copy = dict(c)
            c_copy["is_pareto_optimal"] = bool(pareto_ranks[i] == 1)
            c_copy["pareto_rank"] = int(pareto_ranks[i])
            c_copy["composite_score"] = round(float(composite_scores[i]), 1)
            results.append(c_copy)

        # Sort: Pareto front first, then by composite score descending
        results.sort(key=lambda x: (x["pareto_rank"], -x["composite_score"]))
        return results
