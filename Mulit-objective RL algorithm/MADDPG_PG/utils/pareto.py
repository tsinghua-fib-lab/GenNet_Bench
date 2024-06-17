
from copy import deepcopy
from typing import List, Union

import numpy as np
from scipy.spatial import ConvexHull
from pymoo.indicators.hv import HV


def hypervolume(ref_point: np.ndarray, points: List[np.ndarray]) -> float:
    
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def sparsity(front: List[np.ndarray]) -> float:
    
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value



def get_non_pareto_dominated_inds(candidates: Union[np.ndarray, List], remove_duplicates: bool = True) -> np.ndarray:
    
    candidates = np.array(candidates)
    uniques, indcs, invs, counts = np.unique(candidates, return_index=True, return_inverse=True, return_counts=True, axis=0)

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)

    return np.logical_and(c1, c2) & to_keep


def filter_pareto_dominated(candidates: Union[np.ndarray, List], remove_duplicates: bool = True) -> np.ndarray:
    
    candidates = np.array(candidates)
    if len(candidates) < 2:
        return candidates
    return candidates[get_non_pareto_dominated_inds(candidates, remove_duplicates=remove_duplicates)]


def filter_convex_dominated(candidates: Union[np.ndarray, List]) -> np.ndarray:
    
    candidates = np.array(candidates)
    if len(candidates) > 2:
        hull = ConvexHull(candidates)
        ccs = candidates[hull.vertices]
    else:
        ccs = candidates
    return filter_pareto_dominated(ccs)


def get_non_dominated(candidates: set) -> set:
    
    candidates = np.array(list(candidates))  
    candidates = candidates[candidates.sum(1).argsort()[::-1]]  
    for i in range(candidates.shape[0]):  
        n = candidates.shape[0]  
        if i >= n:  
            break
        non_dominated = np.ones(candidates.shape[0], dtype=bool)  
        
        
        
        non_dominated[i + 1 :] = np.any(candidates[i + 1 :] > candidates[i], axis=1)
        candidates = candidates[non_dominated]  

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(tuple(candidate))  

    return non_dominated


def get_non_dominated_inds(solutions: np.ndarray) -> np.ndarray:
    
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            
            is_efficient[i] = 1
    return is_efficient


class ParetoArchive:
    

    def __init__(self, convex_hull: bool = False):
        
        self.convex_hull = convex_hull
        self.individuals: list = []
        self.evaluations: List[np.ndarray] = []

    def add(self, candidate, evaluation: np.ndarray):
        
        self.evaluations.append(evaluation)
        self.individuals.append(deepcopy(candidate))

        
        if self.convex_hull:
            nd_candidates = {tuple(x) for x in filter_convex_dominated(self.evaluations)}
        else:
            nd_candidates = {tuple(x) for x in filter_pareto_dominated(self.evaluations)}

        
        non_dominated_evals = []
        non_dominated_evals_tuples = []
        non_dominated_individuals = []
        for e, i in zip(self.evaluations, self.individuals):
            if tuple(e) in nd_candidates and tuple(e) not in non_dominated_evals_tuples:
                non_dominated_evals.append(e)
                non_dominated_evals_tuples.append(tuple(e))
                non_dominated_individuals.append(i)
        self.evaluations = non_dominated_evals
        self.individuals = non_dominated_individuals
