import jax.numpy as jnp
import numpy as np
import jax
from typing import Optional, Union, List
from ._utils import *
from ..Kinematics._solvers import dyadic_solve
from ..Geometry._utils import normalize, find_optimal_correspondences, equisample
from ..Kinematics._kin import sort_mechanism

def unscaled_distance(x0s_,
             As_,
             node_types_,
             thetas_,
             target_idx,
             target_curves):
    
    solutions = dyadic_solve(
        As_,x0s_,node_types_,thetas_
    )
    
    n = thetas_.shape[0]
    solved_candidates = solutions[jnp.arange(solutions.shape[0]), target_idx]
    
    has_nans = jnp.isnan(solutions).any(axis=-1).any(axis=-1).any(axis=-1)
    solved_candidates_safe = jnp.nan_to_num(solved_candidates * (~has_nans)[:,None,None], nan=0.0) + target_curves * has_nans[:,None,None]

    normalized_targets = equisample(target_curves, n)
    normalized_candidates = equisample(solved_candidates_safe, n)

    normalized_targets = normalize(normalized_targets, scale=False)
    normalized_candidates = normalize(normalized_candidates, scale=False)
    
    s_target = ((jnp.square(normalized_targets).sum(-1).sum(-1)/n)**0.5)[:,None,None]
    normalized_targets = normalized_targets / s_target
    normalized_candidates = normalized_candidates / s_target
    
    # check if any nans
    has_nans_post_norm = jnp.isnan(normalized_candidates).any(axis=-1).any(axis=-1)
    normalized_candidates = jnp.nan_to_num(normalized_candidates * (~has_nans_post_norm)[:,None,None], nan=0.0) + normalized_targets * has_nans_post_norm[:,None,None]
    
    has_nans = jnp.logical_or(has_nans, has_nans_post_norm)

    optimal_correspondences, distances = find_optimal_correspondences(normalized_candidates,
                                                                      normalized_targets,
                                                                      return_rotation_matrices=False,
                                                                      return_aligned_curves=False,
                                                                      return_distances=True)
    
    
    obj = distances.sum()
    d_final = distances * (~has_nans) + has_nans * jnp.inf

    return obj, d_final

def scaled_distance(x0s_,
             As_,
             node_types_,
             thetas_,
             target_idx,
             target_curves):
    
    solutions = dyadic_solve(
        As_,x0s_,node_types_,thetas_
    )
    
    n = thetas_.shape[0]
    solved_candidates = solutions[jnp.arange(solutions.shape[0]), target_idx]
    
    has_nans = jnp.isnan(solutions).any(axis=-1).any(axis=-1).any(axis=-1)
    solved_candidates_safe = jnp.nan_to_num(solved_candidates * (~has_nans)[:,None,None], nan=0.0) + target_curves * has_nans[:,None,None]

    normalized_targets = equisample(target_curves, n)
    normalized_candidates = equisample(solved_candidates_safe, n)

    normalized_targets = normalize(normalized_targets, scale=True)
    normalized_candidates = normalize(normalized_candidates, scale=True)
    
    # check if any nans
    has_nans_post_norm = jnp.isnan(normalized_candidates).any(axis=-1).any(axis=-1)
    normalized_candidates = jnp.nan_to_num(normalized_candidates * (~has_nans_post_norm)[:,None,None], nan=0.0) + normalized_targets * has_nans_post_norm[:,None,None]
    
    has_nans = jnp.logical_or(has_nans, has_nans_post_norm)

    optimal_correspondences, distances = find_optimal_correspondences(normalized_candidates,
                                                                      normalized_targets,
                                                                      return_rotation_matrices=False,
                                                                      return_aligned_curves=False,
                                                                      return_distances=True)
    
    
    obj = distances.sum()
    d_final = distances * (~has_nans) + has_nans * jnp.inf

    return obj, d_final

def material(x0s_, As_):
    Gs = jnp.square((jnp.expand_dims(x0s_, 1) - jnp.expand_dims(x0s_, 2))).sum(-1)
    material = jnp.sqrt(jnp.where(As_>0, Gs, 0.0)).sum(-1).sum(-1)
    
    obj = material.sum()
    return obj, material

differentiated_distance = jax.value_and_grad(unscaled_distance, has_aux=True)
differentiated_distance_scaled = jax.value_and_grad(scaled_distance, has_aux=True)
differentiated_material = jax.value_and_grad(material, has_aux=True)