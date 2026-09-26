import jax.numpy as np
import jax

def equisample(curves, n):

    l = np.cumsum(np.pad(np.linalg.norm(curves[:,1:,:] - curves[:,:-1,:],axis=-1),((0,0),(1,0))),axis=-1)
    l = l/l[:,-1].reshape(-1,1)

    sampling = np.linspace(-1e-6,1-1e-6,n)
    end_is = jax.vmap(lambda a: np.searchsorted(a.reshape(-1),sampling)[1:])(l)

    end_ids = end_is

    l_end = l[np.arange(end_is.shape[0]).reshape(-1,1),end_is]
    l_start = l[np.arange(end_is.shape[0]).reshape(-1,1),end_is-1]
    ws = (l_end - sampling[1:].reshape(1,-1))/(l_end-l_start)

    end_gather = curves[np.arange(end_ids.shape[0]).reshape(-1,1),end_ids]
    start_gather = curves[np.arange(end_ids.shape[0]).reshape(-1,1),end_ids-1]

    uniform_curves = np.concatenate([curves[:,0:1,:],(end_gather - (end_gather-start_gather)*ws[:,:,None])],1)

    return uniform_curves

def normalize(curves, scale=True):
    
    n = curves.shape[1]
    # center curves
    curves = curves - curves.mean(1)[:,None]
    
    # apply uniform scaling
    if scale:
        s = ((np.square(curves).sum(-1).sum(-1)/n)**0.5)[:,None,None]
        curves = curves/s
    
    return curves

def find_optimal_correspondences(curves,
                                 target_curves,
                                 return_rotation_matrices=True,
                                 return_aligned_curves=False,
                                 return_distances=False):
    n = target_curves.shape[1]
    init_order = np.arange(n)

    base_order_set = init_order[init_order[:,None]-np.zeros_like(init_order)].T
    base_order_set = base_order_set.repeat(2, axis=0)
    permuted_order_set = init_order[init_order[:,None]-init_order].T

    permuted_order_set = np.concatenate([permuted_order_set, np.copy(permuted_order_set[:,::-1])], axis=0)

    permuted_targets = target_curves[
        np.arange(target_curves.shape[0]).repeat(base_order_set.shape[0]).reshape(-1,1),
        base_order_set[None].repeat(target_curves.shape[0], axis=0).reshape(-1,base_order_set.shape[-1])
    ].reshape(-1, base_order_set.shape[0], n, 2)
    
    permuted_candidates = curves[
        np.arange(curves.shape[0]).repeat(permuted_order_set.shape[0]).reshape(-1,1),
        permuted_order_set[None].repeat(curves.shape[0], axis=0).reshape(-1,permuted_order_set.shape[-1])
    ].reshape(-1, permuted_order_set.shape[0], n, 2)
    
    dots = (permuted_targets * permuted_candidates).sum(2).sum(2)
    vars = (permuted_targets[...,1]*permuted_candidates[...,0] - permuted_targets[...,0]*permuted_candidates[...,1]).sum(-1)
    thetas = np.arctan2(vars, dots)
    rotation_matrices = np.stack([np.cos(thetas), -np.sin(thetas), np.sin(thetas), np.cos(thetas)], axis=-1).reshape(-1, permuted_order_set.shape[0], 2, 2)

    rotated_candidates = np.einsum('abij,abkj->abki', rotation_matrices, permuted_candidates)

    d_od = np.linalg.norm(permuted_targets - rotated_candidates, axis=-1).sum(-1)/n * 2 * np.pi

    best_match = np.argmin(d_od, axis=-1)

    best_correspondences = permuted_order_set[best_match]
    best_rotations = rotation_matrices[np.arange(rotation_matrices.shape[0]), best_match]

    if return_rotation_matrices and return_aligned_curves and return_distances:
        return best_correspondences, best_rotations, rotated_candidates[np.arange(rotated_candidates.shape[0]), best_match], d_od.min(axis=-1)
    elif return_rotation_matrices and return_distances:
        return best_correspondences, best_rotations, d_od.min(axis=-1)
    elif return_rotation_matrices and return_aligned_curves:
        return best_correspondences, best_rotations, rotated_candidates[np.arange(rotated_candidates.shape[0]), best_match]
    elif return_distances and return_aligned_curves:
        return best_correspondences, rotated_candidates[np.arange(rotated_candidates.shape[0]), best_match], d_od.min(axis=-1)
    elif return_rotation_matrices:
        return best_correspondences, best_rotations
    elif return_aligned_curves:
        return best_correspondences, rotated_candidates[np.arange(rotated_candidates.shape[0]), best_match]
    elif return_distances:
        return best_correspondences, d_od.min(axis=-1)
    else:
        return best_correspondences