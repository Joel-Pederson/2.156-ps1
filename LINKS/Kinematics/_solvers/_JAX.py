import jax.numpy as np

def solve_rev_vectorized_batch(As,x0s,node_types,thetas):
    
    Gs = np.square((np.expand_dims(x0s,1) - np.expand_dims(x0s,2))).sum(-1)
    
    x = np.zeros([x0s.shape[0],x0s.shape[1],thetas.shape[0],2])
    
    x = x + np.expand_dims(node_types * x0s,2)
    
    m = x[:,0] + np.tile(np.expand_dims(np.swapaxes(np.concatenate([np.expand_dims(np.cos(thetas),0),np.expand_dims(np.sin(thetas),0)],0),0,1),0),[x0s.shape[0],1,1]) * np.expand_dims(np.expand_dims(np.sqrt(Gs[:,0,1]),-1),-1)

    m = np.expand_dims(m,1)
    m = np.pad(m,[[0,0],[1,x0s.shape[1]-2],[0,0],[0,0]],mode='constant')
    x += m

    for k in range(3,x0s.shape[1]):
        
        inds = np.argsort(As[:,k,0:k])[:,-2:]
        
        l_ijs = np.linalg.norm(x[np.arange(x0s.shape[0]),inds[:,0]] - x[np.arange(x0s.shape[0]),inds[:,1]], axis=-1)
        
        gik = np.sqrt(np.expand_dims(Gs[np.arange(x0s.shape[0]),inds[:,0],np.ones(shape=[x0s.shape[0]],dtype=int)*k],-1))
        gjk = np.sqrt(np.expand_dims(Gs[np.arange(x0s.shape[0]),inds[:,1],np.ones(shape=[x0s.shape[0]],dtype=int)*k],-1))
        
        cosphis = (np.square(l_ijs) + np.square(gik) - np.square(gjk))/(2 * l_ijs * gik)
        
        cosphis = np.where(np.tile(node_types[:,k],[1,thetas.shape[0]])==0.0,cosphis,np.zeros_like(cosphis))
                             
        x0i1 = x0s[np.arange(x0s.shape[0]),inds[:,0],np.ones(shape=[x0s.shape[0]]).astype(np.int32)]
        x0i0 = x0s[np.arange(x0s.shape[0]),inds[:,0],np.zeros(shape=[x0s.shape[0]]).astype(np.int32)]
        
        x0j1 = x0s[np.arange(x0s.shape[0]),inds[:,1],np.ones(shape=[x0s.shape[0]]).astype(np.int32)]
        x0j0 = x0s[np.arange(x0s.shape[0]),inds[:,1],np.zeros(shape=[x0s.shape[0]]).astype(np.int32)]
        
        x0k1 = x0s[:,k,1]
        x0k0 = x0s[:,k,0]
        
        s = np.expand_dims(np.sign((x0i1-x0k1)*(x0i0-x0j0) - (x0i1-x0j1)*(x0i0-x0k0)),-1)

        phi = s * np.arccos(cosphis)
        
        a = np.transpose(np.concatenate([np.expand_dims(np.cos(phi),0),np.expand_dims(-np.sin(phi),0)],0),axes=[1,2,0])
        b = np.transpose(np.concatenate([np.expand_dims(np.sin(phi),0),np.expand_dims(np.cos(phi),0)],0),axes=[1,2,0])

        R = np.einsum("ijk...->jki...", np.concatenate([np.expand_dims(a,0),np.expand_dims(b,0)],0))
        
        xi = x[np.arange(x0s.shape[0]),inds[:,0]]
        xj = x[np.arange(x0s.shape[0]),inds[:,1]]
        
        scaled_ij = (xj-xi)/np.expand_dims(l_ijs,-1) * np.expand_dims(gik,-1)
        
        x_k = np.squeeze(np.matmul(R, np.expand_dims(scaled_ij,-1))) + xi
        x_k = np.where(np.tile(np.expand_dims(node_types[:,k],-1),[1,thetas.shape[0],2])==0.0,x_k,np.zeros_like(x_k))

        x_k = np.expand_dims(x_k,1)
        x_k = np.pad(x_k,[[0,0],[k,x0s.shape[1]-k-1],[0,0],[0,0]],mode='constant')
        
        x += x_k
        
    return x