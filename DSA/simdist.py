import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal
import numpy as np
import torch.nn.utils.parametrize as parametrize
from scipy.stats import wasserstein_distance
import ot #optimal transport for multidimensional l2 wasserstein

def pad_zeros(A,B,device):

    with torch.no_grad():
        dim = max(A.shape[0],B.shape[0])
        A1 = torch.zeros((dim,dim)).float()
        A1[:A.shape[0],:A.shape[1]] += A
        A = A1.float().to(device)

        B1 = torch.zeros((dim,dim)).float()
        B1[:B.shape[0],:B.shape[1]] += B
        B = B1.float().to(device)

    return A,B

class LearnableSimilarityTransform(nn.Module):
    """
    Computes the similarity transform for a learnable orthonormal matrix C 
    """
    def __init__(self, n,orthog=True, batch_size=None, device="cpu"):
        """
        Parameters
        __________
        n : int
            dimension of the C matrix
        """
        super(LearnableSimilarityTransform, self).__init__()
        #initialize orthogonal matrix as identity
        # self.C = nn.Parameter(torch.eye(n).float())
        self.orthog = orthog
        if batch_size is None:
            # Single learnable C (n, n)
            C_init = torch.eye(n, device=device).float()
        else:
            # Batch of learnable Cs (batch, n, n)
            C_init = torch.eye(n, device=device).float().expand(batch_size, n, n).clone()

        # Register C as a parameter (this is what parametrizations will wrap)
        self.C = nn.Parameter(C_init)

        
    def forward(self, B):
        if self.orthog:
            return self.C @ B @ self.C.transpose(-1, -2)
        else:
            C_inv = torch.linalg.inv(self.C)  # works with batched C too
            return self.C @ B @ C_inv

class Skew(nn.Module):
    def __init__(self,n,device):
        """
        Computes a skew-symmetric matrix X from some parameters (also called X)
        
        """
        super().__init__()
      
        self.L1 = nn.Linear(n,n,bias = False, device = device)
        self.L2 = nn.Linear(n,n,bias = False, device = device)
        self.L3 = nn.Linear(n,n,bias = False, device = device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X - X.transpose(-1, -2)

class Matrix(nn.Module):
    def __init__(self,n,device):
        """
        Computes a matrix X from some parameters (also called X)
        
        """
        super().__init__()
      
        self.L1 = nn.Linear(n,n,bias = False, device = device)
        self.L2 = nn.Linear(n,n,bias = False, device = device)
        self.L3 = nn.Linear(n,n,bias = False, device = device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X

class CayleyMap(nn.Module):
    """
    Maps a skew-symmetric matrix to an orthogonal matrix in O(n)
    """
    def __init__(self, n, device):
        """
        Parameters
        __________

        n : int 
            dimension of the matrix we want to map
        
        device : {'cpu','cuda'} or int
            hardware device on which to send the matrix
        """
        super().__init__()
        self.register_buffer("Id", torch.eye(n,device = device))

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        if X.dim() == 2:
            # Single matrix
            return torch.linalg.solve(self.Id + X, self.Id - X)
        elif X.dim() == 3:
            # Batch of matrices: need to expand Id
            batch_size, n, _ = X.shape
            Id_batch = self.Id.expand(batch_size, n, n)
            return torch.linalg.solve(Id_batch + X, Id_batch - X)
        else:
            raise ValueError("X must be 2D or 3D tensor")
    
class SimilarityTransformDist:
    """
    Computes the Procrustes Analysis over Vector Fields
    """
    def __init__(self,
                iters = 200, 
                score_method: Literal["angular", "euclidean","wasserstein"] = "angular",
                lr = 0.01,
                device: Literal["cpu","cuda"] = 'cpu',
                verbose = False,
                group: Literal["O(n)","SO(n)","GL(n)"] = "O(n)",
                wasserstein_compare = 'eig'
                ):
        """
        Parameters
        _________
        iters : int
            number of iterations to perform gradient descent
        
        score_method : {"angular","euclidean","wasserstein"}
            specifies the type of metric to use 
            "wasserstein" will compare the singular values or eigenvalues
            of the two matrices as in Redman et al., (2023)

        lr : float
            learning rate

        device : {'cpu','cuda'} or int

        verbose : bool
            prints when finished optimizing
        
        group : {'SO(n)','O(n)', 'GL(n)'}
            specifies the group of matrices to optimize over

        wasserstein_compare : {,'eig',None}
            specifies whether to compare the singular values or eigenvalues
            if score_method is "wasserstein", or the shapes are different
        """

        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.C_star = None
        self.A = None
        self.B = None
        self.group = group
        self.wasserstein_compare = wasserstein_compare

    def fit(self, 
            A, 
            B, 
            iters = None, 
            lr = None, 
            group = None,
            batch = None
            ):
        """
        Computes the optimal matrix C over specified group

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B : np.array or torch.tensor
            second data matrix
        iters : int or None
            number of optimization steps, if None then resorts to saved self.iters
        lr : float or None
            learning rate, if None then resorts to saved self.lr
        group : {'SO(n)','O(n)', 'GL(n)'}
            specifies the group of matrices to optimize over

        Returns
        _______
        None
        """
        if len(A.shape)==2:
            assert A.shape[0] == A.shape[1]
        elif len(A.shape)==3:
            assert A.shape[1] == A.shape[2]

        if len(B.shape)==2:
            assert B.shape[0] == B.shape[1]
        elif len(B.shape)==3:
            assert B.shape[1] == B.shape[2]

        A = A.to(self.device)
        B = B.to(self.device)
        self.A,self.B = A,B
        lr = self.lr if lr is None else lr
        iters = self.iters if iters is None else iters            
        group = self.group if group is None else group

        if group in {"SO(n)", "O(n)"}:
            self.losses, self.C_star, self.sim_net = self.optimize_C(A,
                                                                     B,
                                                                     lr,iters,
                                                                     orthog=True,
                                                                     verbose=self.verbose,
                                                                     batch = batch)
        if group == "O(n)":
            #permute the first row and column of B then rerun the optimization
            if B.ndim == 2:  # single matrix
                n = B.shape[0]
            elif B.ndim == 3:  # batched matrices
                n = B.shape[1]
            P = torch.eye(n,device=self.device)
            if P.shape[0] > 1:
                P[[0, 1], :] = P[[1, 0], :]
            if B.ndim == 2:
                B_perm = P @ B @ P.T
            else:  # batched
                B_perm = P @ B @ P.T
            losses, C_star, sim_net = self.optimize_C(A,
                                                    B_perm,
                                                    lr,iters,
                                                    orthog=True,
                                                    verbose=self.verbose,
                                                    batch=batch)
            if losses[-1] < self.losses[-1]: # losses[-1]=1042.808 Is this too large? ()
                self.losses = losses
                if C_star.ndim == 2:
                    self.C_star = C_star @ P
                else:  # batched case
                    self.C_star = C_star @ P
                self.sim_net = sim_net
        if group == "GL(n)":
            self.losses, self.C_star, self.sim_net = self.optimize_C(A,
                                                                B,
                                                                lr,iters,
                                                                orthog=False,
                                                                verbose=self.verbose,
                                                                batch=batch)

    def optimize_C(self,A,B,lr,iters,orthog,verbose,batch=None): # TODO this is the bottleneck
        #parameterize mapping to be orthogonal
        n = A.shape[-1]
        sim_net = LearnableSimilarityTransform(n,orthog=orthog,batch_size=batch).to(self.device)
        if orthog:
            parametrize.register_parametrization(sim_net, "C", Skew(n,self.device))
            parametrize.register_parametrization(sim_net, "C", CayleyMap(n,self.device))
        else:
            parametrize.register_parametrization(sim_net, "C", Matrix(n,self.device))
        
        simdist_loss = nn.MSELoss(reduction = 'sum')

        optimizer = optim.Adam(sim_net.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        losses = []
        A = A / A.norm(dim=(-2,-1), keepdim=True)
        B = B / B.norm(dim=(-2,-1), keepdim=True)
        for _ in range(iters):
            # Zero the gradients of the optimizer.
            optimizer.zero_grad()      
            # Compute the Frobenius norm between A and the product.
            loss = simdist_loss(A, sim_net(B))

            loss.backward()

            optimizer.step()
            # if _ % 99:
            #     scheduler.step()
            losses.append(loss.item())

        if verbose:
            print("Finished optimizing C")

        C_star = sim_net.C.detach()
        return losses, C_star,sim_net
    
    def score(self,A=None,B=None,score_method=None,group=None,batch=None):
        """
        Given an optimal C already computed, calculate the metric

        Parameters
        __________
        A : np.array or torch.tensor or None
            first data matrix, if None defaults to the saved matrix in fit
        B : np.array or torch.tensor or None
            second data matrix if None, defaults to the savec matrix in fit
        score_method : None or {'angular','euclidean'}
            overwrites the score method in the object for this application
        Returns
        _______

        score : float
            similarity of the data under the similarity transform w.r.t C
        """
        assert self.C_star is not None
        A = self.A if A is None else A
        B = self.B if B is None else B 
        assert A is not None
        assert B is not None
        assert A.shape == self.C_star.shape
        assert B.shape == self.C_star.shape
        score_method = self.score_method if score_method is None else score_method
        group = self.group if group is None else group
        with torch.no_grad():
            if not isinstance(A,torch.Tensor):
                A = torch.from_numpy(A).float().to(self.device)
            if not isinstance(B,torch.Tensor):
                B = torch.from_numpy(B).float().to(self.device)
            C = self.C_star.to(self.device)
        
        # Support both single and batched
        if A.ndim == 2:
            A = A.unsqueeze(0)  # [1, n, n]
        if B.ndim == 2:
            B = B.unsqueeze(0)
        if C.ndim == 2:
            C = C.unsqueeze(0)

        assert A.shape == B.shape == C.shape

        if group in {"SO(n)", "O(n)"}:
            Cinv = C.transpose(-1, -2)
        elif group in {"GL(n)"}:
            Cinv = torch.linalg.inv(C)
        else:
            raise AssertionError("Need proper group name")
        # if score_method == 'angular':    
        #     num = torch.trace(A.T @ C @ B @ Cinv) 
        #     den = torch.norm(A,p = 'fro')*torch.norm(B,p = 'fro')
        #     score = torch.arccos(num/den).cpu().numpy()
        #     if np.isnan(score): #around -1 and 1, we sometimes get NaNs due to arccos
        #         if num/den < 0:
        #             score = np.pi
        #         else:
        #             score = 0
        # else:
        #     score = torch.norm(A - C @ B @ Cinv,p='fro').cpu().numpy().item() #/ A.numpy().size
    
        # return score
        # pass
        if score_method == "angular":
            # numerator = tr(A^T C B C^-1)
            X = A.transpose(-1, -2) @ C @ B @ Cinv   # [batch, n, n]
            num = X.diagonal(dim1=-2, dim2=-1).sum(-1)  # [batch]
            den = torch.norm(A, p='fro', dim=(-2, -1)) * torch.norm(B, p='fro', dim=(-2, -1))
            cos_val = num / den
            # Clamp for numerical safety
            # cos_val = torch.clamp(cos_val, -1.0, 1.0)
            # score = torch.arccos(cos_val).cpu().numpy()
            score_tensor = torch.arccos(num / den)
            if score_tensor.requires_grad:
                pi_tensor = torch.tensor(np.pi, device=score_tensor.device, dtype=score_tensor.dtype)
                zero_tensor = torch.tensor(0.0, device=score_tensor.device, dtype=score_tensor.dtype)
               
                score = torch.where(
                    torch.isnan(score_tensor),
                    torch.where((num / den) < 0, pi_tensor, zero_tensor),
                    score_tensor
                )
            else:
                score = score_tensor.detach().cpu().numpy()
                if np.isnan(score):
                    score = np.pi if (num / den).item() < 0 else 0.0
        else:
            diff = A - C @ B @ Cinv
            norm_tensor = torch.norm(diff, dim=(-2, -1))  # per batch
            if norm_tensor.requires_grad:
                score = norm_tensor
            else:
                score = norm_tensor.detach().cpu().numpy().item()

    
        return score
    
    def fit_score(self,
                A,
                B,
                iters = None, 
                lr = None,
                score_method = None,
                zero_pad = True,
                group = None,
                batch=None,
                n_job=-1
                ):
        """
        for efficiency, computes the optimal matrix and returns the score 

        Parameters
        __________
        A : np.array or torch.tensor
            first data matrix
        B : np.array or torch.tensor
            second data matrix        
        iters : int or None
            number of optimization steps, if None then resorts to saved self.iters
        lr : float or None
            learning rate, if None then resorts to saved self.lr
        score_method : {'angular','euclidean'} or None
            overwrites parameter in the class
        zero_pad : bool
            if True, then the smaller matrix will be zero padded so its the same size
        Returns
        _______

        score : float
            similarity of the data under the similarity transform w.r.t C
            
        """
        score_method = self.score_method if score_method is None else score_method
        group = self.group if group is None else group

        if isinstance(A,np.ndarray):
            A = torch.from_numpy(A).float()
        if isinstance(B,np.ndarray):
            B = torch.from_numpy(B).float()

        offset = 0
        if len(A.shape)==3 and len(B.shape)==3:
            offset = 1
        assert A.shape[0+offset] == B.shape[1+offset] or self.wasserstein_compare is not None
        if A.shape[0+offset] != B.shape[0+offset]:
            if self.wasserstein_compare is None:
                raise AssertionError("Matrices must be the same size unless using wasserstein distance")
            elif self.verbose: #otherwise resort to L2 Wasserstein over singular or eigenvalues
                print(f"resorting to wasserstein distance over {self.wasserstein_compare}")

        if self.score_method == "wasserstein":
            # assert self.wasserstein_compare in {"sv","eig"}
            if batch is None:
                # if self.wasserstein_compare == "sv":
                #     a = torch.svd(A).S.view(-1,1)
                #     b = torch.svd(B).S.view(-1,1)
                # elif self.wasserstein_compare == "eig":
                a = torch.linalg.eig(A).eigenvalues
                a = torch.vstack([a.real,a.imag]).T

                b = torch.linalg.eig(B).eigenvalues
                b = torch.vstack([b.real,b.imag]).T
                # else:
                #     raise AssertionError("wasserstein_compare must be 'sv' or 'eig'")
                device = a.device
                a = a#.cpu()
                b = b#.cpu()
                M = ot.dist(a,b)#.numpy()
                a,b = torch.ones(a.shape[0])/a.shape[0],torch.ones(b.shape[0])/b.shape[0]
                a,b = a.to(device),b.to(device)

                score_star = ot.emd2(a,b,M) 
                #wasserstein_distance(A.cpu().numpy(),B.cpu().numpy())
            else:
                from joblib import Parallel, delayed
                if A.shape[0] != B.shape[0]:
                    raise AssertionError("When using batching, A and B must have the same batch size")
                scores = Parallel(n_jobs=n_job, verbose=1)(
                delayed(wasserstein_pair)(A[i], B[i], self.wasserstein_compare) for i in range(len(A))
            )
            score_star = np.array(scores)

        else:
       
            self.fit(A, B,iters,lr,group,batch=batch)
            score_star = self.score(self.A,self.B,score_method=score_method,group=group, batch=batch)

        return score_star    
    

def wasserstein_pair(A, B, wasserstein_compare="eig"):
    # if wasserstein_compare == "sv":
    #     a = torch.svd(A).S.view(-1,1)
    #     b = torch.svd(B).S.view(-1,1)
    # elif wasserstein_compare == "eig":
    a = torch.linalg.eig(A).eigenvalues
    a = torch.vstack([a.real,a.imag]).T

    b = torch.linalg.eig(B).eigenvalues
    b = torch.vstack([b.real,b.imag]).T
    # else:
    #     raise AssertionError("wasserstein_compare must be 'sv' or 'eig'")
    device = 'cpu'
    a = a.cpu()
    b = b.cpu()
    M = ot.dist(a,b)#.numpy()
    a,b = torch.ones(a.shape[0])/a.shape[0],torch.ones(b.shape[0])/b.shape[0]
    a,b = a.to(device),b.to(device)

    score_star = ot.emd2(a,b,M) 
    return score_star