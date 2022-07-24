# jax-soft-dtw-div

## About
Implementation of Soft-DTW [1] and Soft-DTW Divergence [2] in Jax. 

Thanks to the official implementation [by the authors](https://github.com/google-research/soft-dtw-divergences) in numba 
and an existing implementation in jax [here](https://github.com/khdlr/softdtw_jax).

## Usage
To compute Soft-DTW and Soft-DTW-Divergence between two time-series signals

```python
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from soft_dtw import soft_dtw, soft_dtw_divergence

# Two 3-dimensional time series of lengths 5 and 4, respectively.
X = np.random.randn(5, 3)
Y = np.random.randn(4, 3)

# compute Soft-DTW
value = soft_dtw(X, Y, gamma=1.)
# compute Soft-DTW and the gradient
sdtw_grad_fn = jax.value_and_grad(partial(soft_dtw, gamma=1.0))
value, grad = sdtw_grad_fn(X, Y)

# compute Soft-DTW-Divergence and the gradient
sdtw_div_grad_fn = jax.value_and_grad(partial(soft_dtw_divergence, gamma=1.0))
value, grad = sdtw_div_grad_fn(X, Y)
```

Compositions with jax functions like `grad`, `jit`, `vmap`, etc are supported. 


## References
[1] M. Cuturi, M. Blondel. *'Soft-DTW: a Differentiable Loss Function for Time-Series'*, Proc. ICML, 2017.
    [arxiv](https://arxiv.org/abs/1703.01541)  
[2] M. Blondel, A. Mensch, JP Vert, *"Differentiable Divergences between Time Series"*, Proc. AISTATS, 2021.
    [arxiv](http://proceedings.mlr.press/v130/blondel21a/blondel21a.pdf)