""" deer.py
Code adapted from the original DEER codebase by Lim et al. (2024): https://github.com/machine-discovery/deer 
Based on commit: 17b0b625d3413cb3251418980fb78916e5dacfaa (1/18/24) 
Copyright (c) 2023, Machine Discovery Ltd 
Licensed under the BSD 3-Clause License (see LICENSE file for details).

Modifications for benchmarking and quasi-DEER by Xavier Gonzalez (2024). """

from typing import Callable, Any, Tuple, List, Optional

import jax
import jax.numpy as jnp
import jax.random as jr

import flax.linen

from flax.linen.activation import sigmoid, tanh

from functools import partial


def seq1d(
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    y0: jnp.ndarray,
    xinp: Any,
    params: Any,
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 10000,
    memory_efficient: bool = False,
    quasi: bool = False,
    qmem_efficient: bool = False,  # XG addition
    full_trace: bool = False,  # XG addition
):
    """
    Solve the discrete sequential equation, y[i + 1] = func(y[i], x[i], params) with the DEER framework.

    Arguments
    ---------
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray]
        Function to evaluate the next output signal y[i + 1] from the current output signal y[i].
        The arguments are: output signal y (ny,), input signal x (*nx,) in a pytree, and parameters.
        The return value is the next output signal y[i + 1] (ny,).
    y0: jnp.ndarray
        Initial condition on y (ny,).
    xinp: Any
        The external input signal in a pytree of shape (nsamples, *nx)
    params: Any
        The parameters of the function ``func``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then use the memory efficient algorithm for the DEER iteration.
    quasi: bool
        If True, then make all the Jacobians diagonal. (XG addition)
    qmem_efficient: bool
        If True, use the memory efficient of quasi; if false, call jnp.diag on jac func
        Note: need to use false for eigenworms
    full_trace: bool
        If True, return the full trace of all the Newton iterates for a fixed specification of max_iter (uses a scan)
        if False, return only the final iterate (uses a jax.lax.while_loop)

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the discrete difference equation (nsamples, ny),
        excluding the initial states.
    """
    # set the default initial guess
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros(
            (xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype
        )  # (nsamples, ny)

    def func2(ylist: List[jnp.ndarray], x: Any, params: Any) -> jnp.ndarray:
        # ylist: (ny,)
        return func(ylist[0], x, params)

    def shifter_func(y: jnp.ndarray, shifter_params: Any) -> List[jnp.ndarray]:
        # y: (nsamples, ny)
        # shifter_params = (y0,)
        (y0,) = shifter_params
        y = jnp.concatenate((y0[None, :], y[:-1, :]), axis=0)  # (nsamples, ny)
        return [y]

    # perform the deer iteration
    if quasi:
        yt, samp_iters = deer_iteration(
            inv_lin=diagonal_seq1d_inv_lin,
            p_num=1,
            func=func2,
            shifter_func=shifter_func,
            params=params,
            xinput=xinp,
            inv_lin_params=(y0,),
            shifter_func_params=(y0,),
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=True,
            quasi=quasi,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
        )
    else:
        yt, samp_iters = deer_iteration(
            inv_lin=seq1d_inv_lin,
            p_num=1,
            func=func2,
            shifter_func=shifter_func,
            params=params,
            xinput=xinp,
            inv_lin_params=(y0,),
            shifter_func_params=(y0,),
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=True,
            quasi=quasi,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
        )
    if full_trace:
        return (jnp.vstack((yinit_guess[None, ...], yt)), samp_iters)
    else:
        return (yt, samp_iters)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 9, 10, 11, 12, 13, 14))
def deer_iteration(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray], List[jnp.ndarray]],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,  # gradable as 0
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    quasi: bool = False,  # XG addition
    qmem_efficient: bool = True,  # XG addition
    full_trace: bool = False,  # XG addition
) -> jnp.ndarray:
    """
    Perform the iteration from the DEER framework.

    Arguments
    ---------
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray]
        Inverse of the linear operator.
        Takes the list of G-matrix (nsamples, ny, ny) (p-elements),
        the right hand side of the equation (nsamples, ny), and the inv_lin parameters in a tree.
        Returns the results of the inverse linear operator (nsamples, ny).
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]
        The non-linear function.
        Function that takes the list of y [output: (ny,)] (p elements), x [input: (*nx)] (in a pytree),
        and parameters (any structure of pytree).
        Returns the output of the function.
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]]
        The function that shifts the input signal.
        It takes the signal of shape (nsamples, ny) and produces a list of shifted signals of shape (nsamples, ny).
    p_num: int
        Number of how many dependency on values of ``y`` at different places the function ``func`` has
    params: Any
        The parameters of the function ``func``.
    xinput: Any
        The external input signal of in a pytree with shape (nsamples, *nx).
    inv_lin_params: tree structure of jnp.ndarray
        The parameters of the function ``inv_lin``.
    shifter_func_params: tree structure of jnp.ndarray
        The parameters of the function ``shifter_func``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then do not save the Jacobian matrix for the backward pass.
        This can save memory, but the backward pass will be slower due to recomputation of
        the Jacobian matrix.
    quasi: bool (XG addition)
        If True, then make all the Jacobians diagonal
    full_trace: bool (XG addition)
        If True, then return all the intermediate y values (up to length max_iter)
        If False, then return only the final y value (which may be decided by early stopping up to the tolerance)

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations (nsamples, ny).
    """
    if quasi:
        yt, _, _, _, samp_iters = diagonal_deer_iteration_helper(
            inv_lin=inv_lin,
            func=func,
            shifter_func=shifter_func,
            p_num=p_num,
            params=params,
            xinput=xinput,
            inv_lin_params=inv_lin_params,
            shifter_func_params=shifter_func_params,
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=clip_ytnext,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
        )
        return (yt, samp_iters)
    else:
        yt, _, _, _, samp_iters = deer_iteration_helper(
            inv_lin=inv_lin,
            func=func,
            shifter_func=shifter_func,
            p_num=p_num,
            params=params,
            xinput=xinput,
            inv_lin_params=inv_lin_params,
            shifter_func_params=shifter_func_params,
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=clip_ytnext,
            full_trace=full_trace,
        )
        return (yt, samp_iters)


# we need this function to make the custom vjp
def deer_iteration_eval(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,  # gradable as 0
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    quasi: bool = False,  # XG addition
    qmem_efficient: bool = True,  # XG addition
    full_trace: bool = False,  # XG addition
) -> jnp.ndarray:
    # compute the iteration
    if quasi:
        yt, gts, rhs, func, samp_iters = diagonal_deer_iteration_helper(
            inv_lin=inv_lin,
            func=func,
            shifter_func=shifter_func,
            p_num=p_num,
            params=params,
            xinput=xinput,
            inv_lin_params=inv_lin_params,
            shifter_func_params=shifter_func_params,
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=clip_ytnext,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
        )
    else:
        yt, gts, rhs, func, samp_iters = deer_iteration_helper(
            inv_lin=inv_lin,
            func=func,
            shifter_func=shifter_func,
            p_num=p_num,
            params=params,
            xinput=xinput,
            inv_lin_params=inv_lin_params,
            shifter_func_params=shifter_func_params,
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=clip_ytnext,
        )
    # the function must be wrapped as a partial to be used in the reverse mode
    resid = (
        yt,
        gts,
        rhs,
        xinput,
        params,
        inv_lin_params,
        shifter_func_params,
        jax.tree_util.Partial(inv_lin),
        jax.tree_util.Partial(func),
        jax.tree_util.Partial(shifter_func),
    )
    return (yt, samp_iters), resid


def deer_iteration_bwd(
    # collect non-gradable inputs first
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
    p_num: int,
    max_iter: int,
    memory_efficient: bool,
    clip_ytnext: bool,
    quasi: bool,  # XG addition
    qmem_efficient: bool,  # XG addition
    full_trace: bool,  # XG addition
    # the meaningful arguments
    resid: Any,
    grad_yt: jnp.ndarray,
):
    (
        yt,
        gts,
        rhs0,
        xinput,
        params,
        inv_lin_params,
        shifter_func_params,
        inv_lin,
        func,
        shifter_func,
    ) = resid
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    if gts is None:
        jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
        # recompute gts
        ytparams = shifter_func(yt, shifter_func_params)
        if quasi:
            gts = [-jax.vmap(jnp.diag)(gt) for gt in jacfunc(ytparams, xinput, params)]
        else:
            gts = [-gt for gt in jacfunc(ytparams, xinput, params)]
    _, inv_lin_dual = jax.vjp(inv_lin, gts, rhs0, inv_lin_params)
    _, grad_rhs, grad_inv_lin_params = inv_lin_dual(grad_yt[0]) 
    # grad_rhs: (nsamples, ny)
    ytparams = shifter_func(yt, shifter_func_params)
    _, func_vjp = jax.vjp(func2, ytparams, xinput, params)
    _, grad_xinput, grad_params = func_vjp(grad_rhs)
    grad_shifter_func_params = None
    return grad_params, grad_xinput, grad_inv_lin_params, grad_shifter_func_params, None


deer_iteration.defvjp(deer_iteration_eval, deer_iteration_bwd)


def deer_iteration_helper(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False,  # XG addition
) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    """
    Notes:
        - XG addition: full_trace, to return all the intermediate y values (up to length max_iter)
    """
    # obtain the functions to compute the jacobians and the function
    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 1e-4

    # use the iter function if doing early stopping
    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        gts = [
            -gt for gt in jacfunc(ytparams, xinput, params)
        ]  # [p_num] + (nsamples, ny, ny), meaning its a list of length p num
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) 
        rhs += sum(
            [jnp.einsum("...ij,...j->...i", gt, ytp) for gt, ytp in zip(gts, ytparams)]
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        return err, yt_next, gts, iiter + 1

    # use the scan function to get the full trace
    def scan_func(iter_inp, args):
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        gts = [
            -gt for gt in jacfunc(ytparams, xinput, params)
        ]  # [p_num] + (nsamples, ny, ny)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params)
        rhs += sum(
            [jnp.einsum("...ij,...j->...i", gt, ytp) for gt, ytp in zip(gts, ytparams)]
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        yt_next = jnp.nan_to_num(yt_next)  # XG addition, avoid nans
        new_carry = err, yt_next, gts, iiter + 1
        return new_carry, yt_next

    def cond_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros(
        (yinit_guess.shape[0], yinit_guess.shape[-1], yinit_guess.shape[-1]),
        dtype=dtype,
    )
    gts = [gt] * p_num

    iiter = jnp.array(0, dtype=jnp.int32)
    # decide whether to record full trace or not
    if full_trace:
        _, yt = jax.lax.scan(
            scan_func, (err, yinit_guess, gts, iiter), None, length=max_iter
        )
        samp_iters = max_iter
    else:
        _, yt, gts, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, yinit_guess, gts, iiter)
        )
    if memory_efficient:
        gts = None
    rhs = jnp.zeros_like(gts[0][..., 0])  # (nsamples, ny)
    return yt, gts, rhs, func, samp_iters


def binary_operator(
    element_i: Tuple[jnp.ndarray, jnp.ndarray],
    element_j: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj @ gti
    b = jnp.einsum("...ij,...j->...i", gtj, hti) + htj
    return a, b


def matmul_recursive(
    mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray
) -> jnp.ndarray:
    """
    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape (nsamples - 1, ny, ny)
    vecs: jnp.ndarray
        The vector to be multiplied, shape (nsamples - 1, ny)
    y0: jnp.ndarray
        The initial condition, shape (ny,)

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape (nsamples, ny)
    """
    # shift the elements by one index
    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)[None]  # (1, ny, ny)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, ny, ny)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(binary_operator, elems)
    return yt  # (nsamples, ny)


def seq1d_inv_lin(
    gmat: List[jnp.ndarray], rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = y0.

    Arguments
    ---------
    gmat: jnp.ndarray
        The list of 1 G-matrix of shape (nsamples, ny, ny).
    rhs: jnp.ndarray
        The right hand side of the equation of shape (nsamples, ny).
    inv_lin_params: Tuple[jnp.ndarray]
        The parameters of the linear operator.
        The first element is the initial condition (ny,).

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (nsamples, ny).
    """
    # extract the parameters
    (y0,) = inv_lin_params
    gmat = gmat[0]

    # compute the recursive matrix multiplication and drop the first element
    yt = matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt


# ---------------------------------------------------------------------------#
#                                GRU helpers
#                                  XG addition to work with GRU
# in particular, custom diagonal derivative of the GRU
# ---------------------------------------------------------------------------#

def gru_diagonal_derivative(carry, inputs, params):
    """
    For given carry, inputs, and params, returns the diagonal of the Jacobian of a flax.linen GRU cell

    Source for flax.linen.GRUCell is: https://flax.readthedocs.io/en/v0.5.3/_modules/flax/linen/recurrent.html#GRUCell
    Args:
        carry: previous hidden state, jax.Array (nh,)
        inputs: inputs jax.Array(ninput,)
        params: weight matrices and biases of the GRU cell

    Notes:
    * params['params'] is a dict with keys dict_keys(['ir', 'hr', 'iz', 'hz', 'in', 'hn'])
    * eqx.nn.GRUCell is quite different
    """

    # get the params
    ir = params["params"]["ir"]
    hr = params["params"]["hr"]
    iz = params["params"]["iz"]
    hz = params["params"]["hz"]
    in_ = params["params"]["in"]
    hn = params["params"]["hn"]

    # get the intermediate terms
    rcomp = hn["kernel"].T @ carry + hn["bias"]
    ract = (
        ir["kernel"].T @ inputs + ir["bias"] + hr["kernel"].T @ carry
    )  
    r = sigmoid(ract)

    zact = (
        iz["kernel"].T @ inputs + iz["bias"] + hz["kernel"].T @ carry
    )  
    z = sigmoid(zact)

    n_act = in_["kernel"].T @ inputs + in_["bias"] + r * rcomp
    n = tanh(n_act)

    # get the derivative terms
    dzdh = sigmoid(zact) * (1 - sigmoid(zact)) * jnp.diag(hz["kernel"])
    drdh = sigmoid(ract) * (1 - sigmoid(ract)) * jnp.diag(hr["kernel"])
    dndh = (1 - tanh(n_act) ** 2) * (r * jnp.diag(hn["kernel"]) + drdh * rcomp)

    # add the terms from the product rule
    term1 = -dzdh * n
    term2 = (1 - z) * dndh
    term3 = dzdh * carry
    return term1 + term2 + term3 + z


# ---------------------------------------------------------------------------#
#                                Quasi
#                                  XG addition to do quasi-Newton (i.e. just use diagonalized Jacobians)
# ---------------------------------------------------------------------------#


def diagonal_binary_operator(
    element_i: Tuple[jnp.ndarray, jnp.ndarray],
    element_j: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    XG addition to make the matrix multiplication diagonal
    """
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj * gti
    b = gtj * hti + htj
    return a, b


def diagonal_matmul_recursive(
    mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray
) -> jnp.ndarray:
    """
    XG addition to make the matrix multiplication diagonal

    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape (nsamples - 1, ny) # changed to make the matrices diagonal
    vecs: jnp.ndarray
        The vector to be multiplied, shape (nsamples - 1, ny)
    y0: jnp.ndarray
        The initial condition, shape (ny,)

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape (nsamples, ny)
    """
    # shift the elements by one index
    eye = jnp.ones(mats.shape[-1], dtype=mats.dtype)[None]  # (1, ny)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, ny)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(diagonal_binary_operator, elems)
    return yt  # (nsamples, ny)


def diagonal_seq1d_inv_lin(
    gmat: List[jnp.ndarray], rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = y0.

    Arguments
    ---------
    gmat: jnp.ndarray
        The list of 1 G-matrix of shape (nsamples, ny). NOTE: these G-matrices must be diagonal (XG addition)
    rhs: jnp.ndarray
        The right hand side of the equation of shape (nsamples, ny).
    inv_lin_params: Tuple[jnp.ndarray]
        The parameters of the linear operator.
        The first element is the initial condition (ny,).

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (nsamples, ny).
    """
    # extract the parameters
    (y0,) = inv_lin_params
    gmat = gmat[0]

    # compute the recursive matrix multiplication and drop the first element
    yt = diagonal_matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt


def diagonal_deer_iteration_helper(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False,  # XG addition
    qmem_efficient: bool = True,  # XG addition
) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    # obtain the functions to compute the jacobians and the function
    jacfunc = jax.vmap(
        jax.jacfwd(func, argnums=0), in_axes=(0, 0, None)
    )  # bunch of dense matrices
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 1e-4

    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        # XG change to be more memory efficient
        if qmem_efficient:
            gts = [
                -jax.vmap(gru_diagonal_derivative, in_axes=(0, 0, None))(
                    ytparams[0], xinput, params
                )
            ]
        else:
            gts = [
                -jax.vmap(jnp.diag)(gt)
                for gt in jacfunc(
                    ytparams, xinput, params
                )  # adjusted to deal with scalars
            ]  # [p_num] + (nsamples, ny)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) 
        rhs += sum(
            [
                gt * ytp for gt, ytp in zip(gts, ytparams)
            ]  # adjusted to deal with scalars
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        return err, yt_next, gts, iiter + 1

    def scan_func(iter_inp, args):
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        if qmem_efficient:
            # XG change to be more memory efficient
            gts = [
                -jax.vmap(gru_diagonal_derivative, in_axes=(0, 0, None))(
                    ytparams[0], xinput, params
                )
            ]
        else:
            gts = [
                -jax.vmap(jnp.diag)(gt)
                for gt in jacfunc(
                    ytparams, xinput, params
                )  # adjusted to deal with scalars
            ]  # [p_num] + (nsamples, ny)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) 
        rhs += sum(
            [
                gt * ytp for gt, ytp in zip(gts, ytparams)
            ]  # adjusted to deal with scalars
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        yt_next = jnp.nan_to_num(yt_next)  # XG addition, avoid nans
        new_carry = err, yt_next, gts, iiter + 1
        return new_carry, yt_next

    def cond_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros(
        (yinit_guess.shape[0], yinit_guess.shape[-1]),
        dtype=dtype,
    )
    gts = [gt] * p_num
    iiter = jnp.array(0, dtype=jnp.int32)
    # decide whether to record full trace or not
    if full_trace:
        _, yt = jax.lax.scan(
            scan_func, (err, yinit_guess, gts, iiter), None, length=max_iter
        )
        samp_iters = max_iter
    else:
        _, yt, gts, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, yinit_guess, gts, iiter)
        )
    if memory_efficient:
        gts = None
    rhs = jnp.zeros_like(gts[0])  # (nsamples, ny)
    return yt, gts, rhs, func, samp_iters
