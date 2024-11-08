"""
ssm.py
code to construct ssm modules
adapted from: https://github.com/lindermanlab/S5
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import lecun_normal, normal

import equinox as eqx

from typing import Optional, Callable, Tuple

from elk.models.ssm_init import init_VinvB, init_CV, init_log_steps, trunc_standard_normal
from elk.models.model_utils import OptionalMultiply


def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(
    Lambda_elements,
    Bu_elements,
    C_tilde,
    conj_sym,
    bidirectional,
    Lambda_elements_bwd=None,
    Bu_elements_bwd=None,
):
    """Compute the LxH output of discretized SSM given an LxH input

    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        input_sequence (float32): input sequence of features         (L, H)
        conj_sym (bool):         whether conjugate symmetry is enforced
        bidirectional (bool):    whether bidirectional setup is used,
                              Note for this case C_tilde will have 2P cols
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """

    _, xs = jax.lax.associative_scan(
        binary_operator, (Lambda_elements, Bu_elements)
    )  # xs are the hidden states

    if bidirectional:
        assert (Lambda_elements_bwd is not None) and (
            Bu_elements_bwd is not None
        ), "Must provide bwd kernels."
        _, xs2 = jax.lax.associative_scan(
            binary_operator, (Lambda_elements_bwd, Bu_elements_bwd), reverse=True
        )
        xs = jnp.concatenate((xs, xs2), axis=-1)
    else:
        assert (Lambda_elements_bwd is None) and (
            Bu_elements_bwd is None
        ), "Cannot provide bwd kernels."

    if conj_sym:
        return jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


class S5SSM(eqx.Module):
    """The S5 SSM

    Applies a single application of a linear dynamical system (with hippo initialization)
    Args:
        Lambda_re (complex64): Real part of init diag state matrix  (P,)
        Lambda_im (complex64): Imag part of init diag state matrix  (P,)
        V           (complex64): Eigenvectors used for init           (P,P)
        Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
        H           (int32):     Number of features of input (and output) seq
        P           (int32):     state size for the SSM
        C_init      (string):    Specifies How C is initialized
                     Options: [trunc_standard_normal: sample from truncated standard normal
                                                    and then multiply by V, i.e. C_tilde=CV.
                               lecun_normal: sample from Lecun_normal and then multiply by V.
                               complex_normal: directly sample a complex valued output matrix
                                                from standard normal, does not multiply by V]
        conj_sym    (bool):    Whether conjugate symmetry is enforced
        clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                               constrain real part of eigenvalues to be negative.
                               True recommended for autoregressive task/unbounded sequence lengths
                               Discussed in https://arxiv.org/pdf/2206.11893.pdf.
        bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
        discretization: (string) Specifies discretization method
                         options: [zoh: zero-order hold method,
                                   bilinear: bilinear transform]
        dt_min:      (float32): minimum value to draw timescale values from when
                                initializing log_step
        dt_max:      (float32): maximum value to draw timescale values from when
                                initializing log_step
        step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training
                                on a different resolution for the speech commands benchmark

        D:              (OptionalMutiply from utils), the diagonal feedthrough matrix for the SSM
        use_D:            (bool):
                            if True: D is used
                            if False: D is 0

        x0:             (jax.Array) shape (H,) phantom 0th input, to create h0 (init to 0) (a learnable param)
        use_x0:         (bool) True: use x0 in first layer; False, don't use x0 in any other layer
    """

    H: int
    P: int
    Lambda_re: jax.Array
    Lambda_im: jax.Array
    V: jax.Array
    Vinv: jax.Array
    C_init: str = eqx.field(static=True)
    discretization: str = eqx.field(static=True)
    dt_min: float
    dt_max: float
    variable_observation_interval: bool
    conj_sym: bool
    clip_eigs: bool
    bidirectional: bool
    step_rescale: float

    Lambda: jax.Array
    B: jax.Array
    C1: jax.Array
    C2: Optional[jax.Array]
    C_tilde: jax.Array

    D: eqx.Module  # OptionalMultiply from utils
    use_D: bool

    x0: (
        jax.Array
    )  # learnable first phantom input (to initialize first hidden state h_0)
    use_x0: bool  # whether we should use u0, should only be true for first layer

    log_step: float
    discretize_fn: Callable[
        [jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array]
    ] = eqx.field(static=True)
    Lambda_bar: jax.Array
    B_bar: jax.Array

    ssm_type: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        H,
        P,
        Lambda_re,
        Lambda_im,
        V,
        Vinv,
        C_init,
        discretization,
        dt_min,
        dt_max,
        variable_observation_interval,
        use_D=False,
        conj_sym=False,
        clip_eigs=False,
        bidirectional=False,
        step_rescale=1.0,
        use_x0=False,
    ):
        """Initializes parameters once and performs discretization each time
        the SSM is applied to a sequence
        """
        # self.key = key
        bkey, ckey, c1key, c2key, dkey, lskey = jr.split(key, 6)
        self.H = H
        self.P = P
        self.Lambda_re = Lambda_re
        self.Lambda_im = Lambda_im
        self.V = V
        self.Vinv = Vinv
        self.C_init = C_init
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.variable_observation_interval = variable_observation_interval
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.step_rescale = step_rescale
        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2 * self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        if self.clip_eigs:
            self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()  # it seems like B_init is actually an initializer...
        B_shape = (local_P, self.H)
        self.B = init_VinvB(B_init, bkey, B_shape, self.Vinv)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5**0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(self.C_init)
            )

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = C_init(ckey, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = C_init(ckey, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = init_CV(C_init, c1key, C_shape, self.V)
                self.C2 = init_CV(C_init, c2key, C_shape, self.V)
                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = jnp.concatenate((C1, C2), axis=-1)

            else:
                self.C1 = init_CV(C_init, ckey, C_shape, self.V)
                self.C2 = None
                self.C_tilde = self.C1[..., 0] + 1j * self.C1[..., 1]

        # Initialize feedthrough (D) matrix
        self.use_D = use_D
        self.D = OptionalMultiply(self.H, self.use_D, dkey)

        # Initialize learnable discretization timescale value
        self.log_step = init_log_steps(lskey, (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.discretize_fn = discretize_zoh
        elif self.discretization in ["bilinear"]:
            self.discretize_fn = discretize_bilinear
        else:
            raise NotImplementedError(
                "Discretization method {} not implemented".format(self.discretization)
            )

        if not self.variable_observation_interval:
            self.Lambda_bar, self.B_bar = self.discretize_fn(self.Lambda, B_tilde, step)
        else:
            # If we have variable observation interval, then we will need to discretize on-the-fly.
            self.Lambda_bar, self.B_bar = None, None

        self.ssm_type = "ssm"

        self.use_x0 = use_x0
        self.x0 = jnp.zeros(self.H)

    def __call__(self, input_sequence, integration_timesteps=None):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """

        # begin by appending self.u0 to the input_sequence
        if self.use_x0:
            input_sequence = jnp.vstack([self.x0, input_sequence])  # (L+1, H)
        # These will be over-written in the case that we have a bi-directional model.
        Lambda_bar_elements_bwd = None
        Bu_bar_elements_bwd = None

        # If we have variable observation intervals, then we need to compute the variables on the fly.
        if not self.variable_observation_interval:
            integration_timesteps = jnp.ones((len(input_sequence) - 1))

            @jax.vmap
            def _do_vmapped_discretize(_timestep):
                # print('\nWarning: Discretizing on-the-fly...\n')
                B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
                step = self.step_rescale * jnp.exp(self.log_step[:, 0])
                Lambda_bar, B_bar = self.discretize_fn(
                    self.Lambda, B_tilde, step * _timestep
                )
                return Lambda_bar, B_bar

            # Discretize forward pass.
            fwd_timesteps = jnp.expand_dims(
                jnp.concatenate((jnp.asarray((1,)), integration_timesteps)), -1
            )
            Lambda_bar_elements, B_bar_elements = _do_vmapped_discretize(fwd_timesteps)
            Bu_bar_elements = jax.vmap(lambda u, b: b @ u)(
                input_sequence, B_bar_elements
            )

            if self.bidirectional:
                bwd_timesteps = jnp.expand_dims(
                    jnp.concatenate((integration_timesteps, jnp.asarray((1,)))), -1
                )
                Lambda_bar_elements_bwd, B_bar_elements_bwd = _do_vmapped_discretize(
                    bwd_timesteps
                )
                Bu_bar_elements_bwd = jax.vmap(lambda u, b: b @ u)(
                    input_sequence, B_bar_elements_bwd
                )

        else:
            assert (self.Lambda_bar is None) and (
                self.B_bar is None
            ), "Cannot pre-compute these.  How are these not `None`..."
            assert (
                integration_timesteps is not None
            ), "Must supply integration_timesteps for variable timesteps."

            @jax.vmap
            def _do_vmapped_discretize(_timestep):
                B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
                step = self.step_rescale * jnp.exp(self.log_step[:, 0])
                Lambda_bar, B_bar = self.discretize_fn(
                    self.Lambda, B_tilde, step * _timestep
                )
                return Lambda_bar, B_bar

            # Discretize forward pass.
            fwd_timesteps = jnp.expand_dims(
                jnp.concatenate((jnp.asarray((1,)), integration_timesteps)), -1
            )
            Lambda_bar_elements, B_bar_elements = _do_vmapped_discretize(fwd_timesteps)
            Bu_bar_elements = jax.vmap(lambda u, b: b @ u)(
                input_sequence, B_bar_elements
            )

            if self.bidirectional:
                bwd_timesteps = jnp.expand_dims(
                    jnp.concatenate((integration_timesteps, jnp.asarray((1,)))), -1
                )
                Lambda_bar_elements_bwd, B_bar_elements_bwd = _do_vmapped_discretize(
                    bwd_timesteps
                )
                Bu_bar_elements_bwd = jax.vmap(lambda u, b: b @ u)(
                    input_sequence, B_bar_elements_bwd
                )

        ys = apply_ssm(
            Lambda_bar_elements,
            Bu_bar_elements,
            self.C_tilde,
            self.conj_sym,
            self.bidirectional,
            Lambda_elements_bwd=Lambda_bar_elements_bwd,
            Bu_elements_bwd=Bu_bar_elements_bwd,
        )

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D(u))(input_sequence)
        return ys + Du

    def get_Lambda_B_bar(self, _timestep):
        """
        Function to get discretized Lambda_bar and B_bar
        """
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * jnp.exp(self.log_step[:, 0])
        Lambda_bar, B_bar = self.discretize_fn(self.Lambda, B_tilde, step * _timestep)
        return Lambda_bar, B_bar

    def single_step(self, state, input, timestep=1.0):
        """
        Want to run the SSM forward one unit in time, using a hidden input that is fed in

        Return the updated hidden state and the output

        The initial hidden state in S5 is supposed to be zeros.

        Lambda_bar and D are both diagonal

        state: jax.Array, with shape (P,)
        """
        Lambda_bar, B_bar = self.get_Lambda_B_bar(timestep)
        new_state = Lambda_bar * state + B_bar @ input
        output = (self.C_tilde @ new_state).real + self.D(input)  # (H,)
        return (new_state, output)

    def show_hiddens(self, input_sequence):
        """
        Inputs
        ------
            input_sequence (float 32): input sequence (L,H)

        Returns
        -------
            hs: all intermediate hidden states (L+1, P) or (L,P)
                (depend on self.use_x0 or not)
            ys: the outputs (which have shape H) to be passed through a non-linearity
        """

        @jax.vmap
        def _do_vmapped_discretize(_timestep):
            B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
            step = self.step_rescale * jnp.exp(self.log_step[:, 0])
            Lambda_bar, B_bar = self.discretize_fn(
                self.Lambda, B_tilde, step * _timestep
            )
            return Lambda_bar, B_bar

        if self.use_x0:
            input_sequence = jnp.vstack([self.x0, input_sequence])  # (L+1, H)
        integration_timesteps = jnp.ones((len(input_sequence) - 1))
        fwd_timesteps = jnp.expand_dims(
            jnp.concatenate((jnp.asarray((1,)), integration_timesteps)), -1
        )
        Lambda_bar_elements, B_bar_elements = _do_vmapped_discretize(fwd_timesteps)
        Bu_bar_elements = jax.vmap(lambda u, b: b @ u)(input_sequence, B_bar_elements)
        _, hs = jax.lax.associative_scan(
            binary_operator, (Lambda_bar_elements, Bu_bar_elements)
        )
        ys = jax.vmap(lambda x: (self.C_tilde @ x).real)(hs)
        Du = jax.vmap(lambda u: self.D(u))(input_sequence)

        return hs, ys + Du


class RNN(eqx.Module):
    """
    RNN
    H           (int32):     Number of features of input (and output) seq
    P           (int32):     state size for the SSM
    """

    H: int  # Number of features of input (and output) seq
    P: int  # state size for the SSM
    cell: eqx.Module
    C: jax.Array
    ssm_type: str = eqx.field(static=True)
    stack: bool  # should we prepare this RNN to be stacked?
    use_x0: bool
    # x0: jax.Array
    x0: jax.Array = eqx.field(static=True)  # not learnable, set to zero

    def __init__(
        self, key, H, P, C_init="identity", type="GRUCell", stack=False, use_x0=True
    ):
        self.H = H
        self.P = P
        self.cell = getattr(eqx.nn, type)(self.H, self.P, key=key)
        if C_init in ["identity"]:
            self.C = jnp.eye(self.H, self.P)
        else:
            raise NotImplementedError(f"C_init method {self.C_init} not implemented")
        self.ssm_type = "rnn"
        self.stack = stack
        self.use_x0 = use_x0
        self.x0 = jnp.zeros(self.H)

    def single_step(self, state, input):
        """
        state: jax.Array, with shape (P,)
        """
        new_state = self.cell(input, state)  # (P,)
        if self.stack:
            output = self.C @ new_state  # (H,), currently no D matrix
        else:
            output = new_state
        return (new_state, output)

    def __call__(self, input):
        """
        Had to use an anonymous function in this scan in response to these annoying equinox / jax bugs

        https://github.com/patrick-kidger/equinox/issues/558

        https://github.com/google/jax/issues/13554
        """
        if self.use_x0:
            input = jnp.vstack([self.x0, input])  # (L+1, H)
        hidden_init = jnp.zeros((self.P,))
        hidden, outputs = jax.lax.scan(
            lambda *a: self.single_step(*a), hidden_init, input
        )
        return outputs
