"""
encoders.py
defines the StackedEncoderModel, which stacks multiple sequence layers
adapted from: https://github.com/lindermanlab/S5
"""

import jax
import jax.random as jr
import jax.numpy as jnp

import equinox as eqx

from typing import List

from elk.models.ssm_init import make_DPLR_HiPPO
from elk.models.ssm import S5SSM, RNN
from elk.models.layers import SequenceLayer

class StackedEncoderModel(eqx.Module):
    """Defines a stack of S5 layers or RNN layers to be used as an encoder.
    Args:
        key         (jr.PRNGKey)
        ssm         (eqx.Module): the SSM to be used (i.e. S5 ssm) (can also be an RNN from ssm.py)
        d_input  (int):    number of input channels in the original input
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                                 we usually refer to this size as H
        n_layers    (int32):    the number of layers to stack
        activation  (string):   Type of activation function to use
        dropout     (float32):  dropout rate
        training    (bool):     whether in training mode or not
        id_encoder  (bool):     replace the encoder with the identity
        prenorm     (bool):     apply prenorm if true
        skip        (bool):     include skip connection in the SequenceLayers
        postnorm    (bool):     apply postnorm if true

        NOTES:
          * always use layer norm
          * note that your RNN doesn't accept the phantom x0
    """

    n_layers: int
    ssm: eqx.Module
    encoder: eqx.nn.Linear
    dropout: float
    d_input: int
    d_model: int  # aka H
    activation: str = eqx.field(static=True)
    training: bool
    id_encoder: bool
    prenorm: bool
    skip: bool
    postnorm: bool
    layers: List[eqx.Module]

    def __init__(
        self,
        key,
        ssm,
        d_input,
        n_layers,
        activation="gelu",
        dropout=0.0,
        training=True,
        id_encoder=False,
        prenorm=False,
        skip=True,
        postnorm=True,
        step_rescale=1.0,
    ):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        self.n_layers = n_layers
        self.ssm = ssm

        key, this_key = jr.split(key)
        self.d_model = self.ssm.H
        self.encoder = eqx.nn.Linear(d_input, self.d_model, key=this_key)
        self.dropout = dropout
        self.d_input = d_input
        self.activation = activation
        self.training = training
        self.id_encoder = id_encoder
        self.prenorm = prenorm
        self.skip = skip
        self.postnorm = postnorm

        # Define the layers
        self.n_layers = n_layers

        seq_keys = jr.split(key, self.n_layers)

        # don't use x0 for the RNN

        if self.ssm.ssm_type == "ssm":
            ssm0 = S5SSM(
                seq_keys[0],
                self.ssm.H,
                self.ssm.P,
                self.ssm.Lambda_re,
                self.ssm.Lambda_im,
                self.ssm.V,
                self.ssm.Vinv,
                self.ssm.C_init,
                self.ssm.discretization,
                self.ssm.dt_min,
                self.ssm.dt_max,
                self.ssm.variable_observation_interval,
                use_D=self.ssm.use_D,
                conj_sym=self.ssm.conj_sym,
                clip_eigs=self.ssm.clip_eigs,
                bidirectional=self.ssm.bidirectional,
                step_rescale=self.ssm.step_rescale,
                use_x0=True,
            )

            later_layers = [
                SequenceLayer(
                    seq_keys[i],
                    ssm=self.ssm,
                    dropout=self.dropout,
                    d_model=self.d_model,
                    activation=self.activation,
                    training=self.training,
                    prenorm=self.prenorm,
                    skip=self.skip,
                    postnorm=self.postnorm,
                )
                for i in range(1, self.n_layers)
            ]

            self.layers = [
                SequenceLayer(
                    seq_keys[0],
                    ssm=ssm0,
                    dropout=self.dropout,
                    d_model=self.d_model,
                    activation=self.activation,
                    training=self.training,
                    prenorm=self.prenorm,
                    skip=self.skip,
                    postnorm=self.postnorm,
                )
            ] + later_layers
        if self.ssm.ssm_type == "rnn":
            self.layers = [
                SequenceLayer(
                    seq_keys[i],
                    ssm=self.ssm,
                    dropout=self.dropout,
                    d_model=self.d_model,
                    activation=self.activation,
                    training=self.training,
                    prenorm=self.prenorm,
                    skip=self.skip,
                    postnorm=self.postnorm,
                )
                for i in range(self.n_layers)
            ]

    @staticmethod
    def ensure_2d(x):
        """If x is 1D (L,), reshape it to 2D (L, 1)"""
        if len(x.shape) == 1:
            return x[:, jnp.newaxis]
        return x

    def __call__(self, x, integration_timesteps=None, drop_key=jr.PRNGKey(0)):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
             drop_key (jr.PRNGKey): key for dropout randomness
             integration_timesteps
        Returns:
            output sequence (float32): (L, d_model)
        """
        x = StackedEncoderModel.ensure_2d(x)
        if self.id_encoder:
            pass
        else:
            x = jax.vmap(self.encoder)(
                x
            )  # apply the encoder to each point in the time series

        for layer in self.layers:
            this_key, drop_key = jr.split(drop_key)
            x = layer(x, integration_timesteps, this_key)
        return x

    def single_step(self, prev_states, x, drop_key=jr.PRNGKey(0), no_encode=False):
        """
        For a single input, and for all the hidden states necessary to form the context (up and down the layers)
        Gives the resulting hidden states (up the stack) and the output

        Args:
            prev_states: List[jax.Arrays], each jax.Array needs to have shape (P,)
            x: jax.Array with size (input_data_dim,) input to stacked encoder

        Returns:
            hiddens: List[jax.Arrays], list of next single steps
            x: jax.Array with size (H,) nonlinear output of S5
        """
        assert len(self.layers) == len(
            prev_states
        ), f"The number of layers is {len(self.layers)}, but the length of the provided states is {len(prev_states)}"
        assert (
            len(prev_states[0]) == self.ssm.P
        ), f"The size of a provided state is {len(prev_states[0])}, but the size of a hidden state in the SSM is {self.ssm.P}"
        assert (
            len(x.shape) == 1,
            f"the input x has shape {x.shape}, but it should be 1D",
        )

        if no_encode or self.id_encoder:
            pass
        else:
            x = self.encoder(x)  # apply the encoder to each point in the time series

        hiddens = []
        for layer, state in zip(self.layers, prev_states):
            this_key, drop_key = jr.split(drop_key)
            hidden, x = layer.single_step(state, x, this_key)
            hiddens.append(hidden)
        return jnp.vstack(hiddens), x

    def get_h01(self, drop_key=jr.PRNGKey(0)):
        """
        return very first hidden state from x0
        """
        P = self.ssm.P
        x0 = self.layers[0].ssm.x0
        h01, out = self.layers[0].single_step(jnp.zeros(P), x0, drop_key)
        return h01, out

    def single_step_from_hidden_layer(
        self, prev_states, h, layer_id, drop_key=jr.PRNGKey(0), input=None
    ):
        """
        Starting from the SSM state at a certain layer
        and the previous time points hiddens
        produces the hiddens above
        and the ssm output

        Default is to not have skip connections or feed through matrix D

        Inputs
        ------
        prev_states: jax.Array with shape (n_layers, P)
        h: an SSM hidden state (still needs to be passed through the non-linearity)
        layer_id: int, indicating which layer the hidden state is at
                  starts at 1
        input: the input to the SSM hidden state h at layer_id, needed if using skip connections

        Outputs
        -------
        above_hiddens: List[jax.Arrays] of length n_layers-1
        out: the output of the SSM stack
        """
        this_layer = self.layers[(layer_id - 1)]
        n_layers = len(self.layers)
        this_ssm = this_layer.ssm
        # apply the non-linearity at layer_id
        if input is None:
            input = jnp.zeros(this_ssm.H)
        this_layer = self.layers[(layer_id - 1)]
        out = (this_ssm.C_tilde @ h).real + this_ssm.D(input)
        this_key, drop_key = jr.split(drop_key)
        out = this_layer.nonlinearity(out, this_key)
        if this_layer.skip:
            out = out + input
        if this_layer.postnorm:
            out = this_layer.norm(out)

        above_hiddens = []
        assert len(prev_states) == n_layers
        for i in range(n_layers - layer_id):
            layer = self.layers[layer_id + i]
            state = prev_states[layer_id + i]
            this_key, drop_key = jr.split(drop_key)
            hidden, out = layer.single_step(state, out, this_key)
            above_hiddens.append(hidden)
        return above_hiddens, out

    def single_step_to_hidden_layer(
        self, prev_states, x, target_layer_id, drop_key=jr.PRNGKey(0), no_encode=False
    ):
        """
        For a given input x (which may or may not be encoded into a vector of shape (H,))
        and the previous SSM hidden states prev_states
        Returns the SSM hidden states at depth target_layer_id up the stack
        I think this code is also agnostic to whether or not we use skip connectons and feed through matrix D

        prev_states: jax.Array with shape (n_layers, P)
        """
        if no_encode or self.id_encoder:
            pass
        else:
            x = self.encoder(x)

        assert len(self.layers) >= target_layer_id
        assert len(prev_states) >= target_layer_id
        for layer_id in range(target_layer_id):
            layer = self.layers[layer_id]
            state = prev_states[layer_id]
            this_key, drop_key = jr.split(drop_key)
            hidden, x = layer.single_step(state, x, this_key)

        return hidden

    def show_hiddens(self, input_sequence, drop_key=jr.PRNGKey(0), use_encoder=False):
        """
        Inputs
        ------
        input_sequence: (L,H)

        Outputs
        -------
        all_hiddens: (L+1, n_layers, P) of all hidden states
            the extra hidden is from the phantom 0th state
        """
        if use_encoder:
            ys = jax.vmap(self.encoder)(input_sequence)  # (L,H)
        else:
            ys = input_sequence  # (L,H)
        all_hiddens = []
        for layer in self.layers:
            this_key, drop_key = jr.split(drop_key)
            hs, ys = layer.show_hiddens(ys, this_key)
            all_hiddens.append(hs)
        all_hiddens = jnp.array(all_hiddens)  # (n_layers, L, H)
        all_hiddens = jnp.transpose(all_hiddens, axes=(1, 0, 2))  # (L, n_layers, H)
        return all_hiddens, ys


def init_StackedEncoder(
    key,
    H=3,
    P=4,
    num_features=1,
    blocks=1,
    C_init="trunc_standard_normal",
    discretization="zoh",
    dt_min=0.001,
    dt_max=0.1,
    variable_observation_interval=False,
    conj_sym=False,
    clip_eigs=False,
    bidirectional=False,
    activation_fn="half_glu1",
    n_layers=6,
    prenorm=False,
    use_D=False,
    skip=False,
    postnorm=False,
    ssm_type="ssm",
):
    """
    Helper function to quickly initialize a stacked encoder model
    Note that S5SSM currently has as default use_D = False
    Args:
        H: output dim of StackedEncoder
        P: dimension of the latent LDS
        num_features: number of features recorded at each time stamp
        n_layers: number of layers in the StackedEncoder model
        ssm_type: str, either 'ssm' or 'rnn'

    """
    block_size = int(P / blocks)
    Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
    Vinv = V.conj().T
    key, this_key = jr.split(key)
    if ssm_type == "ssm":
        ssm = S5SSM(
            this_key,
            H,
            P,
            Lambda.real,
            Lambda.imag,
            V,
            Vinv,
            C_init,
            discretization,
            dt_min,
            dt_max,
            variable_observation_interval,
            use_D=use_D,
            conj_sym=conj_sym,
            clip_eigs=clip_eigs,
            bidirectional=bidirectional,
        )
        key, this_key = jr.split(key)
        encoder_ssm = StackedEncoderModel(
            this_key,
            ssm,
            num_features,
            n_layers,
            activation=activation_fn,
            prenorm=prenorm,
            skip=skip,
            postnorm=postnorm,
        )
    elif ssm_type == "rnn":
        rnn = RNN(this_key, 1, H)
        key, this_key = jr.split(key)
        # hard-coding in 1 layer for the rnn
        encoder_ssm = StackedEncoderModel(
            this_key,
            rnn,
            num_features,
            1,
            activation="identity",
            id_encoder=True,
            prenorm=prenorm,
            skip=skip,
            postnorm=postnorm,
        )
    else:
        raise ValueError(
            f"ssm_type must be either 'ssm' or 'rnn', but it is {ssm_type}"
        )
    return encoder_ssm
