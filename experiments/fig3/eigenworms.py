"""
eigenworms.py
refactor of the code at DEER
https://github.com/machine-discovery/deer/tree/main/experiments/04_rnn_eigenworms
note that use_scan True is sequential evaluation, while False is DEER

Copyright (c) 2023, Machine Discovery Ltd
Licensed under the BSD 3-Clause License (see LICENSE file for details).
Modifications to use quasi-DEER by Xavier Gonzalez (2024). 

Run on a 1GB V100
"""
import wandb
import time

import argparse
import os
import sys
from functools import partial
from typing import Tuple, Any, List, Dict, Optional, Sequence, Callable
from glob import glob

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from tqdm import tqdm
from jax._src import prng


from elk.algs.deer import seq1d


# --------------------------------
#
# model functions
#
# --------------------------------


def vmap_to_shape(func: Callable, shape: Sequence[int]) -> Callable:
    rank = len(shape)
    for i in range(rank - 1):
        func = jax.vmap(func)
    return func


def custom_mlp(
    mlp: eqx.nn.MLP, key: prng.PRNGKeyArray, init_method: Optional[str] = "he_uniform"
) -> eqx.nn.MLP:
    """
    eqx.nn.MLP with custom initialisation scheme using jax.nn.initializers
    """
    where_bias = lambda m: [lin.bias for lin in m.layers]
    where_weight = lambda m: [lin.weight for lin in m.layers]

    mlp = eqx.tree_at(where=where_bias, pytree=mlp, replace_fn=jnp.zeros_like)

    if init_method is None:
        return mlp

    if init_method == "he_uniform":
        # get all the weights of the mlp model
        weights = where_weight(mlp)
        # split the random key into different subkeys for each layer
        subkeys = jax.random.split(key, len(weights))
        new_weights = [
            jax.nn.initializers.he_uniform()(subkey, weight.shape)
            for weight, subkey in zip(weights, subkeys)
        ]
        mlp = eqx.tree_at(where=where_weight, pytree=mlp, replace=new_weights)
    else:
        return NotImplementedError("only he_uniform is implemented")
    return mlp


def custom_gru(gru: eqx.nn.GRUCell, key: prng.PRNGKeyArray) -> eqx.nn.GRUCell:
    """
    eqx.nn.GRUCell with custom initialisation scheme using jax.nn.initializers
    """
    where_bias = lambda g: g.bias
    where_bias_n = lambda g: g.bias_n
    where_weight_ih = lambda g: g.weight_ih
    where_weight_hh = lambda g: g.weight_hh

    gru = eqx.tree_at(where=where_bias, pytree=gru, replace_fn=jnp.zeros_like)
    gru = eqx.tree_at(where=where_bias_n, pytree=gru, replace_fn=jnp.zeros_like)

    weight_ih = where_weight_ih(gru)
    weight_hh = where_weight_hh(gru)

    ih_key, hh_key = jax.random.split(key, 2)

    new_weight_ih = jax.nn.initializers.lecun_normal()(ih_key, weight_ih.shape)
    new_weight_hh = jax.nn.initializers.orthogonal()(hh_key, weight_hh.shape)

    gru = eqx.tree_at(where_weight_ih, gru, new_weight_ih)
    gru = eqx.tree_at(where_weight_hh, gru, new_weight_hh)
    return gru


class MLP(eqx.Module):
    model: eqx.nn.MLP

    def __init__(self, ninp: int, nstate: int, nout: int, key: prng.PRNGKeyArray):
        self.model = eqx.nn.MLP(
            in_size=ninp,
            out_size=nout,
            width_size=nstate,
            depth=1,
            activation=jax.nn.relu,
            key=key,
        )
        self.model = custom_mlp(self.model, key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return vmap_to_shape(self.model, x.shape)(x)


class GRU(eqx.Module):
    gru: eqx.Module
    use_scan: bool

    def __init__(self, ninp: int, nstate: int, key: prng.PRNGKeyArray, use_scan: bool):
        self.gru = eqx.nn.GRUCell(input_size=ninp, hidden_size=nstate, key=key)
        self.gru = custom_gru(self.gru, key)
        self.use_scan = use_scan

    def __call__(
        self, inputs: jnp.ndarray, h0: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # h0.shape == (nbatch, nstate)
        # inputs.shape == (nbatch, ninp)
        assert len(inputs.shape) == len(h0.shape)

        states = vmap_to_shape(self.gru, inputs.shape)(inputs, h0)
        return states, states


class SingleScaleGRU(eqx.Module):
    nchannel: int
    nlayer: int
    encoder: MLP
    grus: List[List[GRU]]
    mlps: List[MLP]
    classifier: MLP
    norms: List[eqx.nn.LayerNorm]
    dropout: eqx.nn.Dropout
    dropout_key: prng.PRNGKeyArray
    use_scan: bool
    quasi: bool  # XG addition

    def __init__(
        self,
        ninp: int,
        nchannel: int,
        nstate: int,
        nlayer: int,
        nclass: int,
        key: prng.PRNGKeyArray,
        use_scan: bool,
        quasi: bool,
    ):
        keycount = 1 + (nchannel + 1) * nlayer + 1 + 1  # +1 for dropout
        print(f"Keycount: {keycount}")
        keys = jax.random.split(key, keycount)

        self.nchannel = nchannel
        self.nlayer = nlayer

        assert nstate % nchannel == 0
        gru_nstate = int(nstate / nchannel)

        # encode inputs (or rather, project) to have nstate in the feature dimension
        self.encoder = MLP(ninp=ninp, nstate=nstate, nout=nstate, key=keys[0])

        # nlayers of (scale_gru + mlp) pair
        self.grus = [
            [
                GRU(
                    ninp=nstate,
                    nstate=gru_nstate,
                    key=keys[int(1 + (nchannel * j) + i)],
                    use_scan=use_scan,
                )
                for i in range(nchannel)
            ]
            for j in range(nlayer)
        ]
        self.mlps = [
            MLP(
                ninp=nstate,
                nstate=nstate,
                nout=nstate,
                key=keys[int(i + 1 + nchannel * nlayer)],
            )
            for i in range(nlayer)
        ]
        assert len(self.grus) == nlayer
        assert len(self.grus[0]) == nchannel
        print(
            f"scale_grus random keys end at index {int(1 + (nchannel * (nlayer - 1)) + (nchannel - 1))}"
        )
        print(f"mlps random keys end at index {int((nchannel * nlayer) + nlayer)}")

        # project nstate in the feature dimension to nclasses for classification
        self.classifier = MLP(
            ninp=nstate,
            nstate=nstate,
            nout=nclass,
            key=keys[int((nchannel + 1) * nlayer + 1)],
        )

        self.norms = [
            eqx.nn.LayerNorm((nstate,), use_weight=False, use_bias=False)
            for i in range(nlayer * 2)
        ]
        self.dropout = eqx.nn.Dropout(p=0.2)
        self.dropout_key = keys[-1]

        self.use_scan = use_scan
        self.quasi = quasi  # XG addition

    def __call__(
        self, inputs: jnp.ndarray, h0: jnp.ndarray, yinit_guess: jnp.ndarray
    ) -> jnp.ndarray:
        # encode (or rather, project) the inputs
        inputs = self.encoder(inputs)

        def model_func(carry: jnp.ndarray, inputs: jnp.ndarray, model: Any):
            return model(inputs, carry)[1]  # could be [0] or [1]

        all_samp_iters = []
        for i in range(self.nlayer):
            inputs = jax.vmap(self.norms[i])(inputs)  # XG change

            x_from_all_channels = []

            for ch in range(self.nchannel):
                if self.use_scan:
                    model = lambda carry, inputs: self.grus[i][ch](inputs, carry)
                    x = jax.lax.scan(model, h0, inputs)[1]
                    samp_iters = 1
                elif self.quasi:
                    x, samp_iters = seq1d(
                        model_func,
                        h0,
                        inputs,
                        self.grus[i][ch],
                        yinit_guess,
                        quasi=self.quasi,  # XG addition
                        qmem_efficient=False,  # XG addition
                    )
                else:
                    x, samp_iters = seq1d(
                        model_func,
                        h0,
                        inputs,
                        self.grus[i][ch],
                        yinit_guess,
                        quasi=self.quasi,  # XG addition
                    )
                all_samp_iters.append(samp_iters)
                x_from_all_channels.append(x)

            samp_iters_out = jnp.stack(all_samp_iters)
            x = jnp.concatenate(x_from_all_channels, axis=-1)
            x = jax.vmap(self.norms[i + 1])(  # XG change
                x + inputs
            )  # add and norm after multichannel GRU layer
            x = self.mlps[i](x) + x  # add with norm added in the next loop
            inputs = x
        return (
            self.classifier(x),
            samp_iters_out,
            samp_iters,
        )  # samp_iters_out is averaged over all layers, samp_iters is last layer only


# --------------------------------
#
# data loading
#
# --------------------------------


class EigenWormsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        datafile: str = "neuralrde",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.datafile = datafile
        if datafile == "neuralrde":
            self.train_file = "neuralrde_split/eigenworms_train.pkl"
            self.val_file = "neuralrde_split/eigenworms_val.pkl"
            self.test_file = "neuralrde_split/eigenworms_test.pkl"
        elif datafile == "lem":
            self.train_file = "lem_split/eigenworms_train.pkl"
            self.val_file = "lem_split/eigenworms_val.pkl"
            self.test_file = "lem_split/eigenworms_test.pkl"
        else:
            raise RuntimeError()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        with open(self.train_file, "rb") as f:
            if self.datafile == "neuralrde":
                x, y = pickle.load(f)
                self._train_dataset = TensorDataset(x, y)
            elif self.datafile == "lem":
                self._train_dataset = pickle.load(f)
            else:
                raise RuntimeError()
        with open(self.val_file, "rb") as f:
            if self.datafile == "neuralrde":
                x, y = pickle.load(f)
                self._val_dataset = TensorDataset(x, y)
            elif self.datafile == "lem":
                self._val_dataset = pickle.load(f)
            else:
                raise RuntimeError()
        with open(self.test_file, "rb") as f:
            if self.datafile == "neuralrde":
                x, y = pickle.load(f)
                self._test_dataset = TensorDataset(x, y)
            elif self.datafile == "lem":
                self._test_dataset = pickle.load(f)
            else:
                raise RuntimeError()
        print("LEN TRAIN DATASET", len(self._train_dataset))
        print("LEN VAL DATASET", len(self._val_dataset))
        print("LEN TEST DATASET", len(self._test_dataset))

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch


def prep_batch(
    batch: Tuple[torch.Tensor, torch.Tensor], dtype: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(batch) == 2
    x, y = batch
    x = jnp.asarray(x.numpy(), dtype=dtype)
    y = jnp.asarray(y.numpy())
    return x, y


def count_params(params) -> jnp.ndarray:
    return sum(
        jnp.prod(jnp.asarray(p.shape)) for p in jax.tree_util.tree_leaves(params)
    )


def grad_norm(grads) -> jnp.ndarray:
    flat_grads = jnp.concatenate(
        [jnp.reshape(g, (-1,)) for g in jax.tree_util.tree_leaves(grads)]
    )
    return jnp.linalg.norm(flat_grads)


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics


def get_datamodule(
    dset: str, batch_size: int, datafile: str = "neuralrde"
) -> pl.LightningDataModule:
    dset = dset.lower()
    datafile = datafile.lower()
    if datafile not in ["neuralrde", "lem"]:
        raise NotImplementedError()
    if dset == "eigenworms":
        return EigenWormsDataModule(
            batch_size=batch_size, datafile=datafile  # nseq = 17984, nclass = 5
        )
    else:
        return NotImplementedError("only eigenworms dataset is available")


# --------------------------------
# code to train rnn
# --------------------------------

# # run on cpu
# jax.config.update('jax_platform_name', 'cpu')
# enable float 64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


@partial(jax.jit, static_argnames=("model"))
def rollout(
    model: eqx.Module,
    y0: jnp.ndarray,
    inputs: jnp.ndarray,
    yinit_guess: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    y0 (nstate,)
    inputs (nsequence, ninp)
    yinit_guess (nsequence, nstate)

    return: (nclass,)
    """
    out, samp_iters_all, samp_iters_last = model(inputs, y0, yinit_guess)
    # jax.debug.print(
    #     "inside of rollout, samp_iters is {samp_iters}", samp_iters=samp_iters
    # )
    return out.mean(axis=0), samp_iters_all, samp_iters_last


@partial(jax.jit, static_argnames=("static"))
def loss_fn(
    params: Any,
    static: Any,
    y0: jnp.ndarray,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    yinit_guess: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    y0 (nbatch, nstate)
    yinit_guess (nbatch, nsequence, nstate)
    batch (nbatch, nsequence, ninp) (nbatch,)
    """
    model = eqx.combine(params, static)
    x, y = batch

    # ypred: (nbatch, nclass)
    ypred, samp_iters_all, samp_iters_last = jax.vmap(
        rollout, in_axes=(None, 0, 0, 0), out_axes=(0)
    )(model, y0, x, yinit_guess)
    # jax.debug.print(
    #     "inside of loss_fn, samp_iters is {samp_iters}", samp_iters=samp_iters
    # )

    metrics = compute_metrics(ypred, y)
    loss, accuracy = metrics["loss"], metrics["accuracy"]
    return loss, (accuracy, samp_iters_all, samp_iters_last)


@partial(jax.jit, static_argnames=("static", "optimizer"))
def update_step(
    params: Any,
    static: Any,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    y0: jnp.ndarray,
    yinit_guess: jnp.ndarray,
) -> Tuple[optax.Params, Any, jnp.ndarray, jnp.ndarray]:
    """
    batch (nbatch, nsequence, ninp) (nbatch,)
    y0 (nbatch, nstate)
    yinit_guess (nbatch, nsequence, nstate)
    """
    loss_and_aux, grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
        params, static, y0, batch, yinit_guess
    )
    loss, (accuracy, samp_iters_all, samp_iters_last) = loss_and_aux
    updates, opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    gradnorm = grad_norm(grad)
    # jax.debug.print(
    #     "inside of update_step, samp_iters is {samp_iters}", samp_iters=samp_iters
    # )
    return (
        new_params,
        opt_state,
        loss,
        accuracy,
        gradnorm,
        samp_iters_all,
        samp_iters_last,
    )


def train():
    wandb.init(project="elk", entity="xavier_gonzalez")
    # set up argparse for the hyperparameters above
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--nepochs", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--ninps", type=int, default=6)
    parser.add_argument("--nstates", type=int, default=32)
    parser.add_argument("--nsequence", type=int, default=17984)
    parser.add_argument("--nclass", type=int, default=5)
    parser.add_argument("--nlayer", type=int, default=5)
    parser.add_argument("--nchannel", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--patience_metric", type=str, default="accuracy")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument(
        "--use_scan", action="store_true", help="Doing --use_scan sets it to True"
    )
    parser.add_argument(
        "--quasi", action="store_true", help="Doing --quasi sets it to True"
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="eigenworms",
        choices=[
            "eigenworms",
        ],
    )
    args = parser.parse_args()

    # set seed for pytorch
    # torch.manual_seed(42)
    pl.seed_everything(args.seed)

    ninp = args.ninps
    nstate = args.nstates
    nsequence = args.nsequence
    nclass = args.nclass
    nlayer = args.nlayer
    nchannel = args.nchannel
    batch_size = args.batch_size
    patience = args.patience
    patience_metric = args.patience_metric
    use_scan = args.use_scan
    quasi = args.quasi  # XG addition

    if args.precision == 32:
        dtype = jnp.float32
    elif args.precision == 64:
        dtype = jnp.float64
    else:
        raise ValueError("Only 32 or 64 accepted")
    print(f"dtype is {dtype}")
    print(f"use_scan is {use_scan}")
    print(f"quasi is {quasi}")
    print(f"patience_metric is {patience_metric}")

    # check the path
    logpath = "logs_instance_3"
    path = os.path.join(logpath, f"version_{args.version}")
    # if os.path.exists(path):
    #     raise ValueError(f"Path {path} already exists!")
    os.makedirs(path, exist_ok=True)

    # set up the model and optimizer
    key = jax.random.PRNGKey(args.seed)
    assert nchannel == 1, "currently only support 1 channel"
    model = SingleScaleGRU(
        ninp=ninp,
        nchannel=nchannel,
        nstate=nstate,
        nlayer=nlayer,
        nclass=nclass,
        key=key,
        use_scan=use_scan,
        quasi=quasi,  # XG addition
    )
    model = jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_array(x) else x, model
    )
    y0 = jnp.zeros(
        (batch_size, int(nstate / nchannel)), dtype=dtype
    )  # (nbatch, nstate)
    yinit_guess = jnp.zeros(
        (batch_size, nsequence, int(nstate / nchannel)), dtype=dtype
    )  # (nbatch, nsequence, nstate)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=1), optax.adam(learning_rate=args.lr)
    )
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)
    print(f"Total parameter count: {count_params(params)}")

    # training loop
    step = 0
    dm = get_datamodule(dset=args.dset, batch_size=args.batch_size)
    dm.setup()
    best_val_acc = 0
    best_val_loss = float("inf")
    for epoch in tqdm(range(args.nepochs), file=sys.stderr):
        print(f"starting epoch {epoch}")
        loop = tqdm(
            dm.train_dataloader(),
            total=len(dm.train_dataloader()),
            leave=False,
            file=sys.stderr,
        )
        for i, batch in enumerate(loop):
            try:
                batch = dm.on_before_batch_transfer(batch, i)
            except Exception():
                pass
            batch = prep_batch(batch, dtype)
            t0 = time.time()
            (
                params,
                opt_state,
                loss,
                accuracy,
                gradnorm,
                samp_iters_all,
                samp_iters_last,
            ) = update_step(
                params=params,
                static=static,
                optimizer=optimizer,
                opt_state=opt_state,
                batch=batch,
                y0=y0,
                yinit_guess=yinit_guess,
            )
            t1 = time.time()
            wandb.log(
                {
                    "train_loss": loss,
                    "train_accuracy": accuracy,
                    "gru_gradnorm": gradnorm,
                    "time_per_train_step": t1 - t0,
                    "samp_iters_train": jnp.mean(samp_iters_all),
                    "samp_iters_train_last": jnp.mean(samp_iters_last),
                },
                step=step,
            )
            step += 1

        inference_model = eqx.combine(params, static)
        inference_model = eqx.tree_inference(inference_model, value=True)
        inference_params, inference_static = eqx.partition(
            inference_model, eqx.is_array
        )
        if epoch % 1 == 0:
            val_loss = 0
            nval = 0
            val_acc = 0
            loop = tqdm(
                dm.val_dataloader(),
                total=len(dm.val_dataloader()),
                leave=False,
                file=sys.stderr,
            )
            tval = 0
            for i, batch in enumerate(loop):
                try:
                    batch = dm.on_before_batch_transfer(batch, i)
                except Exception():
                    pass
                batch = prep_batch(batch, dtype)
                tstart = time.time()
                loss, (accuracy, samp_iters, _) = loss_fn(
                    inference_params, inference_static, y0, batch, yinit_guess
                )
                tval += time.time() - tstart
                val_loss += loss * len(batch[1])
                val_acc += accuracy * len(batch[1])
                nval += len(batch[1])
                # break
            tval /= i + 1
            val_loss /= nval
            val_acc /= nval
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "time_per_val_step": tval,
                    "samp_iters_val": jnp.mean(samp_iters),  # averaged over all layers
                },
                step=step,
            )
            if patience_metric == "accuracy":
                if val_acc > best_val_acc:
                    patience = args.patience
                    best_val_acc = val_acc
                    for f in glob(f"{path}/best_model_epoch_*"):
                        os.remove(f)
                    checkpoint_path = os.path.join(
                        path, f"best_model_epoch_{epoch}_step_{step}.pkl"
                    )
                    best_model = eqx.combine(params, static)
                    eqx.tree_serialise_leaves(checkpoint_path, best_model)
                else:
                    patience -= 1
                    if patience == 0:
                        print(
                            f"The validation accuracy stopped improving, training ends here at epoch {epoch} and step {step}!"
                        )
                        # break
            elif patience_metric == "loss":
                if val_loss < best_val_loss:
                    patience = args.patience
                    best_val_loss = val_loss
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                    for f in glob(f"{path}/best_model_epoch_*"):
                        os.remove(f)
                    checkpoint_path = os.path.join(
                        path, f"best_model_epoch_{epoch}_step_{step}.pkl"
                    )
                    best_model = eqx.combine(params, static)
                    eqx.tree_serialise_leaves(checkpoint_path, best_model)
                else:
                    patience -= 1
                    if patience == 0:
                        print(
                            f"The validation loss stopped improving at {best_val_loss} with accuracy {best_val_acc}, training ends here at epoch {epoch} and step {step}!"
                        )
                        # break
            else:
                raise ValueError


if __name__ == "__main__":
    train()
