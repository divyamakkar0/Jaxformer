import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import linen as nn
from einops import rearrange

import tiktoken
from utils import config, modelConfig
import numpy as np
from typing import Optional, Tuple, List
from jaxtyping import Array, PyTree
from functools import partial

from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_dimension: int
    vocab_size: int
    n_head: int
    blocks: int
    layers_per_block: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

class Dense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return nn.Dense(features=self.features, dtype=self.dtype)(x)

class FeedForward(nn.Module):
    model_dimension: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: Array, train=True) -> Array:
        x = Dense(features=self.model_dimension * 4, dtype=self.model_dtype)(x)
        x = nn.selu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = Dense(features=self.model_dimension, dtype=self.model_dtype)(x)
        return x

class RMSNorm(nn.Module):
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:

        x_type =x.dtype
        x = x.astype(jnp.float32)
        rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x / jnp.sqrt(rms + 1e-6)
        x = x.astype(x_type)

        gamma = self.param(
            "gamma", nn.initializers.ones, (1, 1, x.shape[-1]), self.model_dtype
        )
        beta = self.param(
            "beta", nn.initializers.zeros, (1, 1, x.shape[-1]), self.model_dtype
        )

        x = x * gamma + beta

        return x

class Embedding(nn.Module):
    model_dimension: int
    vocab_size: int
    model_dtype: jnp.dtype

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dimension,
            dtype=self.model_dtype,
        )

        self.norm = RMSNorm(model_dtype=self.model_dtype)

    def __call__(self, x: Array, out: bool = False) -> Array:
        if not out:
            x = self.embedding(x)
            if self.is_mutable_collection("params"):
                _ = self.norm(x)
        else:
            x = self.norm(x)
            x = self.embedding.attend(x)

        return x



class RoPE(nn.Module):
    T: int
    model_dim: int

    def setup(self):
        assert self.model_dim % 2 == 0, "model_dim must be even"

        freq = jnp.arange(self.T, dtype=jnp.float32)[:, None] + 1

        pos = jnp.arange(self.model_dim // 2, dtype=jnp.float32)[:, None]
        pos = pos.repeat(2, axis=-1).reshape(1, -1)
        log_theta_base = jnp.log(10000.0)
        theta = jnp.exp(-2 * pos / self.model_dim * log_theta_base)

        self.cos = jnp.cos(freq * theta)
        self.sin = jnp.sin(freq * theta)

    def __call__(
        self,
        x: Array,
        t_start: int,
    ) -> Array:
        B, T, C = x.shape
        x_dtype = x.dtype
        x = x.astype(jnp.float32)

        cos_rope = x * self.cos[t_start : t_start + T, :]

        x_inter = x.reshape((B, T, C // 2, 2))
        x_inter_one = x_inter[..., 0]
        x_inter_two = -1 * x_inter[..., 1]
        x_inter = jnp.stack([x_inter_two, x_inter_one], axis=-1).reshape((B, T, C))

        x_inter = x_inter.reshape((B, T, C))
        sin_rope = x_inter * self.sin[t_start : t_start + T, :]

        x = cos_rope + sin_rope
        x = x.astype(x_dtype)

        return x

class MLA(nn.Module):
    model_dimension: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int
    model_dtype: jnp.dtype
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        cKV_cache: Optional[Array] = None,
        kRT_cache: Optional[Array] = None,
        train=True,
    ) -> Tuple[Array, Tuple[Array, Array]]:

        use_rope = self.dhR > 0

        B, T, C = x.shape

        x = Dense(features = 2 * self.latent_dim, dtype=self.model_dtype)(x)
        cKVt, cqt = jnp.split(x, 2, axis=-1)

        if use_rope:
            t_start = cKV_cache.shape[1] if cKV_cache is not None else 0
            x_k_r = Dense(features=self.dhR, dtype=self.model_dtype)(x)
            x_q_r = Dense(features=self.dhR * self.n_heads, dtype=self.model_dtype)(x)

            rope_k = RoPE(model_dim=self.dhR, T=self.T)
            rope_q = RoPE(model_dim=self.dhR * self.n_heads, T=self.T)

            kRt = rope_k(x_k_r, t_start)

            qRt = rope_q(x_q_r, t_start)
            qRt = rearrange(qRt, "B T (nh d) -> B nh T d", nh=self.n_heads)

        if not train:
            if cKV_cache is not None:
                cKVt = jnp.concatenate([cKV_cache, cKVt], axis=1)
            cKV_cache = cKVt

            if use_rope:
                if kRT_cache is not None:
                    kRt = jnp.concatenate([kRT_cache, kRt], axis=1)
                kRT_cache = kRt

        k, v = jnp.split(
            Dense(features=2 * self.model_dimension, dtype=self.model_dtype)(cKVt), 2, axis=-1
        )
        q = Dense(features=self.model_dimension, dtype=self.model_dtype)(cqt)

        qkv = jnp.concat([q, k, v], axis=0)
        qkv = rearrange(
            qkv,
            "B T (nh dk) -> B nh T dk",
            B=B * 3,
            nh=self.n_heads,
            dk=C // self.n_heads,
        )

        q, k, v = jnp.split(qkv, 3, axis=0)

        if use_rope:


            q = jnp.concatenate([q, qRt], axis=-1)
            kRt = jnp.repeat(kRt[:, None, :, :], self.n_heads, axis=1)

            k = jnp.concatenate([k, kRt], axis=-1)

        def scaledDotProd(q, k, v, mask):
            input_dtype = q.dtype

            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)
            v = v.astype(jnp.float32)

            dk = q.shape[-1]

            w = jnp.einsum("B n T d, B n t d -> B n T t", q, k) * (dk ** -0.5)
            w = jnp.where(mask == 0, -jnp.inf, w)
            w = jax.nn.softmax(w, axis=-1).astype(self.model_dtype)
            output = jnp.einsum("B n T t, B n t d -> B n T d", w, v)

            output = output.astype(input_dtype)
            return output

        local_n_heads = q.shape[1]
        if T == 1:
            mask = jnp.ones((B, local_n_heads, 1, k.shape[2]))
        else:
            mask = jnp.tril(
                jnp.ones((B, local_n_heads, q.shape[2], k.shape[2])),
            )
        output = scaledDotProd(q, k, v, mask)


        output = rearrange(output, "B nh T dk -> B T (nh dk)")

        output = Dense(features=self.model_dimension, dtype=self.model_dtype)(output)
        output = nn.Dropout(rate=self.dropout)(output, deterministic=not train)

        return output, (cKV_cache, kRT_cache)

class Layer(nn.Module):
    model_dimension: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, cache, train=True):
        x_res = x

        x = RMSNorm(model_dtype=self.model_dtype)(x)
        x, cache = MLA(
            model_dimension=self.model_dimension,
            n_heads=self.n_heads,
            T=self.T,
            latent_dim=self.latent_dim,
            dhR=self.dhR,
            model_dtype=self.model_dtype,
            dropout=self.dropout_rate,
        )(x, cKV_cache=cache[0], kRT_cache=cache[1],train=train)

        x = x + x_res
        x_res = x

        x = RMSNorm(model_dtype=self.model_dtype)(x)
        x = FeedForward(
            model_dimension=self.model_dimension,
            dropout_rate=self.dropout_rate,
            model_dtype=self.model_dtype,
        )(x, train=train)
        x = x + x_res

        return x, cache

class Block(nn.Module):
    layers: int
    model_dimension: int
    n_heads: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, cache:Optional[Tuple[Array, Optional[Array]]]=None, train=True):

        cKV_cache = []
        kRT_cache = []

        for i in range(self.layers):
            current_cache = [None, None]
            if cache is not None:
                current_cache[0] = cache[0][i]
                if i < self.layers - 1:
                    current_cache[1] = cache[1][i]

            x, cache_out = Layer(
                model_dimension=self.model_dimension,
                n_heads=self.n_heads,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR if i < self.layers - 1 else 0,
                dropout_rate=self.dropout_rate,
                model_dtype=self.model_dtype
            )(x, current_cache, train=train)

            ckV, kRT = cache_out
            if ckV is not None:
                cKV_cache.append(ckV)
            if kRT is not None:
                kRT_cache.append(kRT)

        if len(cKV_cache) > 0:
            cKV_cache = jnp.stack(cKV_cache, axis=0)
        else:
            cKV_cache = None

        if len(kRT_cache) > 0:
            kRT_cache = jnp.stack(kRT_cache, axis=0)
        else:
            kRT_cache = None

        out_cache = (cKV_cache, kRT_cache)

        return x, out_cache

class Transformer(nn.Module):
    model_dimension: int
    vocab_size: int
    n_head: int
    blocks: int
    layers_per_block: int
    T: int
    latent_dim: int
    dhR: int
    dropout_rate: float = 0.1
    model_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, cache=None, train=True):

        *B, T = x.shape
        x = x.reshape(-1, T)
        embedding = Embedding(
            vocab_size=self.vocab_size,
            model_dimension=self.model_dimension,
            model_dtype=self.model_dtype,
        )

        x = embedding(x)

        cKV_cache = []
        ckRT_cache = []

        for i in range(self.blocks):
            if cache is None:
                layer_cache = None
            else:
                cKV = cache[0][i]
                kRT = cache[1][i] if cache[1] is not None else None
                layer_cache = (cKV, kRT)

            x, cache_out = Block(
                layers=self.layers_per_block,
                model_dimension=self.model_dimension,
                n_heads=self.n_head,
                T=self.T,
                latent_dim=self.latent_dim,
                dhR=self.dhR,
                dropout_rate=self.dropout_rate,
                model_dtype=self.model_dtype
            )(x, layer_cache, train=train)

            if cache_out[0] is not None:
                cKV_cache.append(cache_out[0])
            if cache_out[1] is not None:
                ckRT_cache.append(cache_out[1])

        if len(cKV_cache) > 0:
            cKV_cache = jnp.stack(cKV_cache, axis=0)
        else:
            cKV_cache = None
        if len(ckRT_cache) > 0:
            ckRT_cache = jnp.stack(ckRT_cache, axis=0)
        else:
            ckRT_cache = None
        out_cache = (cKV_cache, ckRT_cache)

        x_out = embedding(x, out=True)
        x_out = x_out.reshape(*B, T, self.vocab_size)

        return x_out, out_cache

    @classmethod
    def get_model(cls, cfg: ModelConfig) -> "Transformer":
        return cls(
            model_dimension=cfg.model_dimension,
            vocab_size=cfg.vocab_size,
            n_head=cfg.n_head,
            blocks=cfg.blocks,
            layers_per_block=cfg.layers_per_block,
            T=cfg.T,
            latent_dim=cfg.latent_dim,
            dhR=cfg.dhR,
            dropout_rate=cfg.dropout_rate,
            model_dtype=cfg.model_dtype
        )

    #TODO: Implement generate method
    def generate(self):
        return NotImplementedError()

class shardedModel:

    def __init__(self, cfg: ModelConfig):
        self.embedding = Embedding(
            vocab_size=cfg.vocab_size,
            model_dimension=cfg.model_dimension,
            model_dtype=cfg.model_dtype
        )

        self.block = Block(
            layers=cfg.layers_per_block,
            model_dimension=cfg.model_dimension,
            n_heads=cfg.n_head,
            T=cfg.T,
            latent_dim=cfg.latent_dim,
            dhR=cfg.dhR,
            dropout_rate=cfg.dropout_rate,
            model_dtype=cfg.model_dtype
        )

        self.cfg = cfg

    def init_weights(self, key, mesh):

        out_spec = shardedModel.get_p_spec([self.embedding, self.block], mesh, self.cfg)

        x_embed = jnp.ones((1, self.cfg.T), dtype=jnp.int32)
        x_layer = jnp.ones((1, self.cfg.T, self.cfg.model_dimension), dtype=jnp.float32)

        layer_devices = mesh.devices.shape[1]

        assert self.cfg.blocks // layer_devices
        layers_per_device = self.cfg.blocks // layer_devices

        key, embed_key = jax.random.split(key, 2)
        key, *layer_keys = jax.random.split(key, layer_devices + 1)
        layer_keys = jnp.array(layer_keys).reshape(layer_devices, 2)

        @jax.jit
        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(None, None), P(None, None, None), P("pp")),
            out_specs=out_spec,
        )
        def init_params(x_embed, x_layer, layer_key):
            layer_key = layer_key.reshape(2,)
            embedding_params = self.embedding.init(embed_key, x_embed, out=False)[
                "params"
            ]
            layer_params = []

            for _ in range(layers_per_device):
                layer_key, init_key = jax.random.split(layer_key)
                current_params = self.block.init(init_key, x_layer, train=False)["params"]
                layer_params.append(current_params)
            layer_params = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *layer_params)

            return embedding_params, layer_params


        params = init_params(x_embed, x_layer, layer_keys)
        params = jax.tree.map(
            lambda x, y: jax.device_put(x, jax.sharding.NamedSharding(mesh, y)),
            params,
            out_spec,
        )

        return params

    def pipe_step(self, params, x, key, train, cache=None):
        embedding_params, layer_params = params

        embeddings = self.embedding.apply({"params": embedding_params}, x, out=False)
        layer_fn = lambda x, params, cache, key: self.block.apply(
            {"params": params},
            x,
            cache=cache,
            train=train,
            rngs=None if not train else {"dropout": key},
        )

        @partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def fwd_fn(x, params, cache, key, state_idx):
            fns = [
                lambda x, *args: jax.lax.stop_gradient(layer_fn(x, *args)),
                lambda x, *args: layer_fn(x, *args),
            ]

            return jax.lax.switch(
                state_idx,
                fns,
                x,
                params,
                cache,
                key,
            )

        layer_out, (current_cache, load) = self.layer_fn(
            fwd_fn, embeddings, layer_params, key, cache=cache
        )

        logits = self.embedding.apply(
            {"params": embedding_params}, layer_out, out=True
        )

        return logits, (current_cache, load)

    def layer_fn(self, fwd_fn, x, params, key, cache=None):
        idx = jax.lax.axis_index("pp")
        n_devices = jax.lax.axis_size("pp")
        microbatch_per_device, B, T, C = x.shape
        microbatch = n_devices * microbatch_per_device
        layers_per_device = params["Layer_0"]["MLA_0"]["Dense_0"]["Dense_0"][
            "kernel"
        ].shape[0]
        layers = layers_per_device * n_devices
        perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]

        outputs = jnp.zeros_like(x) * jnp.nan
        state = jnp.zeros((layers_per_device, B, T, C), dtype=x.dtype) * jnp.nan

        state_idx = jnp.zeros((layers_per_device,), dtype=jnp.int32)

        out_load = None

        cKV_cache = []
        cKRT_cache = []

        for i in range(layers + microbatch - 1):
            batch_idx = i % microbatch_per_device
            layer_idx = (i + 1 - layers) % microbatch_per_device

            state = state.at[0].set(jnp.where(idx == 0, x[batch_idx], state[0]))
            state_idx = state_idx.at[0].set(jnp.where(idx == 0, 1, state_idx[0]))


            key, *dropout_key = jax.random.split(key, layers_per_device + 1)
            dropout_key = jnp.array(dropout_key)

            current_cache = [None, None]
            if cache is not None:
                current_cache[0] = cache[0][i]
                if cache[1] is not None:
                    current_cache[1] = cache[1][i]

            state, layer_cache = jax.vmap(fwd_fn)(
                state,
                params,
                current_cache,
                dropout_key,
                state_idx,
            )

            if layer_cache is not None:
                if layer_cache[0] is not None:
                    cKV_cache.append(layer_cache[0])
                if layer_cache[1] is not None:
                    cKRT_cache.append(layer_cache[1])

            outputs = outputs.at[layer_idx].set(
                jnp.where(
                    idx == (n_devices - 1),
                    state[-1],
                    outputs[layer_idx],
                )
            )

            state_perm = jax.lax.ppermute(
                state[-1],
                axis_name="model",
                perm=perm,
            )

            state = jnp.roll(state, shift=1, axis=0).at[0].set(state_perm)

            state_idx_perm = jax.lax.ppermute(
                state_idx[-1],
                axis_name="pp",
                perm=perm,
            )

            state_idx = jnp.roll(state_idx, shift=1, axis=0).at[0].set(state_idx_perm)

            if batch_idx == microbatch_per_device - 1:
                x = jax.lax.ppermute(
                    x,
                    axis_name="pp",
                    perm=perm,
                )

            if layer_idx == microbatch_per_device - 1:
                outputs = jax.lax.ppermute(
                    outputs,
                    axis_name="pp",
                    perm=perm,
                )

        if len(cKV_cache) > 0:
            cKV_cache = jnp.stack(cKV_cache, axis=0)
        else:
            cKV_cache = None
        if len(cKRT_cache) > 0:
            cKRT_cache = jnp.stack(cKRT_cache, axis=0)
        else:
            cKRT_cache = None
        out_cache = (cKV_cache, cKRT_cache)

        outputs = jax.lax.ppermute(
            outputs,
            axis_name="pp",
            perm=perm,
        )

        return outputs, (out_cache, out_load)

    @staticmethod
    def get_p_spec(model: Tuple[Embedding, Block], mesh, config: ModelConfig) -> Tuple[jax.sharding.NamedSharding, jax.sharding.NamedSharding]:
        T = config.T
        n_blocks = mesh.devices.shape[1]
        n_layers = config.blocks

        embed, layer = model

        x_embed = jnp.ones((1, T), dtype=jnp.int32)
        x_layer = jnp.ones((1, T, embed.model_dimension), dtype=jnp.float32)
        key = jax.random.PRNGKey(0)

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(None, None), P(None, None, None)),
            out_specs=(P("pp")),
        )
        def get_var_spec_shard(x_embed, x_layer):
            embed_shape = embed.init(key, x_embed)["params"]
            layer_shape = []
            for _ in range(n_layers // n_blocks):
                layer_shape.append(layer.init(key, x_layer, train=False)["params"])
            layer_shape = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *layer_shape)

            return embed_shape, layer_shape

        eval_shape = jax.eval_shape(
            get_var_spec_shard,
            x_embed,
            x_layer,
        )

        join_fn = lambda path: " ".join(i.key for i in path).lower()

        def layer_partition(key: Tuple[str, ...], x: Array) -> P:
            path = join_fn(key)
            if "moe" in path and "feedforward" in path:
                if x.ndim == 4:
                    return P("pp", None, None, None)
                if x.ndim == 3:
                    return P("pp", None, None)
            if "gamma" in path or "beta" in path:
                return P("pp", None, None, None)

            if x.ndim == 3:
                return P("pp", None, None)
            return P("pp")

        embed_p_spec = jax.tree.map(
            lambda _: P(),
            eval_shape[0],
        )

        layer_p_spec = jax.tree.map_with_path(
            layer_partition,
            eval_shape[1],
        )

        return embed_p_spec, layer_p_spec