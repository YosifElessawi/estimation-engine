"""Step 5's nets in JAX: the player and the judge.

- Player: one net for every decision. An MLP over Essam's raw view (estimation.jaxenv.encode)
  with a move output over the 120 action ids and the practice questions' outputs (``PRACTICE``).
- Judge, training only: the same view plus the three hidden hands (``HIDDEN`` inputs), valuing
  the player's final place. Thrown away after training; the player never sees hidden cards.

Both are built as Essam Gouda's are in estimation.jaxenv.selfplay (ReLU layers, orthogonal start,
the move output starting near uniform). Saved as plain numpy arrays (the player with Essam's key
names, so his loaders read its body and move output), so the Mac can play it without JAX.
"""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from estimation.jaxenv.encode import SIZE
from estimation.jaxenv.rules import NUM_ACTIONS
from estimation.jaxenv.selfplay import HIDDEN, _body, _head, _mlp, _out

Net = dict[str, Any]

# The practice questions (plan step 5), each an output block of the player, in this order.
PRACTICE: tuple[tuple[str, int], ...] = (
    ("holder", 52 * 3),  # 1. who holds each hidden card: one of the three other seats
    ("my_tricks", 14),  # 2. my tricks this round, 0-13
    ("made", 4),  # 3. will each player make their call (the player first)
    ("other_tricks", 3 * 14),  # 4. each other player's tricks this round
    ("void", 3 * 4),  # 5. is each other seat out of each suit
    ("king", 1),  # 6. my chance of finishing King
    ("round_score", 4),  # 7. each player's round score, over 50
)


def _slices() -> dict[str, slice]:
    out: dict[str, slice] = {}
    start = 0
    for name, size in PRACTICE:
        out[name] = slice(start, start + size)
        start += size
    return out


SLICES = _slices()
PRACTICE_SIZE = sum(size for _, size in PRACTICE)


def parse_size(text: str) -> tuple[int, int]:
    """``"384x3"``: 384 units a layer, 3 layers."""
    try:
        hidden, depth = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"a net size is units x layers, such as 384x3, not {text!r}") from None
    if hidden < 1 or depth < 1:
        raise ValueError(f"a net size needs at least 1 unit and 1 layer, not {text!r}")
    return hidden, depth


def init_player(key: jax.Array, hidden: int, depth: int) -> Net:
    k_body, k_move, k_practice = jax.random.split(key, 3)
    return {
        "body": _mlp(k_body, SIZE, hidden, depth),
        "policy": _head(k_move, hidden, NUM_ACTIONS, 0.01),
        "practice": _head(k_practice, hidden, PRACTICE_SIZE, 1.0),
    }


def init_judge(key: jax.Array, hidden: int, depth: int) -> Net:
    k_body, k_value = jax.random.split(key)
    return {
        "body": _mlp(k_body, SIZE + HIDDEN, hidden, depth),
        "value": _head(k_value, hidden, 1, 1.0),
    }


def apply_player(net: Net, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """[B, 120] move logits (mask the illegal ones before use) and [B, PRACTICE_SIZE] outputs."""
    h = _body(net["body"], x)
    return _out(net["policy"], h), _out(net["practice"], h)


def apply_judge(net: Net, x: jax.Array, hidden: jax.Array) -> jax.Array:
    """[B]: the expected final place reward, from the view and the three hidden hands."""
    h = _body(net["body"], jnp.concatenate([x, hidden.astype(x.dtype)], -1))
    return _out(net["value"], h)[..., 0]


def hidden_hands(holder: jax.Array) -> jax.Array:
    """[B, 156]: the three other seats' cards (relative seats 1-3), from the holder of each card:
    the judge's extra input, as estimation.jaxenv.selfplay.hidden_of builds it in its games."""
    seats = holder[:, None, :] == jnp.arange(3, dtype=holder.dtype)[None, :, None]
    return seats.reshape(holder.shape[0], 3 * 52)


def _save(path: Path, arrays: dict[str, Any], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # readers never see a half-written net
    plain = {k: np.asarray(v, np.float32) for k, v in arrays.items()}
    with tmp.open("wb") as file:
        np.savez_compressed(file, allow_pickle=False, meta=np.array(json.dumps(meta)), **plain)
    tmp.replace(path)


def _body_arrays(net: Net) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for k, layer in enumerate(net["body"]):
        arrays[f"w{k}"], arrays[f"b{k}"] = layer["w"], layer["b"]
    return arrays


def save_player(path: Path, net: Net, meta: dict[str, Any]) -> None:
    arrays = _body_arrays(net)
    arrays["pw"], arrays["pb"] = net["policy"]["w"], net["policy"]["b"]
    arrays["qw"], arrays["qb"] = net["practice"]["w"], net["practice"]["b"]
    _save(path, arrays, {**meta, "practice": [list(p) for p in PRACTICE]})


def save_judge(path: Path, net: Net, meta: dict[str, Any]) -> None:
    arrays = _body_arrays(net)
    arrays["vw"], arrays["vb"] = net["value"]["w"], net["value"]["b"]
    _save(path, arrays, meta)


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"]))
    return arrays, meta


def _body_of(arrays: dict[str, np.ndarray]) -> list[dict[str, jax.Array]]:
    depth = sum(1 for k in arrays if k.startswith("w"))
    return [
        {"w": jnp.asarray(arrays[f"w{k}"]), "b": jnp.asarray(arrays[f"b{k}"])} for k in range(depth)
    ]


def load_player(path: Path) -> tuple[Net, dict[str, Any]]:
    arrays, meta = _load(path)
    if [list(p) for p in PRACTICE] != meta.get("practice"):
        raise ValueError(
            f"{path} has other practice outputs than this code: {meta.get('practice')}"
        )
    net = {
        "body": _body_of(arrays),
        "policy": {"w": jnp.asarray(arrays["pw"]), "b": jnp.asarray(arrays["pb"])},
        "practice": {"w": jnp.asarray(arrays["qw"]), "b": jnp.asarray(arrays["qb"])},
    }
    return net, meta


def load_judge(path: Path) -> tuple[Net, dict[str, Any]]:
    arrays, meta = _load(path)
    net = {
        "body": _body_of(arrays),
        "value": {"w": jnp.asarray(arrays["vw"]), "b": jnp.asarray(arrays["vb"])},
    }
    return net, meta
