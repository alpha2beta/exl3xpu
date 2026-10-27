"""EXL3 routed-expert quantization method (Phase 1: dispatch + expert loading).

Why this file exists
--------------------
``Exl3Config.get_quant_method`` used to return ``None`` for
``RoutedExperts``.  vLLM turns ``None`` into
``UnquantizedFusedMoEMethod`` (``routed_experts.py:219-220``), which allocates
256 x (2 x 2048x512 + 512x2048) fp16 experts per layer -- ~64 GiB for this
model -- before failing on the missing ``.weight`` tensors.  The project rule
is fail-closed, so this module registers a real ``FusedMoEMethodBase`` that
either serves the layer or raises.

Stage boundary
--------------
P1-01 wired dispatch.  P1-02 registers one compressed allocation per tensor
kind per layer (trellis / suh / svh / codebook scalars, no fp16 expert weights)
and loads them through a purpose-built expert loader.  P1-03 packs int64
per-expert descriptors, and P1-04 adds ``verify_expert_integrity`` (stride/
offset tiling, no metadata aliasing, tile-not-dense trellis).  P1-05 streams
shards; Phase 2 adds the eager routed-expert ``apply`` (per-expert grouping,
weighted scatter-accumulate) and an optional intermediate-capture hook.  There
is no unquantized or Triton-backed expert path to fall back to.

Why a custom ``load_weights``
-----------------------------
``RoutedExperts.load_weights`` decides "fused checkpoint tensor" with
``loaded_weight.dim() == 3`` and then chunks along dim 1 for ``w1``/``w3``.
EXL3 trellis tensors are *also* 3-D (``[k/16, n/16, 16K]`` per expert), so the
stock loader would split one expert's trellis into 64 fake experts.  The
method therefore binds a per-layer loader that parses
``<expert>.<proj>.<trellis|suh|svh|codebook>`` explicitly and copies each
tensor into ``[expert, slot]``.  See ``docs/moe_api_audit.md`` §3.2.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    from .vllm_plugin import Exl3Config

#: EXL3 codebook name -> numeric id used by the kernels and the checkpoint.
CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}

try:
    from vllm_xpu_kernels import _moe_C  # noqa: F401

    _HAS_MOE_C = (
        hasattr(torch.ops, "_moe_C")
        and hasattr(torch.ops._moe_C, "remap_hidden_states")
        and hasattr(torch.ops._moe_C, "moe_gather")
    )
except Exception:
    _HAS_MOE_C = False

#: Projections every routed expert must provide in this checkpoint schema.
CHECKPOINT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# One-shot per-path trace (EXL3_TRACE_PATHS=1) so a latched or unreachable
# grouped path cannot stay silent — §11.8 G0 of mtp_moe.md.
_PATH_SEEN: set[str] = set()
_TRACE_PATHS: bool | None = None


def _trace_path(name: str, detail: str = "") -> None:
    global _TRACE_PATHS
    if _TRACE_PATHS is None:
        import os

        _TRACE_PATHS = bool(os.environ.get("EXL3_TRACE_PATHS"))
    if not _TRACE_PATHS or name in _PATH_SEEN:
        return
    _PATH_SEEN.add(name)
    import sys

    print(f"[exl3] path engaged: {name}{detail}", file=sys.stderr, flush=True)


class Exl3MoEUnsupported(NotImplementedError):
    """The MoE layer cannot be served by EXL3.

    Deliberately an exception rather than a ``None`` return: ``None`` makes
    vLLM silently construct ``UnquantizedFusedMoEMethod`` (~64 GiB of fp16
    experts).  Callers may catch this to produce a richer error, but must not
    downgrade it to an unquantized path.
    """


#: Level Zero per-allocation ceiling from the P1-02 plan. Every registered
#: expert tensor must stay below this *and* the per-layer total must too.
MAX_ALLOCATION_BYTES = 4 * 1024**3

#: MoE shard id -> fused slot index inside w13 (w2 has a single slot).
_SHARD_SLOT = {"w1": 0, "w3": 1, "w2": 0}

#: Canonical projection order used by the packed expert descriptors.
_PROJ_ORDER = ("gate_proj", "up_proj", "down_proj")
_PROJ_SHARDS = ("w1", "w3", "w2")

#: Fields of one descriptor row (int64), indexed by
#: ``layer.exl3_moe_descriptors[expert, projection, field]``.
#: ``*_tensor`` indexes ``layer.exl3_moe_tensor_names``; ``*_offset`` is an
#: element offset into that allocation. -1 marks an absent tensor (3inst has no
#: codebook scalar).
DESCRIPTOR_FIELDS = (
    "trellis_tensor",
    "trellis_offset",
    "suh_tensor",
    "suh_offset",
    "svh_tensor",
    "svh_offset",
    "codebook_tensor",
    "codebook_offset",
    "k",
    "n",
    "K",
    "cb",
    "family",
    "slot",
)

#: family id used in descriptor rows.
_FAMILY_ID = {"w13": 0, "w2": 1}


def _s32(value: int) -> int:
    """Interpret an unsigned 32-bit constant as a signed int32 (checkpoint form)."""
    return value - (1 << 32) if value >= (1 << 31) else value


#: Value stored in the per-expert codebook scalar. The tensor's *presence*
#: selects the codebook; its value is the codebook multiplier, so a mismatch
#: means the checkpoint is not what the config claims.
CODEBOOK_MULTIPLIERS = {
    "mcg": _s32(0xCBAC1FED),
    "mul1": _s32(0x83DCD12D),
}


def _trellis_shape(k: int, n: int, K: int) -> tuple[int, int, int]:
    return (k // 16, n // 16, 16 * K)


@dataclass(frozen=True)
class ExpertTensorSpec:
    """One per-layer stacked expert tensor (all experts share the layout)."""

    name: str  # parameter name on RoutedExperts, e.g. "w13_trellis"
    tensor: str  # checkpoint suffix: trellis | suh | svh | <codebook>
    slots: int  # 2 for w13 (gate+up), 1 for w2 (down)
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


class Exl3ExpertStore:
    """Shape/dtype/budget plan for one layer's compressed expert tensors.

    All experts of a layer live in one allocation per tensor kind (eight
    allocations per layer for the mcg checkpoint). At 3.09 bpw that is
    ~192 MiB for ``w13_trellis`` and ~96 MiB for ``w2_trellis``, so each
    allocation and the per-layer total stay far below
    ``MAX_ALLOCATION_BYTES``.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        K: int,
        codebook: str,
    ):
        if num_experts <= 0:
            raise Exl3MoEUnsupported(f"exl3: MoE: bad expert count {num_experts}")
        for label, dim in (
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
        ):
            if dim % 128 != 0:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE: {label}={dim} must be a multiple of 128 "
                    "(trellis tile / grouped-GEMV requirement)"
                )
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.K = K
        self.codebook = codebook

        def spec(name, tensor, slots, shape, dtype):
            return ExpertTensorSpec(name, tensor, slots, tuple(shape), dtype)

        specs = [
            spec(
                "w13_trellis", "trellis", 2,
                (num_experts, 2, *_trellis_shape(hidden_size, intermediate_size, K)),
                torch.int16,
            ),
            spec("w13_suh", "suh", 2, (num_experts, 2, hidden_size), torch.float16),
            spec("w13_svh", "svh", 2, (num_experts, 2, intermediate_size), torch.float16),
            spec(
                "w2_trellis", "trellis", 1,
                (num_experts, 1, *_trellis_shape(intermediate_size, hidden_size, K)),
                torch.int16,
            ),
            spec("w2_suh", "suh", 1, (num_experts, 1, intermediate_size), torch.float16),
            spec("w2_svh", "svh", 1, (num_experts, 1, hidden_size), torch.float16),
        ]
        if codebook != "3inst":
            specs += [
                spec(f"w13_{codebook}", codebook, 2, (num_experts, 2), torch.int32),
                spec(f"w2_{codebook}", codebook, 1, (num_experts, 1), torch.int32),
            ]
        self.specs: tuple[ExpertTensorSpec, ...] = tuple(specs)
        self.total_bytes = sum(s.nbytes for s in self.specs)

        oversized = [s for s in self.specs if s.nbytes >= MAX_ALLOCATION_BYTES]
        if oversized:
            raise Exl3MoEUnsupported(
                "exl3: MoE: single-allocation limit exceeded: "
                + ", ".join(
                    f"{s.name}={s.nbytes / 2**30:.2f} GiB" for s in oversized
                )
                + f" (limit {MAX_ALLOCATION_BYTES / 2**30:.0f} GiB)"
            )
        if self.total_bytes >= MAX_ALLOCATION_BYTES:
            raise Exl3MoEUnsupported(
                f"exl3: MoE: per-layer expert buffers total "
                f"{self.total_bytes / 2**30:.2f} GiB >= "
                f"{MAX_ALLOCATION_BYTES / 2**30:.0f} GiB"
            )

    def get(self, name: str) -> ExpertTensorSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def meta(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "K": self.K,
            "total_bytes": self.total_bytes,
            "max_allocation_bytes": MAX_ALLOCATION_BYTES,
            "tensors": {
                s.name: {
                    "shape": list(s.shape),
                    "dtype": str(s.dtype).removeprefix("torch."),
                    "bytes": s.nbytes,
                }
                for s in self.specs
            },
        }


class Exl3MoEMethod(FusedMoEMethodBase):
    """Quantization method for EXL3-compressed routed experts.

    Holds the per-layer format contract (K, codebook, expert count), registers
    the compressed expert tensors in :meth:`create_weights`, loads them
    through the per-layer expert loader described in the module docstring, and
    verifies packed-arena integrity in :meth:`verify_expert_integrity`.
    Execution (:meth:`apply`) is P1-05+.
    """

    def __init__(self, config: "Exl3Config", layer: "RoutedExperts", prefix: str):
        moe: FusedMoEConfig = layer.moe_config
        super().__init__(moe)

        self.config = config
        self.prefix = prefix
        # Checkpoint storage keys are normalized ("layers.0..."), while vLLM
        # passes the model-qualified prefix ("model.language_model.layers.0...").
        from .vllm_plugin import _norm_key
        self.key_prefix = _norm_key(prefix)
        self.num_local_experts = int(moe.num_local_experts)
        self.hidden_size = int(moe.hidden_dim)
        self.intermediate_size_per_partition = int(moe.intermediate_size_per_partition)

        self._validate_parallel(moe)

        codebook = config.codebook
        if codebook not in CODEBOOK_IDS:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {prefix}: unknown codebook {codebook!r}; "
                f"expected one of {sorted(CODEBOOK_IDS)}"
            )
        self.cb = CODEBOOK_IDS[codebook]

        # Raises if the checkpoint has no / inconsistent / unsupported expert
        # format for this layer.  Resolved before the constructor returns so a
        # bad checkpoint fails at model construction, not at first token.
        self.K = self._resolve_expert_K(layer)
        self._check_kernel_support()

        # Reused read-only ``shard_of_nb`` buffers (one per (device, n)).
        self._nb_cache: dict[tuple[torch.device, int], torch.Tensor] = {}
        # P4-01 zero-copy execution cache (built lazily from the arenas).
        self._exec_cache: dict[str, Any] | None = None
        # Optional P2-08 diagnostic capture (None = disabled, zero overhead).
        self._debug_capture: dict[str, Any] | None = None

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _validate_parallel(moe: FusedMoEConfig) -> None:
        pc = moe.moe_parallel_config
        if pc.tp_size > 1 or pc.ep_size > 1 or getattr(pc, "use_ep", False):
            raise Exl3MoEUnsupported(
                "exl3: MoE tensor/expert parallelism is not supported "
                f"(tp_size={pc.tp_size}, ep_size={pc.ep_size}, use_ep={pc.use_ep}); "
                "run with TP=1 and EP off"
            )

    def _checkpoint_projection_names(self, layer: "RoutedExperts") -> tuple[str, str, str]:
        """Checkpoint-side projection names for gate/up/down.

        ``RoutedExperts`` stores the ckpt names it was built with; defaults
        match the EXL3 Qwen3.5 checkpoint.  Ordering is gate, up, down.
        """
        return (
            getattr(layer, "ckpt_gate_proj_name", CHECKPOINT_PROJECTIONS[0]),
            getattr(layer, "ckpt_up_proj_name", CHECKPOINT_PROJECTIONS[1]),
            getattr(layer, "ckpt_down_proj_name", CHECKPOINT_PROJECTIONS[2]),
        )

    def _resolve_expert_K(self, layer: "RoutedExperts") -> int:
        """Validate the checkpoint schema and return the routed trellis K."""
        storage = self.config.storage
        if not storage:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: checkpoint storage map is empty, so the "
                "routed-expert format cannot be resolved (was "
                "Exl3Config.maybe_update_config called?)"
            )

        prefix = self.key_prefix + "."
        entries = {k: v for k, v in storage.items() if k.startswith(prefix)}
        if not entries:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: no EXL3 expert tensors found for this "
                "layer (partially-quantized or unquantized MoE is not supported)"
            )

        proj_names = set(self._checkpoint_projection_names(layer))
        by_expert: dict[int, dict[str, int]] = {}
        for key, bits in entries.items():
            rest = key[len(prefix):].split(".")
            if len(rest) != 2 or not rest[0].isdigit():
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: unexpected expert tensor name {key!r}"
                )
            expert_id, proj = int(rest[0]), rest[1]
            if proj not in proj_names:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: unexpected projection {proj!r} in {key!r}"
                )
            info = self.config.member_trellis.get(key)
            K = int(info[1]) if info is not None else int(bits)
            by_expert.setdefault(expert_id, {})[proj] = K

        # Completeness: an expert missing a projection would load partially and
        # silently produce garbage, so reject the layer up front.
        for expert_id, projs in sorted(by_expert.items()):
            missing = proj_names - projs.keys()
            if missing:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: expert {expert_id} is missing "
                    f"quantized projections {sorted(missing)}"
                )

        if len(by_expert) != self.num_local_experts:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: checkpoint has {len(by_expert)} experts, "
                f"layer expects {self.num_local_experts}"
            )

        Ks = {K for projs in by_expert.values() for K in projs.values()}
        if len(Ks) != 1:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: mixed trellis K across experts: {sorted(Ks)}"
            )
        return Ks.pop()

    def _check_kernel_support(self) -> None:
        """Fail closed unless the built extension has a kernel for (K, cb)."""
        from . import ops

        try:
            esimd = ops._get_esimd()
        except RuntimeError as exc:  # library missing / ABI mismatch
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: ESIMD op library is not loadable and "
                "routed experts have no Triton fallback"
            ) from exc
        if not esimd or not hasattr(esimd, "exl3_supported"):
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: ESIMD op library is not available "
                "(EXL3_BACKEND=triton disables it) and routed experts have no "
                "fallback"
            )
        if not bool(esimd.exl3_supported(self.K, self.cb)):
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: no ESIMD kernel for K={self.K}, "
                f"codebook={self.config.codebook!r} (cb={self.cb}); rebuild with "
                "-DEXL3_ALL_CODEBOOKS or use a supported bitrate"
            )

    # -- FusedMoEMethodBase contract ---------------------------------------

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """Register the compressed expert tensors for one layer.

        Exactly one allocation per tensor kind:
        ``w13_trellis[E,2,k/16,n/16,16K]``, ``w13_suh[E,2,k]``,
        ``w13_svh[E,2,n]``, ``w2_trellis[E,1,n/16,k/16,16K]``, ``w2_suh``,
        ``w2_svh`` and (for ``mcg``/``mul1``) one scalar ``int32`` per expert
        and slot.  There is no fp16 copy of any expert matrix.
        """
        if int(num_experts) != self.num_local_experts:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: create_weights got num_experts="
                f"{num_experts}, method was built for {self.num_local_experts}"
            )
        if int(hidden_size) != self.hidden_size:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: create_weights got hidden_size="
                f"{hidden_size}, method was built for {self.hidden_size}"
            )
        if int(intermediate_size_per_partition) != self.intermediate_size_per_partition:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: create_weights got "
                f"intermediate_size_per_partition={intermediate_size_per_partition}, "
                f"method was built for {self.intermediate_size_per_partition}"
            )

        store = Exl3ExpertStore(
            num_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            K=self.K,
            codebook=self.config.codebook,
        )
        self.store = store
        self._loaded: set[tuple[str, int, int]] = set()
        self._load_attempted = False

        for spec in store.specs:
            # Allocate on the active device: vLLM constructs the model inside
            # `with torch.device(target_device)` (base_loader.py:67).
            param = Parameter(
                torch.empty(spec.shape, dtype=spec.dtype), requires_grad=False
            )
            param.weight_loader = self._make_weight_loader(spec)
            param.exl3_spec = spec
            layer.register_parameter(spec.name, param)

        tensor_names = [s.name for s in store.specs]
        layer.exl3_moe_tensor_names = tensor_names
        descriptors = self._build_descriptors(layer, tensor_names)
        # Non-persistent: moves with the module, absent from state_dict, and
        # skipped by the dummy loader.
        layer.register_buffer(
            "exl3_moe_descriptors", descriptors, persistent=False
        )

        meta = store.meta()
        meta.update(
            {
                "prefix": self.key_prefix,
                "codebook": self.config.codebook,
                "codebook_id": self.cb,
                "allocations": len(store.specs),
                "tensor_order": tensor_names,
                "descriptor_fields": list(DESCRIPTOR_FIELDS),
                "descriptor_shape": list(descriptors.shape),
            }
        )
        layer.exl3_moe_meta = meta

        # Replace the generic expert loader (see module docstring): trellis
        # tensors are 3-D and would otherwise be treated as fused w13 tensors.
        layer.load_weights = types.MethodType(self._layer_load_weights, layer)

    # -- expert weight loading ---------------------------------------------

    def _make_weight_loader(self, spec: ExpertTensorSpec):
        """Per-parameter loader used by the per-layer expert loader.

        Signature mirrors vLLM's ``weight_loader`` protocol; ``return_success``
        is honoured so the stock mapping bookkeeping keeps working.
        """

        def weight_loader(
            param: Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str = "",
            shard_id: str | None = None,
            expert_id: int | None = None,
            return_success: bool = False,
            **_: Any,
        ):
            self._copy_expert_tensor(
                spec, param, loaded_weight, shard_id, expert_id, weight_name
            )
            return True if return_success else None

        return weight_loader

    def _copy_expert_tensor(
        self,
        spec: ExpertTensorSpec,
        param: Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | None,
        expert_id: int | None,
        weight_name: str,
    ) -> None:
        if shard_id not in _SHARD_SLOT:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: {spec.name} got unknown shard_id "
                f"{shard_id!r} (expected one of {sorted(_SHARD_SLOT)}) for "
                f"{weight_name!r}"
            )
        if spec.name.startswith("w13_") != (shard_id in ("w1", "w3")):
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: shard_id {shard_id!r} does not match "
                f"parameter {spec.name!r}"
            )
        if expert_id is None or not isinstance(expert_id, int):
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: {spec.name} requires an integer "
                f"expert_id, got {expert_id!r} for {weight_name!r}"
            )
        if not 0 <= expert_id < self.num_local_experts:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: expert_id {expert_id} out of range "
                f"for {spec.name} ({self.num_local_experts} experts)"
            )

        slot = _SHARD_SLOT[shard_id]
        expected = spec.shape[2:]
        if tuple(loaded_weight.shape) != expected:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: {weight_name!r} has shape "
                f"{tuple(loaded_weight.shape)}, expected {expected} for "
                f"{spec.name} (expert {expert_id}, slot {slot})"
            )
        if loaded_weight.dtype != spec.dtype:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: {weight_name!r} has dtype "
                f"{loaded_weight.dtype}, expected {spec.dtype} for {spec.name}"
            )

        param.data[expert_id, slot].copy_(loaded_weight)
        self._loaded.add((spec.name, expert_id, slot))

    def _layer_load_weights(
        self, layer: "RoutedExperts", weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[str]:
        """Layer-bound replacement for ``RoutedExperts.load_weights``.

        Consumes names relative to the ``RoutedExperts`` module (the form
        ``AutoWeightsLoader`` passes after stripping the child prefix), e.g.
        ``"0.gate_proj.trellis"``.

        Kept as a plain function returning a generator so the "a real load was
        attempted" flag is set eagerly, even if a caller forgets to iterate.
        """
        self._load_attempted = True
        return self._iter_layer_weights(layer, weights)

    def _iter_layer_weights(
        self, layer: "RoutedExperts", weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[str]:
        gate, up, down = self._checkpoint_projection_names(layer)
        shard_of_proj = {gate: "w1", up: "w3", down: "w2"}

        for name, loaded_weight in weights:
            parts = name.split(".")
            if len(parts) != 3 or not parts[0].isdigit():
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: unexpected routed-expert tensor "
                    f"{name!r} (expected '<expert>.<projection>.<suffix>')"
                )
            expert_id, proj, suffix = int(parts[0]), parts[1], parts[2]
            shard_id = shard_of_proj.get(proj)
            if shard_id is None:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: unexpected projection {proj!r} in "
                    f"{name!r} (expected one of {sorted(shard_of_proj)})"
                )
            family = "w13" if shard_id in ("w1", "w3") else "w2"
            param_name = f"{family}_{suffix}"
            try:
                spec = self.store.get(param_name)
            except KeyError:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: unexpected tensor suffix {suffix!r} "
                    f"in {name!r}; expected "
                    f"{sorted(s.tensor for s in self.store.specs)}"
                ) from None
            param = getattr(layer, param_name)
            param.weight_loader(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
            yield param_name

    # -- packed expert descriptors (P1-03) ----------------------------------

    def _build_descriptors(
        self, layer: "RoutedExperts", tensor_names: list[str]
    ) -> torch.Tensor:
        """Build ``[E, 3, len(DESCRIPTOR_FIELDS)]`` int64 descriptor rows.

        Rows are derived from the allocation plan only (no data), so every
        expert/projection resolves to an explicit ``(allocation, element
        offset, slot, k, n, K, cb)`` rather than an implicit concatenation.
        """
        index = {name: i for i, name in enumerate(tensor_names)}
        rows: list[list[int]] = []
        for expert_id in range(self.num_local_experts):
            for proj_index, shard_id in enumerate(_PROJ_SHARDS):
                family = "w13" if shard_id in ("w1", "w3") else "w2"
                slot = _SHARD_SLOT[shard_id]
                trellis = self.store.get(f"{family}_trellis")
                suh = self.store.get(f"{family}_suh")
                svh = self.store.get(f"{family}_svh")
                if self.config.codebook == "3inst":
                    cb_spec = None
                else:
                    cb_spec = self.store.get(f"{family}_{self.config.codebook}")

                def offset(spec: ExpertTensorSpec) -> int:
                    per_expert = math.prod(spec.shape[2:])
                    return (expert_id * spec.slots + slot) * per_expert

                # NOTE: projection identity follows _PROJ_ORDER/_PROJ_SHARDS;
                # custom checkpoint projection names are resolved by the loader
                # (``_checkpoint_projection_names``), not here.
                if shard_id in ("w1", "w3"):
                    k, n = self.hidden_size, self.intermediate_size_per_partition
                else:
                    k, n = self.intermediate_size_per_partition, self.hidden_size

                rows.append(
                    [
                        index[trellis.name],
                        offset(trellis),
                        index[suh.name],
                        offset(suh),
                        index[svh.name],
                        offset(svh),
                        index[cb_spec.name] if cb_spec is not None else -1,
                        offset(cb_spec) if cb_spec is not None else -1,
                        k,
                        n,
                        self.K,
                        self.cb,
                        _FAMILY_ID[family],
                        slot,
                    ]
                )
        device = getattr(layer, tensor_names[0]).device
        return torch.tensor(rows, dtype=torch.int64, device=device).reshape(
            self.num_local_experts, len(_PROJ_ORDER), len(DESCRIPTOR_FIELDS)
        )

    def descriptor(self, layer: "RoutedExperts", expert_id: int, projection: str) -> dict[str, int]:
        """One descriptor row as a named dict (raises on bad ids)."""
        if projection not in _PROJ_ORDER:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: unknown projection {projection!r}"
            )
        if not 0 <= expert_id < self.num_local_experts:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: expert_id {expert_id} out of range"
            )
        row = layer.exl3_moe_descriptors[
            expert_id, _PROJ_ORDER.index(projection)
        ].tolist()
        return dict(zip(DESCRIPTOR_FIELDS, row))

    def expert_views(
        self, layer: "RoutedExperts", expert_id: int
    ) -> dict[str, dict[str, torch.Tensor | None]]:
        """Actual parameter views for one expert, one dict per projection.

        This is the P1-03/P1-04 bridge: consumers never index the stacked
        allocation by hand, and tests can compare views against the descriptor
        offsets.
        """
        out: dict[str, dict[str, torch.Tensor | None]] = {}
        for projection in _PROJ_ORDER:
            row = self.descriptor(layer, expert_id, projection)
            shard_id = _PROJ_SHARDS[_PROJ_ORDER.index(projection)]
            family = "w13" if shard_id in ("w1", "w3") else "w2"
            slot = row["slot"]
            codebook = None
            if row["codebook_tensor"] >= 0:
                cb_name = layer.exl3_moe_tensor_names[row["codebook_tensor"]]
                codebook = getattr(layer, cb_name)[expert_id, slot]
            out[projection] = {
                "trellis": getattr(layer, f"{family}_trellis")[expert_id, slot],
                "suh": getattr(layer, f"{family}_suh")[expert_id, slot],
                "svh": getattr(layer, f"{family}_svh")[expert_id, slot],
                "codebook": codebook,
                "k": row["k"],
                "n": row["n"],
                "K": row["K"],
                "cb": row["cb"],
            }
        return out

    # -- integrity verification (P1-04) -------------------------------------

    def verify_expert_integrity(self, layer: "RoutedExperts") -> dict[str, Any]:
        """Assert the packed arenas keep per-expert metadata intact (P1-04).

        Structural only (no checkpoint values), so it is safe for
        ``--load-format dummy`` too:

        * each tensor kind owns a distinct allocation;
        * every ``(expert, slot)`` slice owns distinct storage, so ``suh``/``svh``
          scales and Hadamard vectors are never shared between experts or
          between gate and up;
        * registered parameter shapes match the plan (tiled, not dense);
        * the ``[E, slots, ...]`` arena is exactly tiled by the descriptor
          offsets ``(expert*slots + slot) * per_expert``, so codes are always
          addressed through explicit stride/offset tracking;
        * trellis uses the tile layout ``[k/16, n/16, 16K]`` (``w2`` is
          ``[n/16, k/16, 16K]``), never a dense ``[k, n]`` row store;
        * ``suh``/``svh`` lengths equal ``k``/``n`` and codebook entries carry
          the expected multiplier.

        Returns a small summary (used in evidence) and raises
        :class:`Exl3MoEUnsupported` on any violation.
        """
        names = list(getattr(layer, "exl3_moe_tensor_names", []))
        if names != [s.name for s in self.store.specs]:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: expert tensor order {names} does not "
                "match the allocation plan"
            )

        descriptors = getattr(layer, "exl3_moe_descriptors", None)
        expected_shape = (
            self.num_local_experts,
            len(_PROJ_ORDER),
            len(DESCRIPTOR_FIELDS),
        )
        if descriptors is None or tuple(descriptors.shape) != expected_shape:
            got = None if descriptors is None else tuple(descriptors.shape)
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: descriptor shape {got} != {expected_shape}"
            )
        if descriptors.dtype != torch.int64:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: descriptor dtype {descriptors.dtype} "
                "!= int64"
            )

        # 1. A distinct allocation per tensor kind.
        owners: dict[int, str] = {}
        for name in names:
            ptr = getattr(layer, name).data_ptr()
            if ptr in owners:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: {name} aliases the allocation of "
                    f"{owners[ptr]}"
                )
            owners[ptr] = name

        # 2. Registered shapes match the plan; every (expert, slot) slice owns
        #    distinct storage and the arena is exactly tiled.
        for spec in self.store.specs:
            param = getattr(layer, spec.name)
            if tuple(param.shape) != spec.shape:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: {spec.name} has shape "
                    f"{tuple(param.shape)}, expected tiled {spec.shape} (codes must "
                    "not be stored as concatenated dense rows)"
                )
            per = math.prod(spec.shape[2:])
            if param.numel() != self.num_local_experts * spec.slots * per:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: {spec.name} arena size "
                    f"{param.numel()} is not E*slots*per={self.num_local_experts}"
                    f"*{spec.slots}*{per}"
                )
            seen: set[int] = set()
            for expert_id in range(self.num_local_experts):
                for slot in range(spec.slots):
                    ptr = param[expert_id, slot].data_ptr()
                    if ptr in seen:
                        raise Exl3MoEUnsupported(
                            f"exl3: MoE {self.prefix}: {spec.name}[expert="
                            f"{expert_id},slot={slot}] aliases another expert slice"
                        )
                    seen.add(ptr)

        expected_multiplier = CODEBOOK_MULTIPLIERS.get(self.config.codebook)
        for expert_id in range(self.num_local_experts):
            for proj_index, shard_id in enumerate(_PROJ_SHARDS):
                family = "w13" if shard_id in ("w1", "w3") else "w2"
                slot = _SHARD_SLOT[shard_id]
                row = dict(
                    zip(DESCRIPTOR_FIELDS, descriptors[expert_id, proj_index].tolist())
                )
                if row["family"] != _FAMILY_ID[family] or row["slot"] != slot:
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                        f"{proj_index}] family/slot {row['family']}/{row['slot']} "
                        f"!= {_FAMILY_ID[family]}/{slot}"
                    )
                if row["K"] != self.K or row["cb"] != self.cb:
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                        f"{proj_index}] K/cb {row['K']}/{row['cb']} != "
                        f"{self.K}/{self.cb}"
                    )

                if family == "w13":
                    k = self.hidden_size
                    n = self.intermediate_size_per_partition
                else:
                    k = self.intermediate_size_per_partition
                    n = self.hidden_size
                # Trellis tiles are always [k/16, n/16, 16K] (see
                # docs/checkpoint_schema.md); w2 is *not* transposed.
                tile = _trellis_shape(k, n, self.K)
                if row["k"] != k or row["n"] != n:
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                        f"{proj_index}] k/n {row['k']}/{row['n']} != {k}/{n}"
                    )

                for kind, spec_name in (
                    ("trellis", f"{family}_trellis"),
                    ("suh", f"{family}_suh"),
                    ("svh", f"{family}_svh"),
                ):
                    spec = self.store.get(spec_name)
                    per = math.prod(spec.shape[2:])
                    if row[f"{kind}_tensor"] != names.index(spec.name):
                        raise Exl3MoEUnsupported(
                            f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                            f"{proj_index}] {kind}_tensor {row[f'{kind}_tensor']} "
                            f"!= {names.index(spec.name)} ({spec.name})"
                        )
                    expected_offset = (expert_id * spec.slots + slot) * per
                    if row[f"{kind}_offset"] != expected_offset:
                        raise Exl3MoEUnsupported(
                            f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                            f"{proj_index}] {kind}_offset {row[f'{kind}_offset']} "
                            f"!= strided offset {expected_offset}"
                        )
                    if expected_offset + per > getattr(layer, spec.name).numel():
                        raise Exl3MoEUnsupported(
                            f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                            f"{proj_index}] {kind} slice runs past {spec.name}"
                        )

                trellis_spec = self.store.get(f"{family}_trellis")
                if tuple(trellis_spec.shape[2:]) != tile:
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: {family}_trellis tile "
                        f"{tuple(trellis_spec.shape[2:])} != {tile}"
                    )
                if self.store.get(f"{family}_suh").shape[2] != k:
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: {family}_suh length "
                        f"{self.store.get(f'{family}_suh').shape[2]} != k={k}"
                    )
                if self.store.get(f"{family}_svh").shape[2] != n:
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: {family}_svh length "
                        f"{self.store.get(f'{family}_svh').shape[2]} != n={n}"
                    )

                if self.config.codebook == "3inst":
                    if row["codebook_tensor"] != -1 or row["codebook_offset"] != -1:
                        raise Exl3MoEUnsupported(
                            f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                            f"{proj_index}] has a codebook entry for 3inst"
                        )
                    continue
                cb_spec = self.store.get(f"{family}_{self.config.codebook}")
                if row["codebook_tensor"] != names.index(cb_spec.name):
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                        f"{proj_index}] codebook_tensor "
                        f"{row['codebook_tensor']} != {names.index(cb_spec.name)}"
                    )
                if row["codebook_offset"] != expert_id * cb_spec.slots + slot:
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: descriptor[{expert_id},"
                        f"{proj_index}] codebook_offset {row['codebook_offset']} "
                        f"!= {expert_id * cb_spec.slots + slot}"
                    )
                if expected_multiplier is not None and self._load_attempted:
                    value = int(
                        getattr(layer, cb_spec.name).data[expert_id, slot].item()
                    )
                    if value != expected_multiplier:
                        raise Exl3MoEUnsupported(
                            f"exl3: MoE {self.prefix}: {cb_spec.name}[expert="
                            f"{expert_id},slot={slot}] multiplier 0x"
                            f"{value & 0xFFFFFFFF:08X} != 0x"
                            f"{expected_multiplier & 0xFFFFFFFF:08X} "
                            f"({self.config.codebook!r})"
                        )

        return {
            "experts": self.num_local_experts,
            "projections": len(_PROJ_ORDER),
            "allocations": len(names),
            "tiled_slices": sum(
                self.num_local_experts * s.slots for s in self.store.specs
            ),
            "trellis_tile_w13": list(
                self.store.get("w13_trellis").shape[2:]
            ),
            "trellis_tile_w2": list(self.store.get("w2_trellis").shape[2:]),
        }

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        """Settle the layer after loading.

        * fail closed if the checkpoint did not cover every expert/slot (a
          missing slice would otherwise be uninitialised memory that silently
          produces garbage tokens) — only when a real load was attempted, since
          ``--load-format dummy`` never calls ``load_weights``;
        * then run the P1-04 structural integrity check, which is safe on any
          path because it reads the plan and descriptor buffer only.
        """
        if self._load_attempted:
            self._require_complete(layer)
            self._verify_codebook_multipliers(layer)
        self.verify_expert_integrity(layer)

    def _require_complete(self, layer: "RoutedExperts") -> None:
        missing: list[tuple[str, int, int]] = []
        for spec in self.store.specs:
            for expert_id in range(self.num_local_experts):
                for slot in range(spec.slots):
                    if (spec.name, expert_id, slot) not in self._loaded:
                        missing.append((spec.name, expert_id, slot))
        if missing:
            preview = ", ".join(
                f"{name}[expert={e},slot={s}]" for name, e, s in missing[:6]
            )
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: {len(missing)} expert tensor slice(s) "
                f"were not loaded ({preview}"
                f"{', ...' if len(missing) > 6 else ''}); refusing to run with "
                "uninitialised compressed experts"
            )

    def _verify_codebook_multipliers(self, layer: "RoutedExperts") -> None:
        # The per-expert codebook scalar stores the codebook *multiplier*
        # (presence selects the codebook; see ``exl3xpu/ref.py``). A config/
        # checkpoint mismatch (e.g. `mcg` weights with a `mul1` config) would
        # decode nonsense, so verify the value here.
        expected_multiplier = CODEBOOK_MULTIPLIERS.get(self.config.codebook)
        if expected_multiplier is not None:
            for spec in self.store.specs:
                if spec.tensor != self.config.codebook:
                    continue
                param = getattr(layer, spec.name)
                if not bool(torch.all(param.data == expected_multiplier)):
                    bad = int((param.data != expected_multiplier).sum().item())
                    raise Exl3MoEUnsupported(
                        f"exl3: MoE {self.prefix}: {spec.name} contains {bad} "
                        f"codebook multiplier(s) != 0x"
                        f"{expected_multiplier & 0xFFFFFFFF:08X} "
                        f"({self.config.codebook!r} expected); checkpoint and "
                        "config disagree"
                    )

    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        # EXL3 never calls a vLLM fused-MoE kernel, so there is no
        # FusedMoEQuantConfig to publish.
        return None

    # -- eager execution (P2-01) -------------------------------------------

    def _shard_of_nb(self, device: torch.device, n: int) -> torch.Tensor:
        """Cached all-zero ``shard_of_nb`` for a single-group linear."""
        key = (device, n)
        cached = self._nb_cache.get(key)
        if cached is None:
            cached = torch.zeros(n // 128, dtype=torch.int32, device=device)
            self._nb_cache[key] = cached
        return cached

    def _expert_linear(
        self, views: dict[str, Any], projection: str, x: torch.Tensor
    ) -> torch.Tensor:
        """One projection of one expert through the verified fused op."""
        from . import ops

        m = views[projection]
        # Production dense contract is suh:[num_groups, k]; the expert view is
        # the per-expert [k] row (P1-08).
        suh = m["suh"].unsqueeze(0)
        return ops.exl3_linear(
            x,
            m["trellis"],
            suh,
            m["svh"],
            self._shard_of_nb(x.device, m["n"]),
            [0, m["n"]],
            m["K"],
            m["cb"],
        )

    # -- P4-01: zero-copy execution cache ------------------------------------
    def _build_exec_cache(self, layer: "RoutedExperts") -> None:
        """Per-projection arena handles, sliced once (no per-call sync).

        P4-01. The P2-01 ``apply`` built handles per call through
        ``expert_views`` → ``descriptor`` → ``.tolist()``, which forces a
        device→host sync per expert per projection (~24 syncs/token at decode).
        These handles are *views* into the stacked arenas, so indexing them
        with a Python expert id costs no copy and no sync. ``K``/``cb`` are
        layer-constant (validated in the constructor), so they are read from
        ``self`` rather than from the descriptor buffer.
        """
        self._exec_cache = {
            # projection: (trellis[E,...], suh[E,k], svh[E,n], n)
            "gate": (getattr(layer, "w13_trellis")[:, 0],
                     getattr(layer, "w13_suh")[:, 0],
                     getattr(layer, "w13_svh")[:, 0],
                     self.intermediate_size_per_partition),
            "up": (getattr(layer, "w13_trellis")[:, 1],
                   getattr(layer, "w13_suh")[:, 1],
                   getattr(layer, "w13_svh")[:, 1],
                   self.intermediate_size_per_partition),
            "down": (getattr(layer, "w2_trellis")[:, 0],
                     getattr(layer, "w2_suh")[:, 0],
                     getattr(layer, "w2_svh")[:, 0],
                     self.hidden_size),
        }

    def _cached_linear(self, projection: str, expert_id: int, x: torch.Tensor) -> torch.Tensor:
        from . import ops

        tr, suh, svh, n = self._exec_cache[projection]
        return ops.exl3_linear(
            x,
            tr[expert_id],
            suh[expert_id].unsqueeze(0),
            svh[expert_id],
            self._shard_of_nb(x.device, n),
            [0, n],
            self.K,
            self.cb,
        )

    def _expert_mlp_cached(self, expert_id: int, x: torch.Tensor) -> torch.Tensor:
        """Routed-expert MLP via the zero-copy cache (no per-call sync)."""
        import torch.nn.functional as F

        gate = self._cached_linear("gate", expert_id, x)
        up = self._cached_linear("up", expert_id, x)
        act = F.silu(gate) * up
        return self._cached_linear("down", expert_id, act)

    # -- P4-02: grouped M=1 decode path ---------------------------------------
    def _grouped_mlp_decode(
        self,
        layer: "RoutedExperts",
        x1: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """Single-token routed MLP through the grouped kernels (zero host sync).

        P4-02. Replaces the per-expert Python loop (3 launches *per projection*
        * 8 experts = 72 launches/token/layer) with ONE grouped launch per
        projection:

        * gate+up: ``exl3_moe_grouped_gemv`` with ``nslots=2``,
          ``x_unit_stride=0`` (all 16 units share the token row);
        * SiLU-gated product in one device expression (``acts`` is the only
          inter-launch buffer — activations, never weights);
        * down: one grouped launch with ``nslots=1``,
          ``x_unit_stride=Kdim`` (each unit consumes its own activation row).

        Routing stays on the device end-to-end (``expert_ids`` is built from
        ``topk_ids`` without a host read), so this path performs **zero**
        host-device syncs — the two remaining ``.tolist()`` syncs of the P4-01
        fast path disappear. Numerics are bit-for-bit equal to
        ``_expert_mlp_cached`` (tests/test_grouped_gemv.py, 1920/1920
        bit-exact), including the weighted fp32 reduction.

        Returns ``None`` when the grouped op is unavailable for this layer's
        K/codebook so callers can fall back to the eager loop.
        """
        import torch.nn.functional as F

        ids = topk_ids.view(-1)
        if ids.numel() == 0:
            return None
        if getattr(self, "_grouped_supported", None) is None:
            try:
                from . import ops

                esimd = ops._get_esimd()
                self._grouped_supported = bool(esimd) and bool(
                    esimd.exl3_supported(self.K, self.cb)
                )
            except Exception:
                self._grouped_supported = False
        if not self._grouped_supported:
            return None
        try:
            if ids.dtype != torch.int32:
                ids = ids.to(torch.int32)
            dev = x1.device
            inter = self.intermediate_size_per_partition

            # gate+up: units = (expert, slot) pairs, x shared (stride 0).
            gu = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
                x1,
                layer.w13_trellis,
                layer.w13_suh,
                layer.w13_svh,
                ids.repeat_interleave(2),
                self._shard_of_nb(dev, inter),
                2,
                inter,
                self.K,
                self.cb,
                0,
            )
            # gated SiLU on the device, then down with per-unit input rows.
            acts = (F.silu(gu[0::2]) * gu[1::2]).reshape(-1, inter).contiguous()
            dn = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
                acts,
                layer.w2_trellis,
                layer.w2_suh,
                layer.w2_svh,
                ids,
                self._shard_of_nb(dev, self.hidden_size),
                1,
                self.hidden_size,
                self.K,
                self.cb,
                inter,
            )
        except Exception:
            # One failure (unsupported K/codebook, shape guard) disables the
            # grouped path for this layer; the eager loop below stays correct.
            self._grouped_supported = False
            return None
        w = weights.view(-1).to(torch.float32).unsqueeze(1)
        return (dn.squeeze(1).to(torch.float32) * w).sum(dim=0).to(x1.dtype)

    # -- §11.5: token-parallel grouped verify path (MTP verify 1 < T <= 4) -----
    def _grouped_mlp_verify(
        self,
        layer: "RoutedExperts",
        x2: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """Token-parallel grouped verify path for small T <= 4 (zero host sync).

        Unit mapping (§11.5): ``ids = topk_ids.reshape(-1)`` is token-major, so
        gate+up uses ``nslots=2`` with ``expert_ids = ids.repeat_interleave(2)``
        (unit ``u`` -> expert ``ids[u // 2]``, slot ``u % 2``) and a per-unit
        ``x_unit_stride`` fed by ``x2.repeat_interleave(2 * k)``, which maps unit
        ``u`` to token ``u // (2k)``.  Down consumes ``ids`` directly with
        ``x_unit_stride=inter``.  The weighted reduction is fp32 and sums the k
        slots in token-major order, matching P4-02's decode reduction.

        The failure latch deliberately writes ``_verify_supported`` only — a
        verify-only problem (shape, dtype, unsupported K/codebook) must never
        disable ``_grouped_supported``, i.e. the certified P4-02 decode path
        (§11.6).  Returns ``None`` when latched so ``apply()`` falls through to
        the P4-03 / eager multi-token paths.
        """
        import torch.nn.functional as F

        if getattr(self, "_verify_supported", None) is False:
            return None
        if not hasattr(torch.ops.exl3xpu_C, "exl3_moe_grouped_gemv"):
            self._latch_verify("exl3_moe_grouped_gemv op is not registered")
            return None
        try:
            T, k = topk_ids.shape
            dev = x2.device
            inter = self.intermediate_size_per_partition
            ids = topk_ids.reshape(-1).to(torch.int32)

            # gate+up: repeat each token row 2*k times so unit u gets token u // (2k)
            x_rep = x2.repeat_interleave(2 * k, dim=0)
            gu = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
                x_rep,
                layer.w13_trellis,
                layer.w13_suh,
                layer.w13_svh,
                ids.repeat_interleave(2),
                self._shard_of_nb(dev, inter),
                2,
                inter,
                self.K,
                self.cb,
                self.hidden_size,
            )
            acts = (F.silu(gu[0::2]) * gu[1::2]).reshape(-1, inter).contiguous()
            dn = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
                acts,
                layer.w2_trellis,
                layer.w2_suh,
                layer.w2_svh,
                ids,
                self._shard_of_nb(dev, self.hidden_size),
                1,
                self.hidden_size,
                self.K,
                self.cb,
                inter,
            )
            w = weights.reshape(-1).to(torch.float32).unsqueeze(1)
            y = (dn.squeeze(1).to(torch.float32) * w).view(T, k, -1).sum(dim=1)
            return y.to(x2.dtype)
        except Exception as exc:
            # Latch this path only: the eager / P4-03 fallbacks below stay
            # correct, and P4-02 decode keeps serving this layer.
            self._latch_verify(f"{type(exc).__name__}: {exc}")
            return None

    def _latch_verify(self, reason: str) -> None:
        """Disable the §11.5 verify path for this layer, reporting it once."""
        self._verify_supported = False
        if getattr(self, "_verify_latch_logged", False):
            return
        self._verify_latch_logged = True
        import warnings

        warnings.warn(
            f"exl3: grouped verify path disabled for {self.prefix}: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )

    # -- P4-03: grouped small-M prefill path ----------------------------------
    def _grouped_mlp_prefill(
        self,
        layer: "RoutedExperts",
        x2: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """P4-03: Multi-token grouped prefill GEMM for small-M experts.

        Sorts token assignments into expert bins on the device, launches
        exl3_moe_grouped_gemm for all active experts with M_u <= 8 in ONE launch
        for gate+up and ONE launch for down, and falls back to cached linear
        only for hot experts with M_u > 8.
        """
        import torch.nn.functional as F

        if getattr(self, "_grouped_gemm_supported", None) is False:
            return None
        if not hasattr(torch.ops.exl3xpu_C, "exl3_moe_grouped_gemm"):
            self._grouped_gemm_supported = False
            return None

        try:
            T, k = topk_ids.shape
            dev = x2.device
            E = self.num_local_experts

            if _HAS_MOE_C:
                remapped_x = torch.empty(
                    (T * k, self.hidden_size), dtype=x2.dtype, device=dev
                )
                rows_per_expert = torch.zeros(E, dtype=torch.int32, device=dev)
                unperm = torch.empty((T, k), dtype=torch.int32, device=dev)
                ids_i32 = (
                    topk_ids.to(torch.int32)
                    if topk_ids.dtype != torch.int32
                    else topk_ids
                )

                torch.ops._moe_C.remap_hidden_states(
                    x2,
                    None,
                    remapped_x,
                    None,
                    None,
                    rows_per_expert,
                    unperm,
                    ids_i32,
                    E,
                    E,
                )
                active_mask = rows_per_expert > 0
                active_experts = torch.nonzero(active_mask).squeeze(-1).to(torch.int32)
                U = active_experts.shape[0]
                if U == 0:
                    return torch.zeros_like(x2)

                counts = rows_per_expert[active_experts]
                all_offsets = torch.zeros(E, dtype=torch.int32, device=dev)
                all_offsets[1:] = torch.cumsum(rows_per_expert[:-1], dim=0)
                offsets = all_offsets[active_experts]
                x_sorted = remapped_x
                total_tokens = T * k
            else:
                flat_ids = topk_ids.reshape(-1)
                if flat_ids.dtype != torch.int32:
                    flat_ids = flat_ids.to(torch.int32)
                flat_weights = weights.reshape(-1)
                flat_token_ids = (
                    torch.arange(T, device=dev, dtype=torch.int32)
                    .unsqueeze(1)
                    .expand(T, k)
                    .reshape(-1)
                )

                # Sort assignments on device
                sort_order = torch.argsort(flat_ids)
                sorted_expert_ids = flat_ids[sort_order]
                sorted_token_ids = flat_token_ids[sort_order]
                sorted_weights = flat_weights[sort_order]

                # Gather x according to sorted token assignments
                x_sorted = x2.index_select(0, sorted_token_ids)

                # Identify contiguous expert slices
                active_experts, counts = torch.unique_consecutive(
                    sorted_expert_ids, return_counts=True
                )
                active_experts = active_experts.to(torch.int32)
                counts = counts.to(torch.int32)
                U = active_experts.shape[0]
                if U == 0:
                    return torch.zeros_like(x2)

                offsets = torch.zeros(U, dtype=torch.int32, device=dev)
                offsets[1:] = torch.cumsum(counts[:-1], dim=0)
                total_tokens = x_sorted.shape[0]

            inter = self.intermediate_size_per_partition
            shard13 = self._shard_of_nb(dev, inter)
            shard2 = self._shard_of_nb(dev, self.hidden_size)

            # Vector tile selection (MR=8 or MR=16):
            # For small T <= 8, counts <= 8 always holds: bypass host syncs (§11.4).
            if T <= 8:
                max_m_target = 8
            else:
                gt8_count = (counts > 8).sum().item()
                max_m_target = 16 if (gt8_count > U // 2) else 8
                if (counts > max_m_target).any():
                    chunk_exp = []
                    chunk_cnt = []
                    chunk_off = []
                    for exp, cnt, off in zip(
                        active_experts.tolist(), counts.tolist(), offsets.tolist()
                    ):
                        while cnt > max_m_target:
                            chunk_exp.append(exp)
                            chunk_cnt.append(max_m_target)
                            chunk_off.append(off)
                            off += max_m_target
                            cnt -= max_m_target
                        chunk_exp.append(exp)
                        chunk_cnt.append(cnt)
                        chunk_off.append(off)
                    active_experts = torch.tensor(
                        chunk_exp, dtype=torch.int32, device=dev
                    )
                    counts = torch.tensor(chunk_cnt, dtype=torch.int32, device=dev)
                    offsets = torch.tensor(chunk_off, dtype=torch.int32, device=dev)
                    U = active_experts.shape[0]

            ids13 = active_experts.repeat_interleave(2)
            m13 = counts.repeat_interleave(2)
            off13 = offsets.repeat_interleave(2)

            gu = torch.ops.exl3xpu_C.exl3_moe_grouped_gemm(
                x_sorted,
                layer.w13_trellis,
                layer.w13_suh,
                layer.w13_svh,
                ids13,
                shard13,
                m13,
                off13,
                off13,
                total_tokens,
                2,
                inter,
                self.K,
                self.cb,
                max_m_target,
            )
            act = (F.silu(gu[0]) * gu[1]).contiguous()
            dn = torch.ops.exl3xpu_C.exl3_moe_grouped_gemm(
                act,
                layer.w2_trellis,
                layer.w2_suh,
                layer.w2_svh,
                active_experts,
                shard2,
                counts,
                offsets,
                offsets,
                total_tokens,
                1,
                self.hidden_size,
                self.K,
                self.cb,
                max_m_target,
            )

            # Scatter-add weighted outputs back to original token positions
            if _HAS_MOE_C:
                out = torch.empty((T, self.hidden_size), dtype=x2.dtype, device=dev)
                w_f32 = (
                    weights.to(torch.float32)
                    if weights.dtype != torch.float32
                    else weights
                )
                torch.ops._moe_C.moe_gather(out, dn, w_f32, unperm, E)
                return out
            else:
                out = torch.zeros(
                    (T, self.hidden_size), dtype=torch.float32, device=dev
                )
                out.index_add_(
                    0,
                    sorted_token_ids,
                    dn.to(torch.float32) * sorted_weights.unsqueeze(-1),
                )
                return out.to(x2.dtype)
        except Exception as exc:
            _trace_path("prefill-latched", f" {type(exc).__name__}: {exc}")
            self._grouped_gemm_supported = False
            return None

    def _expert_mlp(
        self, layer: "RoutedExperts", expert_id: int, x: torch.Tensor
    ) -> torch.Tensor:
        """Routed-expert MLP: ``down(silu(gate(x)) * up(x))``."""
        import torch.nn.functional as F

        views = self.expert_views(layer, expert_id)
        gate = self._expert_linear(views, "gate_proj", x)
        up = self._expert_linear(views, "up_proj", x)
        act = F.silu(gate) * up
        down = self._expert_linear(views, "down_proj", act)
        if self._debug_capture is not None:
            self._debug_capture.setdefault("experts", {})[int(expert_id)] = {
                "gate": gate.detach(),
                "up": up.detach(),
                "act": act.detach(),
                "down": down.detach(),
            }
        return down

    # -- P2-08 diagnostic hook ---------------------------------------------
    def enable_debug_capture(self) -> None:
        """Capture per-expert/router intermediates on the next ``apply``."""
        self._debug_capture = {}

    def disable_debug_capture(self) -> dict[str, Any] | None:
        captured, self._debug_capture = self._debug_capture, None
        return captured

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Eager routed-expert execution (P2-01).

        vLLM's modular runner already selected the experts and normalized the
        weights, and computes/adds the shared expert itself (``_maybe_combine``
        does ``shared_output + fused_output``). This method therefore computes
        **only** the routed contribution:

            Y[i] = sum_{j in top-k(i)} w[i, j] * MLP_{ids[i, j]}(X[i])

        with the routing weight applied to the *down-projection output* (after
        SiLU/up and the output Hadamard), never to the input activations
        (P2-03). Experts receiving no tokens are skipped. A Python loop over
        active experts is the Gate 2 baseline; grouped kernels come in Phase 4.

        P4-01: the per-expert handles are sliced once into a zero-copy cache
        and the single-token decode path reads the routing decision with one
        host sync (instead of ~27 per-expert descriptor/validation syncs).
        """
        hidden = x.shape[-1]
        x2 = x.reshape(-1, hidden)
        if not x2.is_contiguous():
            x2 = x2.contiguous()

        if x2.shape[0] == 0:
            return torch.zeros_like(x2)

        if topk_ids.shape != topk_weights.shape or topk_ids.dim() != 2:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: topk_ids/topk_weights must be "
                f"[tokens, k]; got {tuple(topk_ids.shape)} / "
                f"{tuple(topk_weights.shape)}"
            )
        if topk_ids.shape[0] != x2.shape[0]:
            raise Exl3MoEUnsupported(
                f"exl3: MoE {self.prefix}: {topk_ids.shape[0]} routing rows for "
                f"{x2.shape[0]} tokens"
            )

        if getattr(self, "_exec_cache", None) is None:
            self._build_exec_cache(layer)

        # Routed output accumulated in fp32, cast back at the end.
        out = torch.zeros(x2.shape, dtype=torch.float32, device=x2.device)
        weights = topk_weights.to(torch.float32)
        tokens = x2.shape[0]

        if self._debug_capture is None and tokens == 1:
            # --- P4-01/P4-02 single-token (decode) fast path ------------------
            # P4-02 first: two grouped kernel launches for all top-k experts
            # with zero host syncs (routing stays on the device). Falls back to
            # the P4-01 loop when the grouped op is unavailable.
            y = self._grouped_mlp_decode(layer, x2, topk_ids, weights)
            if y is not None:
                _trace_path("decode", f" tokens={tokens}")
                return y.reshape(x.shape)
            # One host sync for the routing decision; every expert handle is a
            # zero-copy cache view, so there are no further syncs.
            ids = topk_ids.view(-1).tolist()
            wts = weights.view(-1).tolist()
            for expert_id, wt in zip(ids, wts):
                y = self._expert_mlp_cached(int(expert_id), x2)
                out += y.to(torch.float32) * wt
            return out.to(x2.dtype).reshape(x.shape)

        if self._debug_capture is None and 1 < tokens <= 4:
            # --- §11.5 token-parallel grouped verify fast path (MTP) -----------
            y = self._grouped_mlp_verify(layer, x2, topk_ids, weights)
            if y is not None:
                _trace_path("verify", f" tokens={tokens}")
                return y.reshape(x.shape)

        if self._debug_capture is None and tokens > 1:
            # --- P4-03 grouped prefill fast path ------------------------------
            y = self._grouped_mlp_prefill(layer, x2, topk_ids, weights)
            if y is not None:
                _trace_path("prefill-gemm", f" tokens={tokens}")
                return y.reshape(x.shape)

        # --- general (prefill / multi-token) fallback path ------------------
        _trace_path(
            "eager-fallback",
            f" tokens={tokens} debug_capture={self._debug_capture is not None}",
        )
        if self._debug_capture is not None and topk_ids.numel():
            # Range validation is a device->host sync; keep it on the debug
            # path only (the arena shape is asserted once at load time).
            if int(topk_ids.min()) < 0 or int(topk_ids.max()) >= self.num_local_experts:
                raise Exl3MoEUnsupported(
                    f"exl3: MoE {self.prefix}: routing index outside "
                    f"[0, {self.num_local_experts})"
                )

        for expert_id in torch.unique(topk_ids).tolist():
            token_idx, slot_idx = (topk_ids == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            rows = x2.index_select(0, token_idx)
            if self._debug_capture is not None:
                y = self._expert_mlp(layer, int(expert_id), rows)
            else:
                y = self._expert_mlp_cached(int(expert_id), rows)
            w = weights[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, y.to(torch.float32) * w)

        if self._debug_capture is not None:
            self._debug_capture.update({
                "x": x2.detach(),
                "topk_ids": topk_ids.detach(),
                "topk_weights": topk_weights.detach(),
                "routed_fp32": out.detach(),
            })

        return out.to(x2.dtype).reshape(x.shape)


# ---------------------------------------------------------------------------
# P1-05: shard-by-shard streaming loader
# ---------------------------------------------------------------------------

#: Checkpoint suffixes that make up one routed-expert module.
EXPERT_SUFFIXES = frozenset({"trellis", "suh", "svh", "mcg", "mul1"})


def split_expert_key(key: str) -> tuple[str | None, str | None]:
    """Split a checkpoint key into ``(layer_key, relative_name)``.

    ``layer_key`` is the normalized ``RoutedExperts`` key
    (``"layers.3.mlp.experts"``); ``relative_name`` is the name the per-layer
    loader consumes (``"0.gate_proj.trellis"``). Non-routed keys (shared
    expert, attention, vision, embedding, ``lm_head``, MTP) return
    ``(None, None)``.
    """
    stem, dot, suffix = key.rpartition(".")
    if not dot or suffix not in EXPERT_SUFFIXES:
        return None, None
    from .vllm_plugin import _norm_key

    parts = _norm_key(stem).split(".")
    if (
        len(parts) < 6
        or parts[0] != "layers"
        or parts[2] != "mlp"
        or parts[3] != "experts"
        or not parts[4].isdigit()
    ):
        return None, None
    layer_key = ".".join(parts[:4])
    relative = ".".join(parts[4:]) + "." + suffix
    return layer_key, relative


def stream_expert_shards(
    shard_paths: Iterable[Any],
    layers: dict[str, "RoutedExperts"],
    *,
    on_shard: Any = None,
) -> dict[str, Any]:
    """Pack routed experts into ``layers`` one ``safetensors`` shard at a time.

    P1-05. On the unified-memory Arc part, holding several 8 GiB shards'
    mmaps open while also touching the packed arenas duplicates host and
    device traffic. This loader:

    1. opens exactly one shard at a time with ``safe_open``;
    2. reads each routed-expert tensor lazily via ``get_tensor`` inside a
       generator, so at most one checkpoint tensor's payload is resident in
       addition to the arenas;
    3. feeds it to the layer's per-layer loader (``RoutedExperts.load_weights``
       override), then tears the shard down with ``gc.collect()`` and
       ``torch.xpu.empty_cache()`` before moving on.

    ``layers`` maps a *normalized* layer key (``"layers.3.mlp.experts"``) to
    its ``RoutedExperts`` whose ``create_weights`` has already run.

    Returns a report with:

    * ``consumed`` — routed-expert checkpoint keys handed to a layer loader;
    * ``out_of_scope`` — routed keys for layers not in ``layers`` (deliberately
      skipped, e.g. a ``--layers`` subset; the caller decides whether it
      expects these);
    * ``shards`` — per-shard timing/telemetry (fed to ``on_shard`` as it goes);
    * ``expert_tensors`` — total routed tensors streamed.
    """
    import gc
    import time

    from safetensors import safe_open

    consumed: list[str] = []
    out_of_scope: list[str] = []
    shards: list[dict[str, Any]] = []

    for shard_index, path in enumerate(shard_paths):
        t0 = time.perf_counter()
        n_read = 0
        by_layer: dict[str, list[tuple[str, str]]] = {}
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                layer_key, relative = split_expert_key(key)
                if layer_key is None:
                    continue
                by_layer.setdefault(layer_key, []).append((key, relative))

            for layer_key, entries in by_layer.items():
                layer = layers.get(layer_key)
                if layer is None:
                    out_of_scope.extend(key for key, _ in entries)
                    continue

                def _tensors(entries=entries, f=f):
                    for key, relative in entries:
                        yield relative, f.get_tensor(key)

                list(layer.load_weights(_tensors()))
                consumed.extend(key for key, _ in entries)
                n_read += len(entries)
            # Drop the key map before closing so mmap pages can be reclaimed.
            del by_layer

        gc.collect()
        try:
            torch.xpu.empty_cache()
        except Exception:  # pragma: no cover - CPU-only / no XPU
            pass

        row = {
            "shard_index": shard_index,
            "shard": str(path),
            "expert_tensors": n_read,
            "seconds": time.perf_counter() - t0,
        }
        if on_shard is not None:
            on_shard(row)
        shards.append(row)

    return {
        "consumed": consumed,
        "out_of_scope": out_of_scope,
        "shards": shards,
        "expert_tensors": sum(r["expert_tensors"] for r in shards),
    }
