"""D-Wave QPU backend with explicit minor-embedding, embedding cache and parallel
embedding (spec doc 04 §2, §6; roadmap 3.1, 3.3).

Unlike :class:`qws.solvers.anneal.AnnealingSolver`, which wraps any dimod sampler, this
solver drives a *structured* child sampler directly: ``dwave.system.DWaveSampler`` in
production, or ``dwave.system.testing.MockDWaveSampler`` offline. Offline, it is
therefore exercised with the real topology, coefficient ranges and timing fields;
only the physics of the device is mocked.

Scaling convention (implementation note, spec doc 09 §4):
the problem is divided by
    kappa = max( 1,  max|h| / h_max,  max|J| / J_max,  csf * max|J| / |J_min| )
so that the logical couplings, the local fields AND the chain couplings
(-csf * max|J_scaled|, ferromagnetic in Ocean's sign convention) all fit the device's
ranges. Problems that already fit are submitted unscaled (kappa = 1): if kappa
normalized every problem to the full range, any rescaling applied by temperature
calibration (doc 03 §4) would be undone. ``auto_scale`` is disabled, so kappa is the
only scaling between our problem and the hardware, and ``range_limited`` reports when
it was needed.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..ising import IsingProblem, to_dimod_bqm
from .base import SampleSet, Solver


def graph_key(p: IsingProblem) -> str:
    """Key of a problem's interaction graph (not its coefficients): equal keys share an embedding."""
    m = hashlib.sha256(np.array([p.n], dtype=np.int64).tobytes())
    m.update(np.ascontiguousarray(p.edges[np.lexsort(p.edges.T[::-1])]).tobytes())
    return m.hexdigest()[:16]


@dataclass
class EmbeddingStats:
    qubits: int
    max_chain: int
    mean_chain: float

    @classmethod
    def of(cls, emb: dict) -> "EmbeddingStats":
        lengths = [len(c) for c in emb.values()]
        return cls(int(sum(lengths)), int(max(lengths)), float(np.mean(lengths)))


class QPUAnnealingSolver(Solver):
    """Quantum annealing on a structured D-Wave sampler.

    Args:
        child: Structured sampler with ``nodelist``, ``edgelist`` and ``properties``
            (``h_range``, ``j_range``).
        chain_strength_factor: csf; chain coupling = csf * max|J| after scaling.
        chain_break_threshold: Calls whose mean chain-break fraction exceeds this are
            flagged ``low_fidelity``.
        embedding_seed: Seed for minorminer (deterministic, cacheable embeddings).
        parallel: Pack all problems of one call into a single QPU submission
            (disjoint embeddings) when they fit together.
        sample_kwargs: Passed to the child (annealing_time, num_spin_reversal_transforms, ...).
    """

    name = "anneal_qpu"
    capabilities = frozenset({"minimize", "sample"})

    def __init__(self, child: Any, chain_strength_factor: float = 1.5, chain_break_threshold: float = 0.1,
                 embedding_seed: int = 0, parallel: bool = True, embedding_tries: int = 10,
                 embedding_timeout: float = 60.0, pack_timeout: float = 5.0, max_fill: float = 0.6,
                 **sample_kwargs: Any):
        self.child = child
        self.csf = chain_strength_factor
        self.chain_break_threshold = chain_break_threshold
        self.embedding_seed = embedding_seed
        self.parallel = parallel
        self.embedding_tries = embedding_tries
        self.sample_kwargs = sample_kwargs
        self.embeddings: dict[tuple[str, ...], dict] = {}
        self._failed: set[tuple[str, ...]] = set()       # unions known not to embed (never retried)
        self._groups_cache: dict[tuple[str, ...], list[list[int]]] = {}
        self._tgt = None
        self.embedding_timeout, self.pack_timeout, self.max_fill = embedding_timeout, pack_timeout, max_fill
        props = child.properties
        self.h_range = tuple(props["h_range"])
        self.j_range = tuple(props["j_range"])
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ factories
    @classmethod
    def dwave(cls, solver: str | dict | None = None, **kw: Any) -> "QPUAnnealingSolver":  # pragma: no cover
        """Real QPU. Needs DWAVE_API_TOKEN. ``solver`` defaults to an Advantage (Pegasus) system (R6)."""
        from dwave.system import DWaveSampler

        solver = solver if solver is not None else {"topology__type": "pegasus"}
        return cls(DWaveSampler(solver=solver), **kw)

    @classmethod
    def mock(cls, topology_shape: int = 6, beta_range: tuple[float, float] = (0.1, 4.0), seed: int | None = 0,
             **kw: Any) -> "QPUAnnealingSolver":
        """Offline stand-in: MockDWaveSampler on a Pegasus graph P(topology_shape).

        Its physics substitute is simulated annealing with a FIXED inverse-temperature
        range, so, like a device at fixed physical temperature, the effective temperature
        of the samples depends on the submitted coefficient scale.
        """
        from dwave.system.testing import MockDWaveSampler
        from neal import SimulatedAnnealingSampler

        child = MockDWaveSampler(topology_type="pegasus", topology_shape=[topology_shape],
                                 substitute_sampler=SimulatedAnnealingSampler(),
                                 substitute_kwargs={"beta_range": list(beta_range), "seed": seed},
                                 exact_solver_cutoff=0)
        return cls(child, **kw)

    # ------------------------------------------------------------------ scaling and embedding
    def scale_factor(self, p: IsingProblem) -> float:
        h_max = min(abs(self.h_range[0]), abs(self.h_range[1]))
        j_max, j_min = abs(self.j_range[1]), abs(self.j_range[0])
        maxh = float(np.abs(p.h).max(initial=0.0))
        maxj = float(np.abs(p.J).max(initial=0.0))
        return max(1.0, maxh / h_max, maxj / j_max, self.csf * maxj / j_min)

    def _target_graph(self):
        if self._tgt is None:
            import networkx as nx

            g = nx.Graph()
            g.add_nodes_from(self.child.nodelist)
            g.add_edges_from(self.child.edgelist)
            self._tgt = g
        return self._tgt

    def embed(self, problems: Sequence[IsingProblem], tries: int | None = None, timeout: float | None = None) -> dict:
        """Embedding of the disjoint union of the problems' graphs, cached by graph keys.

        Logical variable ``(k, i)`` is variable ``i`` of problem ``k``. Failures are cached
        too, so a union that does not fit is never retried.
        """
        key = tuple(graph_key(p) for p in problems)
        if key in self.embeddings:
            return self.embeddings[key]
        if key in self._failed:
            raise RuntimeError("embedding previously failed for this union of graphs")
        import minorminer

        src = [((k, int(i)), (k, int(j))) for k, p in enumerate(problems) for i, j in p.edges]
        isolated = [(k, i) for k, p in enumerate(problems) for i in range(p.n)]
        import networkx as nx
        from dwave.embedding import is_valid_embedding

        tgt = self._target_graph()
        src_graph = nx.Graph(src)
        emb: dict = {}
        for attempt in range(tries or self.embedding_tries):
            emb = minorminer.find_embedding(src, tgt, random_seed=self.embedding_seed + attempt,
                                            timeout=timeout or self.embedding_timeout)
            # minorminer can return a non-empty but invalid embedding (e.g. on timeout): verify it
            if emb and not is_valid_embedding(emb, src_graph, tgt):
                emb = {}
            if emb or not src:
                break
        # variables without edges still need a qubit each
        used = {q for chain in emb.values() for q in chain}
        free = (q for q in self.child.nodelist if q not in used)
        for v in isolated:
            if v not in emb:
                try:
                    emb[v] = [next(free)]
                except StopIteration:
                    emb = {}
                    break
        if src and not emb or any(v not in emb for v in isolated):
            self._failed.add(key)
            raise RuntimeError(f"no embedding found for {len(problems)} problem(s) with "
                               f"{sum(p.n for p in problems)} variables")
        self.embeddings[key] = emb
        return emb

    def _groups(self, problems: Sequence[IsingProblem]) -> list[list[int]]:
        """Greedily pack problem indices into submissions that embed together.

        A problem joins the current submission while the union still embeds. Unions
        whose variable count exceeds ``max_fill`` x (number of qubits) are not attempted,
        and packing attempts use one short minorminer run (``pack_timeout``). The
        grouping is cached per tuple of graph keys, so repeated calls (training steps)
        cost nothing.
        """
        if not self.parallel:
            return [[k] for k in range(len(problems))]
        keys = tuple(graph_key(p) for p in problems)
        if keys in self._groups_cache:
            return self._groups_cache[keys]
        budget = self.max_fill * len(self.child.nodelist)
        groups: list[list[int]] = []
        current: list[int] = []
        for k in range(len(problems)):
            trial = current + [k]
            fits = False
            if current and sum(problems[i].n for i in trial) <= budget:
                try:
                    self.embed([problems[i] for i in trial], tries=1, timeout=self.pack_timeout)
                    fits = True
                except RuntimeError:
                    fits = False
            if not current:
                self.embed([problems[k]])  # a single problem must embed (raises otherwise)
                current = [k]
            elif fits:
                current = trial
            else:
                groups.append(current)
                current = [k]
                self.embed([problems[k]])
        if current:
            groups.append(current)
        self._groups_cache[keys] = groups
        return groups

    # ------------------------------------------------------------------ calls
    def _submit(self, problems: Sequence[IsingProblem], num_reads: int, **kw: Any) -> list[SampleSet]:
        import dimod
        from dwave.embedding import embed_bqm, unembed_sampleset
        from dwave.embedding.chain_breaks import majority_vote

        kappas = [self.scale_factor(p) for p in problems]
        union = dimod.BinaryQuadraticModel("SPIN")
        maxj_scaled = 0.0
        for k, (p, kappa) in enumerate(zip(problems, kappas)):
            bqm = to_dimod_bqm(p.scaled(1.0 / kappa))  # Ocean sign convention: h' = -h, J' = -J
            bqm.relabel_variables({i: (k, i) for i in range(p.n)})
            union.update(bqm)
            maxj_scaled = max(maxj_scaled, float(np.abs(p.J).max(initial=0.0)) / kappa)
        emb = self.embed(problems)
        chain_strength = self.csf * maxj_scaled if maxj_scaled > 0 else 0.5 * abs(self.j_range[0])
        target_adj = {q: set() for q in self.child.nodelist}
        for a, b in self.child.edgelist:
            target_adj[a].add(b)
            target_adj[b].add(a)
        embedded = embed_bqm(union, emb, target_adj, chain_strength=chain_strength)
        params = {**self.sample_kwargs, **kw, "num_reads": num_reads, "auto_scale": False}
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not mocked in MockDWaveSampler.*")
            raw = self.child.sample(embedded, **params)
        wall = time.perf_counter() - t0
        ss = unembed_sampleset(raw, emb, union, chain_break_method=majority_vote, chain_break_fraction=True)
        timing = dict(raw.info.get("timing", {}))
        stats = EmbeddingStats.of(emb)
        call = {"problems": len(problems), "variables": sum(p.n for p in problems), "num_reads": num_reads,
                "chain_strength": chain_strength, "embedding": stats.__dict__, "timing": timing,
                "wall_time_s": wall}
        self.calls.append(call)
        cbf_all = np.asarray(ss.record.chain_break_fraction)
        counts = np.asarray(ss.record.num_occurrences)
        out = []
        total_n = sum(p.n for p in problems)
        for k, (p, kappa) in enumerate(zip(problems, kappas)):
            cols = [ss.variables.index((k, i)) for i in range(p.n)]
            samples = np.asarray(ss.record.sample)[:, cols]
            sub_emb = {v: c for v, c in emb.items() if v[0] == k}
            info = {"solver": self.name, "kappa": kappa, "range_limited": kappa > 1.0, "chain_strength": chain_strength,
                    "chain_break_fraction": float(np.average(cbf_all, weights=counts)),
                    "embedding": EmbeddingStats.of(sub_emb).__dict__ if sub_emb else None,
                    "packed_with": len(problems), "wall_time_s": wall * p.n / total_n,
                    # QPU time is shared by every problem in the submission; attributed by size
                    "qpu_access_time_us": float(timing.get("qpu_access_time", 0.0)) * p.n / total_n,
                    "topology": dict(self.child.properties.get("topology", {}))}
            info["low_fidelity"] = info["chain_break_fraction"] > self.chain_break_threshold
            out.append(SampleSet.from_samples(p, samples, counts, info))
        return out

    def _run(self, problems: Sequence[IsingProblem], num_reads: int, **kw: Any) -> list[SampleSet]:
        problems = list(problems)
        results: list[SampleSet | None] = [None] * len(problems)
        for group in self._groups(problems):
            for k, ss in zip(group, self._submit([problems[i] for i in group], num_reads, **kw)):
                results[k] = ss
        return results  # type: ignore[return-value]

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, **kw: Any) -> list[SampleSet]:
        return self._run(problems, num_samples, **kw)

    def minimize(self, problems: Sequence[IsingProblem], num_reads: int = 100, **kw: Any) -> list[SampleSet]:
        return self._run(problems, num_reads, **kw)
