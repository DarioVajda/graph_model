"""Temporal neighborhood sampling over a relational schema.

PyG's ``NeighborSampler`` does the sampling (see PLAN.md 2.1 for why we stopped writing our
own). This module is the thin layer around it that the rest of the pipeline needs:

* **seeding it per (entity, timestamp)** -- relbench's unit of prediction, expressed through
  ``input_time`` so PyG enforces ``neighbor.time <= seed.time`` along the whole path;
* **mapping the output back to ``(table, row, hop)``** so ``row_text`` can fetch pandas rows;
* **re-orienting edges to child -> parent**, the direction the magnetic-Laplacian bias reads;
* **the budget allocator** -- ``num_neighbors`` caps each relation but not the total, and on
  a wide schema one relation will otherwise eat everything (rel-trial's
  ``facilities_studies`` runs to 999 eligible rows against ``eligibilities``' 1, and the
  latter is where the signal is -- PLAN.md 4.2).

The output is ``(nodes, edges)`` and nothing else: no text, no tokens. That keeps this layer
testable against a toy database and keeps serialization choices out of sampling.

`NeighborSampler` is used directly rather than `NeighborLoader`: we want one graph per
(entity, timestamp, version) with a per-example RNG seed, and a DataLoader's batching and
worker pool are pure overhead for that.
"""

import hashlib
from dataclasses import dataclass, field

import torch
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

from .graph_build import STATIC_TIME

# `graph_build` names foreign-key edge types `f2p_<col>`; `ToUndirected` mints the reverse as
# `rev_f2p_<col>`. Sampling walks an edge type from its *destination* to its *source*, so:
#   (child, f2p_x,     parent) -- seeded at the parent, yields children  -> DESCENDING
#   (parent, rev_f2p_x, child) -- seeded at the child,  yields the parent -> ASCENDING
_REV_PREFIX = "rev_"


def _is_descending(edge_type):
    """True if sampling this edge type walks parent -> children."""
    return not edge_type[1].startswith(_REV_PREFIX)


def _stable_seed(*parts):
    """Deterministic 31-bit seed from the sampling identity, stable across machines.

    `hash()` is salted per process in Python 3, so it cannot be used: a rebuild would
    produce different graphs and the `samples_per_node` version index would stop meaning
    anything.
    """
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


@dataclass
class SampledGraph:
    """One neighborhood. `nodes[0]` is always the seed row."""
    nodes: list = field(default_factory=list)   # [(table, row, hop)]
    edges: list = field(default_factory=list)   # [(i, j)] into `nodes`, child -> parent

    def __len__(self):
        return len(self.nodes)

    def tables(self):
        counts = {}
        for table, _, _ in self.nodes:
            counts[table] = counts.get(table, 0) + 1
        return counts


class TemporalSampler:
    """Samples temporally-valid neighborhoods around (entity, timestamp) seeds.

    Build once per (dataset, entity_table, config) and call `sample` per seed.
    """

    def __init__(self, data, entity_table, max_nodes=64, strategy="last",
                 sibling_fanout=0, parent_fanout=1, relation_cap=None, link_tables=(),
                 collapse_links=True):
        """
        `strategy`     -- "last" (the k most recent eligible rows; PLAN.md 5.4's `recent`)
                          or "uniform" (RDL's policy, the control for "is recency doing the
                          work").
        `link_tables`  -- tables that are pure many-to-many joins, from
                          `graph_build.link_tables`. Their rows are kept for topology but
                          **do not consume the node budget**, because they carry no content:
                          a `facilities_studies` row is an id pair and a date, while the
                          facility's name is one hop further out. Without this the budget is
                          spent entirely on junction rows and the dimension tables starve
                          (measured on rel-trial -- PLAN.md 4.2).
        `collapse_links` -- contract those join rows into direct edges once sampling is done.
                          A junction row renders as literally `conditions_studies | date:
                          158d before` -- no content, and a date it shares with the study it
                          hangs off. Six of nineteen nodes in a typical rel-trial document
                          were this. Contracting keeps the relationship and drops the empty
                          node, the same move kgqa makes for unnamed CVT mediators. Set False
                          to ablate.
        `sibling_fanout` -- hop-2 descending fanout. 0 disables the parent -> other-children
                          step; >0 is PLAN.md 5.4's `include_siblings`, which needs no
                          special code because child -> parent -> other children is just a
                          second descending hop.
        `relation_cap` -- per-relation hop-1 cap handed to PyG. Defaults to `max_nodes`, so
                          a single relation *may* fill the whole budget when every other one
                          is empty; the allocator then decides what actually survives. This
                          deliberately oversamples: trimming what we have beats a second
                          sampling pass to discover how much was available.
        """
        if strategy not in ("last", "uniform"):
            raise ValueError(f"unknown strategy {strategy!r}")

        self.data = data
        self.entity_table = entity_table
        self.max_nodes = max_nodes
        self.strategy = strategy
        self.link_tables = set(link_tables)
        self.collapse_links = collapse_links
        cap = relation_cap if relation_cap is not None else max_nodes

        # Per-edge-type, per-hop fanout. Descending and ascending get different budgets
        # because they mean different things: descending is "this entity's records"
        # (unbounded, needs a cap), ascending is "the dimension row this record points at"
        # (exactly one per fkey, so 1 is complete).
        num_neighbors = {}
        for et in data.edge_types:
            if _is_descending(et):
                num_neighbors[et] = [cap, sibling_fanout]
            else:
                num_neighbors[et] = [parent_fanout, parent_fanout]

        self._sampler = NeighborSampler(
            data,
            num_neighbors=num_neighbors,
            time_attr="time",
            temporal_strategy=strategy,
            # NOT "induced": pyg-lib has not implemented induced sampling for heterogeneous
            # graphs, and an `induced` request silently falls through to the torch-sparse
            # backend -- broken in this venv -- raising a misleading "requires either
            # 'pyg-lib' or 'torch-sparse'". See PLAN.md 3.3 finding 1.
            subgraph_type="directional",
        )

    # -- sampling ----------------------------------------------------------

    def sample(self, row, timestamp, version=0, identity=()):
        """Sample the neighborhood of `row` in the entity table as of `timestamp`.

        `timestamp` is unix seconds. `identity` is any extra tuple (dataset, task, split)
        that should make the RNG draw unique; `version` indexes `samples_per_node`.
        """
        # `temporal_strategy="last"` is deterministic, but "uniform" and any future `mixed`
        # policy draw from the global torch RNG. Seed unconditionally so an arm swap cannot
        # silently change reproducibility.
        torch.manual_seed(_stable_seed(*identity, self.entity_table, row, timestamp, version))

        out = self._sampler.sample_from_nodes(NodeSamplerInput(
            input_id=None,
            node=torch.tensor([row], dtype=torch.int64),
            time=torch.tensor([timestamp], dtype=torch.int64),
            input_type=self.entity_table,
        ))

        nodes_by_type, hops_by_type = self._attribute_hops(out)
        adjacency = self._adjacency(out)
        keep = self._allocate(nodes_by_type, hops_by_type, adjacency)
        return self._assemble(out, nodes_by_type, hops_by_type, keep)

    # -- output shaping ----------------------------------------------------

    @staticmethod
    def _attribute_hops(out):
        """`(table -> original row ids, table -> hop per sampled node)`.

        `out.node[t]` is ordered by hop and `out.num_sampled_nodes[t]` gives the size of each
        hop's slice, so hop attribution is a cumulative sum -- no bookkeeping of our own.
        """
        nodes_by_type, hops_by_type = {}, {}
        for table, ids in out.node.items():
            if ids.numel() == 0:
                continue
            per_hop = out.num_sampled_nodes[table]
            hops = torch.empty(ids.numel(), dtype=torch.int64)
            start = 0
            for hop, count in enumerate(per_hop):
                hops[start:start + count] = hop
                start += count
            # Any trailing nodes PyG did not attribute (defensive: shapes have always
            # matched in practice) belong to the last hop.
            if start < ids.numel():
                hops[start:] = len(per_hop) - 1
            nodes_by_type[table] = ids
            hops_by_type[table] = hops
        return nodes_by_type, hops_by_type

    @staticmethod
    def _adjacency(out):
        """Undirected neighbours of every sampled node, keyed by `(table, local_index)`."""
        adj = {}
        for edge_type, src_local in out.row.items():
            if src_local.numel() == 0:
                continue
            dst_local = out.col[edge_type]
            src_table, _, dst_table = edge_type
            for s, d in zip(src_local.tolist(), dst_local.tolist()):
                a, b = (src_table, s), (dst_table, d)
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        return adj

    def _allocate(self, nodes_by_type, hops_by_type, adjacency):
        """Which sampled nodes survive `max_nodes`. Returns {table: bool mask}.

        PLAN.md 5.4's allocator, with one correction found by running it (PLAN.md 4.2):
        the budget buys **content** rows, not rows. Link-table rows ride along for free
        because they are topology, not information.

        1. The seed always survives.
        2. Every non-link candidate, at any hop, is grouped by table and the budget is split
           evenly across those tables, with what a starved table cannot use redistributed to
           the ones still hungry. Scale-free: a wide schema gets fewer rows per table, a
           narrow one more, and no per-dataset constant appears anywhere.
        3. Link rows are then kept only where they actually connect something that survived
           step 2. Ones whose far end was trimmed are dropped by `_drop_orphans`.

        Candidates are pooled across hops rather than filled hop-1-first. On rel-trial the
        content sits at hop 2 (`facilities.name`, `conditions.mesh_term`) behind a hop-1
        junction row, so a hop-ordered budget spends everything before reaching it.

        Within a table, candidates are re-sorted **most recent first**. Do not skip this on
        the grounds that `temporal_strategy="last"` already selected by recency: it returns
        the right *set* but in **ascending** time order, so taking a prefix of the raw
        sampler output keeps the OLDEST rows of the retained window. That fails silently --
        the graph is still valid, still leak-free, just systematically worse.

        Grouping is by *node type* rather than by edge type. In every schema seen so far each
        child table reaches its parent through exactly one fkey, so the two coincide; a
        schema with two fkeys from one table to the same parent (`from_account` /
        `to_account`) would merge them into one share. Revisit if such a dataset arrives.
        """
        keep = {t: torch.zeros(ids.numel(), dtype=torch.bool)
                for t, ids in nodes_by_type.items()}

        seed_mask = hops_by_type[self.entity_table] == 0
        keep[self.entity_table] |= seed_mask
        budget = self.max_nodes - int(seed_mask.sum())
        if budget <= 0:
            return keep

        cand = {}
        for table, hops in hops_by_type.items():
            if table in self.link_tables:
                continue
            idx = torch.nonzero(hops > 0, as_tuple=True)[0]
            if idx.numel():
                # Sort by (hop ascending, time descending): nearer rows outrank farther
                # ones, and within a hop the most recent wins. Both stages are stable sorts
                # applied least-significant key first.
                times = self.data[table].time[nodes_by_type[table][idx]]
                idx = idx[torch.argsort(times, descending=True, stable=True)]
                idx = idx[torch.argsort(hops[idx], stable=True)]
                cand[table] = idx

        for table, n in self._even_split(cand, budget).items():
            keep[table][cand[table][:n]] = True

        # Step 3: reinstate the connective tissue. A link row earns its place when at least
        # one neighbour other than the seed survived -- otherwise it is a dangling stub that
        # says only "this study had *a* facility".
        for table in self.link_tables:
            if table not in nodes_by_type:
                continue
            for local in range(nodes_by_type[table].numel()):
                for (nt, nl) in adjacency.get((table, local), ()):
                    if nt in keep and keep[nt][nl] and not (
                            nt == self.entity_table and hops_by_type[nt][nl] == 0):
                        keep[table][local] = True
                        break
        return keep

    @staticmethod
    def _even_split(cand, budget):
        """Even share per relation, with the unused remainder redistributed round-robin.

        This is the step that stops `facilities_studies` (up to 999 eligible rows) from
        starving `eligibilities` (exactly 1, and full of the criteria text the label depends
        on).
        """
        want = {t: idx.numel() for t, idx in cand.items()}
        taken = {t: 0 for t in cand}
        remaining = budget

        # Repeatedly hand out an equal share; relations that cannot use theirs release it to
        # the next round. Terminates because each round either exhausts the budget or fully
        # satisfies at least one relation.
        while remaining > 0:
            hungry = [t for t in cand if taken[t] < want[t]]
            if not hungry:
                break
            share = max(1, remaining // len(hungry))
            progressed = False
            for table in hungry:
                if remaining <= 0:
                    break
                n = min(share, want[table] - taken[table], remaining)
                if n > 0:
                    taken[table] += n
                    remaining -= n
                    progressed = True
            if not progressed:
                break
        return taken

    def _assemble(self, out, nodes_by_type, hops_by_type, keep):
        """Build the `(nodes, edges)` output, dropping anything orphaned by the trim."""
        # Seed first, so `nodes[0]` is the seed row (PLAN.md 5.2). Then by hop, then by
        # table name, so the node order is a deterministic function of the sample.
        order = []
        for table in sorted(nodes_by_type):
            ids, hops, mask = nodes_by_type[table], hops_by_type[table], keep[table]
            for local in torch.nonzero(mask, as_tuple=True)[0].tolist():
                order.append((int(hops[local]), table, local, int(ids[local])))
        order.sort(key=lambda x: (x[0], x[1], x[2]))

        index = {}          # (table, local) -> position in `nodes`
        nodes = []
        for hop, table, local, row in order:
            index[(table, local)] = len(nodes)
            nodes.append((table, row, hop))

        # Edges, re-oriented to child -> parent and deduplicated. Both orientations of the
        # same underlying fkey edge appear in the sampler output (once under `f2p_x`, once
        # under `rev_f2p_x`), so the dedupe is required, not defensive.
        seen, edges = set(), []
        for edge_type, src_local in out.row.items():
            if src_local.numel() == 0:
                continue
            dst_local = out.col[edge_type]
            src_table, _, dst_table = edge_type
            for s, d in zip(src_local.tolist(), dst_local.tolist()):
                if _is_descending(edge_type):
                    child, parent = (src_table, s), (dst_table, d)
                else:
                    child, parent = (dst_table, d), (src_table, s)
                if child not in index or parent not in index:
                    continue                      # one end was trimmed
                pair = (index[child], index[parent])
                if pair[0] != pair[1] and pair not in seen:
                    seen.add(pair)
                    edges.append(pair)

        graph = SampledGraph(nodes=nodes, edges=edges)
        if self.collapse_links and self.link_tables:
            graph = self._collapse_links(graph)
        return self._drop_orphans(graph)

    def _collapse_links(self, graph):
        """Contract pure join rows into direct edges between what they joined.

        A junction row's neighbours are all *parents* of it, so after contraction there is no
        child/parent relation left to preserve. Orient the replacement edge from the higher
        hop to the lower one, which keeps the convention that edges point back toward the
        seed: `condition (h2) -> study (h0)` reads as "this study has this condition" and
        matches `results (h1) -> drivers (h0)` on rel-f1.
        """
        link_pos = {i for i, (table, _, _) in enumerate(graph.nodes)
                    if table in self.link_tables}
        if not link_pos:
            return graph

        neighbours = {i: set() for i in link_pos}
        kept_edges = []
        for i, j in graph.edges:
            if i in link_pos or j in link_pos:
                if i in link_pos:
                    neighbours[i].add(j)
                if j in link_pos:
                    neighbours[j].add(i)
            else:
                kept_edges.append((i, j))

        hops = [hop for _, _, hop in graph.nodes]
        bridged = set()
        for link, ends in neighbours.items():
            ends = sorted(ends)
            for a_i in range(len(ends)):
                for b_i in range(a_i + 1, len(ends)):
                    a, b = ends[a_i], ends[b_i]
                    # Higher hop -> lower hop; ties keep the ordering already established.
                    edge = (a, b) if hops[a] >= hops[b] else (b, a)
                    bridged.add(edge)

        remap, nodes = {}, []
        for old, node in enumerate(graph.nodes):
            if old not in link_pos:
                remap[old] = len(nodes)
                nodes.append(node)

        seen, edges = set(), []
        for i, j in kept_edges + sorted(bridged):
            pair = (remap[i], remap[j])
            if pair[0] != pair[1] and pair not in seen:
                seen.add(pair)
                edges.append(pair)
        return SampledGraph(nodes=nodes, edges=edges)

    @staticmethod
    def _drop_orphans(graph):
        """Remove nodes with no undirected path to the seed.

        Trimming a hop-1 node can strand the hop-2 parents reached only through it. A
        disconnected component would give the model a row with no stated relationship to
        anything -- and would break the reachability invariant the tests assert.
        """
        n = len(graph.nodes)
        adj = [[] for _ in range(n)]
        for i, j in graph.edges:
            adj[i].append(j)
            adj[j].append(i)

        seen = [False] * n
        stack = [0]
        seen[0] = True
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)

        if all(seen):
            return graph

        remap, nodes = {}, []
        for old, node in enumerate(graph.nodes):
            if seen[old]:
                remap[old] = len(nodes)
                nodes.append(node)
        edges = [(remap[i], remap[j]) for i, j in graph.edges if seen[i] and seen[j]]
        return SampledGraph(nodes=nodes, edges=edges)
