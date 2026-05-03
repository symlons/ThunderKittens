"""
device_stats.py - query actual GPU hardware at runtime and store in SQLite.

Pipeline:
  1. Run `data/device_query` (compiled CUDA binary) → raw key=value stream
  2. Fill gaps with `torch.cuda` properties
  3. Map GPU name → NVIDIA-published peak TFLOPs (nominal specs)
  4. Upsert into `data/reports.db` (SQLite)
  5. Return dict for in-process use

The dashboard and measurements read from this DB instead of hardcoding specs.
"""

import subprocess
import os
import sqlite3
import json
from datetime import datetime, timezone

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(DATA_DIR, "reports.db")




def _compute_peak_specs(props: dict) -> dict:
    """Return peak TFLOP dicts from the CUDA binary (calculated at runtime from SM clocks)."""
    return {
        "bf16_dense_tflops": props.get("peak_bf16_tflops", 0.0),
        "bf16_sparse_tflops": props.get("peak_bf16_sparse_tflops", 0.0),
        "fp32_dense_tflops": props.get("peak_fp32_tflops", 0.0),
        "fp64_dense_tflops": props.get("peak_fp64_tflops", 0.0),
        "tf32_dense_tflops": props.get("peak_tf32_tflops", 0.0),
    }


def _run_cuda_query() -> dict:
    """Execute the compiled device_query binary and parse key=value output."""
    binary = os.path.join(DATA_DIR, "device_query")
    if not os.path.exists(binary):
        return {}
    try:
        out = subprocess.check_output([binary], timeout=10, text=True)
        kv = {}
        for line in out.strip().splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            # Auto-coerce numeric values
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            kv[k] = v
        return kv
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return {}


def _torch_fallback() -> dict:
    """Fill missing properties from torch if CUDA binary fails."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        p = torch.cuda.get_device_properties(0)
        return {
            "name": p.name,
            "device_id": 0,
            "compute_capability_major": p.major,
            "compute_capability_minor": p.minor,
            "multiprocessor_count": p.multi_processor_count,
            "l2_cache_bytes": p.L2_cache_size,
        }
    except Exception:
        return {}


def query_device() -> dict:
    """Merge CUDA binary output with torch fallback and published specs."""
    props = _run_cuda_query()
    for k, v in _torch_fallback().items():
        props.setdefault(k, v)

    # Add published peak specs
    props.update(_compute_peak_specs(props))
    return props


# ---- SQLite database ----

_SCHEMA = """
CREATE TABLE IF NOT EXISTS device_specs (
    id              INTEGER PRIMARY KEY,
    queried_at      TEXT    NOT NULL,
    gpu_name        TEXT    NOT NULL,
    compute_cap     TEXT    DEFAULT '',
    sm_count        INTEGER,
    l2_cache_mb     REAL,
    hbm_bandwidth_tbps  REAL,
    peak_bf16_dense_tflops  REAL,
    peak_bf16_sparse_tflops REAL,
    raw_props       TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS benchmark_config (
    id              INTEGER PRIMARY KEY,
    created_at      TEXT    NOT NULL,
    shape           TEXT    NOT NULL,
    dtype           TEXT    NOT NULL,
    warmup          INTEGER DEFAULT 0,
    iters           INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS layer_info (
    id              INTEGER PRIMARY KEY,
    layer_id        TEXT    NOT NULL,
    label           TEXT    DEFAULT '',
    phase           TEXT    DEFAULT '',
    total_us        REAL
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              INTEGER PRIMARY KEY,
    run_at          TEXT    NOT NULL,
    layer_id        TEXT    NOT NULL,
    kernel_name     TEXT    NOT NULL,
    shape           TEXT    DEFAULT '',
    dtype           TEXT    DEFAULT '',
    total_us        REAL,
    custom_us       REAL,
    custom_tflops   REAL,
    cublas_us       REAL,
    cublas_tflops   REAL,
    baseline_name   TEXT    DEFAULT '',
    baseline_us     REAL,
    speedup_vs_cublas REAL,
    speedup_vs_bas  REAL,
    metadata        TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS correctness_results (
    id              INTEGER PRIMARY KEY,
    checked_at      TEXT    NOT NULL,
    layer_id        TEXT    NOT NULL,
    tensor_name     TEXT    NOT NULL,
    max_diff        REAL,
    mean_diff       REAL,
    threshold       REAL,
    dtypes          TEXT    DEFAULT '',
    shape           TEXT    DEFAULT '',
    passed          INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS compute_graph_nodes (
    id              INTEGER PRIMARY KEY,
    layer_id        TEXT    NOT NULL,
    node_id         TEXT    NOT NULL,
    node_type       TEXT    DEFAULT '',
    label           TEXT    DEFAULT '',
    dtype           TEXT    DEFAULT '',
    shape           TEXT    DEFAULT '',
    formula         TEXT    DEFAULT '',
    description     TEXT    DEFAULT '',
    kernel_name     TEXT    DEFAULT '',
    stage           TEXT    DEFAULT '',
    metadata        TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS compute_graph_edges (
    id              INTEGER PRIMARY KEY,
    layer_id        TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    destination     TEXT    NOT NULL
);
"""


def _ensure_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    _migrate_db(conn)
    conn.commit()
    return conn


def _migrate_db(conn: sqlite3.Connection) -> None:
    """Add optional graph metadata columns for DBs created by older versions."""
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(compute_graph_nodes)").fetchall()
    }
    migrations = {
        "formula": "ALTER TABLE compute_graph_nodes ADD COLUMN formula TEXT DEFAULT ''",
        "description": "ALTER TABLE compute_graph_nodes ADD COLUMN description TEXT DEFAULT ''",
        "kernel_name": "ALTER TABLE compute_graph_nodes ADD COLUMN kernel_name TEXT DEFAULT ''",
        "stage": "ALTER TABLE compute_graph_nodes ADD COLUMN stage TEXT DEFAULT ''",
        "metadata": "ALTER TABLE compute_graph_nodes ADD COLUMN metadata TEXT DEFAULT ''",
    }
    for col, sql in migrations.items():
        if col not in cols:
            conn.execute(sql)


def upsert_device_specs(path: str = DB_PATH) -> dict:
    """Query hardware and store latest row, return dict of specs."""
    props = query_device()
    if not props:
        return {}

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    conn = _ensure_db(path)
    conn.execute("""
        INSERT INTO device_specs (queried_at, gpu_name, compute_cap, sm_count,
                                 l2_cache_mb, hbm_bandwidth_tbps,
                                 peak_bf16_dense_tflops, peak_bf16_sparse_tflops,
                                 raw_props)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts,
        props.get("name", "unknown"),
        f"{props.get('compute_capability_major', '?')}.{props.get('compute_capability_minor', '?')}",
        props.get("multiprocessor_count"),
        (props.get("l2_cache_bytes", 0) / 1024 / 1024),
        props.get("hbm_bandwidth_tbps"),
        props.get("bf16_dense_tflops"),
        props.get("bf16_sparse_tflops"),
        json.dumps(props),
    ))
    conn.commit()
    conn.close()
    return props


def get_latest_specs(path: str = DB_PATH) -> dict | None:
    """Return the most recent device_specs row as a dict (or None)."""
    conn = _ensure_db(path)
    row = conn.execute("""
        SELECT raw_props FROM device_specs ORDER BY id DESC LIMIT 1
    """).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row[0])


def record_benchmark_run(
    layer_id: str,
    kernel_name: str,
    shape: str,
    dtype: str,
    total_us: float,
    custom_us: float,
    custom_tflops: float | None = None,
    cublas_us: float | None = None,
    cublas_tflops: float | None = None,
    baseline_name: str = "",
    baseline_us: float | None = None,
    metadata: str | dict = "",
    path: str = DB_PATH,
) -> None:
    """Insert a benchmark result row."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    speedup_cu = None if (cublas_us is None or custom_us <= 0) else cublas_us / custom_us
    speedup_bl = None if (baseline_us is None or custom_us <= 0) else baseline_us / custom_us
    metadata_s = json.dumps(metadata) if isinstance(metadata, dict) else metadata

    conn = _ensure_db(path)
    conn.execute("""
        INSERT INTO benchmark_runs (run_at, layer_id, kernel_name, shape, dtype,
             total_us, custom_us, custom_tflops, cublas_us, cublas_tflops,
             baseline_name, baseline_us, speedup_vs_cublas, speedup_vs_bas, metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        ts, layer_id, kernel_name, shape, dtype,
        total_us, custom_us, custom_tflops, cublas_us, cublas_tflops,
        baseline_name, baseline_us, speedup_cu, speedup_bl, metadata_s,
    ))
    conn.commit()
    conn.close()


def record_correctness(
    layer_id: str,
    tensor_name: str,
    max_diff: float,
    mean_diff: float,
    threshold: float,
    passed: bool,
    dtypes: str = "",
    shape: str = "",
    path: str = DB_PATH,
) -> None:
    """Insert a correctness result row."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    conn = _ensure_db(path)
    conn.execute("""
        INSERT INTO correctness_results (checked_at, layer_id, tensor_name,
             max_diff, mean_diff, threshold, dtypes, shape, passed)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (ts, layer_id, tensor_name, max_diff, mean_diff, threshold,
          dtypes, shape, 1 if passed else 0))
    conn.commit()
    conn.close()


def record_benchmark_config(shape: str, dtype: str, warmup: int, iters: int, path: str = DB_PATH) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    conn = _ensure_db(path)
    conn.execute("""INSERT INTO benchmark_config (created_at, shape, dtype, warmup, iters)
                    VALUES (?,?,?,?,?)""", (ts, shape, dtype, warmup, iters))
    conn.commit()
    conn.close()


def record_layer(layer_id: str, label: str, phase: str, total_us: float, path: str = DB_PATH) -> None:
    conn = _ensure_db(path)
    conn.execute("DELETE FROM layer_info WHERE layer_id = ?", (layer_id,))
    conn.execute("""INSERT INTO layer_info (layer_id, label, phase, total_us)
                    VALUES (?,?,?,?)""", (layer_id, label, phase, total_us))
    conn.commit()
    conn.close()


def record_compute_graph_nodes(layer_id: str, nodes: list[dict], path: str = DB_PATH) -> None:
    conn = _ensure_db(path)
    conn.execute("DELETE FROM compute_graph_nodes WHERE layer_id = ?", (layer_id,))
    for node in nodes:
        metadata = node.get("metadata", "")
        metadata_s = json.dumps(metadata) if isinstance(metadata, dict) else metadata
        conn.execute("""
            INSERT INTO compute_graph_nodes (
                layer_id, node_id, node_type, label, dtype, shape,
                formula, description, kernel_name, stage, metadata
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            layer_id,
            node["id"],
            node.get("type", ""),
            node.get("label", ""),
            node.get("dtype", ""),
            node.get("shape", ""),
            node.get("formula", ""),
            node.get("description", ""),
            node.get("kernel", node.get("kernel_name", "")),
            node.get("stage", ""),
            metadata_s,
        ))
    conn.commit()
    conn.close()


def record_compute_graph_edges(layer_id: str, edges: list[tuple[str, str]], path: str = DB_PATH) -> None:
    conn = _ensure_db(path)
    conn.execute("DELETE FROM compute_graph_edges WHERE layer_id = ?", (layer_id,))
    for src, dst in edges:
        conn.execute("""INSERT INTO compute_graph_edges (layer_id, source, destination)
                        VALUES (?,?,?)""", (layer_id, src, dst))
    conn.commit()
    conn.close()


# ---- Query helpers ----

def query_all_benchmarks(layer_id: str | None = None, path: str = DB_PATH) -> list[dict]:
    conn = _ensure_db(path)
    q = "SELECT * FROM benchmark_runs"
    args = ()
    if layer_id:
        q += " WHERE layer_id = ?"
        args = (layer_id,)
    q += " ORDER BY id"
    cur = conn.execute(q, args)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def query_all_correctness(layer_id: str | None = None, path: str = DB_PATH) -> list[dict]:
    conn = _ensure_db(path)
    q = "SELECT * FROM correctness_results"
    args = ()
    if layer_id:
        q += " WHERE layer_id = ?"
        args = (layer_id,)
    q += " ORDER BY id"
    cur = conn.execute(q, args)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def query_latest_config(path: str = DB_PATH) -> dict:
    conn = _ensure_db(path)
    cur = conn.execute("SELECT * FROM benchmark_config ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    cols = [d[0] for d in cur.description]
    conn.close()
    if row is None:
        return {}
    return dict(zip(cols, row))


def query_layers(path: str = DB_PATH) -> list[dict]:
    conn = _ensure_db(path)
    cur = conn.execute("SELECT * FROM layer_info ORDER BY id")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def query_compute_graph(layer_id: str, path: str = DB_PATH) -> dict:
    conn = _ensure_db(path)
    ncur = conn.execute(
        "SELECT * FROM compute_graph_nodes WHERE layer_id = ? ORDER BY id",
        (layer_id,),
    )
    ncols = [d[0] for d in ncur.description]
    nodes = [dict(zip(ncols, r)) for r in ncur.fetchall()]
    ecur = conn.execute(
        "SELECT * FROM compute_graph_edges WHERE layer_id = ? ORDER BY id",
        (layer_id,),
    )
    ecols = [d[0] for d in ecur.description]
    edges = [dict(zip(ecols, r)) for r in ecur.fetchall()]
    conn.close()
    return {"nodes": nodes, "edges": edges}


def query_dashboard_data(path: str = DB_PATH) -> dict:
    """Return all data needed for the dashboard in one call."""
    gpu = get_latest_specs(path) or {}
    config = query_latest_config(path)
    layers_list = query_layers(path)
    if not layers_list:
        conn = _ensure_db(path)
        cur = conn.execute("SELECT DISTINCT layer_id FROM benchmark_runs ORDER BY layer_id")
        layers_list = [
            {"layer_id": r[0], "label": r[0], "phase": "", "total_us": None}
            for r in cur.fetchall()
        ]
        conn.close()
    result = {
        "gpu": gpu,
        "config": config,
        "layers": {},
        "correctness": [],
    }
    for ly in layers_list:
        lid = ly["layer_id"]
        runs = query_all_benchmarks(lid, path)
        if runs:
            latest_runs = {}
            for run in runs:
                key = (run["kernel_name"], run.get("baseline_name", ""))
                latest_runs[key] = run
            runs = list(latest_runs.values())
        graph = query_compute_graph(lid, path)
        correctness = query_all_correctness(lid, path)
        if correctness:
            latest_correctness = {}
            for row in correctness:
                latest_correctness[row["tensor_name"]] = row
            correctness = list(latest_correctness.values())
            result["correctness"].extend(correctness)
        result["layers"][lid] = {
            "info": ly,
            "runs": runs,
            "compute_graph": {
                "nodes": graph["nodes"],
                "edges": [(e["source"], e["destination"]) for e in graph["edges"]],
            },
        }
    return result


def init_db(path: str = DB_PATH) -> None:
    """Bootstrap: query hardware once and store it."""
    existing = get_latest_specs(path)
    if existing is None:
        upsert_device_specs(path)


if __name__ == "__main__":
    init_db()
    specs = get_latest_specs()
    print(json.dumps(specs, indent=2))
