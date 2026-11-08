"""uvcore.py - load a messy mesh, repair it, UV unwrap it (xatlas), export GLB
with the UV map and a blank (white) texture.

Performance notes (target: ~400k triangles):
  * trimesh is imported lazily and only for loading (no texture/material parsing).
  * Vertex welding / duplicate-face removal use packed 64-bit keys + 1-D np.unique
    (several times faster than np.unique(axis=0) on N x 3 arrays).
  * xatlas is run as compute_charts() + a single pack_charts() with an analytic
    texels_per_unit, which avoids xatlas' iterative "fit to resolution" search
    (it repacks the whole atlas several times).
  * GLB is written directly with numpy (no trimesh Trimesh/export, no Pillow);
    the white PNG is generated once per process with zlib.
  * Workers are persistent processes, so numpy/xatlas are imported once per
    worker instead of once per file.
"""
import json
import os
import struct
import sys
import time
import zlib
from functools import lru_cache
from pathlib import Path

# Portable mode: use dependencies from the ./libs folder next to this file.
_LIBS = Path(__file__).resolve().parent / "libs"
if _LIBS.is_dir():
    sys.path.insert(0, str(_LIBS))

import numpy as np
import xatlas

TEXTURE_SIZE = 2048      # blank texture is TEXTURE_SIZE x TEXTURE_SIZE
PADDING = 4              # texels between charts
TARGET_FILL = 0.60       # first-guess fraction of atlas covered by raw chart area
# Speed/quality knobs for xatlas (silently ignored if the binding lacks them).
# Larger max_cost -> fewer, bigger charts: faster pack, fewer seams, a bit more stretch.
CHART_OPTIONS = dict(max_iterations=1, max_cost=3.0)
PACK_OPTIONS = dict(bilinear=True, bruteforce=False, block_align=False)

# Performance guard for AI-generated / pathological meshes. xatlas chart
# generation can become very expensive on highly fragmented topology.
FAST_MESH_FACE_LIMIT = 250_000
FAST_MESH_VERTEX_LIMIT = 450_000


# --------------------------------------------------------------------------- load
def load_geometry(path):
    """Load any supported file and return merged (vertices, faces) arrays.
    Scenes with many parts are flattened (transforms applied) into one mesh."""
    import trimesh  # lazy: ~0.5-1 s import, only workers that load need it
    obj = trimesh.load(str(path), process=False, skip_materials=True)
    geoms = obj.dump() if isinstance(obj, trimesh.Scene) else [obj]
    verts, faces, offset = [], [], 0
    for g in geoms:
        if isinstance(g, trimesh.Trimesh) and len(g.vertices) and len(g.faces):
            verts.append(np.asarray(g.vertices, dtype=np.float64))
            faces.append(np.asarray(g.faces, dtype=np.int64) + offset)
            offset += len(g.vertices)
    if not verts:
        raise ValueError("no triangle geometry found in file")
    if len(verts) == 1:
        return verts[0], faces[0]
    return np.vstack(verts), np.vstack(faces)


# ------------------------------------------------------------------------ repair
def _compact(v, f):
    """Drop unreferenced vertices (O(n), no sorting)."""
    used = np.zeros(len(v), dtype=bool)
    used[f.ravel()] = True
    remap = np.cumsum(used) - 1
    return v[used], remap[f]


def repair(v, f):
    """Make a messy mesh safe for unwrapping: drop NaN/inf, out-of-range and
    degenerate/duplicate faces, weld duplicate vertices, remove unused vertices."""
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64)

    f = f[((f >= 0) & (f < len(v))).all(axis=1)]            # bad indices
    finite = np.isfinite(v).all(axis=1)
    if not finite.all():
        f = f[finite[f].all(axis=1)]                         # NaN / inf vertices
    if len(f) == 0:
        raise ValueError("no valid faces after removing bad data")
    v, f = _compact(v, f)

    lo = v.min(axis=0)
    extent = float((v.max(axis=0) - lo).max())
    if extent <= 0:
        raise ValueError("mesh has zero size")

    # Weld: quantise to a 2^21 grid and pack x/y/z into one int64 key.
    step = extent / (2 ** 21 - 2)
    q = np.rint((v - lo) / step).astype(np.int64)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    v = v[first]
    f = inverse.reshape(-1)[f]

    # degenerate faces (repeated vertex or ~zero area)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    if len(f):
        cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        f = f[np.einsum("ij,ij->i", cross, cross) > (extent ** 4) * 1e-28]

    # duplicate faces (same vertex set, any order/winding)
    if len(f):
        fs = np.sort(f, axis=1)
        n = len(v)
        if n ** 3 < 2 ** 62:
            fkey = (fs[:, 0] * n + fs[:, 1]) * n + fs[:, 2]
            _, idx = np.unique(fkey, return_index=True)
        else:
            _, idx = np.unique(fs, axis=0, return_index=True)
        if len(idx) != len(f):
            f = f[np.sort(idx)]
    if len(f) == 0:
        raise ValueError("no valid faces left after repair")

    return _compact(v, f)


def vertex_normals(v, f):
    """Area-weighted vertex normals (pure numpy). Returns float32."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        for ax in range(3):
            n[:, ax] += np.bincount(f[:, k], weights=fn[:, ax], minlength=len(v))
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return (n / length).astype(np.float32)


# ------------------------------------------------------------------------ unwrap
def _apply(opts, **kw):
    for k, val in kw.items():
        if hasattr(opts, k):
            try:
                setattr(opts, k, val)
            except Exception:  # noqa: BLE001
                pass


def _surface_area(v, f):
    c = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    return 0.5 * float(np.sqrt(np.einsum("ij,ij->i", c, c)).sum())


def _fix_uvs(uvs, atlas):
    uvs = np.asarray(uvs, dtype=np.float64)
    if uvs.size and float(uvs.max()) > 1.0001:           # not normalised by the binding
        w = float(getattr(atlas, "width", 0) or 0)
        h = float(getattr(atlas, "height", 0) or 0)
        if w > 0 and h > 0:
            uvs = uvs / np.array([w, h])
        else:
            uvs = uvs / float(uvs.max())
    return np.clip(uvs, 0.0, 1.0)


def _unwrap_fast(v32, f32, area, progress):
    atlas = xatlas.Atlas()
    atlas.add_mesh(v32, f32)
    co = xatlas.ChartOptions()
    _apply(co, **CHART_OPTIONS)
    atlas.compute_charts(co)
    progress(0.80)

    po = xatlas.PackOptions()
    _apply(po, **PACK_OPTIONS)
    po.resolution = TEXTURE_SIZE
    po.padding = PADDING
    tpu = (TARGET_FILL * TEXTURE_SIZE * TEXTURE_SIZE / area) ** 0.5
    for _ in range(6):                                   # shrink until one atlas fits
        po.texels_per_unit = tpu
        atlas.pack_charts(po)
        if atlas.atlas_count <= 1:
            return atlas
        tpu *= 0.88
    raise RuntimeError("could not fit charts into one atlas")


def _unwrap_generic(v32, f32):
    """Fallback for older/different xatlas bindings: let xatlas fit the resolution."""
    atlas = xatlas.Atlas()
    atlas.add_mesh(v32, f32)
    co = xatlas.ChartOptions()
    _apply(co, **CHART_OPTIONS)
    po = xatlas.PackOptions()
    po.resolution = TEXTURE_SIZE
    po.padding = PADDING
    atlas.generate(co, po)
    return atlas


def _fast_projection_uv(v, f, progress=lambda x: None):
    """O(F) UV fallback for very large/pathological meshes.

    Uses six planar projection groups and duplicates vertices at UV seams.
    It is designed for predictable speed, not for the same seam quality as
    xatlas.
    """
    progress(0.30)
    v = np.asarray(v, dtype=np.float32)
    f = np.asarray(f, dtype=np.int64)
    nf = len(f)
    if nf == 0:
        raise ValueError("no faces")

    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    dominant = np.argmax(np.abs(fn), axis=1)
    sign = (fn[np.arange(nf), dominant] < 0).astype(np.int8)
    group = dominant * 2 + sign

    corners = tri.reshape(-1, 3)
    g = np.repeat(group, 3)
    uv = np.empty((len(corners), 2), dtype=np.float32)

    # Six regions: +/-X, +/-Y, +/-Z, packed in a 3x2 grid.
    for gi in range(6):
        mask = (g == gi)
        if not np.any(mask):
            continue
        axis = gi // 2
        if axis == 0:
            raw = corners[mask][:, (1, 2)]
        elif axis == 1:
            raw = corners[mask][:, (0, 2)]
        else:
            raw = corners[mask][:, (0, 1)]

        lo = raw.min(axis=0)
        span = np.maximum(raw.max(axis=0) - lo, 1e-12)
        q = (raw - lo) / span
        col, row = gi % 3, gi // 3
        tw, th, pad = 1.0 / 3.0, 1.0 / 2.0, 0.01
        uv[mask, 0] = col * tw + pad + q[:, 0] * (tw - 2 * pad)
        uv[mask, 1] = row * th + pad + q[:, 1] * (th - 2 * pad)

    vmapping = f.reshape(-1)
    new_f = np.arange(len(vmapping), dtype=np.int64).reshape(-1, 3)
    progress(0.80)
    return vmapping, new_f, uv


def unwrap(v, f, progress=lambda x: None):
    """Run xatlas, or use the O(F) fast fallback for huge meshes."""
    if len(f) > FAST_MESH_FACE_LIMIT or len(v) > FAST_MESH_VERTEX_LIMIT:
        return _fast_projection_uv(v, f, progress)

    v32 = np.ascontiguousarray(v, dtype=np.float32)
    f32 = np.ascontiguousarray(f, dtype=np.uint32)
    try:
        atlas = _unwrap_fast(v32, f32, max(_surface_area(v, f), 1e-30), progress)
        vmapping, indices, uvs = atlas[0]
    except (AttributeError, TypeError, RuntimeError):
        try:
            atlas = _unwrap_generic(v32, f32)
            vmapping, indices, uvs = atlas[0]
        except (AttributeError, TypeError):
            vmapping, indices, uvs = xatlas.parametrize(v32, f32)
            atlas = None
    return (np.asarray(vmapping, dtype=np.int64),
            np.asarray(indices, dtype=np.int64).reshape(-1, 3),
            _fix_uvs(uvs, atlas))


# ------------------------------------------------------------------------ export
@lru_cache(maxsize=4)
def white_png(size):
    """Pure-zlib white RGB PNG (no Pillow)."""
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    row = b"\x00" + b"\xff" * (size * 3)
    co = zlib.compressobj(6)
    idat = b"".join(co.compress(row) for _ in range(size)) + co.flush()
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", idat)
            + chunk(b"IEND", b""))


def write_glb(path, pos, nrm, uv, idx):
    """Write a single-mesh GLB (positions, normals, UVs, white baseColor texture)."""
    pos = np.ascontiguousarray(pos, dtype=np.float32)
    nrm = np.ascontiguousarray(nrm, dtype=np.float32)
    uv = np.array(uv, dtype=np.float32)
    uv[:, 1] = 1.0 - uv[:, 1]                       # glTF UV origin is top-left
    idx = np.ascontiguousarray(idx, dtype=np.uint32).reshape(-1)
    png = white_png(TEXTURE_SIZE)

    blobs = [idx.tobytes(), pos.tobytes(), nrm.tobytes(), np.ascontiguousarray(uv).tobytes(), png]
    views, offset = [], 0
    for i, b in enumerate(blobs):
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(b)}
        if i == 0:
            view["target"] = 34963
        elif i < 4:
            view["target"] = 34962
        views.append(view)
        offset += (len(b) + 3) & ~3
    total = offset

    n, nf = len(pos), len(idx)
    doc = {
        "asset": {"version": "2.0", "generator": "uvcore"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 1, "NORMAL": 2, "TEXCOORD_0": 3},
            "indices": 0, "material": 0, "mode": 4}]}],
        "materials": [{"name": "blank", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": nf, "type": "SCALAR"},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3",
             "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()},
            {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 3, "componentType": 5126, "count": n, "type": "VEC2"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": total}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    length = 12 + 8 + len(js) + 8 + total

    tmp = str(path) + ".part"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, length))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", total, 0x004E4942))
        for b, view in zip(blobs, views):
            fh.write(b)
            fh.write(b"\x00" * (-len(b) % 4))
    os.replace(tmp, path)                           # atomic: no half-written .glb


# -------------------------------------------------------------------------- jobs
def run_job(idx, src, out_path, q):
    """Process one file. Sends ("p", idx, fraction) progress, then
    ("ok", idx, path, seconds) or ("err", idx, text)."""
    t0 = time.time()
    try:
        def p(x):
            q.put(("p", idx, x))

        p(0.02)
        v, f = load_geometry(src)
        p(0.15)
        v, f = repair(v, f)
        p(0.25)
        # xatlas does not need normals. Calculate them only after unwrap so
        # large meshes do not pay for an extra full face-normal/bincount pass.
        vmapping, new_f, uvs = unwrap(v, f, p)
        p(0.90)
        nrm = vertex_normals(v[vmapping], new_f)
        write_glb(out_path, v[vmapping], nrm, uvs, new_f)
        q.put(("ok", idx, str(out_path), time.time() - t0))
    except Exception as e:  # noqa: BLE001
        q.put(("err", idx, f"{type(e).__name__}: {e}"))


def worker_main(in_q, out_q):
    """Persistent worker process: imports happen once, then jobs stream in.
    A native crash only kills this worker; the app detects it and restarts one."""
    while True:
        job = in_q.get()
        if job is None:
            break
        idx, src, out_path = job
        run_job(idx, src, out_path, out_q)
