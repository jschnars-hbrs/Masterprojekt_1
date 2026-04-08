"""
PBRT Trianglemesh Viewer — interactive 3D viewer with live editing.

Run:  streamlit run pbrt_mesh_viewer.py
"""

import re
import numpy as np
import streamlit as st
import plotly.graph_objects as go

EXAMPLE_MESH = """\
Shape "trianglemesh"  "integer indices" [0 1 3 1 2 3 2 3 4 3 4 5]
    "point3 P" [-0.289 -3 0.5
            -0.289 3 0.5
             0 3 1
             0 -3 1
             0.289 3 0.5
              0.289 -3 0.5]"""


def parse_pbrt_trianglemesh(text: str):
    """Parse PBRT trianglemesh text into vertices and face indices."""
    # Extract integer indices
    idx_match = re.search(
        r'"integer\s+indices"\s*\[([^\]]+)\]', text, re.DOTALL
    )
    if not idx_match:
        raise ValueError('Could not find "integer indices" [...] block.')
    indices = np.fromstring(idx_match.group(1).replace("\n", " "), sep=" ", dtype=int)
    if len(indices) % 3 != 0:
        raise ValueError(
            f"Index count ({len(indices)}) is not a multiple of 3."
        )
    faces = indices.reshape(-1, 3)

    # Extract point3 P
    p_match = re.search(
        r'"point3\s+P"\s*\[([^\]]+)\]', text, re.DOTALL
    )
    if not p_match:
        raise ValueError('Could not find "point3 P" [...] block.')
    coords = np.fromstring(p_match.group(1).replace("\n", " "), sep=" ")
    if len(coords) % 3 != 0:
        raise ValueError(
            f"Coordinate count ({len(coords)}) is not a multiple of 3."
        )
    vertices = coords.reshape(-1, 3)

    if faces.max() >= len(vertices):
        raise ValueError(
            f"Index {faces.max()} out of range for {len(vertices)} vertices."
        )

    return vertices, faces


def build_figure(vertices, faces):
    """Build a Plotly figure with the mesh and vertex markers."""
    fig = go.Figure()

    # Solid mesh
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="steelblue",
            opacity=0.7,
            flatshading=True,
            name="mesh",
        )
    )

    # Wireframe edges
    edge_x, edge_y, edge_z = [], [], []
    for tri in faces:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            edge_x += [vertices[tri[a], 0], vertices[tri[b], 0], None]
            edge_y += [vertices[tri[a], 1], vertices[tri[b], 1], None]
            edge_z += [vertices[tri[a], 2], vertices[tri[b], 2], None]
    fig.add_trace(
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="black", width=2),
            name="wireframe",
        )
    )

    # Vertex markers with index labels
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers+text",
            marker=dict(size=4, color="red"),
            text=[str(i) for i in range(len(vertices))],
            textposition="top center",
            textfont=dict(size=10),
            name="vertices",
        )
    )

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
    )
    return fig


# ── Streamlit app ──────────────────────────────────────────────

st.set_page_config(page_title="PBRT Mesh Viewer", layout="wide")
st.title("PBRT Trianglemesh Viewer")

with st.sidebar:
    st.header("Mesh Input")
    mesh_text = st.text_area(
        "Paste PBRT trianglemesh definition:",
        value=EXAMPLE_MESH,
        height=350,
    )

try:
    vertices, faces = parse_pbrt_trianglemesh(mesh_text)
    st.plotly_chart(build_figure(vertices, faces), use_container_width=True)
    with st.sidebar:
        st.success(f"{len(vertices)} vertices, {len(faces)} faces")
except ValueError as exc:
    st.error(f"Parse error: {exc}")
