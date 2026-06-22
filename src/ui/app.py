"""
Comprehensive Interactive Design Copilot — 设计副驾驶系统
=========================================================

Tabs:
  1. Dataset — generate, inspect, quality breakdown
  2. Explain — MCTS subgraph extraction, motif discovery
  3. Floor Plans — view generated floor plans
  4. Pattern Lab — select patterns, place in canvas, GNN scoring
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys, io, json, math, random, time
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
import torch
import numpy as np
import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch

# Chinese font
import matplotlib.font_manager as fm
_cn_fonts = [f.name for f in fm.fontManager.ttflist]
_heiti = 'SimHei' if 'SimHei' in _cn_fonts else 'Microsoft YaHei' if 'Microsoft YaHei' in _cn_fonts else 'sans-serif'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [_heiti, 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

from utils.enums import RoomType
from metrics.quality_metrics import QualityMetrics

ROOM_COLORS = {
    'classroom': '#4C72B0', 'special_classroom': '#55A868', 'music_room': '#C44E52',
    'teacher_office': '#7FB8D0', 'corridor': '#F2EFE8', 'staircase': '#8172B2',
    'toilet': '#C8C8C8', 'storage': '#E8E4D8', 'entrance_hall': '#64B5CD',
}
ROOM_CN = {
    'classroom': '教室', 'special_classroom': '专用教室', 'music_room': '音乐教室',
    'teacher_office': '教师办公', 'corridor': '走道', 'staircase': '楼梯间',
    'toilet': '卫生间', 'storage': '储藏室', 'entrance_hall': '门厅',
}
RT_LIST = ['classroom', 'special_classroom', 'music_room', 'gymnasium', 'library',
           'office', 'teacher_office', 'corridor', 'staircase', 'toilet', 'storage',
           'cafeteria', 'entrance_hall']

st.set_page_config(page_title="Design Copilot", page_icon="🏗️", layout="wide")
st.title("🏗️ 建筑设计副驾驶 — Architectural Design Copilot")

# ══════════════════════════════════════════════════════════════
# Load GNN model
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'model_checkpoint_v12.pt')
    if not os.path.exists(ckpt_path):
        return None
    from models.scorer import SchoolGraphScorer
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = SchoolGraphScorer(hidden_dim=128, num_layers=3)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

gnn = load_model()

# ══════════════════════════════════════════════════════════════
# Tab 1: Dataset
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset", "🔍 Explain", "📐 Floor Plans", "🧪 Pattern Lab"
])

with tab1:
    st.header("Dataset Generation & Quality Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_graphs = st.slider("Graphs to generate", 3, 50, 5, 5)
    with col2:
        school_size = st.selectbox("School size", ["small", "medium", "large"], index=1)
    with col3:
        n_floors = st.slider("Floors", 3, 5, 4)

    if st.button("🚀 Generate & Analyze", type="primary"):
        from data.dataset import SchoolDataset
        out_dir = Path('outputs/ui_test')
        if out_dir.exists():
            import shutil; shutil.rmtree(str(out_dir))

        with st.spinner(f"Generating {n_graphs} graphs..."):
            ds = SchoolDataset(output_dir=str(out_dir))
            r = ds.generate(total_count=n_graphs,
                          school_sizes={school_size: n_graphs},
                          num_floors_range=(n_floors, n_floors),
                          split=(0.7, 0.2, 0.1), base_seed=42, verbose=False)

        st.success(f"Generated {r['actual_generated']} graphs in {r['elapsed_seconds']}s")

        # Quality analysis
        raw = out_dir / 'raw'
        metrics_all = defaultdict(list)
        scores = []
        for pt_file in raw.glob('*.pt'):
            b = torch.load(str(pt_file), weights_only=False)
            scores.append(b['metadata'].get('quality_score', 0))
            for k, v in b['metadata'].get('quality', {}).items():
                metrics_all[k].append(v)

        st.subheader("Quality Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(scores, bins=15, color='#4C72B0', edgecolor='white', alpha=0.8)
        ax.set_xlabel("Quality Score"); ax.set_ylabel("Count")
        ax.set_title(f"Quality Scores (mean={np.mean(scores):.3f}, std={np.std(scores):.3f})")
        st.pyplot(fig)
        plt.close()

        st.subheader("Per-Metric Breakdown")
        metric_names = {
            'daylight_quality': 'Daylight', 'circulation_efficiency': 'Circulation',
            'fire_safety_margin': 'Fire Safety', 'graph_robustness': 'Robustness',
            'path_redundancy': 'Path Redundancy', 'zone_cohesion': 'Zone Cohesion',
            'space_type_diversity': 'Diversity', 'vertical_flow_balance': 'Vertical Balance',
        }
        cols = st.columns(4)
        for i, (key, label) in enumerate(metric_names.items()):
            if key in metrics_all:
                vals = metrics_all[key]
                with cols[i % 4]:
                    st.metric(label, f"{np.mean(vals):.3f}", f"±{np.std(vals):.3f}")

        st.subheader("Room Type Distribution")
        type_counts_all = Counter()
        for pt_file in raw.glob('*.pt'):
            b = torch.load(str(pt_file), weights_only=False)
            x = b['hetero_data']['room'].x
            types = x[:, :13].argmax(dim=1).tolist()
            type_counts_all.update(types)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        labels = [ROOM_CN.get(RT_LIST[i], RT_LIST[i]) for i in range(13) if i in type_counts_all]
        values = [type_counts_all[i] / n_graphs for i in range(13) if i in type_counts_all]
        ax2.barh(labels, values, color='#55A868')
        ax2.set_xlabel("Avg per graph"); ax2.set_title("Room type distribution")
        st.pyplot(fig2)
        plt.close()

# ══════════════════════════════════════════════════════════════
# Tab 2: Explain
# ══════════════════════════════════════════════════════════════

with tab2:
    st.header("Subgraph Explanation — MCTS + WL Kernel")

    if gnn is None:
        st.warning("No GNN model found. Train first (see Dataset tab).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            n_explain = st.slider("Graphs to explain", 2, 10, 3, key="explain_n")
        with col2:
            n_sims = st.slider("MCTS simulations/graph", 30, 200, 50, key="explain_sims")

        if st.button("🔬 Run Subgraph Extraction", type="primary"):
            from explainer.subgraph_runner import SubgraphRunner
            from explainer.clustering import SubgraphClusterer

            with st.spinner(f"Running MCTS on {n_explain} graphs ({n_sims} sims each)..."):
                runner = SubgraphRunner(gnn, dataset_dir='outputs/ui_test', device='cpu')
                subgraphs, meta = runner.run_on_split('test', n_simulations=n_sims,
                                                       max_graphs=n_explain)

            st.success(f"Extracted {len(subgraphs)} subgraphs")

            if len(subgraphs) >= 2:
                st.subheader("Extracted Subgraphs")
                fig3, axes = plt.subplots(1, min(3, len(subgraphs)), figsize=(12, 4))
                if len(subgraphs) == 1: axes = [axes]
                sorted_idx = sorted(range(len(meta)), key=lambda i: meta[i]['reward'], reverse=True)
                for ax_i, i in enumerate(sorted_idx[:3]):
                    ax = axes[ax_i]
                    sg = subgraphs[i]
                    pos = nx.spring_layout(sg, seed=42, k=1.5)
                    node_colors = []
                    for n, attrs in sg.nodes(data=True):
                        rid = attrs.get('room_id', '?')
                        rt = rid.split('_')[0] if '_' in rid else 'room'
                        node_colors.append(ROOM_COLORS.get(rt, '#CCC'))
                    nx.draw(sg, pos, ax=ax, node_color=node_colors, node_size=80,
                           edge_color='#999', width=0.5, with_labels=False)
                    ax.set_title(f"reward={meta[i]['reward']:.4f}\n{meta[i]['subgraph_nodes']}n {meta[i]['subgraph_edges']}e",
                                fontsize=9)
                    ax.axis('off')
                st.pyplot(fig3)
                plt.close()

                # Clustering
                st.subheader("Motif Clusters")
                clusterer = SubgraphClusterer(n_clusters=min(3, len(subgraphs)))
                motifs = clusterer.fit(subgraphs, meta)
                for motif in motifs:
                    with st.expander(f"{motif.motif_id}: {motif.name} (freq={motif.frequency})"):
                        st.write(f"**Nodes:** {motif.avg_nodes:.1f} avg")
                        st.write(f"**Composition:** {motif.room_composition}")
                        st.write(f"**Description:** {motif.description}")

# ══════════════════════════════════════════════════════════════
# Tab 3: Floor Plans
# ══════════════════════════════════════════════════════════════

with tab3:
    st.header("Floor Plan Viewer")

    dataset_dir = st.text_input("Dataset directory", "outputs/dataset_200_v12")
    raw_path = Path(dataset_dir) / 'raw'
    if raw_path.exists():
        files = sorted(raw_path.glob('*.pt'))
        st.write(f"Found {len(files)} graphs")

        idx = st.slider("Graph index", 0, max(0, len(files)-1), 0)
        if idx < len(files):
            b = torch.load(str(files[idx]), weights_only=False)
            hd = b['hetero_data']; m = b['metadata']
            qs = m.get('quality_score', 0)
            x = hd['room'].x
            size = m['school_size']; nf = m['num_floors']
            dims = {'small': (50, 16), 'medium': (62, 18), 'large': (78, 18)}
            bld_w, bld_d = dims.get(size, (62, 18))
            areas = (x[:, 13] * 800).tolist()
            types = x[:, :13].argmax(dim=1).tolist()
            floor_mid = (x[:, 19] * 4).round().long().tolist()
            unique_floors = sorted(set(floor_mid))

            st.write(f"**{size.upper()}** | {nf} floors | {x.shape[0]} rooms | QS={qs:.3f}")

            for fl in unique_floors:
                indices = [i for i, f in enumerate(floor_mid) if f == fl]
                fig, ax = plt.subplots(figsize=(bld_w/10, bld_d/10 + 1))
                margin = 0.5
                corr_h = 2.4; corr_y = (bld_d - corr_h) / 2
                south_h = bld_d - corr_y - corr_h - margin
                north_h = corr_y - margin

                all_placed = []
                south_data = [i for i in indices if types[i] in {0, 5, 6}]
                north_data = [i for i in indices if types[i] not in {0, 5, 6, 7}]
                corr_data = [i for i in indices if types[i] == 7]

                total_corr = sum(areas[i] for i in corr_data)

                # South row
                cx = margin
                for i in south_data:
                    if south_h > 0:
                        w = max(3, areas[i] / south_h)
                        rt = RT_LIST[types[i]]
                        rect = FancyBboxPatch((cx, corr_y+corr_h), w-0.1, south_h,
                                              boxstyle='round,pad=0.05',
                                              facecolor=ROOM_COLORS.get(rt,'#CCC'),
                                              edgecolor='#888', lw=0.3, alpha=0.9)
                        ax.add_patch(rect)
                        if w > 2: ax.text(cx+w/2, corr_y+corr_h+south_h/2,
                                          f"{ROOM_CN.get(rt,rt)}\n{areas[i]:.0f}m2",
                                          ha='center',va='center',fontsize=5)
                        cx += w

                # North row
                cx = margin
                for i in north_data:
                    if north_h > 0:
                        w = max(3, areas[i] / north_h)
                        rt = RT_LIST[types[i]]
                        rect = FancyBboxPatch((cx, margin), w-0.1, north_h,
                                              boxstyle='round,pad=0.05',
                                              facecolor=ROOM_COLORS.get(rt,'#CCC'),
                                              edgecolor='#888', lw=0.3, alpha=0.9)
                        ax.add_patch(rect)
                        cx += w

                # Corridor spine
                if total_corr > 0:
                    rect = FancyBboxPatch((margin, corr_y), bld_w-2*margin, corr_h,
                                          boxstyle='round,pad=0.05',
                                          facecolor=ROOM_COLORS['corridor'],
                                          edgecolor='#999', lw=0.5, alpha=0.9)
                    ax.add_patch(rect)
                    ax.text(bld_w/2, corr_y+corr_h/2, f'Corridor {total_corr:.0f}m2',
                           ha='center',va='center',fontsize=7)

                fl_label = {0:'Ground', 1:'Std', 2:'Std', 3:'Std', 4:'Top'}.get(fl, f'F{fl}')
                ax.add_patch(Rectangle((0,0), bld_w, bld_d, fill=False, edgecolor='#333', lw=2))
                ax.set_xlim(-0.5, bld_w+0.5); ax.set_ylim(-0.5, bld_d+0.5)
                ax.set_aspect('equal'); ax.axis('off')
                ax.set_title(f"Floor {fl_label} ({len(indices)} rooms)", fontsize=10)
                st.pyplot(fig)
                plt.close()
    else:
        st.warning(f"Dataset not found: {raw_path}")

# ══════════════════════════════════════════════════════════════
# Tab 4: Pattern Lab
# ══════════════════════════════════════════════════════════════

with tab4:
    st.header("Pattern Lab — Compose & Score")

    # Pattern templates from v12 dataset analysis
    PATTERN_TEMPLATES = {
        "Classroom Wing (3 cls + corridor + stair)": {
            'rooms': [
                ('classroom', 60, (2, 8)), ('classroom', 60, (7, 8)),
                ('classroom', 60, (12, 8)),
                ('corridor', 180, (5, 5)), ('staircase', 30, (1, 8)),
                ('staircase', 30, (14, 8)),
            ],
            'desc': 'Standard teaching wing: 3 south-facing classrooms connected by corridor with stairs at both ends.'
        },
        "Service Block (toilet + storage + stair)": {
            'rooms': [
                ('toilet', 40, (2, 2)), ('storage', 15, (8, 2)),
                ('staircase', 30, (14, 2)), ('corridor', 60, (5, 4)),
            ],
            'desc': 'Service core with combined restroom, storage, and vertical circulation.'
        },
        "Entrance Zone (hall + corridor + stair)": {
            'rooms': [
                ('entrance_hall', 60, (4, 8)), ('corridor', 120, (5, 5)),
                ('staircase', 30, (1, 8)), ('teacher_office', 45, (12, 8)),
            ],
            'desc': 'Ground floor entrance with lobby, corridor spine, and stair access.'
        },
        "Music + Special Wing": {
            'rooms': [
                ('music_room', 80, (2, 2)), ('special_classroom', 80, (8, 2)),
                ('corridor', 80, (5, 4)), ('staircase', 30, (14, 2)),
            ],
            'desc': 'Special-purpose wing with music room (acoustic isolation needed) and lab space.'
        },
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Pattern Library")
        selected_pattern = st.radio("Select a pattern:", list(PATTERN_TEMPLATES.keys()))
        if selected_pattern:
            pat = PATTERN_TEMPLATES[selected_pattern]
            st.write(f"**Description:** {pat['desc']}")
            st.write(f"**Rooms:** {len(pat['rooms'])}")

            # Preview
            fig_p, ax_p = plt.subplots(figsize=(5, 4))
            for rt, area, (cx, cy) in pat['rooms']:
                w = max(2, math.sqrt(area) * 0.8)
                h = area / w
                rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                      boxstyle='round,pad=0.1',
                                      facecolor=ROOM_COLORS.get(rt, '#CCC'),
                                      edgecolor='#555', lw=0.8, alpha=0.9)
                ax_p.add_patch(rect)
                ax_p.text(cx, cy, f"{ROOM_CN.get(rt,rt)}\n{area:.0f}m2",
                         ha='center', va='center', fontsize=7)
            ax_p.set_xlim(0, 16); ax_p.set_ylim(0, 12)
            ax_p.set_aspect('equal'); ax_p.axis('off')
            ax_p.set_title(f"Preview: {selected_pattern}", fontsize=10)
            st.pyplot(fig_p)
            plt.close()

    with col2:
        st.subheader("Composition Canvas")
        if st.button("➕ Place Pattern on Canvas", type="primary"):
            st.session_state['canvas_rooms'] = st.session_state.get('canvas_rooms', [])
            pat = PATTERN_TEMPLATES[selected_pattern]
            offset_x = random.uniform(2, 6)
            offset_y = random.uniform(1, 3)
            for rt, area, (cx, cy) in pat['rooms']:
                st.session_state['canvas_rooms'].append({
                    'type': rt, 'area': area,
                    'x': cx + offset_x, 'y': cy + offset_y,
                })
            st.success(f"Added {len(pat['rooms'])} rooms from '{selected_pattern}'")
            st.rerun()

        if st.button("🗑️ Clear Canvas"):
            st.session_state['canvas_rooms'] = []
            st.rerun()

        canvas = st.session_state.get('canvas_rooms', [])
        if canvas:
            st.write(f"Canvas: {len(canvas)} rooms")

            fig_c, ax_c = plt.subplots(figsize=(8, 5))
            total_area = 0
            for r in canvas:
                w = max(1.5, math.sqrt(r['area']) * 0.8)
                h = r['area'] / w
                total_area += r['area']
                rect = FancyBboxPatch((r['x']-w/2, r['y']-h/2), w, h,
                                      boxstyle='round,pad=0.1',
                                      facecolor=ROOM_COLORS.get(r['type'], '#CCC'),
                                      edgecolor='#555', lw=0.7, alpha=0.9)
                ax_c.add_patch(rect)
                ax_c.text(r['x'], r['y'], f"{ROOM_CN.get(r['type'],r['type'])}\n{r['area']:.0f}m2",
                         ha='center', va='center', fontsize=6.5)
            ax_c.set_xlim(0, 20); ax_c.set_ylim(0, 14)
            ax_c.set_aspect('equal'); ax_c.axis('off')
            ax_c.set_title(f"Canvas — {len(canvas)} rooms, {total_area:.0f} m2 total")
            st.pyplot(fig_c)
            plt.close()

            if gnn is not None and st.button("🧠 Score with GNN"):
                # Build minimal HeteroData from canvas
                from data.feature_engineering import FeatureEngineer
                from data.room_factory import RoomSpec, RoomNode
                from utils.enums import DaylightLevel, NoiseLevel

                # Create RoomNodes
                rooms = []
                for i, r in enumerate(canvas):
                    try:
                        rt = RoomType(r['type'])
                    except ValueError:
                        rt = RoomType.CLASSROOM
                    spec = RoomSpec(rt, r['type'], (20, 200), (1, 3),
                                   DaylightLevel.HIGH, NoiseLevel.MODERATE,
                                   NoiseLevel.MODERATE, 1.2, 2, [0, 1, 2])
                    rooms.append(RoomNode(f"r{i}", spec, r['area'], 1.5, 0,
                                          floor_range=(0, 0), typical_floor='ground',
                                          centroid=(r['x'], r['y']), zone_id=0))

                fe = FeatureEngineer()
                hd = fe.build_hetero_data(rooms, [], {})

                # GNN score
                with torch.no_grad():
                    score = gnn(hd)
                # Quality metrics
                qm = QualityMetrics.compute_all(hd)
                qs = QualityMetrics.aggregate(qm)

                st.metric("GNN Quality Score", f"{score.item():.3f}")
                st.write(f"Aggregated: {qs:.3f}")

                cols_m = st.columns(4)
                active = ['daylight_quality', 'circulation_efficiency', 'fire_safety_margin',
                         'graph_robustness', 'path_redundancy', 'zone_cohesion']
                for i, k in enumerate(active):
                    if k in qm:
                        with cols_m[i % 4]:
                            st.metric(k[:18], f"{qm[k]:.3f}")
        else:
            st.info("Select a pattern and click 'Place Pattern' to start composing.")

st.sidebar.markdown("---")
st.sidebar.info(f"GNN Model: {'Loaded (R2=0.74)' if gnn else 'Not loaded'}")
st.sidebar.markdown("[GitHub](https://github.com/Chen-apzhu/graph_pattern_learning)")
