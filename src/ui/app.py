"""
Pattern-Driven Interactive Layout — 模式驱动交互式布局

Usage:
    streamlit run src/ui/app.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys, io, json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import streamlit as st
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ui.pattern_engine import PatternEngine, ROOM_COLORS, ROOM_CN

st.set_page_config(page_title="Interactive Pattern Layout", page_icon="🎨", layout="wide")
st.title("🎨 模式驱动交互式布局 — Pattern-Driven Interactive Design")

# ══════════════════════════════════════════════════════════════
# Load GNN model once
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'model_checkpoint.pt')
    if not os.path.exists(ckpt_path):
        return None
    from models.scorer import SchoolGraphScorer
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = SchoolGraphScorer(hidden_dim=128, num_layers=3)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

gnn_model = load_model()

# ══════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════

if 'engine' not in st.session_state:
    st.session_state.engine = PatternEngine()
    st.session_state.engine.load_motifs()

engine = st.session_state.engine

# ══════════════════════════════════════════════════════════════
# Three-column layout
# ══════════════════════════════════════════════════════════════

col_palette, col_canvas, col_score = st.columns([2, 4, 2])

# ── LEFT: Pattern Palette ──────────────────────────────────────
with col_palette:
    st.subheader("📖 建筑模式库")
    st.caption("从数据集中学到的空间模体")

    if not engine.patterns:
        st.warning("未加载模体词典")
    else:
        # Pattern selector
        pattern_names = [p.summary()[:60] for p in engine.patterns]
        selected = st.selectbox("选择模式", range(len(pattern_names)),
                                format_func=lambda i: pattern_names[i])

        if selected is not None:
            pat = engine.patterns[selected]
            st.info(f"**{pat.motif_id}**: {pat.name}")
            st.caption(f"频率: {pat.percentage:.1%} ({pat.frequency}次)")
            # Room list
            for rt, cnt in sorted(pat.room_counts.items(), key=lambda x: -x[1]):
                if cnt >= 0.5:
                    st.caption(f"  {ROOM_CN.get(rt, rt)} × {cnt:.0f}")

            # Placement controls
            st.number_input("放置 X", 0.0, 120.0, engine.placement_x, 2.0,
                           key='place_x', on_change=lambda: setattr(engine, 'placement_x', st.session_state.place_x))
            st.number_input("放置 Y", 0.0, 80.0, engine.placement_y, 2.0,
                           key='place_y', on_change=lambda: setattr(engine, 'placement_y', st.session_state.place_y))

            if st.button("📍 放置到画布", use_container_width=True, type='primary'):
                engine.place_pattern(selected, engine.placement_x, engine.placement_y)
                engine.auto_complete()
                if gnn_model:
                    engine.score_with_gnn(gnn_model)
                st.rerun()

    st.divider()

    # Canvas controls
    st.subheader("🎛️ 画布控制")
    if st.button("🔄 自动补全", use_container_width=True):
        engine.auto_complete()
        if gnn_model:
            engine.score_with_gnn(gnn_model)
        st.rerun()

    if st.button("🗑️ 清空画布", use_container_width=True):
        engine.clear()
        st.rerun()

    st.divider()
    n_rooms = len(engine.placed_rooms)
    types = set(r.room_type for r in engine.placed_rooms)
    st.caption(f"画布: {n_rooms} 个房间, {len(types)} 种类型")

# ── MIDDLE: Canvas ────────────────────────────────────────────
with col_canvas:
    st.subheader("🎨 设计画布")

    fig = engine.render_canvas()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    st.image(buf)
    plt.close(fig)

    # Room list
    if engine.placed_rooms:
        with st.expander(f"📋 房间列表 ({len(engine.placed_rooms)} 个)"):
            for pr in engine.placed_rooms:
                st.caption(
                    f"{pr.label:20s} | "
                    f"{pr.width:.1f}×{pr.height:.1f}m | "
                    f"({pr.x:.0f}, {pr.y:.0f})"
                )

# ── RIGHT: Score + Feedback ───────────────────────────────────
with col_score:
    st.subheader("📊 实时评分")

    if gnn_model is None:
        st.warning("GNN 模型未加载")
    elif engine.gnn_score is not None:
        grade = 'A' if engine.gnn_score >= 0.85 else 'B' if engine.gnn_score >= 0.7 else 'C'
        st.metric("GNN 质量分", f"{engine.gnn_score:.3f}", delta=grade)

        # Score gauge visual
        import numpy as np
        fig_g, ax_g = plt.subplots(figsize=(3, 1.5), dpi=80)
        ax_g.barh([0], [engine.gnn_score], color='#2ECC71' if engine.gnn_score >= 0.7
                  else '#F39C12' if engine.gnn_score >= 0.5 else '#E74C3C')
        ax_g.barh([0], [1.0], color='#EEE', zorder=0)
        ax_g.set_xlim(0, 1)
        ax_g.axis('off')
        buf_g = io.BytesIO()
        fig_g.savefig(buf_g, format='png', dpi=80, bbox_inches='tight')
        st.image(buf_g)
        plt.close(fig_g)
    else:
        st.info("放置房间后自动评分")

    st.divider()
    st.subheader("✅ 设计检查")

    if engine.placed_rooms:
        types_present = set(r.room_type for r in engine.placed_rooms)
        checks = [
            ('classroom', '有教室', 'classroom' in types_present),
            ('corridor', '有走道', 'corridor' in types_present),
            ('staircase', '有楼梯(≥2)', sum(1 for r in engine.placed_rooms if r.room_type == 'staircase') >= 2),
            ('toilet', '有卫生间', 'toilet' in types_present),
        ]
        for _, label, ok in checks:
            st.caption(f"{'✅' if ok else '❌'} {label}")

    st.divider()
    st.subheader("💡 使用说明")
    st.caption("1. 左侧选择建筑模式")
    st.caption("2. 调整放置坐标")
    st.caption("3. 点击「放置到画布」")
    st.caption("4. 系统自动补全配套")
    st.caption("5. 查看实时 GNN 评分")
    st.caption("6. 重复放置构建完整平面")

st.divider()
st.caption("Phase 1-5 · Pattern-Driven Interactive Design · Graph Pattern Learning Project")
