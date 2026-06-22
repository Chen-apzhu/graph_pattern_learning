# Graph Pattern Learning — 项目文档

> 基于多模态异构图与子图解释器的可解释性建筑布局生成与模式提取

**版本**: 0.2.0 | **日期**: 2026-06-22 | **分支**: master | **远程**: https://github.com/Chen-apzhu/graph_pattern_learning

---

## 1. 项目概述

### 研究目标

解决纯数据驱动生成模型在早期概念设计阶段的"黑盒"问题，利用子图解释器提取在严苛约束下依然高效的核心空间模体（Motifs），构建人类可读的《建筑空间模式语言词典》，落地为人机协同的设计副驾驶系统。

### 研究对象

以**中小学教学楼**为核心场景的拓扑关系复杂的公共建筑。

### 技术栈

| 层级 | 技术 | 版本 |
|:---|:---|:---|
| 语言 | Python | 3.12.7 |
| 深度学习 | PyTorch | 2.12.0 (CPU) |
| 图学习 | PyG (Torch Geometric) | 2.8.0 |
| 图论 | NetworkX | 3.3 |
| 可视化 | Matplotlib | 3.9.2 |
| 前端 | Streamlit | 1.37.1 |
| 数值计算 | NumPy | 1.26.4 |
| 配置 | PyYAML | 6.0+ |

---

## 2. 项目结构

```
graph_pattern_learning/
├── src/
│   ├── config/
│   │   ├── building_rules.yaml    # 建筑规范参数、学校规模、占地面积
│   │   └── room_catalog.yaml      # 13种房间类型参数规范
│   ├── data/
│   │   ├── generator.py           # 主编排器：SchoolBuildingGenerator
│   │   ├── room_factory.py        # RoomNode/RoomSpec 工厂 + 目录加载
│   │   ├── room_rules.py          # 房间分布验证 + 最小值强制
│   │   ├── topology_rules.py      # 边生成：物理/声学/视线 + 走廊拓扑多样性
│   │   ├── constraints.py         # 7项约束验证器（含面积完备性）
│   │   ├── feature_engineering.py # RoomNode → PyG HeteroData 特征工程
│   │   └── dataset.py             # 批量数据集生成 + train/val/test 拆分
│   ├── graph/
│   │   ├── school_graph.py        # SchoolGraphData 包装器
│   │   ├── graph_utils.py         # GraphAnalyzer 图分析工具
│   │   └── graph_stats.py         # 图统计计算
│   ├── models/
│   │   ├── encoder.py             # HeteroGNNEncoder (3层 SAGEConv)
│   │   ├── scorer.py              # SchoolGraphScorer (encoder + pool + MLP)
│   │   └── losses.py              # 约束损失函数 + 品质分计算
│   ├── training/
│   │   ├── data_loader.py         # SchoolDataLoader (PyG .pt 加载)
│   │   └── trainer.py            # 训练循环 (MSE + 约束损失)
│   ├── explainer/
│   │   ├── mcts_search.py         # SubgraphX MCTS 子图搜索
│   │   ├── wl_kernel.py           # WL 子树核图相似度
│   │   ├── clustering.py          # 谱聚类 + 模体原型提取
│   │   ├── motif.py              # Motif 数据结构（中英文）
│   │   ├── subgraph_runner.py     # 批量 MCTS 运行器
│   │   └── explanation.py         # 综合解释报告生成
│   ├── metrics/
│   │   └── quality_metrics.py     # 9个品质指标 + 加权聚合
│   ├── ui/
│   │   ├── app.py                 # Streamlit 交互式设计副驾驶
│   │   ├── layout_engine.py       # 正交平面图布局引擎
│   │   ├── pattern_engine.py      # 模式匹配引擎
│   │   └── components.py          # UI 组件
│   └── utils/
│       ├── enums.py               # 枚举定义 (RoomType/EdgeCategory/ZoneType...)
│       ├── constants.py           # 常量 (特征维度/归一化边界)
│       ├── serialization.py       # HeteroData 序列化
│       └── visualization.py       # 图可视化工具
├── tests/                         # 19个单元测试 (TDD)
│   ├── test_enums.py
│   ├── test_room_factory.py
│   ├── test_topology_rules.py
│   ├── test_constraints.py        # 含 area_completeness 测试
│   ├── test_generator.py
│   ├── test_feature_engineering.py
│   ├── test_school_graph.py
│   ├── test_graph_utils.py
│   ├── test_graph_stats.py
│   ├── test_serialization.py
│   └── test_dataset.py (planned)
├── outputs/
│   ├── dataset_200_v12/           # 当前主力数据集 (200图)
│   ├── dataset_200_v11/           # v11 (走廊多样性 + 新指标)
│   ├── model_checkpoint_v12.pt    # GNN 模型 (R²=0.74)
│   ├── explainer/                 # 模体词典 JSON/TXT
│   ├── sample_plans_v12/          # 平面布局图
│   └── presentation/              # 汇报图表
├── PROJECT.md                     # 本文档
├── README.md                      # 快速入门
├── task.md                        # 课题原始需求文档
├── pyproject.toml                 # 项目配置
├── requirements.txt               # 依赖
├── draw_plans.py                  # 平面图生成脚本
├── run_explainer.py               # 可解释管线运行脚本
├── evaluate_quality.py            # 品质评估脚本
├── inspect_dataset.py             # 数据集检查脚本
├── create_presentation.py         # 汇报图表生成
├── create_ortho_plans.py          # 正交平面图生成
└── audit_quality.py               # 品质审计脚本
```

---

## 3. 数据生成流程 (Phase 1)

### 3.1 输入配置

**room_catalog.yaml** — 13种房间类型规范（GB50099-2011）：

| 房间类型 | 面积(m²) | 采光 | 噪声 | 用途 |
|:---|:---|:---|:---|:---|
| classroom | 54-72 | HIGH | MODERATE | 普通教室 |
| special_classroom | 72-96 | HIGH | MODERATE | 专用教室 |
| music_room | 72-90 | MEDIUM | LOUD | 音乐教室 |
| teacher_office | 30-60 | HIGH | MODERATE | 教师办公室 |
| corridor | 12-300 | LOW | NOISY | 走道 |
| staircase | 18-30 | NONE | MODERATE | 楼梯间 |
| toilet | 25-55 | NONE | MODERATE | 卫生间(含男女) |
| storage | 6-15 | NONE | QUIET | 储藏室 |
| entrance_hall | 40-80 | MEDIUM | NOISY | 门厅 |
| gymnasium | 400-800 | — | — | 已从教学楼移除 |
| library | 100-200 | — | — | 已从教学楼移除 |
| office | 15-30 | — | — | 已从教学楼移除 |
| cafeteria | 200-400 | — | — | 已从教学楼移除 |

**building_rules.yaml** — 学校规模模板：

| 规模 | 教室 | 专用 | 音乐 | 教办 | 楼梯 | 卫生间 | 储藏 | 门厅 | 走廊 | 占地面积/层 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| small | 12 | 3 | 1 | 3 | 6 | 3 | 3 | 1 | DYNAMIC | 50×16m = 800m² |
| medium | 24 | 4 | 2 | 5 | 8 | 5 | 5 | 1 | DYNAMIC | 62×18m = 1116m² |
| large | 36 | 6 | 3 | 8 | 10 | 7 | 8 | 1 | DYNAMIC | 78×18m = 1404m² |

### 3.2 生成 Pipeline

```
1. 制定方案 (_generate_school_program)
   └─ 查表得到 room_type → count 映射

2. 标准层拆分 (_resolve_typical_floors + _split_program_to_floors)
   ├─ 3层: ground(floor 0) + teaching(floors 1-2)
   ├─ 5层: ground(0) + teaching(1-3) + top(4)
   └─ 走廊数量动态计算 (13% 目标比, 30/60/10 分配)

3. 生成房间节点 (RoomFactory.generate)
   └─ 面积: 中值±10% 随机采样, 后由 _allocate_areas 精调

4. 面积约束分配 (_allocate_areas)
   ├─ 固定房间从 min 起步 → 缩放到预算 × (1 - target_corr_ratio)
   ├─ target_corr_ratio = uniform(0.10, 0.20) 随机采样
   ├─ 走廊吸收剩余 (上限 25%)
   ├─ 余量按比例均匀分配 (保证面积完备性)
   └─ 所有房间钳制到 spec [min, max]

5. 生成环境节点 (EnvNodeFactory)
   └─ south_facing + main_road_access + playground + green_space

6. 分配位置 (_assign_positions)
   └─ 按标准层垂直切片, 区域内随机撒点

7. 应用拓扑规则 (TopologyRuleEngine)
   ├─ 物理连通: 房间↔走廊, 走廊网络(spine/loop/branch),
   │             交叉连接(40%), 冗余连接(30%), 楼梯↔走廊, 垂直链
   ├─ 声学阻断: 噪声对检查 (noise_gap ≥ 2, dist < 15m)
   └─ 视线/采光: HIGH→南向(100%), HIGH→走廊(40%随机)

8. 约束验证 (ConstraintValidator)
   ├─ fire_exits (硬), daylight (硬), acoustic (硬)
   ├─ connectivity (硬), area_completeness (硬)
   ├─ area_bounds (软), circulation_ratio (软)
   └─ 验证失败 → 重试 (最多5次)

9. 特征工程 (FeatureEngineer.build_hetero_data)
   └─ RoomNode → [N,27] tensor, EnvNode → [M,6] tensor, 5种边类型

10. 保存 + 品质指标计算
    └─ .pt 文件 (hetero_data + metadata + quality + validation)
```

### 3.3 特征维度

```
room features [N, 27]:
  [0:13]   RoomType one-hot (13类)
  [13]     area / 800
  [14]     aspect_ratio
  [15]     occupancy / 300
  [16]     daylight_level / 4
  [17]     noise_level / 4
  [18]     noise_tolerance / 4
  [19]     floor_mid / 4
  [20:26]  ZoneType one-hot (6类)
  [26]     fire_exits_min / 4

environment features [M, 6]:
  [0:4]    EnvNodeType one-hot
  [4:6]    position (x/200, y/150)

Edge types (5):
  (room, physical_connects, room)
  (room, acoustic_blocks, room)
  (room, sight_lines, room)
  (room, sight_lines, environment)
  (room, physical_connects, environment)
```

---

## 4. GNN 评分模型 (Phase 2)

### 4.1 架构

```
HeteroData (room [N,27] + env [M,6] + 5种边)
    ↓
Input Projection: Linear(27→128), Linear(6→128)
    ↓
HeteroConv Layer ×3 (SAGEConv, aggr='mean', 10 edge types incl. reverse)
    ├─ LayerNorm + ReLU + Dropout(0.2)
    └─ Residual skip connection
    ↓
Global Mean Pool (room nodes only) → [128]
    ↓
MLP Head: Linear(128→64) → ReLU → Dropout → Linear(64→1) → Sigmoid
    ↓
Quality Score ∈ [0, 1]
```

### 4.2 训练配置

| 参数 | 值 |
|:---|:---|
| Hidden dim | 128 |
| Layers | 3 |
| Dropout | 0.2 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Loss | MSE(pred, target) + λ×constraint_losses |
| Epochs | 60-80 |
| Batch size | 1 (variable graph sizes) |

### 4.3 性能演进

| 版本 | R² | 关键改进 |
|:---|:---|:---|
| v1 (pass/fail) | 0.004 | 初始版本，标签无方差 |
| v6 (quality metrics) | 0.983 | 首次品质指标，但 circulation=0 始终 |
| v7 (teaching only) | 0.117 | 移除体育馆/食堂/图书馆/办公室 |
| v11 (topology diversity) | 0.632 | 走廊 spine/loop/branch + path_redundancy + zone_cohesion |
| **v12 (current)** | **0.738** | 楼梯数修正 + 走廊面积上限 25% + 房间面积钳制 |

---

## 5. 品质指标体系

### 5.1 当前指标 (6个评分 + 1个仅计算)

| # | 指标 | 公式 | 权重 | 范围 | v12 均值 |
|:---|:---|:---|:---|:---|:---|
| 1 | daylight_quality | 连续分(0/0.5/0.8/1.0) × sight_degree | 1.0 | [0,1] | 0.680 |
| 2 | circulation_efficiency | 1 − \|ratio−0.18\|/0.12 | 1.0 | [0,1] | 0.039 |
| 3 | fire_safety_margin | mean(ReLU(degree−min_req+1)/3) | 1.0 | [0,1] | 0.760 |
| 4 | graph_robustness | min(λ₂(L_norm)×5, 1.0) | 1.0 | [0,1] | 0.176 |
| 5 | path_redundancy | min(branch_ratio×2, 1.0) | 1.0 | [0,1] | 0.951 |
| 6 | zone_cohesion | 1 − \|intra_zone_edge_ratio−0.4\|/0.4 | 1.0 | [0,1] | 0.346 |
| 7 | space_type_diversity | H(room_types)/ln(13) | 0.5 | [0,1] | 0.727 |
| 8 | vertical_flow_balance | 1 − CV(stair_count_per_floor) | 0.5 | [0,1] | 0.362 |
| — | acoustic_comfort | BFS noisy→quiet/rooms (仅计算，不计分) | — | — | — |

**总权重**: 6.0 (4个核心×1.0 + 2个辅助×0.5 + 1个仅计算×0)

### 5.2 参考文献

- GB50099-2011 — 中小学校设计规范 (采光、声学、走廊比)
- GB50016-2014 — 建筑设计防火规范 (消防疏散、连通性)
- SubgraphX (Yuan et al., ICML 2021) — MCTS 子图解释
- WL Kernel (Shervashidze et al., JMLR 2011) — 图相似度

---

## 6. 可解释管线 (Phase 3)

### 6.1 流程

```
高分图集 (test split, top 20%)
    ↓
SubgraphX MCTS (SubgraphRunner)
  ├─ 对每张图: 搜索最小子图 (n_simulations=80)
  ├─ 状态: 房间子集 + 边子集
  ├─ 动作: 删除房间节点 / 删除边
  └─ 奖励: GNN 分数变化 + 稀疏性奖励
    ↓
最优子图集合 (10-30 个)
    ↓
WL Kernel 相似度矩阵 (h=3 迭代)
    ↓
谱聚类 (SubgraphClusterer, k=3-5)
    ↓
模体提取 (Motif)
  ├─ 房间组成统计 (平均计数)
  ├─ 边组成统计
  ├─ 质心图 (簇代表)
  ├─ 中英文描述
  └─ 关联约束引用
    ↓
《建筑空间模式语言词典》(JSON + TXT)
```

### 6.2 当前输出

- `outputs/explainer/motif_dictionary.json` — 结构化模体数据
- `outputs/explainer/motif_dictionary.txt` — 人类可读词典
- `outputs/explainer/mined_patterns.json` — 挖掘模式

---

## 7. 交互界面 (Phase 4 & 5)

### 7.1 Streamlit 应用 (`src/ui/app.py`)

4个功能标签页:

| 标签 | 功能 | 输入 | 输出 |
|:---|:---|:---|:---|
| Dataset | 数据集生成+品质分析 | graph数/规模/层数 | 直方图、指标统计、房型分布 |
| Explain | 子图提取+模体发现 | 图数/MCTS模拟数 | 子图可视化、模体列表 |
| Floor Plans | 平面图浏览 | 数据集路径/图索引 | 逐层平面图 |
| Pattern Lab | 模式组合+GNN评分 | 模式库选择 | 画布+实时品质分 |

启动: `streamlit run src/ui/app.py --server.port 8501`

### 7.2 平面图生成 (`draw_plans.py`)

精确面积布局:
- 走廊: 一条连续脊柱 (不分成多段)
- 房间: `w = area / row_h`, 缩放填满全宽
- 教学层: 面积按物理子层拆分 `area / n_phys`
- 面积精度: 3-8% (墙体厚度)

---

## 8. 关键约束与设计决策

### 8.1 面积完备性约束 (Area Completeness)

```
Σ(room.area) ≈ per_floor_area × num_floors  (容差 ±5%)
按标准层验证 + 全局验证
```

### 8.2 纯教学楼设计

移除的房间类型: gymnasium, cafeteria, library, office
保留的房间类型: 9种 (classroom, special_classroom, music_room, teacher_office, corridor, staircase, toilet, storage, entrance_hall)

### 8.3 走廊拓扑多样性

| 类型 | 概率 | 结构 |
|:---|:---|:---|
| spine | 50% | 线性链 c₀−c₁−c₂−... |
| loop | 30% | 环 spine + tail→head |
| branch | 20% | Y型从中心枢纽辐射 |

附加: 40% 概率横向交叉连接, 30% 概率房间冗余连接

### 8.4 声学指标判定

纯教学楼中声学冲突有限 (音乐教室是唯一大噪声源)，`acoustic_comfort` 指标计算但不计入品质总分。

---

## 9. 运行命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试 (19个)
python tests/test_constraints.py
python tests/test_generator.py
# ... 等

# 生成数据集
python -c "from data.dataset import SchoolDataset; ..."

# 训练 GNN
python -c "from training.trainer import Trainer; ..."

# 生成平面图
KMP_DUPLICATE_LIB_OK=TRUE python draw_plans.py

# 运行可解释管线
KMP_DUPLICATE_LIB_OK=TRUE python run_explainer.py

# 启动交互界面
streamlit run src/ui/app.py --server.port 8501

# Git 推送 (需要代理)
git add . && git commit -m "message" && git push origin master
```

---

## 10. 版本历史

| 版本 | 日期 | 关键变更 |
|:---|:---|:---|
| 0.1.0 | 2026-06-17 | 初始提交: Phase 1-5 全代码架构 |
| 0.1.1 | 2026-06-20 | 面积完备性约束 + 品质指标模块 |
| 0.1.2 | 2026-06-20 | 移除非教学楼房间 + 建筑占地面积修正 |
| 0.1.3 | 2026-06-20 | CORRIDOR_IDX 修复 (8→7) + GNN 首次有效训练 |
| 0.1.4 | 2026-06-21 | 平面图精确面积 (area = w×h) + 卫生间合并 |
| 0.1.5 | 2026-06-21 | 品质指标修复 (daylight + circulation) + 走廊拓扑多样性 |
| 0.1.6 | 2026-06-22 | 精细化指标 (path_redundancy + zone_cohesion) |
| 0.2.0 | 2026-06-22 | 楼梯数修正 + 走廊脊柱可视化 + 走廊面积上限 + 交互界面重构 + GNN R²=0.74 |

---

## 11. 待办事项

- [ ] 增加 MCTS 模拟次数 (需 GPU) 以提取更精细的局部模体
- [ ] 完善模体词典，区分 spine/loop/branch 模式
- [ ] circulation_efficiency 方差偏低 (mean=0.04)，考虑调整走廊比随机范围
- [ ] 添加 GNN 训练 GPU 支持
- [ ] 补充 test_dataset.py 单元测试
- [ ] 将平面图生成集成到 Streamlit 界面
