```markdown
# next_plan.md

# Graph Pattern Learning 下一步研究与开发计划

> 方向聚焦：**基于子图解释的建筑空间模式发现方法**  
> 目标定位：从“规则生成 + GNN 评分 + 可视化原型”升级为“可解释图学习驱动的建筑空间模式发现与设计辅助框架”

**项目名称**: Graph Pattern Learning  
**当前版本**: v0.2.0  
**计划版本**: v0.3.0 → v0.5.0  
**日期**: 2026-06-22  
**核心创新点**: 基于子图解释的空间模式发现方法  
**目标期刊对标**: Advanced Engineering Informatics, Automation in Construction

---

## 1. 下一阶段总体定位

当前项目已经完成了以下基础能力：

1. 中小学教学楼空间图数据生成；
2. 多模态异构图建模；
3. GNN 质量评分模型；
4. SubgraphX 子图解释；
5. WL Kernel 子图相似度计算；
6. 谱聚类模体提取；
7. 初步模式词典生成；
8. Streamlit 交互式设计副驾驶原型。

但当前系统仍存在明显不足：

- 平面几何生成效果较弱；
- 合成数据和真实建筑案例之间缺少校准；
- GNN 主要拟合手写质量指标，建筑学意义不足；
- SubgraphX 提取结果尚未充分转译为建筑空间模式；
- 模体缺少正负对比、反事实验证和专家评价；
- “空间模式发现方法”的学术贡献尚未被突出。

因此，下一阶段的核心目标不是继续单纯提升 GNN R²，而是围绕以下研究主线展开：

```text
建筑空间异构图
    ↓
GNN 质量学习
    ↓
子图解释
    ↓
高影响空间子图提取
    ↓
子图相似度聚类
    ↓
空间模体归纳
    ↓
建筑模式语言词典
    ↓
设计辅助与反事实验证
```

最终形成一个具有明确学术创新性的框架：

> **Explainable Subgraph-based Spatial Pattern Discovery Framework for Educational Building Layouts**

---

## 2. 核心研究问题

下一阶段围绕以下研究问题展开。

### RQ1：建筑空间能否被建模为可解释的异构图结构？

将房间、环境、楼层、走廊、楼梯、采光、声学、疏散等要素编码为异构图，研究其是否能够表达教学楼空间组织的关键语义。

### RQ2：GNN 学到的高质量建筑空间结构能否通过子图解释方法被提取出来？

使用 SubgraphX、GNNExplainer 或 PGExplainer 等方法，从高分图中识别对建筑质量评分贡献最大的空间子图。

### RQ3：子图解释结果能否被聚类为稳定、可复用的空间模体？

通过 WL Kernel、Graph Edit Distance、谱聚类等方法，将相似解释子图归纳为若干类空间模式。

### RQ4：这些空间模体是否具有建筑学意义？

通过真实案例验证、专家评价、指标统计和反事实实验，判断模体是否对应人类可理解的建筑空间组织原则。

### RQ5：空间模体能否反向辅助方案生成与优化？

将发现的空间模体作为设计建议、约束模板或图编辑操作，用于提高新生成方案的质量。

---

## 3. 总体技术路线

下一阶段建议采用以下技术路线。

```text
Phase A: 数据与图表示修正
    ↓
Phase B: GNN 质量模型稳定化
    ↓
Phase C: 子图解释增强
    ↓
Phase D: 模体聚类与语义归纳
    ↓
Phase E: 模式语言词典构建
    ↓
Phase F: 反事实验证与专家评价
    ↓
Phase G: 模体引导的设计辅助原型
```

---

## 4. 阶段 A：数据与图表示修正

### 4.1 目标

保证子图解释的输入数据是可信的。当前系统中存在走廊面积异常、楼层面积分配不稳定、几何平面与图拓扑弱耦合等问题。这些问题会直接影响 GNN 学习和子图解释结果。

### 4.2 核心任务

#### A1. 楼层级面积审计

新增每层面积审计输出。

每张图应记录：

```json
{
  "graph_id": "sample_0001",
  "scale": "medium",
  "num_floors": 5,
  "gross_area_total": 5580,
  "room_area_total": 4520,
  "corridor_area_total": 860,
  "corridor_ratio_global": 0.154,
  "floors": [
    {
      "floor": 0,
      "floor_area_budget": 1116,
      "room_area_sum": 940,
      "corridor_area_sum": 176,
      "corridor_ratio": 0.158,
      "num_rooms": 14,
      "num_classrooms": 4,
      "num_stairs": 2,
      "num_toilets": 1
    }
  ]
}
```

新增约束：

```text
0.10 <= floor_corridor_ratio <= 0.25
0.10 <= global_corridor_ratio <= 0.25
```

#### A2. 修正走廊面积异常

当前示例中出现 Corridor 687㎡ 的异常情况。下一步需要定位以下问题：

- 走廊是否被用于吸收面积剩余；
- 标准层面积和物理楼层面积是否错配；
- 绘图时是否将走廊强制拉伸为整层宽度；
- corridor node area 与 geometry area 是否不一致；
- ground floor program 是否过少导致大面积空余被走廊吞并。

修正原则：

```text
走廊面积 = 走廊宽度 × 走廊中心线长度
```

而不是：

```text
走廊面积 = 楼层剩余面积
```

#### A3. 房间面积与房间类型严格一致

严格执行 `room_catalog.yaml` 的面积上下限。

例如：

```yaml
teacher_office:
  min_area: 30
  max_area: 60
```

如果出现多个教师办公室聚合，应明确表达为：

```text
教师办公组：3 rooms, total 117㎡
```

而不是单个：

```text
教师办公：117㎡
```

#### A4. 增加楼层功能模板

将随机楼层拆分升级为基于建筑类型的楼层模板。

示例：

```yaml
floor_templates:
  ground:
    required:
      entrance_hall: 1
      staircase: 2
      toilet: 1
    preferred:
      classroom: 4
      teacher_office: 1

  teaching:
    required:
      staircase: 2
      toilet: 1
    preferred:
      classroom: 6
      teacher_office: 1

  top:
    required:
      staircase: 2
      toilet: 1
    preferred:
      special_classroom: 2
      music_room: 1
      storage: 1
```

#### A5. 补充图节点与边类型

当前图结构中建议增加以下语义：

新增节点类型：

```text
floor
service_core
corridor_segment
entrance
exterior_boundary
```

新增边类型：

```text
(room, located_on, floor)
(room, adjacent_to, room)
(room, served_by, service_core)
(room, has_daylight_from, exterior_boundary)
(corridor_segment, connects, corridor_segment)
(staircase, vertical_connects, staircase)
```

这样子图解释结果能够更清晰地表达建筑语义。

### 4.3 阶段输出

- `src/data/floor_audit.py`
- `outputs/audit/floor_area_report.json`
- 修正后的数据集 `outputs/dataset_200_v13/`
- 更新后的图 schema 文档 `docs/graph_schema.md`

---

## 5. 阶段 B：GNN 质量模型稳定化

### 5.1 目标

当前 GNN 主要拟合手写质量分。下一步应让 GNN 评分更适合支持子图解释，而不是只追求 R²。

### 5.2 关键改进

#### B1. 质量标签拆分为多任务输出

当前模型输出单一质量分：

```text
Quality Score ∈ [0, 1]
```

建议改为多任务输出：

```text
daylight_quality
circulation_efficiency
fire_safety_margin
graph_robustness
path_redundancy
zone_cohesion
space_type_diversity
vertical_flow_balance
overall_quality
```

模型结构：

```text
HeteroGNN Encoder
    ↓
Shared Graph Embedding
    ↓
Multi-head MLP
    ├── daylight head
    ├── circulation head
    ├── fire safety head
    ├── robustness head
    ├── zone cohesion head
    └── overall quality head
```

优势：

- 可以解释“哪个子图影响哪个质量维度”；
- 模体词典可以记录模式的指标贡献；
- 有助于发现不同类型的空间模式。

#### B2. 标签分布均衡

当前 circulation_efficiency 均值过低，说明标签分布可能不合理。需要：

- 查看每个指标的均值、方差、分布；
- 对低方差指标进行重标定；
- 避免 GNN 学到退化目标；
- 对高分、中分、低分样本进行均衡采样。

#### B3. 引入 pairwise ranking loss

除了 MSE，加入排序损失：

```text
如果 Q_i > Q_j，则模型应满足 pred_i > pred_j
```

损失形式：

```text
L = L_MSE + λ L_rank + μ L_constraint
```

其中：

```text
L_rank = max(0, margin - (pred_i - pred_j))
```

意义：

- 更符合设计方案优劣比较；
- 有助于解释高质量结构；
- 降低绝对标签误差对模型的影响。

#### B4. 模型校准

增加以下评估：

```text
R²
MAE
RMSE
Spearman correlation
Kendall tau
top-k precision
high-quality retrieval accuracy
```

对于模式发现任务，尤其应关注：

```text
top 20% 高质量图识别准确率
```

### 5.3 阶段输出

- `src/models/multitask_scorer.py`
- `src/training/ranking_loss.py`
- `outputs/model_checkpoint_v13_multitask.pt`
- `outputs/evaluation/gnn_quality_report.json`

---

## 6. 阶段 C：子图解释方法增强

### 6.1 目标

将当前 SubgraphX 从“可运行”升级为“可信、稳定、可验证”的空间子图解释模块。

### 6.2 子图解释对象

解释不应只针对 overall quality，也应针对不同质量维度。

解释目标包括：

```text
overall_quality
daylight_quality
circulation_efficiency
fire_safety_margin
path_redundancy
zone_cohesion
vertical_flow_balance
```

例如：

```text
解释 daylight_quality → 提取南向教室 + 外墙采光边模体
解释 fire_safety_margin → 提取楼梯—走廊—教室组团模体
解释 zone_cohesion → 提取同类教学空间聚集模体
```

### 6.3 正负子图解释

当前主要提取高分图中的重要子图。下一步需要同时提取：

```text
Positive motifs: 高分图中提升质量的子图
Negative motifs: 低分图中降低质量的子图
Neutral motifs: 对质量影响不显著的常见子图
```

实现方式：

```text
高分样本 top 20% → positive explanations
低分样本 bottom 20% → negative explanations
中间样本 middle 20% → neutral references
```

### 6.4 反事实解释

对每个解释子图进行反事实验证。

#### 删除式反事实

```text
ΔQ_remove = Q(G) - Q(G \ S)
```

如果删除子图后分数显著下降，则说明该子图是正模式。

#### 插入式反事实

```text
ΔQ_insert = Q(G + S) - Q(G)
```

如果将子图插入低分图后分数提高，则说明该子图具有可迁移设计价值。

#### 替换式反事实

```text
ΔQ_replace = Q(G with S_a replaced by S_b) - Q(G)
```

用于比较不同空间组织方式。

### 6.5 子图稳定性评估

同一个图在不同随机种子、不同 MCTS 模拟次数下，解释结果应保持稳定。

评估指标：

```text
node_overlap
edge_overlap
Jaccard similarity
explanation_score_variance
motif_cluster_consistency
```

实验设置：

```text
n_simulations = [40, 80, 120, 200]
random_seed = [0, 1, 2, 3, 4]
```

### 6.6 与其他解释方法对比

建议加入以下 baseline：

| 方法 | 用途 |
|---|---|
| SubgraphX | 主方法 |
| GNNExplainer | 对比解释方法 |
| PGExplainer | 参数化解释器 |
| Integrated Gradients | 特征重要性对比 |
| Random Subgraph | 随机基线 |
| Degree-based Subgraph | 高度数节点基线 |

比较维度：

```text
fidelity
sparsity
stability
counterfactual impact
expert interpretability
```

### 6.7 阶段输出

- `src/explainer/counterfactual.py`
- `src/explainer/stability.py`
- `src/explainer/explainer_baselines.py`
- `outputs/explainer/positive_subgraphs.json`
- `outputs/explainer/negative_subgraphs.json`
- `outputs/explainer/counterfactual_report.json`
- `outputs/explainer/stability_report.json`

---

## 7. 阶段 D：子图聚类与空间模体归纳

### 7.1 目标

将大量解释子图转化为稳定、可命名、可复用的建筑空间模体。

### 7.2 子图相似度方法

当前使用 WL Kernel。下一步建议扩展为多种相似度组合：

```text
S_total = α S_WL + β S_node_type + γ S_edge_type + δ S_metric_effect + η S_geometry
```

其中：

- `S_WL`: Weisfeiler-Lehman 子树核相似度；
- `S_node_type`: 房间类型组成相似度；
- `S_edge_type`: 边类型组成相似度；
- `S_metric_effect`: 对质量指标贡献的相似度；
- `S_geometry`: 几何实现方式相似度。

### 7.3 聚类方法

建议比较：

| 方法 | 特点 |
|---|---|
| Spectral Clustering | 当前方法，适合相似度矩阵 |
| Agglomerative Clustering | 层次结构清晰 |
| HDBSCAN | 可自动发现簇数量和噪声点 |
| K-Medoids | 可找到真实代表样本 |

输出每个簇的：

```text
cluster_id
size
representative_subgraph
average_node_composition
average_edge_composition
average_quality_effect
stability_score
expert_label
```

### 7.4 模体原型提取

每个模体应包含三个层次：

#### 图结构原型

```text
room types
edge types
degree pattern
central nodes
boundary nodes
```

#### 建筑语义原型

```text
空间组成
功能关系
交通组织
采光关系
声学关系
消防关系
```

#### 几何实现原型

```text
常见平面形态
走廊类型
房间排列方式
楼梯位置
服务核位置
```

### 7.5 模体命名规则

每个模体需要由机器统计和人工语义共同命名。

示例：

```text
M01: 双端楼梯—线性教室组团
M02: 楼梯—卫生间—储藏服务核
M03: 南向教室—单侧走廊采光模体
M04: 音乐教室隔离模体
M05: 环形走廊冗余疏散模体
M06: 教师办公—普通教室邻近模体
```

### 7.6 阶段输出

- `src/explainer/motif_clusterer_v2.py`
- `src/explainer/motif_prototype.py`
- `outputs/explainer/motif_clusters.json`
- `outputs/explainer/motif_prototypes.json`
- `outputs/explainer/motif_cluster_visualization/`

---

## 8. 阶段 E：建筑空间模式语言词典

### 8.1 目标

将子图解释结果转译为人类可读、可复用、可验证的建筑空间模式语言。

### 8.2 模式词典结构

每个模式建议采用以下结构：

```json
{
  "motif_id": "M01",
  "name_cn": "双端楼梯—线性教室组团",
  "name_en": "Dual-Stair Linear Classroom Cluster",
  "pattern_type": "positive",
  "applicable_building": "primary/secondary school teaching building",
  "graph_signature": {
    "nodes": {
      "classroom": 6,
      "corridor": 1,
      "staircase": 2,
      "toilet": 1
    },
    "edges": {
      "physical_connects": 9,
      "sight_lines": 6,
      "vertical_connects": 2
    }
  },
  "spatial_description_cn": "普通教室沿线性走廊成组布置，两端设置楼梯，卫生间靠近楼梯服务核。",
  "spatial_description_en": "Classrooms are organized along a linear corridor with staircases placed at both ends and toilets located near the service cores.",
  "quality_effect": {
    "overall_quality": 0.12,
    "fire_safety_margin": 0.18,
    "circulation_efficiency": 0.09,
    "zone_cohesion": 0.14
  },
  "counterfactual_evidence": {
    "remove_delta_q": -0.11,
    "insert_delta_q": 0.08
  },
  "design_guidelines": [
    "适用于普通教学层。",
    "楼梯宜布置于走廊两端。",
    "卫生间宜靠近楼梯或服务核。",
    "普通教室宜沿主要采光面连续布置。"
  ],
  "code_references": [
    "GB50099-2011",
    "GB50016-2014"
  ],
  "representative_graph_ids": [
    "graph_0031",
    "graph_0074",
    "graph_0128"
  ],
  "confidence": {
    "frequency": 0.34,
    "stability": 0.82,
    "expert_score": 4.3
  }
}
```

### 8.3 模式分类

建议将模式分为：

```text
交通组织模式
采光组织模式
消防疏散模式
服务核模式
教学组团模式
声学隔离模式
竖向组织模式
负面反模式
```

### 8.4 正模式与反模式

#### 正模式示例

```text
双端楼梯—线性教室组团
南向教室—单侧走廊模式
服务核邻近模式
环形走廊冗余模式
```

#### 反模式示例

```text
超长单向死走廊
音乐教室紧邻普通教室
楼梯集中于一端
卫生间远离教学组团
高采光需求房间无外墙接触
```

### 8.5 阶段输出

- `outputs/explainer/motif_dictionary_v2.json`
- `outputs/explainer/motif_dictionary_v2.md`
- `outputs/explainer/anti_pattern_dictionary.md`
- `docs/pattern_language_spec.md`

---

## 9. 阶段 F：验证体系设计

### 9.1 目标

证明发现的模体不是模型噪声，而是具有建筑学意义、统计稳定性和设计可迁移性的空间模式。

### 9.2 验证维度

#### F1. 模型忠实度 Fidelity

衡量解释子图是否真正影响 GNN 输出。

```text
Fidelity_remove = Q(G) - Q(G \ S)
Fidelity_keep = Q(S) / Q(G)
```

#### F2. 稀疏性 Sparsity

解释子图应尽量小而有效。

```text
Sparsity = 1 - |S| / |G|
```

#### F3. 稳定性 Stability

同一图在不同随机种子下解释结果应相似。

```text
Stability = mean Jaccard(S_i, S_j)
```

#### F4. 频率 Frequency

模体应在高质量图中反复出现。

```text
Frequency = count(motif in high-quality graphs) / count(high-quality graphs)
```

#### F5. 区分度 Discriminativeness

正模体应更多出现在高分图中，反模式应更多出现在低分图中。

```text
Discriminativeness = P(motif | high-quality) - P(motif | low-quality)
```

#### F6. 反事实影响 Counterfactual Impact

删除正模体应降低分数，插入正模体应提高分数。

```text
Impact_remove = mean(Q(G) - Q(G without motif))
Impact_insert = mean(Q(G with motif) - Q(G))
```

#### F7. 专家可解释性 Expert Interpretability

建筑专家对模式进行盲评。

评价维度：

```text
功能合理性
交通组织合理性
采光合理性
消防疏散合理性
可复用性
表达清晰度
```

评分采用 1–5 Likert scale。

### 9.3 真实案例验证

建立小规模真实教学楼案例库。

建议数量：

```text
30–100 个真实教学楼平面
```

每个案例提取：

```text
room graph
corridor graph
stair locations
room types
floor geometry
quality metrics
```

验证发现的 motif 是否也存在于真实案例中。

输出：

```text
motif occurrence in synthetic high-quality set
motif occurrence in real building set
motif occurrence in synthetic low-quality set
```

### 9.4 Baseline 对比

建议设置以下 baseline：

| Baseline | 说明 |
|---|---|
| Random Subgraph | 随机选择同规模子图 |
| Degree-based Subgraph | 选择高度数节点组成子图 |
| Rule-based Pattern | 人工规则定义模式 |
| GNNExplainer | 通用 GNN 解释方法 |
| PGExplainer | 参数化解释器 |
| SubgraphX | 主方法 |
| SubgraphX + Counterfactual | 增强方法 |

比较指标：

```text
fidelity
sparsity
stability
expert score
counterfactual impact
motif discriminativeness
```

### 9.5 阶段输出

- `outputs/evaluation/explanation_fidelity.json`
- `outputs/evaluation/motif_stability.json`
- `outputs/evaluation/motif_real_case_validation.json`
- `outputs/evaluation/expert_review_results.csv`
- `outputs/evaluation/baseline_comparison.md`

---

## 10. 阶段 G：模体引导的设计辅助

### 10.1 目标

将发现的空间模式从“解释结果”转化为“设计辅助工具”。

### 10.2 模体作为设计建议

在 Streamlit Pattern Lab 中显示：

```text
当前方案缺少：
- 双端楼梯疏散模式
- 卫生间服务核邻近模式
- 南向教室组团模式

建议添加：
- 在走廊东端增加楼梯
- 将卫生间移动至楼梯附近
- 将普通教室移动至南向采光边
```

### 10.3 模体作为图编辑操作

每个 motif 定义一组 graph edit operations：

```json
{
  "motif_id": "M01",
  "operations": [
    {
      "type": "add_node",
      "node_type": "staircase"
    },
    {
      "type": "add_edge",
      "edge_type": "physical_connects",
      "from": "staircase",
      "to": "corridor"
    },
    {
      "type": "move_node",
      "node_type": "toilet",
      "target": "near_staircase"
    }
  ]
}
```

### 10.4 模体引导优化

生成候选方案时，可以加入 motif reward：

```text
Q_total = Q_GNN + λ Q_motif + μ Q_code + η Q_geometry
```

其中：

```text
Q_motif = 正模体匹配得分 - 反模式匹配惩罚
```

### 10.5 阶段输出

- `src/ui/motif_advisor.py`
- `src/ui/graph_edit_operations.py`
- `src/ui/pattern_lab_v2.py`
- `outputs/demo/motif_guided_design_examples/`

---

## 11. 几何生成的最低改进要求

虽然下一阶段创新点聚焦于“基于子图解释的空间模式发现”，但几何结果不能过于粗糙，否则会削弱论文可信度。

### 11.1 最低目标

不追求完全自动生成优秀建筑方案，但至少避免明显错误：

- 走廊面积异常；
- 房间超出面积上限；
- 房间像色块随意堆叠；
- 无外窗、无门、无楼梯表达；
- 平面图无法对应图拓扑；
- 教室不能有效采光；
- 楼梯和卫生间位置不合理。

### 11.2 推荐改进

采用 corridor skeleton + room module 的简化生成。

```text
1. 选择平面类型：单廊、双廊、环廊、枝状
2. 固定走廊宽度
3. 沿走廊两侧布置标准房间模块
4. 教室优先放南侧或外墙侧
5. 楼梯布置在端部或服务核
6. 卫生间靠近楼梯
7. 音乐教室远离普通教室
8. 输出门窗和外墙
```

### 11.3 新增几何指标

```text
room_aspect_validity
exterior_wall_access
south_facing_classroom_ratio
corridor_width_validity
mean_travel_distance
max_escape_distance
service_core_accessibility
geometry_graph_consistency
```

---

## 12. 版本规划

## v0.3.0：数据与解释可信化

目标：修正数据问题，建立稳定解释流程。

任务：

- [ ] 修正 corridor area 异常；
- [ ] 增加 floor-level area audit；
- [ ] 增加楼层功能模板；
- [ ] 多任务 GNN scorer；
- [ ] 子图解释支持多质量指标；
- [ ] 正负子图解释；
- [ ] 子图稳定性评估；
- [ ] 初版 counterfactual 删除实验。

输出：

```text
dataset_200_v13
model_checkpoint_v13_multitask.pt
positive_subgraphs.json
negative_subgraphs.json
stability_report.json
counterfactual_report.json
```

---

## v0.4.0：空间模体发现与模式词典

目标：形成完整的空间模式发现方法。

任务：

- [ ] WL Kernel + 多特征相似度；
- [ ] 谱聚类 / 层次聚类对比；
- [ ] 模体原型提取；
- [ ] 正模式和反模式词典；
- [ ] 模体质量贡献统计；
- [ ] 模体可视化；
- [ ] 初步专家命名。

输出：

```text
motif_clusters.json
motif_prototypes.json
motif_dictionary_v2.json
anti_pattern_dictionary.md
pattern_language_spec.md
```

---

## v0.5.0：验证与设计辅助

目标：完成论文实验闭环和交互式应用验证。

任务：

- [ ] 收集并标注真实教学楼案例；
- [ ] 真实案例 motif occurrence 验证；
- [ ] baseline explainer 对比；
- [ ] 专家盲评；
- [ ] motif-guided graph editing；
- [ ] Pattern Lab v2；
- [ ] 模体引导生成案例展示。

输出：

```text
real_case_graph_dataset
baseline_comparison.md
expert_review_results.csv
motif_real_case_validation.json
pattern_lab_v2_demo
```

---

## 13. 论文贡献表达建议

下一步论文应避免只强调“生成平面图”，而应突出以下贡献。

### Contribution 1：建筑空间异构图表达

提出一种面向教学楼的 code-aware heterogeneous spatial graph，用于统一表示：

```text
房间类型
功能分区
交通连接
声学隔离
采光关系
环境节点
消防需求
楼层组织
```

### Contribution 2：基于 GNN 的空间质量学习

构建多任务 HeteroGNN，用于学习不同空间质量维度，并支持高质量空间图检索。

### Contribution 3：基于子图解释的空间模体发现

提出一套从 GNN 解释到建筑模式语言的转换流程：

```text
Subgraph explanation
→ Subgraph similarity
→ Motif clustering
→ Prototype extraction
→ Semantic labeling
→ Pattern dictionary
```

### Contribution 4：反事实验证的模式可信度评估

通过删除、插入、替换等反事实实验，验证空间模体对质量评分的贡献。

### Contribution 5：人机协同设计辅助原型

将发现的空间模式转化为设计建议和图编辑操作，支持交互式方案优化。

---

## 14. 面向 AEI / AIC 的策略

### 14.1 AEI 投稿策略

AEI 更适合当前方向。建议主打：

```text
Explainable AI
Graph learning
Engineering knowledge discovery
Design decision support
Pattern language induction
```

推荐论文题目方向：

```text
Subgraph Explanation-based Spatial Pattern Discovery in Educational Building Layouts Using Heterogeneous Graph Neural Networks
```

AEI 需要强化：

- 图表示方法；
- 子图解释方法；
- 模式发现框架；
- 可解释性验证；
- 专家评价；
- 设计知识转译。

### 14.2 AIC 投稿策略

AIC 更关注自动化设计和工程应用。若投 AIC，需要进一步强化：

- 几何平面生成；
- 规范合规；
- CAD/BIM 输出；
- 真实工程案例；
- 与优化算法或生成模型对比；
- 设计效率提升。

当前阶段不建议直接冲 AIC，除非完成拓扑—几何联合生成和真实案例验证。

---

## 15. 近期优先级清单

### P0：必须立即修复

- [ ] Corridor area 异常；
- [ ] Teacher office 面积超限；
- [ ] 楼层面积审计；
- [ ] 走廊比楼层级约束；
- [ ] 几何面积和节点面积一致性；
- [ ] 每层 program 合理性。

### P1：支撑核心创新

- [ ] 多任务 GNN；
- [ ] 正负子图解释；
- [ ] 反事实删除实验；
- [ ] 子图稳定性评估；
- [ ] motif clustering v2；
- [ ] pattern dictionary v2。

### P2：支撑论文验证

- [ ] 真实案例库；
- [ ] baseline explainer；
- [ ] 专家评价；
- [ ] 模体频率和区分度统计；
- [ ] 反事实插入实验。

### P3：提升展示效果

- [ ] corridor skeleton layout；
- [ ] room module layout；
- [ ] 门窗表达；
- [ ] service core visualization；
- [ ] Pattern Lab v2。

---

## 16. 最终目标成果

下一阶段完成后，项目应形成以下成果。

### 16.1 方法成果

```text
A subgraph explanation-based spatial pattern discovery framework
```

### 16.2 数据成果

```text
Synthetic school building graph dataset
Real-case educational building graph benchmark
```

### 16.3 模型成果

```text
Multitask Heterogeneous GNN Quality Scorer
Subgraph-based Spatial Motif Explainer
Counterfactual Motif Validator
```

### 16.4 知识成果

```text
Architectural Spatial Pattern Language Dictionary
Positive Motif Library
Negative Anti-pattern Library
```

### 16.5 系统成果

```text
Interactive Pattern Lab
Motif-guided Design Advisor
Floor Plan Visualization Tool
```

---

## 17. 总结

下一阶段的核心不是继续追求更高的 GNN R²，也不是单纯增加 MCTS 模拟次数，而是建立一个可信的研究闭环：

```text
空间图建模
→ 质量学习
→ 子图解释
→ 模体聚类
→ 模式语言
→ 反事实验证
→ 设计辅助
```

项目的最佳学术定位应为：

> **基于可解释异构图学习的建筑空间模式发现方法。**

相较于直接宣称“自动生成高质量建筑平面”，该定位更加稳健，也更符合当前项目已有基础。  
若能够补充真实案例验证、专家评价、反事实实验和模体引导设计，该方向具备冲击 AEI 的潜力；若进一步完成规范合规的拓扑—几何联合生成和 CAD/BIM 输出，则可进一步面向 AIC。
```