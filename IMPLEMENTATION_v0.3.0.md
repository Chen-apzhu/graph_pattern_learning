# v0.3.0 代码实施计划

> 基于 `PLAN.md` §4-6 (Phase A + B + C 启动)

---

## 映射关系

| PLAN.md 阶段 | 本计划 | 优先级 |
|:---|:---|:---|
| A1-A3 面积修正 | Step 1-3 | **P0** |
| A4 楼层模板 | Step 4 | **P0** |
| B1 多任务 GNN | Step 5 | P1 |
| B2 标签均衡 | Step 6 | P1 |
| B3 ranking loss | Step 7 | P2 |
| C1 多维度解释 | Step 8 | P1 |
| C3 反事实 | Step 9 | P2 |

---

## Step 1: 新建 `src/data/floor_audit.py`

逐层面积审计器，输出符合 PLAN.md A1 格式的 JSON 报告。

```python
class FloorAuditor:
    def audit_graph(hetero_data, metadata) -> dict
    def audit_dataset(dataset_dir) -> list
```

输出路径: `outputs/audit/floor_area_report.json`

## Step 2: 走廊面积修正 (`generator.py`)

- `_allocate_areas`: 走廊预算吸收改为「走廊宽 × 长」逻辑，不再充填全部剩余
- 走廊面积上限收紧: 22%（全局）+ 25%（逐层）
- 余量均匀分给固定房间，允许轻微超出 spec max

## Step 3: `constraints.py` 新增逐层走廊比约束

- `check_floor_corridor_ratio`: `0.10 <= ratio <= 0.25`（硬约束）
- `area_bounds` 升级为硬约束
- 注册到 `validate_all()` 和 `hard_constraints_passed()`

## Step 4: `building_rules.yaml` + `generator.py` 楼层模板

- YAML 新增 `floor_templates` 段 (ground/teaching/top 的 required + preferred)
- `_split_program_to_floors` 重写: 先满足 required，再按比例分配 preferred

## Step 5: 新建 `src/models/multitask_scorer.py`

```
HeteroGNNEncoder (共享) → Shared Embedding [128]
  ├─ daylight_head:      Linear(128→64→Sigmoid)
  ├─ circulation_head:   Linear(128→64→Sigmoid)
  ├─ fire_safety_head:   Linear(128→64→Sigmoid)
  ├─ robustness_head:    Linear(128→64→Sigmoid)
  ├─ zone_cohesion_head: Linear(128→64→Sigmoid)
  ├─ redundancy_head:    Linear(128→64→Sigmoid)
  └─ overall_head:       Linear(128→64→Sigmoid)
```

同步修改: `data_loader.py` (多目标标签), `trainer.py` (多任务 loss)

## Step 6: `quality_metrics.py` 标签均衡

- `circulation_efficiency`: `/0.15` 代替 `/0.12`
- `daylight_quality`: 增加随机微扰提升方差
- 新增 `compute_label_distribution()` 工具

## Step 7: 新建 `src/models/ranking_loss.py`

```python
def pairwise_ranking_loss(preds, targets, margin=0.05) -> Tensor
```

`trainer.py` 损失更新为: `L = L_MSE + 0.1*L_rank + 0.05*L_constraint`

## Step 8: `explainer/mcts_search.py` 多维度解释

- `search()` 新增 `target_metric: str = 'overall'`
- 奖励函数基于指定 metric head 的输出变化
- `subgraph_runner.py`: 批量运行 ×7 指标，输出 `positive_subgraphs.json`

## Step 9: 新建 `src/explainer/counterfactual.py`

```python
class CounterfactualValidator:
    def remove_test(model, graph, subgraph) -> dict  # ΔQ = Q(G) - Q(G\S)
    def insert_test(model, graph, subgraph, target) -> dict
    def batch_test(model, graphs, subgraphs) -> list
```

---

## 实施顺序

```
1 → floor_audit.py
2 → generator.py 走廊修正
3 → constraints.py 逐层约束
4 → building_rules.yaml 楼层模板 + generator.py 适配
  └─ 重新生成 dataset_200_v13 + 运行 floor_audit 验证
5 → multitask_scorer.py + data_loader/trainer 适配
6 → quality_metrics.py 标签均衡
7 → ranking_loss.py + trainer 集成
  └─ 训练 v13 多任务模型
8 → mcts_search.py target_metric + subgraph_runner 批量
9 → counterfactual.py
  └─ 生成 positive/negative subgraphs + counterfactual report
```

---

## v0.3.0 产出清单

| 文件 | 类型 |
|:---|:---|
| `src/data/floor_audit.py` | 新 |
| `src/models/multitask_scorer.py` | 新 |
| `src/models/ranking_loss.py` | 新 |
| `src/explainer/counterfactual.py` | 新 |
| `outputs/audit/floor_area_report.json` | 新 |
| `outputs/dataset_200_v13/` | 更新 |
| `outputs/model_checkpoint_v13_multitask.pt` | 新 |
| `outputs/explainer/positive_subgraphs.json` | 新 |
| `outputs/explainer/negative_subgraphs.json` | 新 |
| `outputs/explainer/counterfactual_report.json` | 新 |
| `src/config/building_rules.yaml` | 修改 |
| `src/data/generator.py` | 修改 |
| `src/data/constraints.py` | 修改 |
| `src/training/data_loader.py` | 修改 |
| `src/training/trainer.py` | 修改 |
| `src/explainer/mcts_search.py` | 修改 |
| `src/explainer/subgraph_runner.py` | 修改 |
