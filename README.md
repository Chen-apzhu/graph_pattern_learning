# Graph Pattern Learning — Phase 1: 数据生成与图表征

基于多模态异构图与子图解释器的可解释性建筑布局生成与模式提取

## 项目结构

```
graph_pattern_learning/
├── src/
│   ├── config/       # YAML 配置文件（房间目录、建筑规范参数）
│   ├── data/         # 数据生成模块（工厂、拓扑规则、约束、特征工程）
│   ├── graph/        # 图表示模块（PyG HeteroData 包装、图分析、统计）
│   └── utils/        # 工具模块（枚举、常量、序列化）
├── tests/            # 单元测试（TDD 风格）
├── outputs/          # 生成的图数据集（.pt 文件）
├── task.md           # 项目任务文档
└── requirements.txt
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行所有测试
python tests/test_enums.py
python tests/test_room_factory.py
python tests/test_topology_rules.py
python tests/test_constraints.py
python tests/test_generator.py
python tests/test_school_graph.py
python tests/test_graph_utils.py
python tests/test_graph_stats.py
```

## 生成合成数据集

```python
from data.generator import SchoolBuildingGenerator
from utils.serialization import save_hetero_data

# 创建生成器
gen = SchoolBuildingGenerator(seed=42)

# 生成中等规模学校（3层，24班/600人）
result = gen.generate(num_floors=3, school_size='medium')

# 查看摘要
print(result.summary())

# 转换为 PyG HeteroData 并保存
hetero_data = gen.to_hetero_data(result)
gen.save(result, output_dir='outputs')
```

## Phase 1 特性

- ✅ 13 种学校房间类型（教室、走廊、楼梯间、音乐教室等）
- ✅ 3 种边类型（物理连通、声学阻断、视线采光）
- ✅ 4 种环境节点（南向日照、主干道接驳、操场、绿化）
- ✅ 硬约束验证（消防疏散、采光合规、声学隔离、连通性）
- ✅ 软约束检查（面积范围、交通核比例）
- ✅ PyG HeteroData 集成
- ✅ 确定性随机种子

## 参考规范

- GB50099-2011 — 中小学校设计规范
- GB50016-2014 — 建筑设计防火规范
