# PiFlow 项目说明文档

## 📖 项目简介

**PiFlow (Principle Flow)** 是一个基于信息论框架的原理感知科学发现系统，采用多智能体协作的方式进行自动化科学研究。该系统将科学发现视为一个结构化的不确定性减少问题，通过基础科学原理的指导，实现更加系统化和理性的科学问题探索。

### 核心特性

- **预算限制的迭代假设检验**：在有限资源下进行高效的科学探索
- **完整的智能体运行日志**：记录完整的研究过程
- **彩色终端输出**：不同智能体使用不同颜色，便于追踪
- **即插即用**：可以快速适配到各种科学发现场景

### 应用领域

PiFlow 已在以下三个科学领域得到验证：
- 🔬 **纳米螺旋材料**：发现具有高手性性质的纳米螺旋几何结构
- 🧬 **生物分子**：ChEMBL化学分子性质预测
- ⚡ **超导体**：超导临界温度(Tc)预测

---

## 📁 项目结构

```
PiFlow/
├── README.md                    # 项目说明
├── PiFlow.pdf                   # 项目论文
├── requirements.txt             # 环境配置文件
├── inference.py                 # 主程序入口
│
├── configs/                     # 配置文件目录
│   ├── demo_config_for_model.yaml   # 模型配置（API、智能体设置）
│   └── demo_config_for_task.yaml    # 任务配置（研究任务定义）
│
├── src/                         # 核心源代码
│   ├── agents/                  # 智能体定义
│   │   ├── experiment.py        # 实验智能体
│   │   ├── hypothesis.py        # 假设智能体
│   │   ├── plan.py              # 规划智能体
│   │   └── user.py              # 用户代理智能体
│   ├── group/                   # 智能体组管理
│   │   ├── manage.py            # 组聊天管理
│   │   ├── selector.py          # 智能体选择器
│   │   └── workflow.py          # 原理流工作流（PrincipleFlow）
│   └── utils/                   # 工具函数
│       ├── config.py            # 配置加载
│       ├── console.py           # 控制台输出
│       └── process.py           # 进程管理
│
├── envs/                        # 预测模型环境
│   ├── start_server.py          # 统一服务器启动脚本
│   ├── unified_server.py        # 统一预测服务器实现
│   ├── check_tools.py           # 工具检查脚本
│   ├── AgenX_Chembl35/          # ChEMBL化学分子预测模型
│   ├── AgenX_Nanohelix/         # 纳米螺旋g-factor预测模型
│   └── AgenX_Supercon/          # 超导体Tc预测模型
│
├── tools/                       # 工具函数接口
│   ├── __init__.py              # 工具注册和管理
│   ├── tool_chembl35.py         # ChEMBL预测工具
│   ├── tool_nanomaterial.py     # 纳米材料预测工具
│   └── tool_predict_Tc.py       # 超导Tc预测工具
```

---

## 📄 主要文件说明

#### `inference.py`
**主程序入口文件**，负责整个系统的初始化和运行。

#### `configs/demo_config_for_task.yaml`
**任务配置文件**，定义具体的科学发现任务。

**示例任务**：发现具有最高g-factor（手性因子）的纳米螺旋结构

#### `configs/demo_config_for_model.yaml`
**模型配置文件**，定义智能体及其使用的语言模型。


### 3. 服务器相关

#### `envs/start_server.py`
**预测服务器启动脚本**，启动统一的预测服务。

#### `envs/unified_server.py`
**统一预测服务器实现**，使用Flask框架。

---

## 🤖 智能体系统

### 智能体架构

PiFlow采用多智能体协作架构，每个智能体负责科学发现流程的不同阶段：

#### 1. **Planner Agent（规划智能体）**
**文件位置**：`src/agents/plan.py`

**工作流程**：
1. 理解PrincipleFlow的建议
2. 明确当前值与目标值的差距
3. 结合物理化学原理进行分析
4. 陈述科学原理
5. 指导Hypothesis Agent下一步行动

#### 2. **Hypothesis Agent（假设智能体）**
**文件位置**：`src/agents/hypothesis.py`

**职责**：
- 基于科学原理提出可测试的假设
- 每次迭代只提出一个明确的假设
- 将假设与物理化学原理关联
- 根据Planner的建议调整假设
- 提出具体的实验候选方案

#### 3. **Experiment Agent（实验智能体）**
**文件位置**：`src/agents/experiment.py`

**工作流程**：
1. 接收假设和实验参数
2. 调用相应的预测工具
3. 获取预测结果
4. 格式化并返回实验数据

#### 4. **User Proxy Agent（用户代理智能体）**
**文件位置**：`src/agents/user.py`

**职责**：
- 代表人类研究者
- 促进人类与智能体系统的交互
- 提供上下文和澄清信息
- 确保研究目标明确传达

---

## 🔧 核心组件

### 1. PrincipleFlow（原理流）
**文件位置**：`src/group/workflow.py`

**功能**：
- PiFlow的核心算法实现
- 基于信息论的不确定性减少
- 提供科学原理指导
- 优化探索策略

### 2. 组管理（Group Management）
**文件位置**：`src/group/manage.py`

**功能**：
- 管理智能体之间的通信
- 实现Hypothesis-Validation机制
- 控制对话流程和轮次
- 记录运行日志

### 3. 工具系统（Tools）
**文件位置**：`tools/`

工具是智能体与外部预测模型交互的接口。

#### `tool_nanomaterial.py`
- 功能：纳米螺旋材料g-factor预测
- 输入参数：fiber_radius, helix_radius, n_turns, pitch
- 输出：g-factor值（手性因子）
- API端点：`POST /prediction_nanohelix`

#### `tool_chembl35.py`
- 功能：ChEMBL化学分子性质预测
- 输入参数：SMILES字符串
- 输出：预测的分子性质
- API端点：`POST /prediction_chembl`

#### `tool_predict_Tc.py`
- 功能：超导体临界温度预测
- 输入参数：材料组成和结构信息
- 输出：临界温度Tc
- API端点：`POST /prediction_supercon`

**工具注册机制**：
使用`@register_tool`装饰器在`tools/__init__.py`中自动注册工具，系统会自动加载所有注册的工具。

---

## 🗂️ 预测模型环境

### 1. AgenX_Chembl35
**ChEMBL化学分子预测模型**

**目录结构**：
```
AgenX_Chembl35/
├── config.yaml          # 模型配置
├── inference.py         # 推理脚本
├── launch.py            # 启动脚本
├── server.py            # 独立服务器
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── src/                 # 源代码
│   ├── model.py         # 模型定义
│   └── preprocessing.py # 数据预处理
└── models/              # 训练好的模型
    ├── best_loss_model.pt
    └── best_r2_model.pt
```

**功能**：基于SMILES字符串预测化学分子的生物活性。

### 2. AgenX_Nanohelix
**纳米螺旋g-factor预测模型**

**目录结构**：
```
AgenX_Nanohelix/
├── inference.py         # 推理脚本
├── launch.py            # 启动脚本
├── train.py             # 训练脚本
├── predict.py           # 预测脚本
├── core/                # 核心模块
│   └── models.py        # 模型定义
├── data/                # 数据集
│   ├── train_g_0603.csv
│   └── test_g_0603.csv
├── models/              # 训练好的模型
│   ├── nanohelix_mlp_model.pkl
│   ├── nanohelix_scaler_X.pkl
│   └── nanohelix_scaler_y.pkl
└── figures/             # 结果图表
    └── r2_plot.pdf
```

**功能**：
- 基于结构参数预测纳米螺旋的g-factor（手性因子）
- 输入：fiber_radius, helix_radius, n_turns, pitch
- 输出：g-factor值（范围通常为0.0-1.8）

### 3. AgenX_Supercon
**超导体临界温度预测模型**

**目录结构**：
```
AgenX_Supercon/
├── README.md            # 模型说明
├── config.yaml          # 模型配置
├── launch_supercon.py   # 启动脚本
├── _train_supercon.py   # 训练脚本
├── _inference.py        # 推理脚本
├── src/                 # 源代码
│   ├── data_processor.py # 数据处理
│   └── model.py          # 模型定义
├── data/                 # 数据集
│   └── superconductor_data.tsv
├── models/               # 训练好的模型
│   ├── best_supercon_model.pth
│   ├── best_supercon_model_r2.pth
│   └── supercon_processor.pkl
└── figures/              # 结果图表
    ├── supercon_metrics.pdf
    └── supercon_predictions.pdf
```

**功能**：预测超导材料的临界温度Tc。

---

## 🚀 使用方法

### 环境安装

1. **克隆项目**：
```bash
git clone https://github.com/kianakaslanna/PiFlow
cd PiFlow
```

2. **创建Conda环境**：
```bash
conda create --name piflow python=3.11
conda activate piflow
pip install -r requirements.txt
```

### 配置API

在配置文件中设置OpenAI兼容的API：

1. 编辑 `configs/demo_config_for_model.yaml`
2. 修改每个智能体的`api_config`部分：
   - `base_url`: API端点
   - `model_name`: 模型名称
   - `api_key`: API密钥

3. 编辑 `configs/demo_config_for_task.yaml`
4. 修改`environment`部分的API配置

**推荐模型**：Alibaba Cloud的QwenMax等支持工具调用的大模型。

### 运行项目

```bash
# 1. 修改API信息（在配置文件中）
# 2. 修改start_server.py（如需要）
# 3. 启动预测服务器
python envs/start_server.py

# 4. 在另一个终端运行主程序
python inference.py
```
