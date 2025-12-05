# PiFlow

---

## 使用说明

### 1. 环境安装

```bash
git clone https://github.com/kianakaslanna/PiFlow
cd PiFlow
conda create --name piflow python=3.11
conda activate piflow
pip install -r requirements.txt
```

### 2. 启动 vLLM 模型服务

```bash
# 这里使用自己的模型和端口，以及显卡
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model /home/qtt/dataset/weights/qwen3-8b \
    --host 127.0.0.1 \
    --port 10501 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### 3. 运行实验

```bash
python inference.py --principled --output run_output
```

---

## 配置说明

### 模型配置 (`configs/demo_config_for_model.yaml`)

配置各智能体使用的 LLM：

```yaml
agents:
  planner:
    api_config:
      base_url: "http://127.0.0.1:10501/v1"
      model_name: "/path/to/your/model"
      api_key: "EMPTY"
```

### 任务配置 (`configs/demo_config_for_task.yaml`)

定义科学发现任务和参数范围：

```yaml
task: |
  发现具有最高 g-factor 的纳米螺旋结构

objective_value: "g-factor"
memory_buffer_size: 10
```

---

## 系统架构

### 智能体协作流程

```
User Task → Planner (分析建议) → Hypothesis (提出假设)
         ↑                                          ↓
   PrincipleFlow ← Experiment (验证实验) ←───────────┘
```

### 核心组件

- **PrincipleFlow**: 基于信息论的探索策略优化算法 (`src/group/workflow.py`)
- **智能体管理**: 协调多智能体通信和任务流转 (`src/group/manage.py`)
- **工具系统**: 与预测模型交互的接口 (`tools/`)
- **预测服务**: 统一的 Flask 服务器 (`envs/unified_server.py`)

---


### 查看运行记录

```bash
cat run_output/running_notes.json | python -m json.tool | less
```

---

## 项目结构

```
PiFlow/
├── inference.py              # 主程序入口
├── configs/                  # 配置文件
├── src/
│   ├── agents/              # 智能体定义
│   ├── group/               # 协作管理
│   └── utils/               # 工具函数
├── envs/                    # 预测模型环境
└── tools/                   # 工具接口
```

