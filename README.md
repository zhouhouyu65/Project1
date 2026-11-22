# Project 1 · NumPy Image Encoder — 数值计算作业

深圳大学《Python 程序设计》Project 1（NumPy 数值计算练习）实验作业完整提交物。
仅依赖 **Python + NumPy**，全部结果由计算得到，未写死任何数值；代码含文档字符串与逐行注释。

## GitHub 仓库

> 请将本目录推送到你的 GitHub 仓库，并把仓库链接填入实验报告封面。

```bash
git init
git add .
git commit -m "Project 1: NumPy numerical exercises"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

## 目录结构

```
.
├── numpy_homework.py          # 12 道题的完整实现 + 运行结果打印
├── numpy_homework_output.txt  # 代码的实际运行输出（核对用）
├── requirements.txt           # 依赖
├── README.md                  # 本文件
└── report/
    ├── main.tex               # 实验报告源码（Overleaf / XeLaTeX 编译）
    └── Experiment_Report.pdf  # 编译后的报告（提交用 PDF）
```

## 环境

- Python 3.x
- NumPy：`pip install -r requirements.txt`

## 运行

```bash
python numpy_homework.py
```

按 `NP-E1 … NP-M4` 顺序打印全部题目的结果，展示值保留 6 位小数（核对允许绝对误差 1e-6）。

## 报告编译（Overleaf）

`report/main.tex` 为深大实验报告 LaTeX 模板。在 Overleaf 新建项目上传后，
**编译器选 XeLaTeX** 即可编译，正文 < 5 页（less is more）。提交时附 `Experiment_Report.pdf`。

## 题目一览

| 编号 | 主题 | 关键 NumPy 用法 |
|------|------|-----------------|
| NP-E1 | 数组、形状与切片 | `shape`、`A[1]`、`A[:, -2:]`、布尔掩码 `B[B<0]=0` |
| NP-E2 | 逐元素乘 / 点乘 / 长度 | `a*b`(数组)、`a@b`(标量)、`np.linalg.norm` |
| NP-E3 | 两向量夹角 | `arccos(clip(a@b/(|a||b|),-1,1))`、零向量抛 `ValueError` |
| NP-E4 | 两点距离 | 欧氏 `sqrt(sum(d^2))`、曼哈顿 `sum(abs(d))` |
| NP-E5 | 单位向量 | `a/||a||`、`np.allclose` 校验、不改原数组 |
| NP-E6 | 沿 axis 汇总 | `sum(axis=0)`、`mean(axis=1)`、`max()` |
| NP-E7 | 广播 | `X*scale+bias` 一行表达式，无显式循环 |
| NP-E8 | 图像 batch | `np.stack`、通道切片、`np.transpose` 转 BHWC |
| NP-M1 | 批量夹角 | 矩阵乘 + 按行范数，零向量行返回 `nan` |
| NP-M2 | 余弦 Top-k | 相似度排序，`argsort(kind="stable")` 稳定并列 |
| NP-M3 | 距离矩阵 | 广播 `(N,1,D)-(1,M,D)`，`argmin` 取最近 |
| NP-M4 | 前向计算 | `X@W1.T+b1 -> ReLU -> @W2.T+b2` |

## 讲评要点

- 代码遵循"充分注释 + 文档字符串"：每个函数给出输入/输出 shape 与关键表达式解释。
- 边界输入处理：零向量（E3/E5/M1/M2 抛 `ValueError` 或返回 `nan`）、全 0 行（M1 返回 `nan`）。
- 所有结果均由计算得出，未硬编码。
