"""numpy_homework.py

Project 1 - NumPy 数值计算练习
================================
本文件实现《NumPy 数值计算练习（题目版）》中的全部 12 道题目：
NP-E1~NP-E8（基础题）与 NP-M1~NP-M4（中等题）。

设计约定（与题目一致）：
  * 仅依赖 Python 与 NumPy，统一使用 `import numpy as np`；
  * 结果全部由计算得到，不写死任何数值；
  * 展示结果保留 6 位小数，核对时允许绝对误差 1e-6；
  * 函数题保留题目指定函数名；其余题按编号分函数组织，便于独立运行与讲评；
  * 每个函数均给出输入/输出 shape 说明与关键表达式注释。

运行：`python numpy_homework.py` 即顺序打印全部题目的运行结果。
"""

import numpy as np

# 统一浮点显示精度：6 位小数
np.set_printoptions(precision=6, suppress=True)


# --------------------------------------------------------------------------- #
# NP-E1 数组、形状与切片
# --------------------------------------------------------------------------- #
def np_e1():
    """二维数组的 ndim/shape/size、行/列切片与副本修改。

    输入: 3x4 数组 A。
    要点: shape 元组描述维度大小；A[1] 取第 2 行 -> shape (4,)；
          A[:, -2:] 取最后 2 列 -> shape (3, 2)；
          副本 B 用布尔掩码 B[B < 0] = 0 置零负数，A 不受影响。
    """
    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80,  0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])
    print("[NP-E1]")
    print("  ndim/shape/size:", A.ndim, A.shape, A.size)
    row1 = A[1]                       # 第 2 行 -> shape (4,)
    cols = A[:, -2:]                  # 最后 2 列 -> shape (3, 2)
    print("  row1        :", row1, "shape", row1.shape)
    print("  last2cols   :", cols.tolist(), "shape", cols.shape)
    B = A.copy()                      # 显式复制，避免修改影响 A
    B[B < 0] = 0                      # 布尔索引：负数置 0
    print("  B(负->0)    :", B.tolist())
    print("  A(保持原样) :", A.tolist())


# --------------------------------------------------------------------------- #
# NP-E2 逐元素乘法、点乘与长度
# --------------------------------------------------------------------------- #
def np_e2():
    """区分逐元素乘(数组)与点乘(标量)，并用两种方法求欧氏长度。

    a * b -> 同 shape (3,) 的数组；a @ b -> 标量。
    长度: sqrt(sum(x**2)) 与 np.linalg.norm 结果一致。
    """
    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])
    print("[NP-E2]")
    print("  a*b      :", a * b, "shape", (a * b).shape)
    print("  a@b      :", a @ b, "shape", np.asarray(a @ b).shape)
    na = np.sqrt(np.sum(a ** 2))      # 平方-求和-开方
    nb = np.sqrt(np.sum(b ** 2))
    print("  len(a,b) :", na, nb)
    print("  norm核对 :", np.linalg.norm(a), np.linalg.norm(b))


# --------------------------------------------------------------------------- #
# NP-E3 两个向量的夹角
# --------------------------------------------------------------------------- #
def angle_deg(a, b):
    """返回同维非零一维向量 a、b 的夹角（单位：度）。

    cos(theta) = (a @ b) / (||a|| * ||b||)，clip 到 [-1,1] 后取 arccos。
    任一输入为零向量时抛出 ValueError。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("输入不能为零向量")
    cos = np.clip((a @ b) / (na * nb), -1.0, 1.0)   # 数值稳健
    return np.degrees(np.arccos(cos))


def np_e3():
    a = np.array([1.00, -1.00])
    b = np.array([1.00, 0.00])
    print("[NP-E3]")
    print("  angle_deg :", angle_deg(a, b))


# --------------------------------------------------------------------------- #
# NP-E4 两个点的距离
# --------------------------------------------------------------------------- #
def np_e4():
    """把三维数组当作坐标点，计算欧氏距离与曼哈顿距离。"""
    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])
    d = p - q                          # 差向量 shape (3,)
    print("[NP-E4]")
    print("  p-q        :", d)
    print("  euclid     :", np.sqrt(np.sum(d ** 2)))
    print("  manhattan  :", np.sum(np.abs(d)))


# --------------------------------------------------------------------------- #
# NP-E5 单位向量
# --------------------------------------------------------------------------- #
def unit_vector(a):
    """返回与 a 同方向、长度为 1 的新数组；零向量抛 ValueError。"""
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("输入不能为零向量")
    return a / n                       # 广播除法，不改原数组


def np_e5():
    a = np.array([-1.20, 0.40, 0.80])
    u = unit_vector(a)
    print("[NP-E5]")
    print("  unit       :", u)
    print("  norm       :", np.linalg.norm(u),
          "allclose1:", np.allclose(np.linalg.norm(u), 1.0))
    print("  orig保持   :", a)


# --------------------------------------------------------------------------- #
# NP-E6 沿不同 axis 汇总
# --------------------------------------------------------------------------- #
def np_e6():
    """按列(axis=0)、按行(axis=1)汇总与整体最大值。

    sum(axis=0) 跨行求和 -> shape (3,)，每列为该列两元素之和；
    mean(axis=1) 跨列求均值 -> shape (2,)，每行为该行均值。
    """
    A = np.array([[0.20, -0.50, 1.10],
                  [0.80,  0.40, -0.30]])
    s0 = A.sum(axis=0)
    m1 = A.mean(axis=1)
    print("[NP-E6]")
    print("  sum axis=0 :", s0, "shape", s0.shape, "-> 各列 [col0+col0, ...]")
    print("  mean axis=1:", m1, "shape", m1.shape, "-> 各行均值")
    print("  max        :", A.max())


# --------------------------------------------------------------------------- #
# NP-E7 广播：一次处理所有行
# --------------------------------------------------------------------------- #
def np_e7():
    """用一条表达式 Y = X * scale + bias 完成逐列缩放加偏置（无显式循环）。

    scale/bias 形如 (3,)，与 X 的每一行(长度 3)按最后一维对齐后广播到两行。
    """
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])
    Y = X * scale + bias               # (2,3)*(3,) -> (2,3)
    print("[NP-E7]")
    print("  Y          :", Y.tolist(), "shape", Y.shape)


# --------------------------------------------------------------------------- #
# NP-E8 图像 batch
# --------------------------------------------------------------------------- #
def np_e8():
    """两张 (C,H,W) 图像堆叠为 batch，取蓝色通道并转置为 (B,H,W,C)。"""
    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30,  0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])
    b = a + 0.10
    batch = np.stack([a, b], axis=0)   # 新增第 0 轴为图像编号 -> (2,3,2,2)
    print("[NP-E8]")
    print("  batch shape:", batch.shape)
    print("  blue C=2   :", [batch[i, 2].tolist() for i in range(2)])
    moved = np.transpose(batch, (0, 2, 3, 1))   # CHW -> HWC
    print("  BHWC shape :", moved.shape)


# --------------------------------------------------------------------------- #
# NP-M1 批量夹角
# --------------------------------------------------------------------------- #
def angles_to(query, X):
    """返回 X 每一行与 query 的夹角数组(度)，shape (N,)。

    用矩阵乘法 query @ X.T 一次得到所有点积，按行求长度做归一化，零向量行返回 nan。
    """
    query = np.asarray(query, dtype=float)
    X = np.asarray(X, dtype=float)
    qn = np.linalg.norm(query)
    if qn == 0:
        raise ValueError("query 不能为零向量")
    xn = np.linalg.norm(X, axis=1)             # (N,) 每行长度
    valid = xn > 0
    dots = X @ query                           # (N,) 一次矩阵乘法
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = dots / (xn * qn)
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
    return np.where(valid, ang, np.nan)        # 零向量行置 nan


def np_m1():
    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00],
                  [-0.60, 0.80, 0.00],
                  [0.80,  0.60, 0.00],
                  [0.00,  0.00, 0.00]])
    ang = angles_to(query, X)
    print("[NP-M1]")
    print("  angles(deg):", ang, "shape", ang.shape)


# --------------------------------------------------------------------------- #
# NP-M2 余弦相似度 Top-k 检索
# --------------------------------------------------------------------------- #
def top_k_cosine(query, gallery, k):
    """按与 query 的余弦相似度对 gallery 行排序，返回 Top-k 行索引与相似度。

    零向量行不参与排序；query 为零向量或 k 非法时抛 ValueError；
    分数相同保持原行顺序（稳定排序）。
    """
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k 必须为正整数")
    qn = np.linalg.norm(query)
    if qn == 0:
        raise ValueError("query 不能为零向量")
    gn = np.linalg.norm(gallery, axis=1)
    valid = np.where(gn > 0)[0]
    if k > valid.size:
        raise ValueError("k 超过有效行数")
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = (gallery @ query) / (gn * qn)    # (M,) 余弦相似度
    ranked = valid[np.argsort(-sim[valid], kind="stable")]  # 高分在前且并列稳定
    top = ranked[:k]
    return top, sim[top]


def np_m2():
    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([[0.60, -0.80, 0.00],
                        [0.30, -0.40, 0.00],
                        [0.80,  0.60, 0.00],
                        [-0.60, 0.80, 0.00],
                        [0.00,  0.00, 0.00]])
    ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])
    idx, sc = top_k_cosine(query, gallery, 3)
    print("[NP-M2]")
    print("  top-k idx  :", idx, "ids:", ids[idx].tolist())
    print("  similarity :", np.round(sc, 6).tolist())


# --------------------------------------------------------------------------- #
# NP-M3 点对距离矩阵
# --------------------------------------------------------------------------- #
def np_m3():
    """广播构造 (|A|,|B|,dim) 差数组，一次得到距离矩阵 D 及每行最近 B 行。"""
    A = np.array([[0.20, -0.40],
                  [0.80,  0.30],
                  [-0.50, 0.70]])
    B = np.array([[0.10, 0.20],
                  [-0.60, -0.20]])
    diff = A[:, None, :] - B[None, :, :]       # (3,1,2)-(1,2,2) -> (3,2,2)
    D = np.sqrt(np.sum(diff ** 2, axis=2))     # (3,2) D[i,j]=|A[i]-B[j]|
    nearest = np.argmin(D, axis=1)             # 每行最近 B 行索引
    print("[NP-M3]")
    print("  diff shape :", diff.shape)
    print("  D          :", np.round(D, 6).tolist(), "shape", D.shape)
    print("  nearest    :", nearest.tolist())


# --------------------------------------------------------------------------- #
# NP-M4 一次前向计算
# --------------------------------------------------------------------------- #
def np_m4():
    """线性 -> ReLU -> 线性的两层前向：Z=XW1^T+b1, H=max(Z,0), Y=HW2^T+b2。"""
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80],
                   [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60],
                   [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])
    Z = X @ W1.T + b1               # (2,3)@(3,2)+(2,) -> (2,2)
    H = np.maximum(Z, 0.0)          # ReLU：负数置 0
    Y = H @ W2.T + b2               # (2,2)@(2,2)+(2,) -> (2,2)
    print("[NP-M4]")
    print("  Z          :", np.round(Z, 6).tolist(), Z.shape)
    print("  H          :", np.round(H, 6).tolist())
    print("  Y          :", np.round(Y, 6).tolist(), Y.shape)


if __name__ == "__main__":
    for fn in (np_e1, np_e2, np_e3, np_e4, np_e5, np_e6,
               np_e7, np_e8, np_m1, np_m2, np_m3, np_m4):
        fn()
        print()
