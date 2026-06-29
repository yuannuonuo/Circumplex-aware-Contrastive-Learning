# -*- coding: utf-8 -*-
"""
values.py —— Schwartz 19 类基本价值的标签体系与"圆形连续体"工具。

设计核心：Schwartz 理论中这 19 个价值并不是相互独立、彼此等距的标签，而是排
布在一个"环形动机连续体"（circular motivational continuum）上。环上相邻的价值
心理含义相近、容易混淆；环上正对面的价值含义相反、几乎不会混。本文件把这个几何
结构编码下来，使模型"把相邻价值判错"的代价小于"把对立价值判错"。

注意：数据集里的标签分隔符是 EN DASH（U+2013，"–"），不是普通连字符 "-"。
所有标签都按数据原文逐字匹配。
"""

import numpy as np  # 仅用 numpy 做向量化的距离/权重计算

# 环形顺序（沿 Schwartz 圆顺时针排列）。列表里的"位置下标"本身就代表它在环上的角度，
# 因此两个类别 id 之差（取模 19）就是它们在环上的距离。这个顺序也正是数据集里标签的原文写法。
CIRCULAR_ORDER = [
    "Self-direction–thought",        # 0  自我导向-思想：自由地培养自己的想法与能力
    "Self-direction–action",         # 1  自我导向-行动：自由地决定自己的行为
    "Stimulation",                   # 2  刺激：兴奋、新奇与变化
    "Hedonism",                      # 3  享乐：快乐与感官满足
    "Achievement",                   # 4  成就：依社会标准取得成功
    "Power–dominance",               # 5  权力-支配：通过控制他人获得权力
    "Power–resources",               # 6  权力-资源：通过控制物质/社会资源获得权力
    "Face",                          # 7  面子：维护公众形象、避免羞辱带来的安全与权力
    "Security–personal",             # 8  安全-个人：自身直接环境的安全
    "Security–societal",             # 9  安全-社会：更广泛社会的安全与稳定
    "Tradition",                     # 10 传统：维护文化/家庭/宗教传统
    "Conformity–rules",              # 11 顺从-规则：遵守规则、法律与正式义务
    "Conformity–interpersonal",      # 12 顺从-人际：避免惹恼或伤害他人
    "Humility",                      # 13 谦逊：认识到自身在宏大体系中的渺小
    "Benevolence–dependability",     # 14 仁善-可靠：做内群体中可信赖的一员
    "Benevolence–caring",            # 15 仁善-关怀：致力于内群体成员的福祉
    "Universalism–concern",          # 16 普世-关切：承诺对所有人的平等、正义与保护
    "Universalism–nature",           # 17 普世-自然：保护自然环境
    "Universalism–tolerance",        # 18 普世-包容：接纳并理解与自己不同的人
]

NUM_CLASSES = len(CIRCULAR_ORDER)   # 类别总数 = 19
assert NUM_CLASSES == 19            # 断言：保证标签表完整无缺

# 标签字符串 -> 连续整数 id 的映射；id 即环上位置，便于后续算圆形距离
LABEL2ID = {lab: i for i, lab in enumerate(CIRCULAR_ORDER)}
# id -> 标签字符串 的反向映射，用于把预测下标还原成文字标签
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

# 四个"高阶价值组"。仅用于误差诊断（看错误是否落在同一大组内），不参与训练目标。
HIGHER_ORDER = {
    "Openness-to-change": [          # 开放变化
        "Self-direction–thought", "Self-direction–action",
        "Stimulation", "Hedonism",
    ],
    "Self-enhancement": [            # 自我增强
        "Achievement", "Power–dominance", "Power–resources", "Face",
    ],
    "Conservation": [                # 保守
        "Security–personal", "Security–societal", "Tradition",
        "Conformity–rules", "Conformity–interpersonal", "Humility",
    ],
    "Self-transcendence": [          # 自我超越
        "Benevolence–dependability", "Benevolence–caring",
        "Universalism–concern", "Universalism–nature",
        "Universalism–tolerance",
    ],
}
# 把"类别 id -> 高阶组名"展平成一个字典，方便用 id 直接查它属于哪个大组
ID2GROUP = {LABEL2ID[v]: g for g, vs in HIGHER_ORDER.items() for v in vs}


def circular_distance_matrix(num_classes: int = NUM_CLASSES) -> np.ndarray:
    """返回 [C, C] 的整数矩阵，元素 (i,j) 是 i、j 两类在环上的最短距离。"""
    idx = np.arange(num_classes)                      # 生成 0..C-1 的下标向量
    d = np.abs(idx[:, None] - idx[None, :])           # 广播相减取绝对值，得到"直线距离"
    return np.minimum(d, num_classes - d)             # 与"绕另一圈的距离"取较小者 = 环形最短距离


def circular_soft_labels(num_classes: int = NUM_CLASSES,
                         sigma: float = 1.0,
                         peak: float = 0.85) -> np.ndarray:
    """构造 [C, C] 的"圆形软标签"矩阵：第 y 行就是真实类别为 y 时使用的软目标分布。

    真实类别保留 peak 的概率质量；剩下的 (1 - peak) 按"圆形距离的高斯权重"分给其它
    类别——离得越近分到越多，正对面几乎为 0。这样"判成邻居"的惩罚小于"判成对面"。

    参数：
        sigma：高斯带宽，越大则相邻类别分到的质量越多（标签越"软"）。
        peak ：真实类别保留的概率质量（对角线值）。
    """
    d = circular_distance_matrix(num_classes).astype(np.float64)  # 取圆形距离并转浮点
    w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))        # 高斯核：距离越大权重越小
    np.fill_diagonal(w, 0.0)                           # 先把对角线（自身）清零，单独处理 peak
    w *= (1.0 - peak) / w.sum(axis=1, keepdims=True)   # 把每行非对角权重归一化到总和 = (1 - peak)
    np.fill_diagonal(w, peak)                          # 再把对角线（真实类别）填入 peak
    return w.astype(np.float32)                        # 转 float32 便于喂给 PyTorch


if __name__ == "__main__":                            # 直接运行本文件时做一个自检
    M = circular_soft_labels()                        # 生成软标签矩阵
    np.set_printoptions(precision=3, suppress=True, linewidth=200)  # 打印格式
    print("每行求和是否为 1:", np.allclose(M.sum(1), 1.0))          # 校验概率归一化
    print("类别 0 (Self-direction-thought) 的软标签 Top5:")
    for j in np.argsort(-M[0])[:5]:                   # 取第 0 行概率最大的 5 个类别
        print(f"  {M[0, j]:.3f}  {ID2LABEL[j]}")       # 打印它们的概率与标签名
