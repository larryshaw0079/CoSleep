"""
@Time    : 2021/4/20 18:50
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : cmd.py
@Software: PyCharm
@Desc    : 
"""


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    res = ((x1 - x2) ** 2).sum().sqrt()
    return res


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = (sx1 ** k).mean(0)
    ss2 = (sx2 ** k).mean(0)

    return l2diff(ss1, ss2)


def cmd(x1, x2, n_moments=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = dm
    for i in range(n_moments - 1):
        # moment diff of centralized samples
        scms = scms + moment_diff(sx1, sx2, i + 2)
    return scms
