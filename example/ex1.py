import numpy as np
import matplotlib.pyplot as plt
import nums.fdm as fdm

# 设置空间网格
N = 100
x = np.linspace(-1, 1, N, endpoint=False)
h = x[1] - x[0]

# 定义函数
f1 = x
f2 = np.sin(np.pi * x)

# 真值导数
df1_exact = np.ones_like(x)
df2_exact = np.pi * np.cos(np.pi * x)


# 计算导数
df1_1st = fdm.fdm1_f2p_pbc(f1, h)
df2_1st = fdm.fdm1_f2p_pbc(f2, h)

df1_2nd = fdm.fdm1_c3p_pbc(f1, h)
df2_2nd = fdm.fdm1_c3p_pbc(f2, h)

df1_np = fdm.fdm1_c3p_npbc(f1, h)
df2_np = fdm.fdm1_c3p_npbc(f2, h)

# 绘图
fig, axs = plt.subplots(2, 3, figsize=(18, 8))

axs[0, 0].plot(x, df1_exact, 'k--', label='Exact')
axs[0, 0].plot(x, df1_1st, 'r', label='1st-order FD')
axs[0, 0].set_title("f1(x) = x: 1st-order derivative")
axs[0, 0].legend()

axs[0, 1].plot(x, df1_exact, 'k--', label='Exact')
axs[0, 1].plot(x, df1_2nd, 'b', label='2nd-order FD')
axs[0, 1].set_title("f1(x) = x: 2nd-order derivative")
axs[0, 1].legend()

axs[1, 0].plot(x, df2_exact, 'k--', label='Exact')
axs[1, 0].plot(x, df2_1st, 'r', label='1st-order FD')
axs[1, 0].set_title("f2(x) = sin(pi x): 1st-order derivative")
axs[1, 0].legend()

axs[1, 1].plot(x, df2_exact, 'k--', label='Exact')
axs[1, 1].plot(x, df2_2nd, 'b', label='2nd-order FD')
axs[1, 1].set_title("f2(x) = sin(pi x): 2nd-order derivative")
axs[1, 1].legend()

axs[0, 2].plot(x, df1_exact, 'k--', label='Exact')
axs[0, 2].plot(x, df1_np, 'r', label='Numerical')
axs[0, 2].set_title("f1(x) = x: 1st-order derivative non-periodic")
axs[0, 2].legend()

axs[1, 2].plot(x, df2_exact, 'k--', label='Exact')
axs[1, 2].plot(x, df2_np, 'r', label='Numerical')
axs[1, 2].set_title("f2(x) = sin(pi x): 1st-order derivative non-periodic")
axs[1, 2].legend()

plt.tight_layout()
plt.show()
