import matplotlib.pyplot as plt
import numpy as np

# 数据准备
x = np.linspace(-5, 5, 100)  # x轴数据范围
y = np.linspace(-5, 5, 100)  # y轴数据范围
x_mesh, y_mesh = np.meshgrid(x, y)  # 创建网格
z = np.sin(np.sqrt(x_mesh**2 + y_mesh**2))  # 曲面高度

# 创建3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 绘制3D曲面图
ax.plot_surface(x_mesh, y_mesh, z, cmap="viridis")

# 设置坐标轴标签
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 显示图形
plt.show()
