import matplotlib.pyplot as plt
 
# 点的坐标
x = [0, 1, 2, 3, 4]
# 点的值
y = [10, 20, 25, 30, 35]
# 点的颜色
colors = ['r', 'g', 'b', 'm', 'c']
 
# 使用plot函数绘制点，并指定颜色
plt.plot(x, y, color=colors)
 
# 显示图形
plt.show()