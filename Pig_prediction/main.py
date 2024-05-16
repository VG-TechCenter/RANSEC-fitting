# Author: VG-Tech
# Date: 2024-02-29 17:19:26
# File Name: main
# Aim for: Pig prediction

import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

import pandas as pd

# 读取Excel文件
excel_file = 'data.xlsx'
data = pd.read_excel(excel_file)

# 查看数据前几行
print(data.head())

# 提取特征和目标变量

X = data[['Length', 'Width', 'Height']]  # 长宽高作为特征
y = data['Weight']                        # 体重作为目标变量


# 初始化 RANSACRegressor 模型
ransac = RANSACRegressor()

# 拟合模型
ransac.fit(X, y)

# 预测体重
predicted_weights = ransac.predict(X)

# 输出拟合得到的模型参数
print("RANSAC模型的系数:", ransac.estimator_.coef_)
print("RANSAC模型的截距:", ransac.estimator_.intercept_)

# 可视化拟合结果
plt.scatter(X['Length'], y, label='Ground Truth')
plt.scatter(X['Length'], predicted_weights, color='r', label='Predicted')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error

# 计算均方根误差
rmse = np.sqrt(mean_squared_error(y, predicted_weights))
print('均方根误差（RMSE）:', rmse)

# 绘制误差分布直方图
errors = y - predicted_weights
plt.hist(errors, bins='auto', edgecolor='black')  # 使用'auto'来自动选择直方图的数量
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.grid(True)
plt.show()

# 创建序号
indices = range(len(y))

# 绘制散点图
plt.scatter(indices, y, label='Ground Truth')
plt.scatter(indices, predicted_weights, color='r', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Weight')
plt.legend()
plt.title('Ground Truth vs Predicted')
plt.show()