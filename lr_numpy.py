
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification

X, y = make_classification(random_state=2)
# 只取两个特征值, 二维特征值方便可视化
X = X.T[:2, :]
y = np.expand_dims(y, axis=0)
print("X", X.shape)
print("y", y.shape)

#  形成网格, 我们之后用来画分类边界
interval = 0.2
x_min, x_max = X[0, :].min() - .5, X[0, :].max() + .5
y_min, y_max = X[1, :].min() - .5, X[1, :].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, interval),
                     np.arange(y_min, y_max, interval))

# ---------------数据分布图
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=cm_bright, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel("theta_1")
plt.ylabel("theta_2")
plt.show()

# ------------ 激活函数 sigmoid
def sigmoid(y):
    return 1 / (1 + np.exp(-y))

# ---------- 前向传播
# 初始化theta为全0
theta = np.zeros([2, 1])
# 初始化偏置为0
bias = np.zeros([1])

# 进行前向传播计算并求出损失
def forward(X, theta, bias):
    z = np.dot(theta.T, X) + bias
    y_hat = sigmoid(z)
    return y_hat
def compute_loss(y, y_hat):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# ----------- 反向传播
def backward(x, y, y_hat, theta): 
    m = x.shape[-1]
    # theta的梯度
    delta_theta = np.dot(x, (y_hat - y).T) / m
    # bias的梯度
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias

# ----------训练
epochs=1000
learning_rate = 0.1
for i in range(epochs):
    # 前向传播
    y_hat = forward(X, theta, bias)
    # 损失值
    loss = np.mean(compute_loss(y, y_hat)) # 一批数据损失值的均值
    if i % 100 == 0:  
        print('step:{}, loss:{}'.format(i, loss))
    
    # 梯度下降
    delta_theta, delta_bias = backward(X, y, y_hat, theta)
    
    # 参数更新
    theta -= learning_rate * delta_theta
    bias -= learning_rate * delta_bias


# 画等高线图
data = np.c_[xx.ravel(), yy.ravel()].T
# 计算出区域内每一个点的模型预测值
Z = forward(data, theta, bias)
Z = Z.reshape(xx.shape)

# 定义画布大小
plt.figure(figsize=(10,8))
# 画等高线
plt.contourf(xx, yy, Z, 10, cmap=plt.cm.RdBu, alpha=.8)
# 画轮廓
contour = plt.contour(xx, yy, Z, 10, colors="k", linewidths=.5)
plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=cm_bright, edgecolors='k')
# 标出等高线对应的数值
plt.clabel(contour, inline=True, fontsize=10)
plt.show()
