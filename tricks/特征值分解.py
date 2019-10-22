import numpy as np
x=np.mat(np.array([[2,2],[1,3]]))
print(x)
print(np.linalg.det(x))
# 特征值分解, 特征向量竖着看
a, b = np.linalg.eig(x)
print(a)
print(b)

# 奇异值分解
s,v,d = np.linalg.svd(x)
# print(f"{s}\n\n{v}\n\n{d}\n")