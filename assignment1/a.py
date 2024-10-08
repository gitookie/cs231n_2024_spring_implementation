import numpy as np

# x = np.array([1, 2, 3, 4, 5])
# # print(np.square(x))
# # print(x ** 2)
# a = np.array([1, 2, 2, 3, 3, 4])
# unique, counts = np.unique(a, return_counts=True)
# print(unique)
# print(counts)
# print(np.argmax(counts))
# a = np.array([[1, 1, 1], [3, 3, 3], [4, 4, 4]])
# b = np.array([[2, 2, 2], [1, 1, 1]])
# result = np.dot(a, b.T)
# print(result)
# print(result.shape)
# x = np.ones((3, 10, 2))
# y = np.ones((4, 2, 10))
# print((np.dot(x, y)).shape)
# a = np.array([1, 2, 3])
# a = a[:, np.newaxis]
# b = np.array([1, 0, 1, 0])
# b = b[np.newaxis, :]
# result = np.dot(a, b)
# print(result.shape)
# res2 = a * b
# print(res2.shape)
# print(result)
# print(res2)
# all_scores = np.array([[1, 2, 3],
#                        [4, 5, 6]])
# print(all_scores.shape)
# y = np.array([2, 1])
# # x_idx = np.arange(2)
# # result = all_scores[x_idx, y]
# # print(result)
# y = y[:, np.newaxis]
# all_scores *= y

scores = np.array([[-1, 2, 3],
                   [3, -2, 1]])
# y_pred = np.argmax(scores, axis=1)
# print(y_pred)
# flags = scores > 0
# print(flags)
# out = scores * flags
# print(out)
# print(np.maximum(scores, 0))
num_idx = np.arange(2)
y = np.array([0, 1])
scores[num_idx, y] = 0
print(scores)