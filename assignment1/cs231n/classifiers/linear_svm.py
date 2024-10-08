from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    # 这里用朴素方法求梯度，我的做法是引入一个grad_components来标记样本在哪些类上
    # 产生了分类损失，当然也可以在循环里判断完是否产生分类损失以后直接计算相应梯度
    grad_components = np.zeros((num_train, num_classes))    # 我加的
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        grad_component = np.zeros(num_classes)              # 我加的
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                grad_component[j] = 1                       # 我加的 
                grad_component[y[i]] -= 1                   # 我加的
                loss += margin
        grad_components[i] = grad_component                 # 我加的 

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
        X_i = X[i]
        corresponding_grad_component = grad_components[i]
        X_i = X_i[:, np.newaxis]    # (D, 1)
        corresponding_grad_component = corresponding_grad_component[np.newaxis, :]
        # (1, C) 
        dW += X_i * corresponding_grad_component / num_train
        dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1.分类损失部分
    all_scores = np.dot(X, W)   # (N, C)
    num_idx = np.arange(X.shape[0])
    correct_class_scores = all_scores[num_idx, y]   # 取出每个样本
    # 对应的正确类别上的分数    (N, )

    # 下面这一小部分就是hinge loss定义了
    all_scores -= correct_class_scores[:, np.newaxis]
    all_scores += np.ones_like(all_scores)
    mask = all_scores > 0
    mask[num_idx, y] = 0    # 这里直接确保每个样本的正确类别不会产生
    # 分类损失，下面就不用单独减去了，而且再下面求梯度的时候也会方便
    all_scores *= mask
    loss += np.sum(all_scores)
    # loss -= X.shape[0]  # 这一项是因为每个样本在它对应的正确类别上是不会产生
    # 分类损失的，但上面的计算却包括了它，因此这里要再减掉

    loss /= X.shape[0]  # 求均值别忘了

    # 2.正则化项部分
    loss + reg * np.sum(W ** 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1.分类损失部分的grad
    mask = mask.astype(np.int32)    # 上面mask是用于逻辑运算，
    # 是bool类型的，下面要进行数的运算，故转成int32
    minus_cnts = np.sum(mask.astype(np.int32), axis=1)
    # 这一项是统计每个样本产生了多少次分类损失，产生了多少次，就对应有
    # 多少个正确类别上的梯度
    mask[num_idx, y] -= minus_cnts
    mask = mask[:, np.newaxis, :]   # 核心
    X = X[:, :, np.newaxis]         # 核心
    grad = X * mask
    dW += np.sum(grad, axis=0)
    dW /= X.shape[0]

    # 2.正则化项部分的grad
    dW += 2 * W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
