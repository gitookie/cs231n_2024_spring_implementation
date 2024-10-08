from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_flatten = x.reshape(x.shape[0], -1)   # 把输入展平
    out = np.dot(x_flatten, w) + b

    # 参考实现
    # out = x.reshape(len(x), -1) @ w + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)
    x_flatten = x.reshape(x.shape[0], -1)
    dw = np.dot(x_flatten.T, dout)
    db = np.sum(dout, axis=0)

    # 下面是另一份参考实现，但试了一下和我的效果没区别（而且本质也是一样的啊）
    # dx = (dout @ w.T).reshape(x.shape)
    # dw = x.reshape(len(x), -1).T @ dout
    # db = dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x * (x > 0)

    # 参考实现
    # out = np.maximum(x, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0
    num_train = x.shape[0]
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_x)
    mask = np.arange(num_train)
    correct_class_scores = exp_scores[mask, y]
    loss -= np.sum(np.log(1e-20 + correct_class_scores / (np.sum(exp_scores, axis=1))))
    # 1e-20太恶心了，这个避免分子为0的小数不够小的话会影响精度
    loss /= num_train
    dx = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    dx[mask, y] -= 1
    dx /= num_train

    # 参考实现
    # N = len(y) # number of samples

    # P = np.exp(x - x.max(axis=1, keepdims=True)) # numerically stable exponents
    # P /= P.sum(axis=1, keepdims=True)            # row-wise probabilities (softmax)

    # loss = -np.log(P[range(N), y]).sum() / N     # sum cross entropies as loss

    # P[range(N), y] -= 1
    # dx = P / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # print('x shape:', x.shape)
        cur_mean = np.mean(x, axis=0, keepdims=True)
        cur_std = np.std(x, axis=0, keepdims=True)
        normalized_x = (x - cur_mean) / ((cur_std ** 2 + eps) ** 0.5)
        out = gamma[np.newaxis, :] * normalized_x + beta[np.newaxis, :]
        # print('cur mean shape:', cur_mean.shape)
        # print('running mean shape:', running_mean.shape)
        running_mean = running_mean * momentum + (1 - momentum) * cur_mean
        running_var = running_var * momentum + (1 - momentum) * (cur_std ** 2)
        cache = (normalized_x, cur_std, cur_mean, eps, gamma, x)   # normalized_x是用来计算gamma梯度的
        # cur_std和eps以及gamma是用来计算x的梯度的


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        normalized_x = (x - running_mean) / (eps + (running_var ** 0.5))
        out = gamma[np.newaxis, :] * normalized_x + beta[np.newaxis, :]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 先取出cache里的东西
    normalized_x, std, mean, eps, gamma, x = cache
    dgamma = np.sum(dout * normalized_x, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_hat = (dout * gamma[np.newaxis, :])    # (N, D)
    stable_std = np.sqrt(std ** 2 + eps)
    N = dout.shape[0]
    # dx_hat_over_dx = -normalized_x * (np.sum(normalized_x / stable_std, axis=0, keepdims=True)) / N
    # dx = dx_hat * dx_hat_over_dx

    # # 试试拆成三个部分来看
    # dstd = -np.sum(dx_hat * normalized_x / stable_std, axis=0, keepdims=True)
    # dmu = -np.sum(dx_hat / stable_std, axis=0, keepdims=True)
    # dx1 = dstd * normalized_x / N    # std对x的梯度
    # dx2 = dmu * np.ones_like(x) / N               # mu对x的梯度
    # dx = dx_hat / stable_std + dx1 + dx2 
    # 参照版三个部分法
    # var = std ** 2
    # dsigma = -0.5 * np.sum(dx_hat * (x - mean), axis=0) * np.power(var + eps, -1.5)
    # dstd = -np.sum(dx_hat * normalized_x / stable_std, axis=0, keepdims=True)
    # # dmu = -np.sum(dx_hat / np.sqrt(var + eps), axis=0) - 2 * dsigma * np.sum(x - mean, axis=0) / N
    # dmu = -np.sum(dx_hat / np.sqrt(var + eps), axis=0)
    # # dx = dx_hat / np.sqrt(var + eps) + 2.0 * dsigma * (x - mean) / N + dmu / N
    # dx = dx_hat / np.sqrt(var + eps) + dstd * normalized_x / N + dmu / N
    
    # 这是按照计算图推导出来的，没怎么整理，尤其是中间的dz1，dz2那些，不照着计算图看
    # 基本看不太明白
    dx_1 = dx_hat / stable_std
    dz1 = np.sum(dx_hat * (x - mean), axis=0, keepdims=True)
    dz2 = -dz1 / (stable_std ** 2)
    dz3 = 0.5 * dz2 / stable_std
    dz4 = np.ones_like(x) * dz3 / N
    dz5 = 2 * dz4 * (x - mean)
    dx_2 = dz5
    dx_3 = 1 / N * np.ones_like(x) * (np.sum(-dx_hat / stable_std, axis=0, keepdims=True) + np.sum(-dz5, axis=0, keepdims=True))
    dx = dx_1 + dx_2 + dx_3




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 先取出cache里的东西
    normalized_x, std, mean, eps, gamma, x = cache
    dgamma = np.sum(dout * normalized_x, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_hat = (dout * gamma[np.newaxis, :])    # (N, D)
    stable_std = np.sqrt(std ** 2 + eps)
    N = dout.shape[0]

    # 试试拆成三个部分来看
    # dstd = -np.sum(dx_hat * normalized_x / stable_std, axis=0, keepdims=True)
    # dmu = -np.sum(dx_hat / stable_std, axis=0, keepdims=True)
    # dx1 = dstd * normalized_x / N    # std对x的梯度
    # dx2 = dmu * np.ones_like(x) / N               # mu对x的梯度
    # dx = dx_hat / stable_std + dx1 + dx2 
    dx = (dx_hat * N - normalized_x * np.sum(dx_hat * normalized_x, axis=0, keepdims=True) \
    - np.sum(dx_hat, axis=0, keepdims=True)) / (N * stable_std)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    cur_mean = np.mean(x, axis=1, keepdims=True)
    cur_std = np.std(x, axis=1, keepdims=True)
    normalized_x = (x - cur_mean) / ((cur_std ** 2 + eps) ** 0.5)
    out = gamma[np.newaxis, :] * normalized_x + beta[np.newaxis, :]
    cache = (normalized_x, cur_std, cur_mean, eps, gamma, x)   # normalized_x是用来计算gamma梯度的
    # cur_std和eps以及gamma是用来计算x的梯度的

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 先取出cache里的东西
    normalized_x, std, mean, eps, gamma, x = cache
    dgamma = np.sum(dout * normalized_x, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_hat = (dout * gamma[np.newaxis, :])    # (N, D)
    # 上面的梯度是和bn的求法一样的

    stable_std = np.sqrt(std ** 2 + eps)
    N = dout.shape[0]
    D = dout.shape[1]
    dx_1 = dx_hat / stable_std
    dz1 = np.sum(dx_hat * (x - mean), axis=1, keepdims=True)  # (N, 1)
    dz2 = -dz1 / (stable_std ** 2)    # (N, 1)
    dz3 = 0.5 * dz2 / stable_std      # (N, 1) 
    dz4 = np.ones_like(x) * dz3 / D   # (N, D)
    dz5 = 2 * dz4 * (x - mean)        # (N, D)
    dx_2 = dz5
    dx_3 = 1 / D * np.ones_like(x) * (np.sum(-dx_hat / stable_std, axis=1, keepdims=True) + np.sum(-dz5, axis=1, keepdims=True))
    dx = dx_1 + dx_2 + dx_3

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p   # 实现的时候是把/p的操作放到了
        # mask上。思考一下如果是放到out那一行，会不会影响后面求梯度？
        out = x * mask    # 所谓inverted dropout就是指这样的，在训练的时候，前向
        # 的期望输出就是正常输出，这样在预测的时候就不需要再进行一个乘法了


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pad = conv_param['pad']
    stride = conv_param['stride']
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    padded_x = np.pad(x, pad_width, mode='constant', constant_values=0)
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))
    for i in range(F):   # 先实现朴素版本，就一个一个卷积核来看
        for n in range(N):
        # print(i)
        # print(type(w))
          ker = w[i]  # (C, HH, WW)
          # ker = np.expand_dims(ker, axis=0)  # (1, C, HH, WW)
          for height in range(0, H_out):
              for width in range(0, W_out):
                  out[n, i, height, width] = np.sum(\
                      ker * padded_x[n, :, height*stride:height*stride+HH, \
                      width*stride:width*stride+WW])

    # 另一种实现，虽然也是朴素的用循环，但少了一重，主要是不用遍历样本了
    # 每次取出一个卷积核，就对所有的样本都进行卷积（反正迟早要进行这个卷积的）
    # 主要是一个求和，这时不能全部求和，而是把每个样本的卷积结果进行求和，得到的
    # 结果的形状应该是长度为N的一维数组；而恰好np.sum可以指定沿多个轴进行求和，只需要
    # 用一个元组传入参数
    # for width in range(W_out):
    #     for height in range(H_out):
    #         for i in range(F):
    #             ker = w[i]
    #             ker = np.expand_dims(ker, axis=0)
    #             out[:, i, height, width] = np.sum(ker * padded_x[:, :, height*stride:height*stride+HH, \
    #                           width*stride:width*stride+WW], axis=(1, 2, 3))
    
    # 别忘了加bias。。。不然查了半天也不知道为什么错。。。
    out += b[None, :, None, None]
        



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    _, C, HH, WW = w.shape
    dw = np.zeros_like(w)   # (F, C, HH, WW)
    db = np.sum(dout, axis=(0, 2, 3))
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    padded_x = np.pad(x, pad_width, mode="constant", constant_values=0)
    N, F, H_out, W_out = dout.shape
    dx = np.zeros_like(x)
    dx_padded = np.zeros_like(padded_x)
    for i in range(F):
      for width in range(W_out):
          for height in range(H_out):
              dw[i] += np.sum((dout[:, i, height, width].reshape((-1, 1, 1, 1))\
                                * padded_x[:, :, \
              height*stride:height*stride+HH, width*stride:width*stride+WW]\
              ), axis=0)
              # 我这里还引入一个dx_padded，最后再把dx_padded中的一部分拿出来当成dx
              # 可能还是拖慢了一点速度
              dx_padded[:, :, height*stride:height*stride+HH, width*stride:width*stride+WW] += np.expand_dims(w[i], axis=0) * (dout[:, i, height, width].reshape((-1, 1, 1, 1)))
    dx = dx_padded[:, :, pad:-pad, pad:-pad]
    

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int((H - pool_height) / stride + 1)
    W_out = int((W - pool_width) / stride + 1)
    out = np.zeros((N, C, H_out, W_out))
    for width in range(W_out):
        for height in range(H_out):
            out[:, :, height, width] = np.max(x[:, :, height*stride:height*stride+pool_height\
            , width*stride:width*stride+pool_width], axis=(2, 3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int((H - pool_height) / stride + 1)
    W_out = int((W - pool_width) / stride + 1)
    dx = np.zeros_like(x)   # (N, C, H, W)
    for width in range(W_out):
        for height in range(H_out):
            # mask判断哪些元素是被用上了的（这里的实现是允许一个pool里有多个
            # 最大值的，然后都会更新它们）
            # 它是在判断一个小区域里哪些元素需要更新
            mask = (x[:, :, height*stride:height*stride+pool_height\
            , width*stride:width*stride+pool_width] == np.max(x\
            [:, :, height*stride:height*stride+pool_height\
            , width*stride:width*stride+pool_width], axis=(2, 3), keepdims=True))

            dx[:, :, height*stride:height*stride+pool_height\
            , width*stride:width*stride+pool_width] = dout[:, :, height, width][:, :, None, None] * mask
      # 勉勉强强实现了，但有点太复杂了，能不能简化点？

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 如果调用之前实现的bn的代码，那么其实train/test mode的问题也可以不用管了，
    # 只需要由之前的代码负责区分即可
    # 这也是为什么running mean和running var的形状是一维的长度为D的数组
    # 因为一开始就是打算用之前的bn实现的，所以会有展平的一步

    N, C, H, W = x.shape
    # transformed_x = np.transpose(x, (1, 0, 2, 3)).reshape((-1, C))
    # out, cache = batchnorm_forward(transformed_x, gamma, beta, bn_param)
    # out = np.transpose(out.reshape((C, N, H, W)), (1, 0, 2, 3))
    transformed_x = np.transpose(x, (0, 2, 3, 1)).reshape(-1, C)
    out, cache = batchnorm_forward(transformed_x, gamma, beta, bn_param)
    out = np.transpose(out.reshape(N, H, W, C), (0, 3, 1, 2))
    # out = out.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    transformed_dout = np.transpose(dout, (0, 2, 3, 1)).reshape(-1, C)
    dx, dgamma, dbeta = batchnorm_backward(transformed_dout, cache)
    dx = np.transpose(dx.reshape(N, H, W, C), (0, 3, 1, 2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 这个norm的方法类似于layernorm，所以不需要区分train/test（从原理上看即可）
    N, C, H, W = x.shape
    transformed_x = x.reshape(N, G, -1)
    mean = np.mean(transformed_x, axis=-1, keepdims=True)
    std = np.std(transformed_x, axis=-1, keepdims=True)
    stable_std = np.sqrt(std ** 2 + eps)
    normalized_x = (transformed_x - mean) / stable_std      # (N, G, C*H*W/G)
    normalized_x = normalized_x.reshape(N, C, H, W)
    out = normalized_x * gamma + beta
    cache = (normalized_x, transformed_x, std, mean, gamma, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    normalized_x, transformed_x, std, mean, gamma, eps = cache
    # 注意要求的dgamma，dbeta的形状，都是四维的，所以都要keepdims
    dgamma = np.sum(dout * normalized_x, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    N, G, _ = transformed_x.shape
    dx_hat = (dout * gamma)    # (N, C, H, W)
    dx_hat = dx_hat.reshape(N, G, -1)
    
    stable_std = np.sqrt(std ** 2 + eps)
    _, C, H, W = dout.shape
    dx_1 = dx_hat / stable_std
    dz1 = np.sum(dx_hat * (transformed_x - mean), axis=-1, keepdims=True)  # (N, G, 1)
    dz2 = -dz1 / (stable_std ** 2)    # (N, G, 1)
    dz3 = 0.5 * dz2 / stable_std      # (N, G, 1) 
    dz4 = np.ones_like(transformed_x) * dz3 / transformed_x.shape[-1]   # (N, G, 1)
    dz5 = 2 * dz4 * (transformed_x - mean)        # (N, D)
    dx_2 = dz5
    dx_3 = 1 / transformed_x.shape[-1] * np.ones_like(transformed_x) * (np.sum(-dx_hat / stable_std, axis=-1, keepdims=True) + np.sum(-dz5, axis=-1, keepdims=True))
    dx = dx_1 + dx_2 + dx_3
    dx = dx.reshape(N, C, H, W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
