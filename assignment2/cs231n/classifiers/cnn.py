from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 注意，卷积层的W和b可不是指一个卷积核，而是包含了多个卷积核。实现的时候也是这么实现的
        # 至于一个一个的卷积核，我们是这么理解的，但实现起来肯定不是真的一个一个来
        C, H, W = input_dim
        W1 = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        b1 = np.zeros(num_filters)

        # 算一下中间的affine layer的输入维度（它前面说，卷积得到的结果的H和W与输入保持一致）
        # 但是它只说了conv保持H和W不变，没说pool
        # （一般都pool了，也不会专门再去padding保持形状不变吧）
        C_out = num_filters
        H_out = H / 2
        W_out = W / 2
        middle_output_dim = int(C_out * H_out * W_out)

        W2 = np.random.normal(0, weight_scale, (middle_output_dim, hidden_dim))
        b2 = np.zeros(hidden_dim)

        # 最后一个分类层的维度已经固定了
        W3 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b3 = np.zeros(num_classes)

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scores, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # print('scores shape:', scores.shape)
        # print('W2 shape:', W2.shape)
        scores, cache2 = affine_relu_forward(scores, W2, b2)
        # 前面实现affine layer的时候，输入就假设的是，每个样本都可能有若干维度
        # 所以里面会有展平的操作
        # 这样比较符合实际情况，因为图像一般存储的时候都不会展平，而是有三维/二维形状
        # 前面考虑清楚了，这里实现cnn也统一起来了，会方便很多
        scores, cache3 = affine_forward(scores, W3, b3)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)
        dout, dW3, db3 = affine_backward(dout, cache3)
        dout, dW2, db2 = affine_relu_backward(dout, cache2)
        dout, dW1, db1 = conv_relu_pool_backward(dout, cache1)
        
        # print(W3)
        loss += 0.5 * self.reg * np.sum(W3 ** 2)
        dW3 += self.reg * W3
        loss += 0.5 * self.reg * np.sum(W2 ** 2)
        dW2 += self.reg * W2
        loss += 0.5 * self.reg * np.sum(W1 ** 2)
        dW1 += self.reg * W1

        grads['W3'] = dW3
        grads['W2'] = dW2
        grads['W1'] = dW1
        grads['b3'] = db3
        grads['b2'] = db2
        grads['b1'] = db1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
