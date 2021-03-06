import tensorflow as tf
from operations import OPS, ECONV_WEIGHT_SHAPE, relu_conv_bn, batch_norm, factorized_reduce

slim = tf.contrib.slim

# For given input costructs weighted sum of f(_inp) for every f from OPS
# _inp - input tensor
# weights - 1D vector of shape (len(OPS),) defining weight of each operation
# C - number of channels (used in operations)
# stride - stride (used in operations)

def mixed_op(_inp, weights, C, stride):
    op_weights, econv_weights = weights
    op_res = list()
    for k in OPS.keys():
        res = OPS[k](_inp, C, stride, False, econv_weights)
        if 'pool' in k:
            res = batch_norm(res, False)
        op_res.append(res)
        
    op_res = tf.stack(op_res, axis=-1)
    weighted = op_res * tf.reshape(op_weights, (1, 1, 1, 1, -1))
    return tf.reduce_sum(weighted, axis=-1)

# Constructs single cell
# s0 - output tensor of previous cell
# s1 - output tensor of before previous cell
# steps - number of intermediate steps within cell
# reduction - whether or not this cell is reduction cell
# reduction_prev - whether or not previouse cell is reduction cell
# weights (optional) - predefined weight matrix for cell operations

def cell(s0, s1, steps, C, reduction, reduction_prev, weights):
    
    op_weights, econv_weights = weights
    
    assert op_weights.shape == (steps + 2, steps + 2, len(OPS.keys()))
    assert econv_weights.shape[:2] == (steps + 2, steps + 2)
        
    if reduction_prev:
        s0 = factorized_reduce(s0, C)
    else:
        s0 = relu_conv_bn(s0, C, (1, 1), 1, 'SAME')
    s1 = relu_conv_bn(s1, C, (1, 1), 1, 'SAME')
    
    states = [s0, s1]
    for i in range(steps):
        res = list()
        for j in range(len(states)):
            stride = 2 if reduction and j < 2 else 1
            res.append(mixed_op(states[j], (op_weights[i + 2, j], econv_weights[i + 2, j]), C, stride))
        states.append(tf.reduce_sum(res, axis=0))
    
    return tf.concat(states[2:], axis=-1)
    
if __name__ == '__main__':
    s0 = tf.placeholder(tf.float32, (None, 32, 32, 64))
    s1 = tf.placeholder(tf.float32, (None, 32, 32, 64))
    
    steps = 7
    op_weights = slim.variable(
        'op_weights',
        shape=(steps + 2, steps + 2, len(OPS)),
        initializer=tf.random_normal_initializer(),
    )
    
    econv_weights = slim.variable(
        'econv_weights',
        shape=(steps + 2, steps + 2, ECONV_WEIGHT_SHAPE),
        initializer=tf.random_normal_initializer(),
    )
        
    y = cell(s0, s1, steps, 64, False, False, (op_weights, econv_weights))
    print(y.shape)