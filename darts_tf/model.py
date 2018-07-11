import tensorflow as tf
from operations import *
from cell import *

slim = tf.contrib.slim

# Constructs DARTS model
# _inp - input tensor
# Cs - list of number of channels for cells
# reduction - list of boolean values, defining which cells are reduction
# steps - number of intermediate steps within each cell

def model(_inp, Cs, reductions, steps):
    
    econv_weights_shape = request_weights_shape((1, 1), (7, 7), (2, 2))
    
    alpha_normal = slim.variable(
        'alpha_normal',
        shape=(steps + 2, steps + 2, len(OPS)),
        initializer=tf.random_normal_initializer(stddev=1e-3),
    )
    normal_weights = tf.nn.softmax(alpha_normal, axis=2, name='normal_op_weights')
    
    alpha_normal_econv = slim.variable(
        'alpha_normal_econv',
        shape=(steps + 2, steps + 2, econv_weights_shape),
        initializer=tf.random_normal_initializer(stddev=1e-3),
    )
    normal_econv_weights = tf.nn.softmax(alpha_normal_econv, axis=2, name='normal_econv_weights')
    
    alpha_reduction = slim.variable(
        'alpha_reduction',
        shape=(steps + 2, steps + 2, len(OPS)),
        initializer=tf.random_normal_initializer(stddev=1e-3),
    )
    reduction_weights = tf.nn.softmax(alpha_reduction, axis=2, name='reduction_op_weights')
    
    alpha_reduction_econv = slim.variable(
        'alpha_reduction_econv',
        shape=(steps + 2, steps + 2, econv_weights_shape),
        initializer=tf.random_normal_initializer(stddev=1e-3),
    )
    reduction_econv_weights = tf.nn.softmax(alpha_reduction_econv, axis=2, name='reduction_econv_weights')
    
    cell_outputs = [_inp]
    for i in range(len(Cs)):
        C = Cs[i]
        reduction = reductions[i]
        reduction_prev = reductions[max(0, i - 1)]
        alpha = reduction_weights if reduction else normal_weights
        alpha_econv = reduction_econv_weights if reduction else normal_econv_weights
        
        s0 = cell_outputs[max(0, i - 1)]
        s1 = cell_outputs[i]
        _out = cell(s0, s1, steps, C, reduction, reduction_prev, (alpha, alpha_econv))
        cell_outputs.append(_out)
        
    arch = [alpha_normal, alpha_normal_econv, alpha_reduction, alpha_reduction_econv]
        
    return cell_outputs[-1], arch

if __name__ == '__main__':
    _inp = tf.placeholder(tf.float32, (None, 28, 28, 3))
    
    _inp = slim.conv2d(_inp, 64, (5, 5), stride=(2, 2), padding='VALID')
    
    _out, alpha = model(_inp, [64, 128, 128, 256, 256], [False, True, False, True, False], 2)
    
    print('Output tensor shape: ', _out.shape)
    print('Alpha variables:')
    for a in alpha:
        print(a)