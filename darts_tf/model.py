import tensorflow as tf
from operations import *
from cell import *

slim = tf.contrib.slim

def model(_inp, Cs, reductions, steps):
    alpha_normal = slim.variable(
        'alpha_normal',
        shape=(steps + 2, steps + 2, len(OPS)),
        initializer=tf.random_normal_initializer(),
    )
    normal_weights = tf.nn.softmax(alpha_normal, axis=2, name='normal_op_weights')
    
    alpha_reduction = slim.variable(
        'alpha_reduction',
        shape=(steps + 2, steps + 2, len(OPS)),
        initializer=tf.random_normal_initializer(),
    )
    reduction_weights = tf.nn.softmax(alpha_reduction, axis=2, name='reduction_op_weights')
    
    cell_outputs = [_inp]
    for i in range(len(Cs)):
        C = Cs[i]
        reduction = reductions[i]
        reduction_prev = reductions[max(0, i - 1)]
        alpha = reduction_weights if reduction else normal_weights
        
        s0 = cell_outputs[max(0, i - 1)]
        s1 = cell_outputs[i]
        _out = cell(s0, s1, steps, C, reduction, reduction_prev, alpha)
        cell_outputs.append(_out)
        
    return cell_outputs[-1]

if __name__ == '__main__':
    _inp = tf.placeholder(tf.float32, (None, 28, 28, 3))
    
    _inp = slim.conv2d(_inp, 64, (5, 5), stride=(2, 2), padding='VALID')
    
    _out = model(_inp, [64, 128, 128, 256, 256], [False, True, False, True, False], 2)
    
    print(_out)