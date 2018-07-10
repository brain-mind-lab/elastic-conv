import tensorflow as tf
from elastic_conv2d import elastic_conv2d

slim = tf.contrib.slim

OPS = {
    'none': (lambda _inp, C, stride, affine: zeros(_inp, stride)),
    'avg_pool_3x3': (lambda _inp, C, stride, affine: tf.nn.avg_pool(_inp, (1, 3, 3, 1), (1, stride, stride, 1), padding='SAME')),
    'max_pool_3x3': (lambda _inp, C, stride, affine: tf.nn.max_pool(_inp, (1, 3, 3, 1), (1, stride, stride, 1), padding='SAME')),
    'skip_connect': (lambda _inp, C, stride, affine: tf.identity(_inp) if stride == 1 else factorized_reduce(_inp, C, affine)),
    'elastic_conv': (lambda _inp, C, stride, affine: elastic_conv(_inp, C, stride, affine))
}

#OPS = {
#    'none': (lambda _inp, C, stride, affine: zeros(_inp, stride)),
#    'avg_pool_3x3': (lambda _inp, C, stride, affine: tf.nn.avg_pool(_inp, (1, 3, 3, 1), (1, stride, stride, 1), padding='SAME')),
#    'max_pool_3x3': (lambda _inp, C, stride, affine: tf.nn.max_pool(_inp, (1, 3, 3, 1), (1, stride, stride, 1), padding='SAME')),
#    'skip_connect': (lambda _inp, C, stride, affine: tf.identity(_inp) if stride == 1 else factorized_reduce(_inp, C, affine)),
#    'sep_conv_3x3': (lambda _inp, C, stride, affine: sep_conv(_inp, C, (3, 3), stride=stride, padding='SAME', affine=affine)),
#    'sep_conv_5x5': (lambda _inp, C, stride, affine: sep_conv(_inp, C, (5, 5), stride=stride, padding='SAME', affine=affine)),
#    'sep_conv_7x7': (lambda _inp, C, stride, affine: sep_conv(_inp, C, (7, 7), stride=stride, padding='SAME', affine=affine)),
#    'dil_conv_3x3': (lambda _inp, C, stride, affine: dil_conv(_inp, C, (3, 3), stride=stride, padding='SAME', dilation_rate=2, affine=affine)),
#    'dil_conv_5x5': (lambda _inp, C, stride, affine: dil_conv(_inp, C, (5, 5), stride=stride, padding='SAME', dilation_rate=2, affine=affine)),
#    'conv_7x1_1x7': (lambda *args: conv_7x1_1x7(*args)),
#}

def elastic_conv(_inp, C, stride, affine):
    _out = _inp
    _out = tf.nn.relu(_out)
    _out = elastic_conv2d(
        _out,
        filters=C,
        min_kernel=(1, 1),
        max_kernel=(7, 7),
        stride=(stride, stride),
        padding='SAME',
        activation_fn=None
    )
    _out = batch_norm(_out, affine)
    return _out
    
    
def batch_norm(_inp, affine=True):
    if affine:
        return slim.batch_norm(_inp)
    mean, variance = tf.nn.moments(_inp, axes=[0, 1, 2])
    return tf.nn.batch_normalization(_inp, mean, variance, None, None, 1e-8)
    
def zeros(_inp, stride):
    if stride == 1:
        return _inp * 0
    return _inp[:, ::stride, ::stride] * 0
    
def factorized_reduce(_inp, C, affine=True):
    assert C % 2 == 0
    _out = tf.nn.relu(_inp)
    _conv1 = slim.conv2d(_out, C // 2, (1, 1), stride=(2, 2), padding='VALID', normalizer_fn=(lambda x: x))
    _conv2 = slim.conv2d(_out, C // 2, (1, 1), stride=(2, 2), padding='VALID', normalizer_fn=(lambda x: x))
    return batch_norm(tf.concat([_conv1, _conv2], axis=3), affine)
    

def sep_conv(_inp, C, kernel_size, stride, padding, affine=True):
    _out = _inp
    _out = tf.nn.relu(_out)
    _out = slim.separable_conv2d(
        _out,
        num_outputs=C,
        kernel_size=kernel_size,
        depth_multiplier=C,
        stride=(stride, stride),
        padding=padding,
        activation_fn=None,
        normalizer_fn=(lambda x: x)
    )
    _out = batch_norm(_out, affine)
    _out = slim.separable_conv2d(
        _out,
        num_outputs=C,
        kernel_size=kernel_size,
        depth_multiplier=C,
        stride=(1, 1),
        padding=padding,
        activation_fn=None,
        normalizer_fn=(lambda x: x)
    )
    _out = batch_norm(_out, affine)
    
    return _out

def dil_conv(_inp, C, kernel_size, stride, padding, dilation_rate, affine=True):
    _out = _inp
    _out = tf.nn.relu(_out)
    _out = slim.separable_conv2d(
        _out,
        num_outputs=C,
        kernel_size=kernel_size,
        depth_multiplier=C,
        stride=(1, 1),
        padding=padding,
        rate=dilation_rate,
        activation_fn=None,
        normalizer_fn=(lambda x: x)
    )
    
    if stride != 1:
        _out = slim.conv2d(
            _out,
            num_outputs=C,
            kernel_size=kernel_size,
            stride=(stride, stride),
            padding=padding,
            activation_fn=None,
            normalizer_fn=(lambda x: x)
        )
    _out = batch_norm(_out, affine)
    
    return _out

def conv_7x1_1x7(_inp, C, stride, affine=True):
    _out = _inp
    _out = tf.nn.relu(_out)
    _out = slim.conv2d(_out, C, (1, 7), stride=(1, stride), activation_fn=None, padding='SAME', normalizer_fn=(lambda x: x))
    _out = slim.conv2d(_out, C, (7, 1), stride=(stride, 1), activation_fn=None, padding='SAME', normalizer_fn=(lambda x: x))
    return batch_norm(_out, affine)
    
def relu_conv_bn(_inp, C, kernel_size, stride, padding, affine=True):
    _out = _inp
    _out = tf.nn.relu(_out)
    _out = slim.conv2d(_out, C, kernel_size, stride=(stride, stride), padding=padding, activation_fn=None, normalizer_fn=(lambda x: x))
    return batch_norm(_out, affine)
    
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, (32, 64, 64, 128))
    print(x)
    for k in OPS.keys():
        y = OPS[k](x, 128, 1, False)
        print(k, 's=1', y.shape)
    for k in OPS.keys():
        y = OPS[k](x, 128, 2, False)
        print(k, 's=2', y.shape)
        
        
    print('relu_conv_bn', 's=1', relu_conv_bn(x, 128, (3, 3), 1, 'SAME').shape)
    print('relu_conv_bn', 's=2', relu_conv_bn(x, 128, (3, 3), 2, 'SAME').shape)