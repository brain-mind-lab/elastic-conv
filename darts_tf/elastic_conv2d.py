import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def request_weights_shape(min_kernel, max_kernel, stride=None):
    try:
        return np.arange(min_kernel[0], max_kernel[0] + 1, stride[0]).shape[0] * \
               np.arange(min_kernel[1], max_kernel[1] + 1, stride[1]).shape[0]
    except:
        return 2

class weight_to_size:
    def __init__(self, min_kernel, max_kernel, stride=None):
        self.d = dict()
        c = 0
        for i in range(min_kernel[0], max_kernel[0] + 1, stride[0]):
            for j in range(min_kernel[1], max_kernel[1] + 1, stride[1]):
                self.d[c] = (i, j)
                c += 1
                
    def __getitem__(self, key):
        if key.shape[0] == 2: # if kind is regression
            return tuple(np.round(key).astype(int))
        return self.d[key.argmax()]

# Dumb workaround
uses = 0

# _inp - input tensor
# filters - number of convolution filters
# min_kernel - kernel size lower bound 
# max_kernel - kernel size upped bound 
# kind - relaxation type. Must be regression or classification
# kernel_stride - step of kernel size interpolation. Used only when kind=classification
# temperature - slopeness of relaxation. Used only when kind=regression
# alpha_var - optional external variable for storing kernel size
# activation_fn - identical to slim.conv2d
# padding - identical to slim.conv2d
# stride - identical to slim.conv2d
# scope - identical to slim.conv2d
# alpha_collection_name - collection name for storing alpha variables

def elastic_conv2d(
    _inp, 
    filters, 
    min_kernel, 
    max_kernel, 
    kind='regression', 
    kernel_stride=(2, 2), 
    temperature=50, 
    alpha_var=None, 
    activation_fn=tf.nn.relu, 
    padding='VALID',
    stride=(1, 1),
    scope='conv', 
    alpha_collection_name='alpha'
):
    global uses
    with tf.variable_scope(scope + '_%d' % uses):
        uses += 1
        
        # Define variable for storing kernel weights and bias
        kernel = slim.variable(
            'kernel', 
            shape=(max_kernel + (_inp.shape[3].value, filters)),
            initializer=tf.glorot_normal_initializer()
        )
        bias = slim.variable(
            'bias', 
            shape=(filters, ),
            initializer=tf.zeros_initializer()
        )
        
        if kind == 'classification':
            # Create mask tensor with shape (max_tensor[0], max_tensor[1], filters)
            masks = list()
            for h in range(min_kernel[0], max_kernel[0] + 1, kernel_stride[0]):
                for w in range(min_kernel[1], max_kernel[1] + 1, kernel_stride[1]):
                    mask = np.zeros(shape=max_kernel)
                    conv_idx = [
                        (max_kernel[0]) // 2 - h // 2,
                        (max_kernel[1]) // 2 - w // 2
                    ]
                    mask[conv_idx[0]:conv_idx[0]+h, conv_idx[1]:conv_idx[1]+w] = 1
                    masks.append(mask)
            masks = np.stack(masks, axis=2)

            masks = tf.constant(masks, dtype=tf.float32)

            if alpha_var is None:
                # Define variable for storing kernel size weights
                kernel_weights_var = slim.variable(
                    'kernel_weights_raw',
                    shape=(masks.shape[2],),
                    initializer=tf.random_normal_initializer()
                )
                # Add variable to collection of alpha variables
                tf.add_to_collection(alpha_collection_name, kernel_weights_var)
            else:
                kernel_weights_var = alpha_var

            # Reweight with softmax
            kernel_weights = tf.nn.softmax(kernel_weights_var, name='kernel_weights')

            # Compute resulting mask
            mask = tf.expand_dims(tf.reduce_sum(masks * tf.reshape(kernel_weights, (1, 1, -1)), axis=2, keepdims=True, name='kernel_mask'), axis=3)
            
            # Specify return variable
            alpha = kernel_weights_var
            
        elif kind == 'regression':

            # Define variable for storing kernel soft size
            # Default initializer is uniform distribution on [0, 1], where:
            # 0 = min_kernel[i], 1 = min_kernel[i]
            if alpha_var is None:
                kernel_size_var = slim.variable(
                    'kernel_size_raw',
                    shape=(len(min_kernel),),
                    initializer=tf.random_uniform_initializer()
                )
                # Add variable to collection of alpha variables
                tf.add_to_collection(alpha_collection_name, kernel_size_var)
            else:
                kernel_size_var = alpha_var
            
            # Convert [0, 1] to [min_kernel, max_kernel]
            kernel_size = tf.identity((tf.constant(max_kernel, dtype=tf.float32) - tf.constant(min_kernel, dtype=tf.float32)) * tf.nn.sigmoid(kernel_size_var) + min_kernel, name='kernel_size')
            
            # Compute mask from kernel_size
            mesh = np.stack(np.meshgrid(np.arange(max_kernel[0]), np.arange(max_kernel[1])), axis=2)
            mesh = np.abs(mesh - np.reshape(np.array(max_kernel) // 2, (1, 1, -1)))
            mesh = tf.constant(value=mesh, dtype=tf.float32)
            mesh = tf.reshape(kernel_size / 2, (1, 1, -1)) - mesh

            mask = temperature * tf.reduce_min(mesh, axis=2)
            mask = tf.reshape(tf.nn.sigmoid(mask, name='kernel_mask'), max_kernel + (1, 1))
                    
            # Specify return variable
            alpha = kernel_size_var
            

        # Apply mask to resulting tensor and convolve as usual
        conv = tf.nn.convolution(_inp, kernel * mask, padding, strides=stride)
        conv = tf.nn.bias_add(conv, bias)
        if not (activation_fn is None):
            conv = activation_fn(conv)
        
        return conv
    
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, (1, 32, 32, 3))
    y, v = elastic_conv2d(x, 64, (1, 1), (5, 5), kernel_stride=(1, 1), activation_fn=tf.nn.relu, padding='VALID')
    
    print('Input tensor shape: ', x.shape)
    print('Output tensor shape: ', y.shape)
    print('Alpha shape: ', v.shape)