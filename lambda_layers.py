import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_IFG(x):
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = tf.split(x, [1,1,1,1,1,1,1,1,1], 1)
    
    c01 = c1 - c0
    c02 = c2 - c0
    c03 = c3 - c0
    c04 = c4 - c0
    
    c12 = c2 - c1
    c13 = c3 - c1
    c14 = c4 - c1
    c15 = c5 - c1
    
    c23 = c3 - c2
    c24 = c4 - c2
    c25 = c5 - c2
    c26 = c6 - c2
    
    c34 = c4 - c3
    c35 = c5 - c3
    c36 = c6 - c3
    c37 = c7 - c3
    
    c45 = c5 - c4
    c46 = c6 - c4
    c47 = c7 - c4
    c48 = c8 - c4
    
    c56 = c6 - c5
    c57 = c7 - c5
    c58 = c8 - c5
    
    c67 = c7 - c6
    c68 = c8 - c6
    
    c78 = c8 - c7
    
    c = tf.concat([c01, c02, c03, c04, c12, c13, c14, c15, c23, c24, c25, c26, c34, c35, c36, c37, c45, c46, c47, c48, c56, c57, c58, c67, c68, c78], 1)
    
    return c

def merge_TS(x):
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = tf.split(x, [1,1,1,1,1,1,1,1,1], 1)
    
    c = tf.concat([c0, c1, c2, c3, c4], 1)    
    return c

def merge_TS_end(x):
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = tf.split(x, [1,1,1,1,1,1,1,1,1], 1)
    
    c = tf.concat([c5, c6, c7, c8], 1)    
    return c

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
