import tensorflow as tf

def weight_variable(shape, name=None):
    return tf.Variable(
        tf.random.truncated_normal(shape, stddev=0.1),
        name=name
    )

def bias_variable(shape, name=None):
    return tf.Variable(
        tf.constant(0.1, shape=shape),
        name=name
    )

def show_all_variables():
    total_params = 0
    for var in tf.trainable_variables():
        shape = var.shape.as_list()
        params = 1
        for dim in shape:
            params *= dim
        print(f"{var.name}: {shape} ({params} params)")
        total_params += params
    print(f"Total number of parameters: {total_params:,}")