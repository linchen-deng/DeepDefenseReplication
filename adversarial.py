import tensorflow as tf
import numpy as np

# calculate deepfool perturbation
def calculate_r(model, inputs):
    W1 = tf.identity(model.W1)
    W2 = tf.identity(model.W2)

    # original = tf.reshape(input,[-1, model.pixel_size])
    x = tf.reshape(inputs, [-1, model.pixel_size])
    k_pred = tf.argmax(model(x), 1).numpy()[0]
    k_true = k_pred
    r_out = tf.constant(0, shape=[model.pixel_size], dtype=tf.float32)

    counter = 0
    while k_pred == k_true and counter <= 5:
        counter = counter + 1
        a = tf.nn.relu(tf.add(tf.linalg.matmul(x, W1), model.b1))
        a_tile = np.transpose(np.array((np.tile(a != 0, (W2.shape[1], 1))), dtype='float32'))
        tmp1 = tf.multiply(a_tile, W2)
        gradient = tf.linalg.matmul(W1, tmp1)
        delta_w = np.transpose(np.delete(gradient, k_true, axis=1)) - gradient[:, k_true]

        fx = model(x)
        delta_f = np.delete(fx, k_true) - fx[:, k_true]
        w_norm = np.sum(np.abs(delta_w) ** 2, axis=-1) ** (0.5)
        standard = np.abs(delta_f) / w_norm
        index = tf.argmin(standard).numpy()
        r = (standard[index] + 1e-4) / w_norm[index] * delta_w[index,]
        x += 1.02 * r
        k_pred = tf.argmax(model(x), 1).numpy()[0]
        r_out += r
    return (x, 1.02 * r_out)


# calculate deepfool noise
def deepfool(model, inputs):
    all_x = np.zeros(inputs.shape, 'float32')
    all_r_out = np.zeros(inputs.shape, 'float32')
    for i in range(inputs.shape[0]):
        (all_x[i,], all_r_out[i,]) = calculate_r(model, inputs[i, :])
    return (all_x, all_r_out)


# calculate fgs
def FGS(model, inputs, labels, epsilon, regu=False, delta_x=False):
    inputs = tf.convert_to_tensor(inputs)
    batch_logits = model.call(inputs)
    true_idx = (tf.argmax(batch_logits, 1) == tf.argmax(labels, 1))
    false_idx = (tf.argmax(batch_logits, 1) != tf.argmax(labels, 1))

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        logits = model.call(inputs)
        if regu:
            loss = model.loss(logits, labels, delta_x, inputs, true_idx, false_idx)
        else:
            loss = model.loss(logits, labels)

    gradient = tape.gradient(loss, inputs)

    return epsilon * tf.math.sign(gradient)
