import numpy as np
import tensorflow as tf

# Vanilla MLP
class MLP(tf.keras.Model):
    def __init__(self, pixel_size):
        self.pixel_size = pixel_size
        self.num_classes = 10
        self.hidden_size = 100
        self.batch_size = 64

        super(MLP, self).__init__()
        self.W1 = tf.Variable(tf.random.truncated_normal([self.pixel_size, self.hidden_size], stddev=0.01))
        self.b1 = tf.Variable(tf.random.truncated_normal([self.hidden_size], stddev=0.01))
        self.W2 = tf.Variable(tf.random.truncated_normal([self.hidden_size, self.num_classes], stddev=0.01))
        self.b2 = tf.Variable(tf.random.truncated_normal([self.num_classes], stddev=0.01))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs):
        output1 = tf.add(tf.linalg.matmul(inputs, self.W1), self.b1)
        relu_output1 = tf.nn.relu(output1)
        logits = tf.add(tf.linalg.matmul(relu_output1, self.W2), self.b2)

        return logits

    def loss(self, logits, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# fine_tuning model
class MLP_tuning(tf.keras.Model):
    def __init__(self, pixel_size, orig_model):
        self.pixel_size = pixel_size
        self.num_classes = 10
        self.hidden_size = 100
        self.batch_size = 64

        super(MLP_tuning, self).__init__()
        self.W1 = tf.Variable(tf.identity(orig_model.W1))
        self.b1 = tf.Variable(tf.identity(orig_model.b1))
        self.W2 = tf.Variable(tf.identity(orig_model.W2))
        self.b2 = tf.Variable(tf.identity(orig_model.b2))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    def call(self, inputs):
        output1 = tf.add(tf.linalg.matmul(inputs, self.W1), self.b1)
        relu_output1 = tf.nn.relu(output1)
        logits = tf.add(tf.linalg.matmul(relu_output1, self.W2), self.b2)

        return logits

    def loss(self, logits, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# deep defense adversarial training
class regu_MLP(tf.keras.Model):
    def __init__(self, pixel_size, orig_model):
        self.pixel_size = pixel_size
        self.num_classes = 10
        self.hidden_size = 100
        self.batch_size = 100
        self.c = 25
        self.d = 5
        self.l = 15

        super(regu_MLP, self).__init__()
        self.W1 = tf.Variable(tf.identity(orig_model.W1))
        self.b1 = tf.Variable(tf.identity(orig_model.b1))
        self.W2 = tf.Variable(tf.identity(orig_model.W2))
        self.b2 = tf.Variable(tf.identity(orig_model.b2))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    def call(self, inputs):
        output1 = tf.add(tf.linalg.matmul(inputs, self.W1), self.b1)
        relu_output1 = tf.nn.relu(output1)
        logits = tf.add(tf.linalg.matmul(relu_output1, self.W2), self.b2)

        return logits

    def loss(self, logits, labels, delta_x, inputs, true_idx, false_idx):
        original_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        x_div_del = np.linalg.norm(delta_x, axis=1) / np.linalg.norm(inputs, axis=1)
        delta_loss = np.exp(-self.c * x_div_del * np.array(true_idx)) + np.exp(self.d * x_div_del * np.array(false_idx))
        loss = original_loss + self.l * tf.convert_to_tensor(delta_loss)

        return tf.reduce_sum(loss)

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
