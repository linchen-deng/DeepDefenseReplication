from sklearn.metrics import roc_curve, auc
from itertools import cycle
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from adversarial import FGS

# preprocess data
def preprocess(x_train, y_train, x_test, y_test):
    # normalize the inputs
    x_train = x_train / 255.0
    x_train = x_train.astype(np.float32)

    x_test = x_test / 255.0
    x_test = x_test.astype(np.float32)

    # Onehot labels
    y_train = tf.one_hot(y_train, depth=10)

    y_test = tf.one_hot(y_test, depth=10)

    return (x_train, y_train, x_test, y_test)
# train function
def train(model, train_inputs, train_labels):
    # shuffle
    indices = tf.range(0, train_inputs.shape[0])
    new_indices = tf.random.shuffle(indices)
    new_inputs = tf.gather(train_inputs, new_indices)
    new_labels = tf.gather(train_labels, new_indices)

    for x in range(0, new_inputs.shape[0], model.batch_size):
        batch_inputs = new_inputs[x: x + model.batch_size, :]
        batch_labels = new_labels[x: x + model.batch_size]

        # Optimize gradients
        with tf.GradientTape() as tape:
            predictions = model.call(batch_inputs)
            loss = model.loss(predictions, batch_labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def regu_train(regu_model, inputs, labels, delta_x):
    for x in range(0, inputs.shape[0], regu_model.batch_size):
        batch_inputs = inputs[x: x + regu_model.batch_size, :]
        batch_labels = labels[x: x + regu_model.batch_size, :]
        batch_delta_x = delta_x[x: x + regu_model.batch_size, :]

        # collect the index of correct prediction
        batch_logits = regu_model.call(batch_inputs)
        true_idx = (tf.argmax(batch_logits, 1) == tf.argmax(batch_labels, 1))
        false_idx = (tf.argmax(batch_logits, 1) != tf.argmax(batch_labels, 1))

        # Optimize gradients
        with tf.GradientTape() as tape:
            logits = regu_model.call(batch_inputs)
            loss = regu_model.loss(logits, batch_labels, batch_delta_x, batch_inputs, true_idx, false_idx)

        gradients = tape.gradient(loss, regu_model.trainable_variables)
        regu_model.optimizer.apply_gradients(zip(gradients, regu_model.trainable_variables))
# test function
def test(model, test_inputs, test_labels):
    acc = []
    for x in range(0, test_inputs.shape[0], model.batch_size):
        batch_inputs = test_inputs[x: x + model.batch_size, :]
        batch_labels = test_labels[x: x + model.batch_size]

        predictions = model.call(batch_inputs)
        acc.append(model.accuracy(predictions, batch_labels))

    return np.mean(acc)

def rho2(test_all_r_out, test_inputs):
    return tf.reduce_mean(tf.norm(test_all_r_out, axis = 1 ) /tf.norm(test_inputs, axis = 1)).numpy()


def roc(model, inputs, labels, model_name):
    n_classes = labels.shape[1]

    logits = model.call(inputs)

    probs = np.array(tf.nn.softmax(logits))

    # Compute ROC curve and ROC area for each class

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves

    plt.figure()
    colors = cycle(['blue', 'darkorange'])
    for i, color in zip([2,3], colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for class 2 and class 3 of '+ model_name)
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


def plot_acc_on_FGS(model, regu_model, finetune_model, test_inputs, y_test, delta_x):
    acc_model, acc_regu, acc_adv, acc_finetune = [], [], [], []
    pert_test_orig = FGS(model, test_inputs, y_test, 1)
    pert_test_regu = FGS(regu_model, test_inputs, y_test, 1, True, delta_x)
    pert_test_tuning = FGS(finetune_model, test_inputs, y_test, 1)
    for k in np.arange(0 ,0.05 ,0.001):
        acc_model.append(test(model, k* pert_test_orig + test_inputs, y_test))
        acc_regu.append(test(regu_model, k * pert_test_regu + test_inputs, y_test))
        acc_finetune.append(test(finetune_model, k * pert_test_tuning + test_inputs, y_test))
    diff = np.abs(np.array(acc_regu)-0.5)
    ref_100_idx = np.where(diff==min(diff))
    epsilon_ref_100 = np.arange(0, 0.05, 0.001)[ref_100_idx]
    epsilon_ref_50 = epsilon_ref_100 / 2
    epsilon_ref_20 = epsilon_ref_100 / 5

    plt.plot(np.arange(0, 0.05, 0.001), acc_regu, lw=3)
    plt.plot(np.arange(0, 0.05, 0.001), acc_finetune, lw=3)
    plt.plot(np.arange(0, 0.05, 0.001), acc_model, lw=3)

    plt.axvline(x=epsilon_ref_20, color="black", linestyle=":", lw=1)
    plt.axvline(x=epsilon_ref_50, color="black", linestyle="--", lw=1)
    plt.axvline(x=epsilon_ref_100, color="black", linestyle="-", lw=1)

    plt.legend(('acc_regu', 'acc_finetune', 'acc_vanilla', "0.2Ɛref", "0.5Ɛref", "1.0Ɛref"),
               loc='upper right', prop={'size': 12})

    plt.title('Accuracy on FGS')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()
    return epsilon_ref_100, epsilon_ref_50, epsilon_ref_20
