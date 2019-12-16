from keras.datasets import fashion_mnist
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from itertools import cycle

from models import MLP, MLP_tuning, regu_MLP
from adversarial import deepfool, FGS
from helper import preprocess, train, regu_train, test, rho2, plot_acc_on_FGS, roc


def main():
    # read data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # read data
    (x_train, y_train, x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    # Reshape the data inputs such that we can put those inputs into MLP
    train_inputs = np.reshape(x_train, (-1, 28 * 28))
    test_inputs = np.reshape(x_test, (-1, 28 * 28))

    orig_model = MLP(28 * 28)

    # train original model
    epochs = 10
    print("Training Original MLP...")
    for i in range(epochs):
        train(orig_model, train_inputs, y_train)
        test_acc = test(orig_model, test_inputs, y_test)
        print("Epoch: {} ------ Testing accuracy: {}".format(i+1, test_acc))


    # calculate fgs and deepfool for original model, on both test set and training set
    print("Creating DeepFool images set... will take aobut 5 mins")
    (train_adv_orig, train_r_orig) = deepfool(orig_model, train_inputs)
    (test_adv_orig, test_r_orig) = deepfool(orig_model, test_inputs)

    # fine tuning
    tuning_model = MLP_tuning(28 * 28, orig_model)
    epochs = 5
    print("Training Fine Tuning MLP...")
    for i in range(epochs):
        train(tuning_model, train_adv_orig, y_train)
        tuning_test_acc = test(tuning_model, test_adv_orig, y_test)
        print("Epoch: {} ------ Testing accuracy: {}".format(i+1, tuning_test_acc))



    # train deepdefense model
    regu_model = regu_MLP(28 * 28, orig_model)
    epochs = 5
    print("Training Deep Defense MLP...")
    for i in range(epochs):
        regu_train(regu_model, train_adv_orig, y_train, train_r_orig)
        regu_test_acc = test(regu_model, test_adv_orig, y_test)
        print("Epoch: {} ------ Testing accuracy: {}".format(i+1, regu_test_acc))


    # keep training original model for comparison
    epochs = 5
    print("Training MLP for 5 more epochs...")
    for i in range(epochs):
        train(orig_model, train_inputs, y_train)
        test_accu = test(orig_model, test_inputs, y_test)
        print("Epoch: {} ------ Testing accuracy: {}".format(i+1, test_accu))



    ################### Evaluation #########################
    # ROC curve on deepfool testing image generated from origianl MLP model
    roc1 = roc(orig_model, test_adv_orig, y_test, "Vanilla MLP")
    roc2 = roc(tuning_model, test_adv_orig, y_test, "Fine tuning MLP")
    roc3 = roc(regu_model, test_adv_orig, y_test, "Deep Defense MLP")
    AUC = pd.DataFrame({"Vanilla MLP": list(roc1.values()),
                        "Fine-Tune MLP": list(roc2.values()),
                        "Deep Defense": list(roc3.values())}, index=["label " + str(i + 1) for i in range(10)])
    print("Area Under the Curve:")
    print(AUC)

    # testing acc on benign images
    benign_test_acc=pd.DataFrame({
        "Vanilla MLP":test(orig_model, test_inputs, y_test),
        "Fine-Tune MLP":test(tuning_model, test_inputs, y_test),
        "Deep Defense":test(regu_model, test_inputs, y_test)
    },index=["TestAcc"])


    # rho2 scores
    (test_adv_orig2, test_r_orig2) = deepfool(orig_model, test_inputs)
    (test_adv_tuning, test_r_tuning) = deepfool(tuning_model, test_inputs)
    (test_adv_regu, test_r_regu) = deepfool(regu_model, test_inputs)

    regu_rho2 = rho2(test_r_regu, test_inputs)
    tuning_rho2 = rho2(test_r_tuning, test_inputs)
    orig_rho2 = rho2(test_r_orig2, test_inputs)
    rho2_all = pd.DataFrame({"Vanilla MLP":orig_rho2,
        "Fine-Tune MLP":tuning_rho2,
        "Deep Defense":regu_rho2},index=["Rho2 Score"])

    # plot accuracy on FGS images
    epsilon_ref_100, epsilon_ref_50, epsilon_ref_20 = plot_acc_on_FGS(orig_model, regu_model, tuning_model, test_inputs,
                                                                      y_test,
                                                                      test_adv_orig)
    epsilon_list = [epsilon_ref_20, epsilon_ref_50, epsilon_ref_100]

    # calculating testing accuracy of vanilla, regu, and finetune on FGS examples with these three epsilon values
    pert_test_orig = FGS(orig_model, test_inputs, y_test, 1)
    pert_test_regu = FGS(regu_model, test_inputs, y_test, 1, True, test_adv_orig)
    pert_test_tuning = FGS(tuning_model, test_inputs, y_test, 1)

    FGS_orig_test_acc = list(map(lambda x: test(orig_model, x * pert_test_orig + test_inputs, y_test), epsilon_list))
    FGS_regu_test_acc = list(map(lambda x: test(regu_model, x * pert_test_regu + test_inputs, y_test), epsilon_list))
    FGS_tuning_test_acc = list(
        map(lambda x: test(tuning_model, x * pert_test_tuning + test_inputs, y_test), epsilon_list))

    acc_fgs = pd.DataFrame({"Vanilla MLP": FGS_orig_test_acc,
                           "Fine-Tune MLP": FGS_tuning_test_acc,
                           "Deep Defense": FGS_regu_test_acc}, index=["eps_ref@0.2", "eps_ref@0.5", "eps_ref@1.0"])
    result_table = pd.concat([benign_test_acc, rho2_all, acc_fgs], ignore_index = False).transpose()
    print(result_table)



if __name__ == "__main__":
    main()