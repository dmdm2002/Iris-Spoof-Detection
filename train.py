import tensorflow as tf
import numpy as np
import os
import datetime
import time

from makeDataset import MkDataset
from model import Model

# cyclegan data
batchsz = 1
Acnt = 4554
Bcnt = 5018

path = 'Z:/2nd_paper/dataset/ND/ROI/Series_Model_output/'

A_path = '1-fold/A/'
B_path = '1-fold/B/'

iris_path = 'iris'
iris_upper_path = 'iris_upper'
iris_lower_path = 'iris_lower'

A_iris = f'{path}{A_path}{iris_path}'
A_iris_upper = f'{path}{A_path}{iris_upper_path}'
A_iris_lower = f'{path}{A_path}{iris_lower_path}'

B_iris = f'{path}{B_path}{iris_path}'
B_iris_upper = f'{path}{B_path}{iris_upper_path}'
B_iris_lower = f'{path}{B_path}{iris_lower_path}'


classes = ['fake', 'live']
Mkds = MkDataset()

"""A dataset Path list"""
A_iris_Fdata = Mkds.make_path_list(A_iris, classes[0])
A_iris_Tdata = Mkds.make_path_list(A_iris, classes[1])

A_iris_upper_Fdata = Mkds.make_path_list(A_iris_upper, classes[0])
A_iris_upper_Tdata = Mkds.make_path_list(A_iris_upper, classes[1])

A_iris_lower_Fdata = Mkds.make_path_list(A_iris_lower, classes[0])
A_iris_lower_Tdata = Mkds.make_path_list(A_iris_lower, classes[1])

A_iris_path = np.concatenate((A_iris_Fdata, A_iris_Tdata), axis=None)
A_iris_upper_path = np.concatenate((A_iris_upper_Fdata, A_iris_upper_Tdata), axis=None)
A_iris_lower_path = np.concatenate((A_iris_lower_Fdata, A_iris_lower_Tdata), axis=None)

"""B dataset Path list"""
B_iris_Fdata = Mkds.make_path_list(B_iris, classes[0])
B_iris_Tdata = Mkds.make_path_list(B_iris, classes[1])

B_iris_upper_Fdata = Mkds.make_path_list(B_iris_upper, classes[0])
B_iris_upper_Tdata = Mkds.make_path_list(B_iris_upper, classes[1])

B_iris_lower_Fdata = Mkds.make_path_list(B_iris_lower, classes[0])
B_iris_lower_Tdata = Mkds.make_path_list(B_iris_lower, classes[1])

B_iris_path = np.concatenate((B_iris_Fdata, B_iris_Tdata), axis=None)
B_iris_upper_path = np.concatenate((B_iris_upper_Fdata, B_iris_upper_Tdata), axis=None)
B_iris_lower_path = np.concatenate((B_iris_lower_Fdata, B_iris_lower_Tdata), axis=None)

"""make A dataset"""
# make label dataset
A_labels = Mkds.get_label(A_iris_path)
A_labels = tf.one_hot(A_labels, 2)
A_labels_ds = tf.data.Dataset.from_tensor_slices(A_labels)

# make image dataset
A_iris_ds = Mkds.make_ds(A_iris_path)
A_iris_upper_ds = Mkds.make_ds(A_iris_upper_path)
A_iris_lower_ds = Mkds.make_ds(A_iris_lower_path)

# zip three roi image dataset, label dataset
A_ds = tf.data.Dataset.zip((A_iris_ds, A_iris_upper_ds, A_iris_lower_ds, A_labels_ds))

"""make B dataset"""
# make label dataset
B_labels = Mkds.get_label(B_iris_path)
B_labels = tf.one_hot(B_labels, 2)
B_labels_ds = tf.data.Dataset.from_tensor_slices(B_labels)

# make image dataset
B_iris_ds = Mkds.make_ds(B_iris_path)
B_iris_upper_ds = Mkds.make_ds(B_iris_upper_path)
B_iris_lower_ds = Mkds.make_ds(B_iris_lower_path)

# zip three roi image dataset, label dataset
B_ds = tf.data.Dataset.zip((B_iris_ds, B_iris_upper_ds, B_iris_lower_ds, B_labels_ds))

"""load model & restore ckp"""

# load models
Model = Model()
iris_model = Model.baseModel()
iris_upper_model = Model.baseModel()
iris_lower_model = Model.baseModel()

# set optimizer, loss_function, acc_metric
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
loss_mean = tf.keras.metrics.Mean()

# load fusion model
fusion_model = Model.fusionModel_shufflenet(iris_model, iris_upper_model, iris_lower_model)

print(fusion_model.summary())

"""call back"""
# set tensorborad

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'Z:/2nd_paper/backup/Ablation/Networks/1st_Proposed_Method/ND/Series_Model_output/1-fold/try_1/logs'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# set CheckPoint
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=fusion_model)
checkpoint_path = "Z:/2nd_paper/backup/Ablation/Networks/1st_Proposed_Method/ND/Series_Model_output/1-fold/try_1/ckp"
manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)
# ckpt.restore('E:/backup/ckp/Proposed/fusion_model/1-fold/2-Dense-addCNN/ckpt-3')
# iris_model.save_weights(checkpoint_path.format(epoch=0))

"""traning loop"""
# print(fusion_model.summary())

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = fusion_model(x, training=True)
        loss_val = loss_fn(y, logits)

    grads = tape.gradient(loss_val, fusion_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, fusion_model.trainable_variables))
    train_acc_metric.update_state(y, logits)
    loss_mean.update_state(loss_val)

    return loss_val


@tf.function
def test_step(x, y):
    val_logits = fusion_model(x, training=False)
    test_acc_metric.update_state(y, val_logits)


epochs = 50
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    A_ds_shuffle = Mkds.configure_for_performance(A_ds, Acnt, shuffle=True)
    A_ds_it = iter(A_ds_shuffle)

    B_ds_shuffle = Mkds.configure_for_performance(B_ds, Bcnt, shuffle=False)
    B_ds_it = iter(B_ds_shuffle)

    for step in range(Acnt//2):
        iris_img, iris_uppper_img, iris_lower_img, iris_label = next(A_ds_it)
        inputs = [iris_img, iris_uppper_img, iris_lower_img]
        loss_val = train_step(inputs, iris_label)
        ckpt.step.assign_add(1)

        if step % 200 == 0:
            print(
                "Traning Loss at step %d: %.4f"
                % (step, float(loss_val))
            )

    result_loss = loss_mean.result()
    train_acc = train_acc_metric.result()
    save_path = manager.save()
    print(f"LOSS : {result_loss}")
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    for i in range(Bcnt//2):
        iris_img, iris_uppper_img, iris_lower_img, iris_label = next(B_ds_it)
        inputs = [iris_img, iris_uppper_img, iris_lower_img]
        test_step(inputs, iris_label)

    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    loss_mean.reset_states()

    print("Validation acc: %.4f" % (float(test_acc),))

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', result_loss, step=epoch)
        tf.summary.scalar('Accuracy', train_acc, step=epoch)
        tf.summary.scalar('test_Accuracy', test_acc, step=epoch)

    train_acc_metric.reset_states()
    print("Time taken: %.2fs" % (time.time() - start_time))

