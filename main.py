#! /usr/bin/env python3
import time
import os.path
import tensorflow as tf
import warnings
import shutil
from distutils.version import LooseVersion

# Project modules:
import helper
import project_tests as tests

# OUTPUT COLOR FORMATS:
REG = "\033[m"
BOLD_GREEN = '\033[1;32;m'

# Hyper-parameters:
LEARN_RATE = 9e-5   # Tried 1e-4
KEEP_PROB = 0.5
EPOCHS = 25         # Tried 5, 10, 50
BATCH_SIZE = 4      # Tried 5, 10, 15

# Model parameters:
MODEL_NAME = 'gals_model'
TRAINING = True

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.

    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"

    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load the VGG16 model and grab the needed tensors:
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    print(BOLD_GREEN, end='')
    print("----[load_vgg] VGG Model loaded!")
    print(REG)

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.

    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify

    :return: The Tensor for the last layer of output
    """
    # Convolution layer's kernel parameters:
    std_dev = 0.01
    l2_reg_rate = 1e-3

    # Reducing number of classes (filters) of each layer to our wanted number of classes:
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 1e-4)
    l3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes, kernel_size=1, padding='same',
                                   name='l3_conv_1x1',
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate))

    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01)
    l4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes, kernel_size=1, padding='same',
                                   name='l4_conv_1x1',
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate))

    l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, padding='same',
                                   name='l7_conv_1x1',
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate))

    # 1st upsampling (de-convolution) x2:
    output_up_1 = tf.layers.conv2d_transpose(l7_conv_1x1, num_classes, kernel_size=4, strides=2, padding='same',
                                             name='output_up_1',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate))

    # Skip connection - 1st upsampled & layer 4:
    input_skip_4 = tf.add(output_up_1, l4_conv_1x1)

    # 2nd upsampling (de-convolution) x2:
    output_up_2 = tf.layers.conv2d_transpose(input_skip_4, num_classes, kernel_size=4, strides=2, padding='same',
                                             name='output_up_2',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate))

    # Skip connection - 2nd upsampled & layer 3:
    input_skip_3 = tf.add(output_up_2, l3_conv_1x1)

    # 3rd upsampling (de-convolution) x8:
    output = tf.layers.conv2d_transpose(input_skip_3, num_classes, kernel_size=16, strides=8, padding='same',
                                        name='final_output',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_rate))
    print(BOLD_GREEN, end='')
    print("----[layers] Model layers ready!")
    print(REG)

    return output


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.

    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify

    :return: Tuple of (logits, train_op, total_loss)
    """
    # Reshape for return:
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Cross-entropy loss:
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    # l2-regularization loss:
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # Calculating total loss:
    total_loss = cross_entropy_loss + sum(reg_losses)

    # Minimizing the loss using Adam Optimizer:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss)

    print(BOLD_GREEN, end='')
    print("----[optimize] Optimizer ready!")
    print(REG)

    return logits, train_op, total_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.

    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Initialize variables:
    sess.run(tf.global_variables_initializer())

    # Initialize total training time:
    total_time = 0

    # Running over the eopchs:
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = []  # Initializing training loss
        batch = 1
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: image,
                correct_label: label,
                keep_prob: KEEP_PROB,
                learning_rate: LEARN_RATE})

            print('Epoch: %d/%d - Batch: %d - Loss: %.3f' % (epoch + 1, epochs, batch, loss))

            batch += 1
            train_loss.append(loss)

        epoch_time = time.time()
        total_time += epoch_time - start_time
        print('==== Epoch: %d/%d took %.3f seconds' % (epoch + 1, epochs, epoch_time - start_time))

    print(BOLD_GREEN, end='')
    print("----[train_nn] Training complete!")
    print("----... and it only took %.3f seconds" % total_time)
    print(REG)


tests.test_train_nn(train_nn)


def save_model(sess, model_saver, model_out_dir):
    """
    Saves the FCN model.

    :param sess: TF Session
    :param model_saver: TF model saver
    :param model_out_dir: Directory to save the model in
    """
    # Create a folder for the models:
    if os.path.exists(model_out_dir):
        shutil.rmtree(model_out_dir)
    os.makedirs(model_out_dir)

    # Saving the model:
    model_saver.save(sess, os.path.join(model_out_dir, MODEL_NAME))

    # builder = tf.saved_model.builder.SavedModelBuilder("%s/%s" % (model_out_dir, MODEL_NAME))
    # builder.add_meta_graph_and_variables(sess, ["vgg16"])
    # builder.save()

    print(BOLD_GREEN, end='')
    print('Training finished. Saving model to: {}'.format(model_out_dir))
    print(REG)


def load_saved_model(sess, model_name, model_dir):
    """
    Load the relevant tensors from a saved model.

    :param sess: TF session
    :param model_name: Name of the model
    :param model_dir: Directory of the model

    :return: model_tensors: Dictionary of input_image, keep_prob and logits tensors.
    """
    # Initialize tensors dictionary:
    model_tensors = {}

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    logits_op_name = 'final_output/BiasAdd'

    # Load my saved model:
    saver = tf.train.import_meta_graph('%s/%s.meta' % (model_dir, model_name))
    saver.restore(sess, tf.train.latest_checkpoint('%s' % model_dir))

    # tf.saved_model.loader.load(sess, ["vgg16"], "%s/%s" % (model_dir, MODEL_NAME))

    # Grab needed tensors:
    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    logits_tensor = graph.get_operation_by_name(logits_op_name).outputs[0]
    logits_tensor = tf.reshape(logits_tensor, (-1, 2))

    # Adding tensors to dictionary:
    model_tensors['input_image'] = vgg_input_tensor
    model_tensors['keep_prob'] = vgg_keep_prob_tensor
    model_tensors['logits'] = logits_tensor

    print(BOLD_GREEN, end='')
    print("----Model %s loaded!" % model_name)
    print(REG)

    return model_tensors


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'  # Training & Testing Kitti dataset directory
    runs_dir = './runs'  # Results from running trained model directory

    tests.test_for_kitti_dataset(data_dir)

    # Saved model directory
    model_dir = './model'
    model_out_dir = os.path.join(model_dir, str(time.time()))

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        if TRAINING:
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')

            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            # Getting the input image, keep probability and layers 3, 4, 7 tensors from VGG-16:
            input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

            # Getting out the modified last layer's tensor (after upsampling and adding skip layers):
            nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

            # Tensor placeholders for the optimize function:
            correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
            learning_rate = tf.placeholder(tf.float32)

            # Getting the logits, training optimizer and loss:
            logits, train_op, loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

            # Training:
            train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, loss, input_image,
                     correct_label, keep_prob, learning_rate)

            # Saving the model:
            save_model(sess, tf.train.Saver(), model_out_dir)

        else:  # Load a saved model instead of training a new one:
            # Load saved model
            model_tensors = load_saved_model(sess, MODEL_NAME, '%s/1559734400.1497886' % model_dir)
            input_image = model_tensors['input_image']
            keep_prob = model_tensors['keep_prob']
            logits = model_tensors['logits']

        # Save inference data using helper.save_inference_samples:
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()