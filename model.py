import tensorflow as tf


class Model():
    def __init__(self):
        super(Model, self).__init__()
        self.input_shape = (224, 224, 3)

    def channel_shuffle(self, x, groups):
        _, width, height, channels = x.get_shape().as_list()
        group_ch = channels // groups
        x = tf.keras.layers.Reshape([width, height, group_ch, groups])(x)
        x = tf.keras.layers.Permute([1, 2, 4, 3])(x)
        x = tf.keras.layers.Reshape([width, height, channels])(x)

        return x

    def shuffle_unit(self, x, groups, channels, strides):
        y = x

        x = tf.keras.layers.Conv2D(channels // 4, kernel_size=1, strides=(1, 1), padding='same', groups=groups)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = self.channel_shuffle(x, groups)

        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if strides == (2, 2):
            channels = channels - y.shape[-1]

        x = tf.keras.layers.Conv2D(channels, kernel_size=1, strides=(1, 1), padding='same', groups=groups)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if strides == (1, 1):
            x = tf.keras.layers.Add()([x, y])

        if strides == (2, 2):
            y = tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')(y)
            x = tf.keras.layers.concatenate([x, y])

        x = tf.keras.layers.ReLU()(x)

        return x

    def Shuffle_Net(self, nclasses, start_channels, x):
        groups = 2
        # input = tf.keras.layers.Input(input_shape)

        # x = tf.keras.layers.Conv2D(24, kernel_size=1)
        x = tf.keras.layers.Conv2D(24, kernel_size=3, strides=(2, 2), padding='same', use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)

        repetitions = [7, 3]

        for i, repetition in enumerate(repetitions):
            channels = start_channels * (2 ** i)
            x = self.shuffle_unit(x, groups, channels, strides=(2, 2))

            for j in range(repetition):
                x = self.shuffle_unit(x, groups, channels, strides=(1, 1))

        x = tf.keras.layers.GlobalAveragePooling2D(name='shuffle_average')(x)

        output = tf.keras.layers.Dense(nclasses)(x)

        return output

    def baseModel(self):
        input_shape = (224, 224, 3)
        baseModel = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
        baseModel.trainable = False

        x = baseModel.output

        x = tf.keras.layers.GlobalAveragePooling2D(name='fusion_layer')(x)

        preds = tf.keras.layers.Dense(2)(x)
        model = tf.keras.Model(inputs=baseModel.input, outputs=preds)

        return model

    def fusionModel_shufflenet(self,  iris_model, iris_upper_model, iris_lower_model):

        iris_model = tf.keras.models.Model(inputs=iris_model.input,
                                           outputs=iris_model.get_layer('pool2_conv').output)

        iris_upper_model = tf.keras.models.Model(inputs=iris_upper_model.input,
                                                 outputs=iris_upper_model.get_layer('pool2_conv').output)

        iris_lower_model = tf.keras.models.Model(inputs=iris_lower_model.input,
                                                 outputs=iris_lower_model.get_layer('pool2_conv').output)

        iris_model.trainable = False
        iris_upper_model.trainable = False
        iris_lower_model.trainable = False

        iris = tf.keras.layers.Input(self.input_shape, name='iris')
        iris_upper = tf.keras.layers.Input(self.input_shape, name='iris_upper')
        iris_lower = tf.keras.layers.Input(self.input_shape, name='iris_lower')

        encoded_iris = iris_model(inputs=iris)
        encoded_upper = iris_upper_model(inputs=iris_upper)
        encoded_lower = iris_lower_model(inputs=iris_lower)

        fusion = tf.keras.layers.Concatenate(name='concate_layer')([encoded_iris, encoded_upper, encoded_lower])

        prediction = self.Shuffle_Net(2, 200, fusion)

        fusion_model = tf.keras.Model(inputs=[iris, iris_upper, iris_lower], outputs=prediction)

        return fusion_model