import tensorflow as tf
import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import *
import os


dire = os.getcwd()
train_dir = dire+"\\classification_dataset_v1.3\\train"
val_dir = dire+"\\classification_dataset_v1.3\\test"
class_name = ["Bacterial_spot", "Early_blight", "Healthy",
               "Late_blight", "Leaf_Mold", "Septoria_leaf_spot",
               "Spider_mites-Two-spotted_spider_mite", "Target_Spot",
               "Tomato_mosaic_virus", "Tomato_Yellow_Leaf_Curl_Virus"]
CONFIG = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
}


# final data (augmented)

# call resize in model
resize = tf.keras.Sequential([
    tf.keras.layers.Resizing(256, 256),
    tf.keras.layers.Rescaling(1./255)
])

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super().__init__()

        self.dotted_shortcut = (stride != 1)

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, activation='relu', strides=stride, padding='same')
        self.batch = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, activation='relu', strides=1, padding='same')
        # self.conv2 = tf.keras.layers.BatchNormalization(self.conv2)

        self.activation = tf.keras.layers.ReLU()
        if self.dotted_shortcut:
            self.conv3 = tf.keras.layers.Conv2D(filters=filters, activation='relu', kernel_size=1, strides=stride)
            # self.conv3 = tf.keras.layers.BatchNormalization(self.conv3)

    def call(self, inp, train=True):
        x = self.conv1(inp, training=train)
        x = self.batch(x, training=train)
        x = self.conv2(x, training=train)
        x = self.batch(x, training=train)

        # https://arxiv.org/pdf/1512.03385  (page 4)
        # second way https://www.youtube.com/watch?v=cwWFKL0wzi4&t=81s (5:31)
        # Dotted
        if self.dotted_shortcut:
            adding = self.conv3(inp, training=train)  # we modify the input to have the same shape as the output and then we add them
            adding = self.batch(adding, training=train)
            adding = tf.keras.layers.Add()([x, adding])

        # Normal
        else:
            adding = tf.keras.layers.Add()([x, inp])

        return self.activation(adding)

@keras.saving.register_keras_serializable()
class ResNet34(tf.keras.models.Model):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same")
        self.max_pool = tf.keras.layers.MaxPool2D(3, 2)

        self.conv2_1 = ResidualBlock(64)
        self.conv2_2 = ResidualBlock(64)
        self.conv2_3 = ResidualBlock(64)

        self.conv3_1 = ResidualBlock(128, stride=2)
        self.conv3_2 = ResidualBlock(128)
        self.conv3_3 = ResidualBlock(128)
        self.conv3_4 = ResidualBlock(128)

        self.conv4_1 = ResidualBlock(256, stride=2)
        self.conv4_2 = ResidualBlock(256)
        self.conv4_3 = ResidualBlock(256)
        self.conv4_4 = ResidualBlock(256)
        self.conv4_5 = ResidualBlock(256)
        self.conv4_6 = ResidualBlock(256)

        self.conv5_1 = ResidualBlock(512, stride=2)
        self.conv5_2 = ResidualBlock(512)
        self.conv5_3 = ResidualBlock(512)

        self.avg_pool = tf.keras.layers.GlobalAvgPool2D()
        self.fc = tf.keras.layers.Dense(len(class_name), activation=tf.keras.activations.softmax)

    def call(self, x, train=True):
        x = resize(x)
        x = self.conv1(x)
        x = self.max_pool(x)

        x = self.conv2_1(x, training=train)
        x = self.conv2_2(x, training=train)
        x = self.conv2_3(x, training=train)

        x = self.conv3_1(x, training=train)
        x = self.conv3_2(x, training=train)
        x = self.conv3_3(x, training=train)
        x = self.conv3_4(x, training=train)

        x = self.conv4_1(x, training=train)
        x = self.conv4_2(x, training=train)
        x = self.conv4_3(x, training=train)
        x = self.conv4_4(x, training=train)
        x = self.conv4_5(x, training=train)
        x = self.conv4_6(x, training=train)

        x = self.conv5_1(x, training=train)
        x = self.conv5_2(x, training=train)
        x = self.conv5_3(x, training=train)

        x = self.avg_pool(x)
        x = self.fc(x)
        return x


model_path = "C:\\Users\\Gamer\\farm_prototype\\leaf_doctor\\model_v2\\model_v2\\model.keras"
path = "C:\\Users\\Gamer\\farm_prototype\\leaf_doctor\\test_result\\IMG_20240801_194543.jpg__1.jpg"
model = tf.keras.models.load_model(model_path)
pred = model.predict(path)
print(pred)
