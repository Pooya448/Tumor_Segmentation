from tensorflow.keras.utils import Sequence


class Generator(Sequence):
    def __init__(self, file_set, batch_size, labels, num_classes, w_max = 256, h_max = 256, preprocess=None):
        self.file_set = file_set
        self.batch_size = batch_size
        self.w_max = w_max
        self.h_max = h_max
        self.labels = labels
        self.num_classes = num_classes
        self.preprocess = preprocess
        self.classes = self.get_classes()

    def __len__(self):
        return int(np.ceil(len(self.file_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.file_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.file_set[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = [self.read_x(filepath) for filepath in batch_x]
        y = [self.read_y(filepath) for filepath in batch_y]
        return np.array(x), np.array(y)

    def read_x(self, filepath):
        image_string = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.reshape(image, [256, 256, 3])

        return image

    def read_y(self, filepath):
        paths = str.split(filepath, os.path.sep)
        filename = paths[-1]
        y = float(self.labels[filename])
        return y

    def get_classes(self):
        l = []
        for f in self.file_set:
            paths = str.split(f, os.path.sep)
            filename = paths[-1]
            y = int(self.labels[filename])
            l.append(y)
        return l
