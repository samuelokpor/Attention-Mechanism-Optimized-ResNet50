import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import os
import imghdr

data_dir  = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)

        except Exception as e:
            print('Issue with image {}'.format(image_path))

#load data from memory
data = tf.keras.utils.image_dataset_from_directory('data', image_size=(224,224), batch_size=8)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
print(f'labels are {batch[1]}')

#viz training data
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

plt.show()

#one-hot encode the labels
labels = tf.one_hot(batch[1], depth=3)

#preprocess data/scalling
data = data.map(lambda x,y: (x/255, y))
print(data.as_numpy_iterator().next()[0].max())

#split data

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size).map(lambda x, y: (x, tf.one_hot(y, depth=3)))
val = data.skip(train_size).take(val_size).map(lambda x, y: (x, tf.one_hot(y, depth=3)))
test = data.skip(train_size+val_size).take(test_size).map(lambda x, y: (x, tf.one_hot(y, depth=3)))
print(f"{len(train)}, {len(val)},{len(test)}")
