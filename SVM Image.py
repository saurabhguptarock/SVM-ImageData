from pathlib import Path
import numpy as np
from keras.preprocessing import image

p = Path('./images/')
dirs = p.glob('*')

labels_dict = {'cat': 0, 'dog': 1, 'horse': 2, 'human': 3}
image_data = []

labels = []
for folder_name in dirs:
    label = str(folder_name).split('\\')[-1][:-1]

    for img_path in folder_name.glob('*.jpg'):
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[label])

image_data = np.array(image_data, dtype='float32') / 255
labes = np.array(labels)

import random
combined = list(zip(image_data, labels))
random.shuffle(combined)

image_data[:], labels[:] = zip(*combined)

image_data = image_data.reshape(image_data.shape[0], -1)
from sklearn import svm
svc = svm.SVC(kernel='linear', C=1)
svc.fit(image_data, labels)
print(svc.score(image_data, labels))
