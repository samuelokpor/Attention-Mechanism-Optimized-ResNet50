import cv2
from keras.metrics import Precision, Recall, CategoricalAccuracy
import numpy as np
from dataset import test

from fuionresnet50 import fusedresnetinstance



fusedresnetinstance.load_weights('modelfused2.h5')



pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = fusedresnetinstance.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy: {acc.result().numpy()}')

img = cv2.imread('test/bike.jpg')
resize = cv2.resize(img, (224, 224))
resize = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
resize = resize / 255.0

yhat = fusedresnetinstance.predict(np.expand_dims(resize, 0))
pred_label = np.argmax(yhat)

if pred_label == 0:
    vehicle = "Bicycle"
elif pred_label == 1:
    vehicle = "Car"
elif pred_label ==2:
    vehicle = "Truck"

cv2.putText(img, vehicle, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()