# Лабораторная работа #4.
## Использование техник аугментации данных для улучшения сходимости процесса обучения нейронной сети на примере решения задачи классификации Oregon Wildlife
### 2a. Манипуляции с яркостью и контрастом
[Train_a](https://github.com/NikitaShulgan/Laba4/blob/main/train_a.py)
```
def augment(image, label):
  bright = tf.image.adjust_brightness(image, delta=0.5)
  contrast = tf.image.adjust_contrast(bright, contrast_factor=5)
  return contrast, label
```
#### owl-1617140030.3593705 ```delta=0.1, contrast_factor=1 ```
#### owl-1617140912.2121003 ```delta=0.2, contrast_factor=2 ```
#### owl-1617141971.0594199 ```delta=0.3, contrast_factor=3 ```

![image](https://user-images.githubusercontent.com/80168174/113075389-7dd36680-91d5-11eb-91b8-b0bf748f9531.png)


#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/a_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/a_epoch_loss.svg">

### 2b. Поворот изображения на случайный угол
[Train_b](https://github.com/NikitaShulgan/Laba4/blob/main/train_b.py)

[Tensorboard](http://79.170.108.141:6007/#scalars&_smoothingWeight=0&runSelectionState=eyJ2YWxpZGF0aW9uIjp0cnVlLCJvd2wtMTYxNjU5MjAyMC4yMzUwOTkzL3RyYWluIjpmYWxzZSwib3dsLTE2MTY1OTIwMjAuMjM1MDk5My92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY1OTI0OTUuMjczNjAyNy90cmFpbiI6ZmFsc2UsIm93bC0xNjE2NTkyNDk1LjI3MzYwMjcvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2OTMxMTMyLjY1ODgzNS90cmFpbiI6ZmFsc2UsIm93bC0xNjE2OTMxMTMyLjY1ODgzNS92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY5MzIxNzAuMDMxNTIzNS90cmFpbiI6ZmFsc2UsIm93bC0xNjE2OTMyMTcwLjAzMTUyMzUvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2OTMzMDg5LjcxMDk4MjgvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjkzMzA4OS43MTA5ODI4L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNjkzNDI2My4zNTE5NDkyL3RyYWluIjpmYWxzZSwib3dsLTE2MTY5MzQyNjMuMzUxOTQ5Mi92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY5MzUyNTkuODIzNTc4MS90cmFpbiI6ZmFsc2UsIm93bC0xNjE2OTM1MjU5LjgyMzU3ODEvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2OTM2NzkzLjE3NzU4Ni90cmFpbiI6ZmFsc2UsIm93bC0xNjE2OTM2NzkzLjE3NzU4Ni92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY5MzgwNjYuOTY2NjcxNS90cmFpbiI6ZmFsc2UsIm93bC0xNjE2OTM4MDY2Ljk2NjY3MTUvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2OTM5NTU1Ljg3NzY0NTMvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjkzOTU1NS44Nzc2NDUzL3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNjk0MTUxOS45MjM4ODM0L3RyYWluIjpmYWxzZSwib3dsLTE2MTY5NDE1MTkuOTIzODgzNC92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY5NDU4ODYuMzYyNTc1NS90cmFpbiI6ZmFsc2UsIm93bC0xNjE2OTQ1ODg2LjM2MjU3NTUvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2OTQ3NjQ4LjY0MDA1Mi90cmFpbiI6ZmFsc2UsIm93bC0xNjE2OTQ3NjQ4LjY0MDA1Mi92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTY5NjI0MTMuNDkwNDkwNC90cmFpbiI6ZmFsc2UsIm93bC0xNjE2OTYyNDEzLjQ5MDQ5MDQvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2OTYzNDQ0LjUwNzUxL3RyYWluIjpmYWxzZSwib3dsLTE2MTY5NjM0NDQuNTA3NTEvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE2OTY0NzM3LjE3ODA4NjMvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNjk2NDczNy4xNzgwODYzL3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNzExMDU2Ni42NjEzMDkyL3RyYWluIjpmYWxzZSwib3dsLTE2MTcxMTA2NDEuNDYxNTM4Ni90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTEwODA2LjUyNzE0NC90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTI3ODU3LjczMzA4NTIvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNzEyNzg1Ny43MzMwODUyL3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNzEyODc5MS4wMzgzMzU4L3RyYWluIjpmYWxzZSwib3dsLTE2MTcxMjg3OTEuMDM4MzM1OC92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTcxMzAwNTEuMTUwMDI1OC90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTMwMDUxLjE1MDAyNTgvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE3MTMxOTMwLjY4OTA2MjYvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNzEzMTkzMC42ODkwNjI2L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNzEzMzA1NS4wNDIxNzU1L3RyYWluIjpmYWxzZSwib3dsLTE2MTcxMzMwNTUuMDQyMTc1NS92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTcxMzcxOTIuODg2OTM1Ny90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTM3MTkyLjg4NjkzNTcvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE3MTM4MTA4LjM4MzIyNjQvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNzEzODEwOC4zODMyMjY0L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNzEzOTAwOS43MjAzOTMyL3RyYWluIjpmYWxzZSwib3dsLTE2MTcxMzkwMDkuNzIwMzkzMi92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTcxNDAwMzAuMzU5MzcwNS90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTQwMDMwLjM1OTM3MDUvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE3MTQwOTEyLjIxMjEwMDMvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNzE0MDkxMi4yMTIxMDAzL3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNzE0MTk3MS4wNTk0MTk5L3RyYWluIjpmYWxzZSwib3dsLTE2MTcxNDE5NzEuMDU5NDE5OS92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTcxNDMwOTMuOTE1NTk0OC90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTQzMDkzLjkxNTU5NDgvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTcxNDQxODQuNjIyNjExMy90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTQ0MTg0LjYyMjYxMTMvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTcxNDU2NjIuMDEwNjYwNC90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTQ1NjYyLjAxMDY2MDQvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTcxNDcwNjEuNzcyNDM1Ny90cmFpbiI6ZmFsc2UsIm93bC0xNjE3MTQ3MDYxLjc3MjQzNTcvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE3MTQ3OTM4LjAxMDEwNTQvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNzE0NzkzOC4wMTAxMDU0L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNzE0ODgwMC40MDIzMDM3L3RyYWluIjpmYWxzZSwib3dsLTE2MTcxNDg4MDAuNDAyMzAzNy92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTcxNDk2MzYuMTExNTE0L3RyYWluIjpmYWxzZSwib3dsLTE2MTcxNDk2MzYuMTExNTE0L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNzE1MDUwNS41MDM2MzA0L3RyYWluIjpmYWxzZSwib3dsLTE2MTcxNTA1MDUuNTAzNjMwNC92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTcxNTEyNjYuNDkyNzQxL3RyYWluIjpmYWxzZSwib3dsLTE2MTcxNTEyNjYuNDkyNzQxL3ZhbGlkYXRpb24iOmZhbHNlfQ%3D%3D)
```
img_augmentation = keras.Sequential(
    [
        preprocessing.RandomRotation(factor=0.65)
    ]
)
```

#### owl-1617143093.9155948 ``` factor=0.15 ```
#### owl-1617144184.6226113 ``` factor=0.35 ```
#### owl-1617145662.0106604 ``` factor=0.65 ```

![image](https://user-images.githubusercontent.com/80168174/113076081-fe469700-91d6-11eb-86c2-1dbefb4ec722.png)

validation имеет наилучшее качество

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/b_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/b_epoch_loss.svg">

### 2c. Использование случайной части изображения

[Train_c](https://github.com/NikitaShulgan/Laba4/blob/main/train_c.py)

```
def augment(image, label):
  crop = tf.image.random_crop(image, [RESIZE_TO, RESIZE_TO, 3])
  return crop, label
  
example['image'] = tf.image.resize(example['image'], tf.constant([250, 250]))

```

#### owl-1617147061.7724357 ``` tf.constant([225, 225])) ```
#### owl-1617147938.0101054 ``` tf.constant([235, 235])) ```
#### owl-1617148800.4023037 ``` tf.constant([250, 250])) ```

![image](https://user-images.githubusercontent.com/80168174/113076731-5b8f1800-91d8-11eb-9543-a1c4ff351b5d.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/c_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/c_epoch_loss.svg">

### 2d. Добавление случайного шума

[Train_d](https://github.com/NikitaShulgan/Laba4/blob/main/train_d.py)

```
x = tf.keras.layers.GaussianNoise(stddev=0.3)(inputs)
```

#### owl-1617149636.111514 ``` GaussianNoise(stddev=0.1) ```
#### owl-1617150505.5036304 ``` GaussianNoise(stddev=0.2) ```
#### owl-1617151266.492741 ``` GaussianNoise(stddev=0.3) ```

![image](https://user-images.githubusercontent.com/80168174/113077161-48307c80-91d9-11eb-8a14-ea98fca8556e.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/d_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/d_epoch_loss.svg">

### 4. Обучить нейронную сеть с использованием оптимальных техник аугментации данных 2a-d совместно

[Train_4](https://github.com/NikitaShulgan/Laba4/blob/main/train_4.py)

```
img_augmentation = keras.Sequential(
    [
        preprocessing.RandomRotation(factor=0.3)
    ]
)

def augment(image, label):
  bright = tf.image.adjust_brightness(image, delta=0.1)
  contrast = tf.image.adjust_contrast(bright, contrast_factor=2)
  crop = tf.image.random_crop(contrast, [RESIZE_TO, RESIZE_TO, 3])
  return crop, label
  
example['image'] = tf.image.resize(example['image'], tf.constant([230, 230]))

x = tf.keras.layers.GaussianNoise(stddev=0.05)(inputs)
```

#### owl-1617137192.8869357 

```
preprocessing.RandomRotation(factor=0.95)
bright = tf.image.adjust_brightness(image, delta=0.5)
contrast = tf.image.adjust_contrast(bright, contrast_factor=5)
tf.constant([230, 230]))
GaussianNoise(stddev=0.3)(inputs)
```

#### owl-1617138108.3832264

```
preprocessing.RandomRotation(factor=0.2)
bright = tf.image.adjust_brightness(image, delta=0.2)
contrast = tf.image.adjust_contrast(bright, contrast_factor=2)
tf.constant([230, 230])
GaussianNoise(stddev=0.1)
```

#### owl-1617139009.7203932

```
preprocessing.RandomRotation(factor=0.3)
bright = tf.image.adjust_brightness(image, delta=0.1)
contrast = tf.image.adjust_contrast(bright, contrast_factor=2)
tf.constant([230, 230]))
GaussianNoise(stddev=0.05)(inputs)
```

![image](https://user-images.githubusercontent.com/80168174/113077731-5fbc3500-91da-11eb-95da-71d7c7b0db3d.png)

у validation наилучшее значение

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/4_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/4_epoch_loss.svg">
