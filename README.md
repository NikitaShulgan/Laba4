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

### Итог
Наилучший результат у алгоритма с параметрами яркости и контастности равными ```delta=0.2, contrast_factor=2 ``` соответсвенно. По сравнении с validation улучшение 0.8%.

### 2b. Поворот изображения на случайный угол
[Train_b](https://github.com/NikitaShulgan/Laba4/blob/main/train_b.py)

[Tensorboard]()
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
#### owl-1617169107.6798835 ``` factor=0.05 ```
#### owl-1617170328.0750272 ``` factor=0.01 ```

![image](https://user-images.githubusercontent.com/80168174/113117431-176f3800-9217-11eb-95b1-3814aff23a61.png)

validation имеет наилучшее качество среди оранжевых графиков

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/b_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/b_epoch_loss.svg">

### Итог
Использование ``` RandomRotation ``` в алгоритмах не помогло улучшить результат, а только ухудшило его.

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

### Итог
Лучше всего себя показал алгоритм где было увеличено изображение до 225x225. Улучшение по сравнению с validation 0.36%.

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

### Итог
Лучше всего себя показал алгоритм со стандартным отклонением распределения шума ``` stddev=0.2 ```. В сравнении с validation улучшение 0.26%.

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

#### owl-1617171172.1995904

```
RandomRotation(factor=0.01)
bright = tf.image.adjust_brightness(image, delta=0.3)
contrast = tf.image.adjust_contrast(bright, contrast_factor=3)
tf.constant([225, 225])
GaussianNoise(stddev=0.1)
```

#### owl-1617172100.647061

```
preprocessing.RandomRotation(factor=0.01)
bright = tf.image.adjust_brightness(image, delta=0.2)
contrast = tf.image.adjust_contrast(bright, contrast_factor=2)
tf.constant([225, 225]))
GaussianNoise(stddev=0.05)
```

![image](https://user-images.githubusercontent.com/80168174/113112894-6070bd80-9212-11eb-8da5-fc6064e0647d.png)

у validation наилучшее значение из оранжевых графиков

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/4_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/4_epoch_loss.svg">

### Итог 
Лучше всего себя показал алгоритм owl-1617172100.647061 с параметрами ``` RandomRotation(factor=0.01), delta=0.2, contrast_factor=2, [225, 225], stddev=0.05 ```. Улучшение в сравнении с validation 0.13%.

### Анализ результатов
  Во 3 из 4х случаях мы смогли добиться улучшения результатов. Не получилось добиться улучшение используя ``` RandomRotation ```. В совместном использовании техник аументации удалось добиться улучшения на 0.13%. Неверный подбор параметров может ухудшить результаты алгоритма.
  
