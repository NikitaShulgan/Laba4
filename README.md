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
```

```
####  ``` ```
#### owl-1617140912.2121003 ```delta=0.2, contrast_factor=2 ```
#### owl-1617141971.0594199 ```delta=0.3, contrast_factor=3 ```
