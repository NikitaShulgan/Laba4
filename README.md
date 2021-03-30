# Лабораторная работа #4.
## Использование техник аугментации данных для улучшения сходимости процесса обучения нейронной сети на примере решения задачи классификации Oregon Wildlife
### 2a. Манипуляции с яркостью и контрастом
```
def augment(image, label):
  bright = tf.image.adjust_brightness(image, delta=0.5)
  contrast = tf.image.adjust_contrast(bright, contrast_factor=5)
  return contrast, label
```
#### owl-1616962413.4904904 ```delta=0.1, contrast_factor=2 ```
#### owl-1616963444.50751 ```delta=0.2, contrast_factor=3 ```
#### owl-1616964737.1780863 ```delta=0.5, contrast_factor=5 ```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/a_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba4/main/For_Readme/a_epoch_loss.svg">
