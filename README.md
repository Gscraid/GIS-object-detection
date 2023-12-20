# GIS-object-detection
This project uses a neural net to find all geographic object on a picture of the map.

Перед началом работы нужно установить все необходимые библиотеки:

python -m pip install -r requirements.txt

В каталоге images размещаются изображения для обучения,а в каталоге annotations - 
разметка к этим изображениям. Разметка осуществляется при помощи сайта cvat.ai в формате COCO.

Для обучения нейросети нужно воспользоваться либо скриптом train.py либо ноутбуком Train_and_inference.ipynb.

python train.py --classes 35 --epochs 100

В результате обучения модель разместиться в каталоге models.

Для запуска приложения нужно ввести в командную строку:

python app.py

Результаты работы программы сохраняются в каталоге results.