import json
import numpy as np
import matplotlib.pyplot as plt


id_to_ru_category = {
      1: "Памятники архитектуры",
      2: "Музеи",
      3: "Церкви",
      4: "Мечеть",
      5: "Крупные библиотеки",
      6: "Театры",
      7: "Филармония",
      8: "Дворцы и дома культуры",
      9: "Кинотеатры",
      10: "Спортивные сооружения",
      11: "Стадионы",
      12: "Органы закон. власти",
      13: "Гос. совет",
      14: "Гор. администрация",
      15: "Мед. учреждения",
      16: "Исполкомы районов города",
      17: "ВУЗы",
      18: "Средние спец. уч. заведения",
      19: "Гостиницы",
      20: "Рестораны",
      21: "Кафе",
      22: "Торговые центры",
      23: "Рынки",
      24: "Доб быта",
      25: "Почтомат",
      26: "Кассы продажи билетов",
      27: "ЖД вокзал",
      28: "Автовокзал",
      29: "Платные стоянки",
      30: "Трансагенство",
      31: "Туристические организации",
      32: "Центр. переговорный пункт",
      33: "Развлекательные центры",
      34: "Цирк",
      35: "Агенство воздушных сообщений"
    }

def results_to_json(predictions, json_path_file):
    predictions_dict = {}
    num_objects = predictions["labels"].shape[0]
    for i in range(num_objects):
        predictions_dict[i] = {
            "boxes": predictions["boxes"][i].tolist(),
            "label": predictions["labels"][i].tolist(),
            "score": predictions["scores"][i].tolist(),
            "category": id_to_ru_category[predictions["labels"][i]]
        }
    with open(json_path_file, "w", encoding="utf8") as outfile:
        json.dump(predictions_dict, outfile, ensure_ascii=False)

def visualize_detections(
    image, boxes, classes, scores, figsize=(50, 50), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        if (score > 0.9):
          text = "{}: {:.2f}".format(id_to_ru_category[_cls], score)
          x1, y1, x2, y2 = box
          w, h = x2 - x1, y2 - y1
          patch = plt.Rectangle(
              [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
          )
          ax.add_patch(patch)
          ax.text(
              x1,
              y1,
              text,
              bbox={"facecolor": color, "alpha": 0.4},
              clip_box=ax.clipbox,
              clip_on=True,
          )
    plt.savefig('results/result.png')
    plt.show()
    return ax