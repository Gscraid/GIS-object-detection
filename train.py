import os
import torch
import torch.utils.data
import torchvision
import numpy as np

from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]["category_id"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def create_model(num_classes, pretrained=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Параметры для запуска приложения по обнаружению объектов')
    parser.add_argument("-c", "--classes", type=int, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True)

    args = parser.parse_args()
    train_data_dir = 'images'
    train_coco = 'annotations/instances_default.json'

    # create own Dataset
    my_dataset = myOwnDataset(root=train_data_dir,
                              annotation=train_coco,
                              transforms=get_transform()
                              )
    # Batch size
    train_batch_size = 2

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 35 classes; Only target class or background
    num_classes = args.classes + 1
    num_epochs = args.epochs
    model = create_model(num_classes)
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    len_dataloader = len(data_loader)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        loss = []
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            loss.append(losses.cpu().detach().numpy())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        loss = np.array(loss).mean()
        if epoch % int(num_epochs / 10) == 0:
          torch.save(model.state_dict(), f'models/epoch_{epoch}_{loss}.pth')
        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}, mean_loss: {loss}')
