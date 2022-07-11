import torch
import torchvision
from engine import train_one_epoch, evaluate
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, ToPILImage
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import voc
import sys
import os

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def init_model(device):
    num_classes = 2
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes = ((32, 64, 128, 256, 512),),
        aspect_ratios = ((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names = ['0'],
        output_size = 7,
        sampling_ratio = 2)
    model = FasterRCNN(
        backbone,
        num_classes = 2,
        rpn_anchor_generator = anchor_generator,
        box_roi_pool= roi_pooler)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr = 0.005,
        momentum = 0.9, 
        weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = 3,
        gamma = 0.1)
    return model, optimizer, lr_scheduler

def main() -> int:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, optimizer, lr_scheduler = init_model(device)
    if os.path.exists("model.pt"):
        checkpoint = torch.load("model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"loaded from checkpoint: {epoch}")

    data = voc.VOCCustom("train", transforms = get_transform(train=False))
    test = voc.VOCCustom("test", transforms = get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size = 2,
        shuffle = True,
        num_workers = 4,
        collate_fn = collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size = 2,
        shuffle = True,
        num_workers = 4,
        collate_fn = collate_fn)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, test_loader, device=device)

    torch.save({
        "epoch": num_epochs + epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": 0.4},
        "model.pt")

if __name__ == '__main__':
    sys.exit(main())