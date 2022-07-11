import cv2
import window
import torch
import torchvision

@torch.no_grad()
def infer(img):
    global model
    img_tensor = torchvision.transforms.ToTensor()(img)
    return model([img_tensor.to(device)])

def get_boxes(prediction, score, iou) -> list:
    score_idxs = [i for i, v in enumerate(prediction[0]["scores"]) if v > score]
    iou_idxs = torchvision.ops.batched_nms(
        prediction[0]["boxes"], 
        prediction[0]["scores"],
        prediction[0]["labels"], 
        iou).tolist()
    return list(set(score_idxs).intersection(set(iou_idxs)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint = torch.load("model.pt")
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
    sizes = ((32, 64, 128, 256, 512),),
    aspect_ratios = ((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names = ['0'],
    output_size = 7,
    sampling_ratio = 2)
model = torchvision.models.detection.FasterRCNN(
    backbone = backbone,
    num_classes = 2,
    rpn_anchor_generator = anchor_generator,
    box_roi_pool= roi_pooler)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()
d2r_window = window.Window("Diablo II: Resurrected")

while True:
    img = d2r_window.capture()
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    prediction = infer(img)
    idxs = get_boxes(prediction, 0.4, 0.2)

    img = torch.Tensor(img)
    drawn_boxes = torchvision.utils.draw_bounding_boxes(
        img.type(torch.uint8).permute(2, 0, 1),
        prediction[0]["boxes"][idxs], 
        colors="red")
    img = cv2.cvtColor(drawn_boxes.numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    cv2.imshow('output', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

'''
    img, _ = test[0]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    
    img = read_image(test.images[0])

    drawn_boxes = draw_bounding_boxes(img, prediction[0]["boxes"][0:5], colors="red")
    drawn_boxes = ToPILImage()(drawn_boxes.to('cpu'))
    drawn_boxes.show()
'''