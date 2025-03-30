import os
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# Import helper functions (engine.py, utils.py, etc.) from TorchVision tutorial.
import engine
import utils

# ---------------------------------------------------------------------
# 1. Custom Dataset Definition (Using PennFudan Dataset)
# ---------------------------------------------------------------------
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        image = torchvision.io.read_image(img_path)
        mask = torchvision.io.read_image(mask_path)

        # Get all object IDs (skip background - assumed as first unique value)
        obj_ids = torch.unique(mask)[1:]
        num_objs = len(obj_ids)

        # Create binary masks for each object using broadcasting technique
        masks = (mask == obj_ids[:, None, None]).to(torch.uint8)

        # Get bounding boxes from binary masks
        boxes = torchvision.ops.masks_to_boxes(masks)

        # Only one class: person (assign label 1)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.imgs)

# ---------------------------------------------------------------------
# 2. Data Transformations using the New Transforms API
# ---------------------------------------------------------------------
from torchvision.transforms.v2 import functional as F2
import torchvision.transforms.v2 as T2

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T2.RandomHorizontalFlip(0.5))
    transforms.append(T2.ToDtype(torch.float, scale=True))
    transforms.append(T2.ToPureTensor())
    return T2.Compose(transforms)

# ---------------------------------------------------------------------
# 3. Model Definitions: Faster R-CNN with Different Backbones
# ---------------------------------------------------------------------
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes, variant="resnet50"):
    if variant == "resnet50":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    elif variant == "mobilenet_v2":
        # Create a backbone using MobileNetV2 features
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280  # Needed for Faster R-CNN construction
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )
        model = FasterRCNN(backbone, num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    else:
        raise ValueError("Invalid model variant specified!")
    
    if variant == "resnet50":
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # For Mask R-CNN, replace the mask predictor too
        if hasattr(model.roi_heads, "mask_predictor"):
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# ---------------------------------------------------------------------
# 4. Training and Evaluation Pipeline
# ---------------------------------------------------------------------
def train_model(variant="resnet50", num_epochs=2):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 2  # Only one object class: person (plus background)

    # Load dataset
    dataset = PennFudanDataset("data/PennFudanPed", get_transform(train=True))
    dataset_test = PennFudanDataset("data/PennFudanPed", get_transform(train=False))

    # Split dataset into train/test (reserve 50 images for testing)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = Subset(dataset, indices[:-50])
    dataset_test = Subset(dataset_test, indices[-50:])

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes, variant)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Fine-tuning loop
    for epoch in range(num_epochs):
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        eval_results = engine.evaluate(model, data_loader_test, device=device)
        print(f"Epoch {epoch} evaluation:\n{eval_results}")
    
    return model

# ---------------------------------------------------------------------
# 5. HOG-Based Detector (with Possible Parameter Tuning)
# ---------------------------------------------------------------------
def hog_detector(img_path, scale=1.05, winStride=(8, 8), padding=(8, 8)):
    """
    Runs an OpenCV HOG-based pedestrian detector on the given image.
    The parameters (winStride, padding, scale) can be tuned for optimal performance.
    """
    img = cv2.imread(img_path)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(gray, winStride=winStride, padding=padding, scale=scale)

    for (x, y, w, h), weight in zip(rects, weights):
        # Draw bounding boxes and display confidence scores
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, f"{weight:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    img_result = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_result

# (Optional) Custom evaluation for HOG-based detections could be implemented here.
# For a fair comparison, one would compute IoU metrics against ground truth boxes from PennFudan.

# ---------------------------------------------------------------------
# 6. Main Routine: Training, Testing, and Visual Comparison
# ---------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate and Compare Object Detection Methods on the PennFudan Dataset"
    )
    parser.add_argument("--image", type=str, default="data/PennFudanPed/PNGImages/FudanPed00046.png",
                        help="Path to a sample image for visual comparison")
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs")
    parser.add_argument("--variant", type=str, default="resnet50", choices=["resnet50", "mobilenet_v2"],
                        help="Select Faster R-CNN variant (baseline or alternative backbone)")
    args = parser.parse_args()

    # -----------------------------------------------------------
    # 6.1 Fine-tune and Evaluate Faster R-CNN Models
    # -----------------------------------------------------------
    print(f"Training Faster R-CNN with variant '{args.variant}' for {args.epochs} epochs...")
    model = train_model(variant=args.variant, num_epochs=args.epochs)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    img = Image.open(args.image).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).to(device)
    with torch.no_grad():
        prediction = model([img_tensor])
    prediction = [{k: v.cpu() for k, v in t.items()} for t in prediction][0]

    # Draw predictions on the image (only display boxes with a score > 0.8)
    img_np = np.array(img)
    for box, score in zip(prediction["boxes"], prediction["scores"]):
        if score > 0.8:
            box = box.numpy().astype(int)
            cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(img_np, f"{score:.2f}", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # -----------------------------------------------------------
    # 6.2 Run HOG Detection on the Sample Image
    # -----------------------------------------------------------
    hog_result = hog_detector(args.image, scale=1.05, winStride=(8, 8), padding=(8, 8))

    # -----------------------------------------------------------
    # 6.3 Visual Comparison: Display Results Side by Side
    # -----------------------------------------------------------
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Fine-Tuned Faster R-CNN Detection")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("HOG-based Detection")
    plt.imshow(hog_result)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()