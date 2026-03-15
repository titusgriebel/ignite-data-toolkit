# Generic imports
import torch
import os
from glob import glob
import imageio.v3 as imageio
import numpy as np
from tqdm import tqdm
from yolov5.utils.general import non_max_suppression, xyxy2xywh

CLASSES = [
    "pd-l1 negative tumor cell",
    "pd-l1 positive tumor cell",
    "non-tumor cell",
]

CONFIG = {
    "pdl1": {
        "tile_size": 512,
        "conf_thres": 0.5,
        "iou_thres": 0.6,
        "model_path": "pdl1/pdl1/pdl1_detection_weights.pt"
    },
    "nuclei": {
        "tile_size": 128,
        "conf_thres": 0.6,
        "iou_thres": 0.4,
        "model_path": "pdl1/nuclei/nuclei_detection_weights.pt"
    }
}


def _process_image(image, model, tile_size=128, conf_thres=0.5, iou_thres=0.5, max_det=1500):

    stride = tile_size

    _, height, width = image.shape
    tiles = []

    # Generate tiles
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            tile = image[:, y: y + tile_size, x: x + tile_size]
            tiles.append(tile)

    tiles = torch.stack(tiles)

    pred = model(tiles)

    pred = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=None,
        agnostic=True,
        max_det=max_det,
    )

    # Process predictions for each tile
    image_output = []
    for i, p in enumerate(pred):
        if p is not None:
            xyxy, _, c = p[:, :4], p[:, 4], p[:, 5]
            xywh = xyxy2xywh(xyxy)
            xyc = torch.cat(
                (xywh[:, :2], c.unsqueeze(1)), dim=1
            ).cpu()  # Centered (x,y) coordinates + class

            # Adjust coordinates to global image space
            tile_x = (i % (width // stride)) * stride
            tile_y = (i // (width // stride)) * stride
            xyc[:, 0] += tile_x
            xyc[:, 1] += tile_y

            image_output.append(xyc)

    output = torch.cat(image_output, dim=0) if image_output else torch.empty((0, 3))

    return output


def infer_on_image_pdl1_positive(model_root, image_dir, output_dir, pred_target):

    # Make output folders if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(os.path.join(image_dir, "*"))

    model_path = os.path.join(model_root, CONFIG[pred_target]["model_path"])

    # Initialize model
    torch_hub_dir = os.environ.get("TORCH_HOME", None)
    if torch_hub_dir:
        torch.hub.set_dir(torch_hub_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Predicting using {device} device")

    model = torch.hub.load(
        "yolov5",
        "custom",
        path=model_path,
        trust_repo=True,
        source="local",
        force_reload=False,
    ).to(device)

    for image_path in tqdm(image_paths):
        img = torch.from_numpy(imageio.imread(image_path)).permute(2, 0, 1).to(device)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        xyc = _process_image(
            img,
            model,
            tile_size=CONFIG[pred_target]["tile_size"],
            conf_thres=CONFIG[pred_target]["conf_thres"],
            iou_thres=CONFIG[pred_target]["iou_thres"],
        )
        xyc = xyc.cpu().numpy()
        np.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{pred_target}.npy"), xyc)
