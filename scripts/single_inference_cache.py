import time

from GroundingDINO.groundingdino.util.inference import (
    predict,
    batch_predict,
    preprocess_caption,
)
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.misc import clean_state_dict
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from PIL import Image
import torch
import GroundingDINO.groundingdino.datasets.transforms as T
import numpy as np


GROUNDING_DINO_CONFIG = (
    "/home/farouk-gpu/zia-vision/zia_vision/deploy/segmenter/grounding_dino_config.py"
)
GROUNDING_DINO_MODEL = "/home/farouk-gpu/models/checkpoint_best_regular-2.pth"
CAPTION = "single . multipack . price . promo ."


def load_model(
    model_config_path: str, model_checkpoint_path: str, device: str = "cuda"
):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(
        model_checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


class Resize(object):
    def __init__(self, size):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        return T.resize(img, target, self.size)


def pre_process_image(image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            # Resize((800, 1200)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_np: np.ndarray = np.asarray(image)
    image_transformed, _ = transform(image, None)
    return image_np, image_transformed


def main(grounding_model, image_tensor: torch.tensor, text_dict: dict = None):
    boxes_list, logits_list, phrases_list = predict(
        grounding_model,
        image_tensor,
        caption=CAPTION,
        box_threshold=0.3,
        text_threshold=0.25,
        text_dict=text_dict,
        device="cuda",
    )


if __name__ == "__main__":

    image_file = "/home/farouk-gpu/Grounded-Segment-Anything/scripts/data/a7d6a974-652b-4818-b7db-8fb6e9dbc896.jpg"
    batch_size = 4
    num_requests = 2
    device = "cuda"

    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_MODEL,
        device=device,
    ).to(device)

    grounding_model.eval()
    image = Image.open(image_file).convert("RGB")
    image_np, image_tensor = pre_process_image(image)
    image_tensor = image_tensor.to(device)

    caption = preprocess_caption(caption=CAPTION)
    text_dict_cached = grounding_model.encode_captions([caption], device=device)

    num_requests = 32
    start_time = time.time()
    for _ in range(num_requests):
        main(grounding_model, image_tensor, text_dict=None)
    print(f"Running inference with single image and no cache {time.time()-start_time}")

    start_time = time.time()
    caption = preprocess_caption(CAPTION)
    cached_text_dict = grounding_model.encode_captions(
        captions=[caption], device=device
    )

    for _ in range(num_requests):
        main(grounding_model, image_tensor, text_dict=cached_text_dict)
    print(f"Running inference with single image and no cache {time.time()-start_time}")
