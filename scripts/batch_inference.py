import time
from GroundingDINO.groundingdino.util.inference import batch_predict, preprocess_caption
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


def pre_process_batch_images(
    images: list[Image.Image],
) -> tuple[list[np.ndarray], torch.Tensor]:
    transform = T.Compose(
        [
            Resize((800, 1200)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    images_np = [np.asarray(image) for image in images]
    tensors_list = [transform(img, None)[0] for img in images]
    images_tensor = torch.stack(tensors_list)
    return images_np, images_tensor


def main(grounding_model, images_tensor: torch.tensor, text_dict: dict=None):
    boxes_list, logits_list, phrases_list = batch_predict(
        grounding_model,
        images_tensor,
        caption=CAPTION,
        box_threshold=0.3,
        text_threshold=0.25,
        device="cuda",
        text_dict=text_dict,
    )




if __name__ == "__main__":
    image_file = "/home/farouk-gpu/Grounded-Segment-Anything/test-batching/data/a7d6a974-652b-4818-b7db-8fb6e9dbc896.jpg"
    batch_size = 8
    device = "cuda"
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_MODEL,
        device=device,
    ).to(device)
    
    
    caption = preprocess_caption(CAPTION)
    
    
    
    image_files = [image_file] * batch_size
    images = [Image.open(image_file).convert("RGB") for image_file in image_files]
    images_np, images_tensor = pre_process_batch_images(images)
    # text_dict = grounding_model.encode_caption(captions=[caption]*len(images), device=device)
    print("###### Running with Batches And Cachet Text Dict #############")
    for _ in range(5):
        start_inference = time.time()
        main(grounding_model, images_tensor, text_dict=text_dict)
        end_inference = time.time()
        print(f"Inference completed in {end_inference - start_inference:.2f} seconds.")
        
        
    print("###### Running with Batches Without Cachet Text Dict #############")
    for _ in range(5):
        start_inference = time.time()
        main(grounding_model, images_tensor, text_dict=None)
        end_inference = time.time()
        print(f"Inference completed in {end_inference - start_inference:.2f} seconds.") 
        
    print("###### Running with one image at time #############")

    image_files = [image_file] 
    images = [Image.open(image_file).convert("RGB") for image_file in image_files]
    text_dict = grounding_model.encode_caption(captions=[caption]*len(images), device=device)
    images_np, images_tensor = pre_process_batch_images(images)
    for _ in range(5):
        start_inference = time.time()
        for _ in range(batch_size):
            main(grounding_model, images_tensor, text_dict=text_dict)
        end_inference = time.time()
        print(f"Inference completed in {end_inference - start_inference:.2f} seconds.")