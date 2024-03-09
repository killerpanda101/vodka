from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


class BlipImageCaptioning:
    def __init__(self, model_path="models/BLIP"):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        # Initialize the processor and model
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(
            self.device
        )

    def load_image(self, image_path):
        """
        Loads an image and converts it to RGB.
        """
        return Image.open(image_path).convert("RGB")

    def generate_description(self, image, max_new_tokens=512):
        """
        Processes the image and generates a description.
        """
        inputs = (
            self.processor(image, return_tensors="pt").to(torch.int32).to(self.device)
        )
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        description = self.processor.decode(output[0], skip_special_tokens=True)
        return description


if __name__ == "__main__":
    blip = BlipImageCaptioning()
    img = blip.load_image("images/telephone_booth.jpg")
    description = blip.generate_description(img)
    print(description)
