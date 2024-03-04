from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import cv2


class ImageToMasks:
    def __init__(self, model_name="vit_h", checkpoint="./models/sam_vit_h_4b8939.pth"):
        # Determine the best available device
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.model_name = model_name
        self.checkpoint = checkpoint
        self.sam = sam_model_registry[self.model_name](checkpoint=self.checkpoint)
        self.sam = self.sam.to(device=self.device)

    def get_masks(self, image, count=5):
        # get the masks and sort by greatest area
        mask_generator = SamAutomaticMaskGenerator(model=self.sam)
        masks = mask_generator.generate(image)
        masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        masks = masks[:count]
        return masks

    def get_cutouts(self, image):
        masks = self.get_masks(image)
        cutouts = []
        for i, mask in enumerate(masks):
            # Create an empty image with the same size as the image
            segmented_img = np.ones_like(image)
            # Apply the mask to copy the segmented object
            for c in range(3):  # Assuming 3 channels (RGB)
                segmented_img[:, :, c] = image[:, :, c] * mask["segmentation"]

            cutouts.append(segmented_img)

        return cutouts

    def save_cutouts(self, image, output_folder):
        cutouts = self.get_cutouts(image)

        for i, cutout in enumerate(cutouts):
            # Save the segmented object image
            filename = f"{output_folder}/segment_{i+1}.png"
            cv2.imwrite(filename, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
            print(f"Saved: {filename}")


if __name__ == "__main__":
    image = cv2.imread("images/telephone_booth.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam = ImageToMasks()
    sam.save_cutouts(image, "./temp")
