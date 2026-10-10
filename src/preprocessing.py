from PIL import Image
import torchvision.transforms as transforms


TARGET_HEIGHT = 64


class ResizeKeepAspectRatio:

    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, image):

        width, height = image.size

        new_width = int(
            width * self.target_height / height
        )

        image = image.resize(
            (new_width, self.target_height),
            Image.Resampling.BILINEAR
        )

        return image


transform = transforms.Compose([
    ResizeKeepAspectRatio(TARGET_HEIGHT),

    transforms.ToTensor(),

    # Convert roughly from [0, 1] to [-1, 1]
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])