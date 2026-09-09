from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def strong_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandAugment(num_ops=6, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def weak_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def strong_view1_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
        transforms.RandAugment(num_ops=4, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def strong_view2_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandAugment(num_ops=6, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
