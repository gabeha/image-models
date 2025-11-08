#!/usr/bin/env python3
"""
BiRefNet Background Removal Script

This script uses BiRefNet to remove backgrounds from images.
Usage: python main.py <input_image_path> [output_image_path]
"""

import argparse
import os
import sys
from PIL import Image
import torch
from torchvision import transforms

# Add parent directory to path for BiRefNet imports
sys.path.insert(0, "./")


def setup_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load and setup the BiRefNet model from Hugging Face.

    Args:
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, device)
    """
    try:
        from transformers import AutoModelForImageSegmentation

        print(f"Loading BiRefNet model on {device}...")

        # Load the pre-trained BiRefNet model from Hugging Face
        model = AutoModelForImageSegmentation.from_pretrained(
            "zhengpeng7/BiRefNet", trust_remote_code=True
        )

        torch.set_float32_matmul_precision("high")

        model.to(device)
        model.eval()

        print("BiRefNet model loaded successfully!")
        return model, device

    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have the required dependencies installed:")
        print("pip install torch torchvision transformers pillow")
        sys.exit(1)


def create_transform():
    """
    Create the image transformation pipeline for BiRefNet.

    Returns:
        Transform composition
    """
    return transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def remove_background(image_path, output_path=None, model=None, device=None):
    """
    Remove background from an image using BiRefNet.

    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (optional)
        model: Pre-loaded BiRefNet model (optional)
        device: Device to run inference on (optional)

    Returns:
        PIL Image with transparent background
    """
    # Load model if not provided
    if model is None or device is None:
        model, device = setup_model()

    # Load and prepare the image
    print(f"Processing image: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path)
    image = image.convert("RGB") if image.mode != "RGB" else image
    original_size = image.size

    # Transform the image
    transform = create_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    print("Running inference...")
    # Use appropriate dtype based on device (CPU doesn't support float16 well)
    if device == "cuda":
        with torch.amp.autocast(
            device_type=device, dtype=torch.float16
        ), torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().to(torch.float32).cpu()
    else:
        # On CPU, don't use autocast to avoid dtype issues
        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().to(torch.float32).cpu()

    pred = preds[0].squeeze()

    # Convert prediction to PIL Image
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil_resized = pred_pil.resize(original_size, Image.LANCZOS)

    # Create image with transparent background
    image_with_alpha = image.copy()
    image_with_alpha.putalpha(pred_pil_resized)

    # Save the result
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + "_no_bg.png"

    image_with_alpha.save(output_path, "PNG")
    print(f"Background removed! Output saved to: {output_path}")

    return image_with_alpha


def main():
    """Main function to handle command-line arguments and process images."""
    parser = argparse.ArgumentParser(
        description="Remove background from images using BiRefNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.jpg
  python main.py input.jpg output.png
  python main.py image.png -o result.png
        """,
    )

    parser.add_argument("input", help="Path to the input image")

    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to the output image (optional, defaults to <input>_no_bg.png)",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        dest="output_alt",
        help="Alternative way to specify output path",
    )

    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage instead of GPU"
    )

    args = parser.parse_args()

    # Determine output path
    output_path = args.output or args.output_alt

    # Determine device
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model once
    model, device = setup_model(device)

    # Process the image
    remove_background(args.input, output_path, model, device)


if __name__ == "__main__":
    main()
