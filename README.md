# BiRefNet Background Removal

This project uses BiRefNet (Bilateral Reference for High-Resolution Dichotomous Image Segmentation) to remove backgrounds from images.

## Installation

### 1. Create a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: On first run, the BiRefNet model (444MB) will be automatically downloaded from Hugging Face and cached locally.

## Usage

### Basic Usage

Remove background from an image (output will be saved as `<input>_no_bg.png`):

```bash
python main.py input.jpg
```

### Specify Output Path

```bash
python main.py input.jpg output.png
```

Or using the `-o` flag:

```bash
python main.py input.jpg -o output.png
```

### Force CPU Usage

If you want to use CPU instead of GPU:

```bash
python main.py input.jpg --cpu
```

## How It Works

1. **Model Loading**: The script loads the pre-trained BiRefNet model from Hugging Face (`zhengpeng7/BiRefNet`)
2. **Image Preprocessing**: Input images are resized to 1024x1024 and normalized
3. **Inference**: The model predicts a mask that separates foreground from background
4. **Post-processing**: The mask is applied to the original image as an alpha channel
5. **Output**: The result is saved as a PNG with transparent background

## Features

- Automatic GPU/CPU detection and usage
- Supports various image formats (JPG, PNG, etc.)
- Outputs PNG with transparent background
- High-quality segmentation using BiRefNet
- Command-line interface for easy usage

## Model Information

BiRefNet implements Bilateral Reference for High-Resolution Dichotomous Image Segmentation, offering an efficient approach to segmenting images with distinct foreground and background.

- **Model**: zhengpeng7/BiRefNet
- **Paper**: [BiRefNet GitHub Repository](https://github.com/zhengpeng7/BiRefNet)
- **Performance**: Optimized for high-resolution image segmentation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended for faster processing)

## Examples

```bash
# Remove background from a photo
python main.py photo.jpg

# Process multiple images
python main.py image1.jpg -o result1.png
python main.py image2.png -o result2.png

# Use CPU only (if no GPU available)
python main.py portrait.jpg --cpu
```

## Output

The output image will have:

- Original image content with background removed
- Transparent background (alpha channel)
- PNG format for transparency support
- Same resolution as the input image
