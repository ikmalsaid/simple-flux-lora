# simple-flux-lora

A simple utility to prepare image and caption datasets for Flux LoRA training using kohya_ss scripts. This tool helps automate the process of preparing training datasets for LoRA models with features like automatic image resizing, caption generation, and training script generation.

## Features

- **Automatic Image Processing**
  - Resizes images while maintaining aspect ratio
  - Supports multiple image formats (PNG, JPG, JPEG, WebP, BMP)
  - Limits maximum images to prevent VRAM issues

- **Flexible Caption Handling**
  - Uses existing caption files (.txt) if available
  - Optional automatic caption generation using Florence-2-large model
  - Support for trigger keywords (required)
  - Manual caption options:
    - Load from text file (one caption per line)
    - Provide as a list when using Python API
  - Mutually exclusive caption methods (only one can be used at a time)

- **Training Configuration**
  - Automatic generation of dataset.toml
  - Creates optimized training scripts based on VRAM size
  - Configurable training parameters
  - Support for different VRAM configurations (12GB, 16GB, 24GB+)

- **Quality of Life Features**
  - Detailed logging of operations
  - Progress tracking
  - Error handling and validation
  - Performance statistics
  - Immediate processing and saving of images/captions

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ikmalsaid/simple-flux-lora.git
   cd simple-flux-lora
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   ```

2. Install required dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Basic usage:
```bash
python app.py source_images_dir output_name \
  --trigger_keyword "your_trigger_word" \
  --base_model "path/to/base/model" \
  --text_encoder "path/to/text/encoder" \
  --clip_model "path/to/clip/model" \
  --vae_model "path/to/vae/model"
```

Full options:
```bash
python app.py source_images_dir output_name \
  --trigger_keyword "your_trigger_word" \
  --base_model "path/to/base/model" \
  --text_encoder "path/to/text/encoder" \
  --clip_model "path/to/clip/model" \
  --vae_model "path/to/vae/model" \
  --auto_caption \
  --caption_file "path/to/captions.txt" \
  --image_size 512 \
  --num_repeats 10 \
  --max_train_epochs 16 \
  --save_every_n_epochs 4 \
  --vram "12G" \
  --max_images 150
```

### Python API Usage

```python
from simple_flux_lora import SimpleFluxLora

lora = SimpleFluxLora()

# Using auto-caption
lora.create_dataset(
    source_image_dir="path/to/images",
    output_name="my_dataset",
    trigger_keyword="my_trigger",
    auto_caption=True,
    ...
)

# Using manual captions list
lora.create_dataset(
    source_image_dir="path/to/images",
    output_name="my_dataset",
    trigger_keyword="my_trigger",
    manual_captions=["caption1", "caption2", "caption3"],
    ...
)

# Using caption file
lora.create_dataset(
    source_image_dir="path/to/images",
    output_name="my_dataset",
    trigger_keyword="my_trigger",
    caption_file="path/to/captions.txt",
    ...
)
```

### Arguments

Required Arguments:
- `source_dir`: Directory containing source images
- `output_name`: Name for the output folder and LoRA model
- `--trigger_keyword`: Required trigger keyword for captions
- `--base_model`: Path to the base model
- `--text_encoder`: Path to the text encoder model
- `--clip_model`: Path to the CLIP model
- `--vae_model`: Path to the VAE model

Optional Arguments:
- `--auto_caption`: Enable automatic image captioning
- `--caption_file`: Path to a text file containing manual captions (one per line)
- `--image_size`: Target image size (default: 512)
- `--num_repeats`: Number of times to repeat the dataset (default: 10)
- `--max_train_epochs`: Maximum number of training epochs (default: 16)
- `--save_every_n_epochs`: Save model every N epochs (default: 4)
- `--vram`: GPU VRAM size configuration (choices: "12G", "16G", "24G", default: "12G")
- `--max_images`: Maximum number of images to process (default: 150)

Note: `--auto_caption` and `--caption_file` are mutually exclusive. Only one caption method can be used at a time.

## Output Structure

The tool creates the following structure in your output directory:
```
output_name/
├── dataset/
│   ├── image1.png
│   ├── image1.txt
│   ├── image2.png
│   ├── image2.txt
│   └── ...
├── dataset.toml
└── train.bat
```

## Training

After dataset preparation:

1. Make sure you have kohya_ss scripts installed
   ```bash
   git clone -b sd3 https://github.com/kohya-ss/sd-scripts
   cd sd-scripts
   ```

2. Install required dependencies in the same virtual environment (recommended):
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Navigate to your output directory

4. Run the generated training script:
   ```bash
   train.bat
   ```

## Notes

- The tool automatically handles image resizing while maintaining aspect ratios
- Existing caption files (.txt) in the source directory will be preserved
- Auto-captioning uses the Florence-2-large model for high-quality captions
- Training scripts are optimized based on your specified VRAM configuration
- Maximum image limit helps prevent VRAM issues during training
- Images and captions are processed and saved immediately for better fault tolerance
- Only one caption method can be used at a time (auto-caption, manual captions, or caption file)
- Trigger keyword is now required for all caption methods

## License

See [LICENSE](LICENSE) for details.
