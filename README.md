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
  - Support for trigger keywords
  - Manual caption override capability

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
python dataset.py source_images_dir output_name \
  --base_model "path/to/base/model" \
  --text_encoder "path/to/text/encoder" \
  --clip_model "path/to/clip/model" \
  --vae_model "path/to/vae/model"
```

Full options:
```bash
python dataset.py source_images_dir output_name \
  --base_model "path/to/base/model" \
  --text_encoder "path/to/text/encoder" \
  --clip_model "path/to/clip/model" \
  --vae_model "path/to/vae/model" \
  --trigger_keyword "your_trigger_word" \
  --auto_caption \
  --image_size 512 \
  --num_repeats 10 \
  --max_train_epochs 16 \
  --save_every_n_epochs 4 \
  --vram "12G" \
  --max_images 150
```

### Arguments

Required Arguments:
- `source_dir`: Directory containing source images
- `output_name`: Name for the output folder and LoRA model
- `--base_model`: Path to the base model
- `--text_encoder`: Path to the text encoder model
- `--clip_model`: Path to the CLIP model
- `--vae_model`: Path to the VAE model

Optional Arguments:
- `--trigger_keyword`: Optional trigger keyword for captions
- `--auto_caption`: Enable automatic image captioning
- `--image_size`: Target image size (default: 512)
- `--num_repeats`: Number of times to repeat the dataset (default: 10)
- `--max_train_epochs`: Maximum number of training epochs (default: 16)
- `--save_every_n_epochs`: Save model every N epochs (default: 4)
- `--vram`: GPU VRAM size configuration (choices: "12G", "16G", "24G", default: "12G")
- `--max_images`: Maximum number of images to process (default: 150)

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

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
