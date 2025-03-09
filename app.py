import os
import time
import torch
import shutil
import argparse
from PIL import Image
from colorpaws import configure
from transformers import AutoProcessor, AutoModelForCausalLM

class SimpleFluxLora:
    """Copyright (C) 2025 Ikmal Said. All rights reserved."""
    def __init__(self):
        """Initialize SimpleFluxLora"""
        self.logger = configure(self.__class__.__name__, log_on=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16
        self.captioning_model = None
        self.processor = None
        self.logger.info(f"{self.__class__.__name__} is ready!")

    def __setup_output_folder(self, output_name):
        """Create and validate output directory"""
        output_folder = os.path.abspath(output_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder

    def __process_captions(self, images, trigger_keyword, auto_caption, manual_captions):
        """Process and generate captions based on specified method"""
        if manual_captions is not None:
            if len(manual_captions) != len(images):
                raise ValueError(f"Number of manual captions ({len(manual_captions)}) "
                               f"does not match number of images ({len(images)})")
            self.logger.info("Using provided manual captions for all images")
            return manual_captions
            
        self.logger.info(f"Auto-captioning is {'enabled' if auto_caption else 'disabled'}")
        if not auto_caption:
            self.logger.info(f"Using trigger keyword '{trigger_keyword}' for images without existing captions")
            
        # First check for existing caption files
        captions = []
        images_needing_captions = []
        existing_caption_count = 0
        
        for image in images:
            caption_path = os.path.splitext(image)[0] + ".txt"
            if os.path.exists(caption_path):
                with open(caption_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # If file exists and has content
                        existing_caption_count += 1
                        captions.append(content)
                        continue
            
            # No valid caption file found
            if not auto_caption:
                captions.append(trigger_keyword if trigger_keyword else "")
            else:
                images_needing_captions.append(image)
                captions.append(None)
        
        if existing_caption_count > 0:
            self.logger.info(f"Found and using {existing_caption_count} existing caption(s) from source directory")
        
        # Generate new captions only if auto_caption is enabled and there are images needing captions
        if auto_caption and images_needing_captions:
            self.logger.info(f"Generating new captions for {len(images_needing_captions)} image(s) without existing captions...")
            new_captions = self.__generate_captions(images_needing_captions, trigger_keyword)
            # Replace None values with new captions
            caption_idx = 0
            for i in range(len(captions)):
                if captions[i] is None:
                    captions[i] = new_captions[caption_idx]
                    caption_idx += 1
        
        self.logger.info("Final caption distribution:")
        self.logger.info(f"- {existing_caption_count} image(s) using existing captions")
        if auto_caption:
            self.logger.info(f"- {len(images_needing_captions)} image(s) using newly generated captions")
        else:
            self.logger.info(f"- {len(images) - existing_caption_count} image(s) using trigger keyword")
            
        return captions

    def __generate_captions(self, image_paths, trigger_keyword=None):
        """Generate captions for multiple images"""
        
        model_name = "multimodalart/Florence-2-large-no-flash-attn"

        if self.captioning_model is None and self.processor is None:
            self.logger.info(f"Loading captioning model on {self.device}")
            self.captioning_model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=self.torch_dtype, 
                    trust_remote_code=True
                ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(
                    model_name, 
                    trust_remote_code=True
                )
            
        captions = []
        total_images = len(image_paths)
        for i, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Generating caption for image {i}/{total_images}: {os.path.basename(image_path)}")
            caption = self.__generate_single_caption(image_path, trigger_keyword)
            captions.append(caption)
        return captions

    def __generate_single_caption(self, image_path, trigger_keyword=None):
        """Generate caption for a single image"""
        image = Image.open(image_path).convert("RGB")
        prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        
        generated_ids = self.captioning_model.generate(
            input_ids=inputs["input_ids"], 
            pixel_values=inputs["pixel_values"], 
            max_new_tokens=1024, 
            num_beams=3
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        return f"{trigger_keyword} {caption_text}" if trigger_keyword else caption_text

    def __resize_image(self, image_path, output_path, size):
        """Resize an image to the specified size while maintaining aspect ratio"""
        with Image.open(image_path) as img:
            width, height = img.size
            if width < height:
                new_width = size
                new_height = int((size/width) * height)
            else:
                new_height = size
                new_width = int((size/height) * width)
            self.logger.info(f"Resizing {image_path}: {new_width}x{new_height}")
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_resized.save(output_path)

    def __generate_dataset_toml(self):
        """Generate the dataset.toml configuration"""
        return f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {self.image_size}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{os.path.join(os.path.abspath(self.destination_folder), "dataset")}'
  class_tokens = '{self.trigger_keyword}'
  num_repeats = {self.num_repeats}"""

    def __get_optimizer_config(self):
        """Get optimizer configuration based on VRAM size"""
        if self.vram == "16G":
            return """--optimizer_type adafactor ^
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" ^
  --lr_scheduler constant_with_warmup ^
  --max_grad_norm 0.0 ^"""
        elif self.vram == "12G":
            return """--optimizer_type adafactor ^
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" ^
  --split_mode ^
  --network_args "train_blocks=single" ^
  --lr_scheduler constant_with_warmup ^
  --max_grad_norm 0.0 ^"""
        else:  # 20G+
            return """--optimizer_type adamw8bit ^"""

    def __generate_train_script(self, base_model, text_encoder, clip_model, vae_model):
        """Generate the training batch script"""
        dest = os.path.abspath(self.destination_folder)
        dest_name = os.path.basename(dest)
        toml_path = os.path.join(dest, 'dataset.toml')
        optimizer_config = self.__get_optimizer_config()

        return f"""accelerate launch ^
  --mixed_precision bf16 ^
  --num_cpu_threads_per_process 1 ^
  sd-scripts/flux_train_network.py ^
  --pretrained_model_name_or_path "{base_model}" ^
  --clip_l "{clip_model}" ^
  --t5xxl "{text_encoder}" ^
  --ae "{vae_model}" ^
  --cache_latents_to_disk ^
  --save_model_as safetensors ^
  --sdpa --persistent_data_loader_workers ^
  --max_data_loader_n_workers 2 ^
  --seed 42 ^
  --gradient_checkpointing ^
  --mixed_precision bf16 ^
  --save_precision bf16 ^
  --network_module networks.lora_flux ^
  --network_dim 4 ^
  {optimizer_config}
  --learning_rate 8e-4 ^
  --cache_text_encoder_outputs ^
  --cache_text_encoder_outputs_to_disk ^
  --fp8_base ^
  --highvram ^
  --max_train_epochs {self.max_train_epochs} ^
  --save_every_n_epochs {self.save_every_n_epochs} ^
  --dataset_config "{os.path.abspath(toml_path)}" ^
  --output_dir "{dest}" ^
  --output_name "{dest_name}" ^
  --timestep_sampling shift ^
  --discrete_flow_shift 3.1582 ^
  --model_prediction_type raw ^
  --guidance_scale 1 ^
  --loss_type l2"""

    def __create_dataset_files(self, images, captions, output_folder, size):
        """Create dataset files with resized images and captions"""
        dataset_folder = os.path.join(output_folder, "dataset")
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            
        for image, caption in zip(images, captions):
            if not image or not os.path.exists(image):
                continue
                
            # Get source caption file path
            source_caption_path = os.path.splitext(image)[0] + ".txt"
            new_image_path = shutil.copy(image, dataset_folder)
            new_caption_path = os.path.splitext(new_image_path)[0] + ".txt"
            
            # Skip resizing for non-image files
            if not new_image_path.lower().endswith(('.txt')):
                self.__resize_image(new_image_path, new_image_path, size)
            
            # Handle caption: copy existing file if it exists and has content, otherwise create new one
            if os.path.exists(source_caption_path):
                with open(source_caption_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # If source file exists and has content, copy it
                        shutil.copy2(source_caption_path, new_caption_path)
                        continue
            
            # If we get here, either no source caption file exists or it was empty
            if caption:
                with open(new_caption_path, 'w') as f:
                    f.write(caption)

    def __generate_training_configs(self, output_folder, base_model, text_encoder, 
                                  clip_model, vae_model):
        """Generate training configuration files"""
        # Generate dataset.toml
        toml_content = self.__generate_dataset_toml()
        with open(os.path.join(output_folder, "dataset.toml"), 'w', encoding='utf-8') as f:
            f.write(toml_content)
            
        # Generate training script
        script_content = self.__generate_train_script(base_model, text_encoder, 
                                                    clip_model, vae_model)
        script_name = "train.bat" if os.name == 'nt' else "train.sh"
        with open(os.path.join(output_folder, script_name), 'w', encoding='utf-8') as f:
            f.write(script_content)

    def __get_images_from_directory(self, directory):
        """Get list of image files from directory"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        images = []
        
        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")
            
        for file in os.listdir(directory):
            if file.lower().endswith(valid_extensions):
                images.append(os.path.join(directory, file))
                
        images.sort()
        return images

    def create_dataset(self, source_image_dir, output_name, trigger_keyword=None, auto_caption=False,
                       base_model=None, text_encoder=None, clip_model=None, vae_model=None, 
                       manual_captions=None, image_size=512, num_repeats=10, max_train_epochs=16,
                       save_every_n_epochs=4, vram="12G", max_images=150):
        """
        Main function to create a complete dataset with training files
        
        Args:
            source_image_dir: Directory path containing the source images
            output_name: Name for the output folder and lora model
            base_model: Path to base model
            text_encoder: Path to text encoder
            clip_model: Path to CLIP model
            vae_model: Path to VAE model
            trigger_keyword: Optional trigger keyword for captions
            auto_caption: Whether to auto-generate captions
            manual_captions: Optional list of manual captions
            image_size: Target image size
            num_repeats: Number of times to repeat the dataset
            max_train_epochs: Maximum number of training epochs
            save_every_n_epochs: Number of epochs between saving the model
            vram: GPU VRAM size (12G, 16G, 24G)
            max_images: Maximum number of images to process
        """
        start_time = time.time()
        
        self.save_every_n_epochs = save_every_n_epochs
        self.max_train_epochs = max_train_epochs
        self.trigger_keyword = trigger_keyword
        self.destination_folder = output_name
        self.num_repeats = num_repeats
        self.max_images = max_images
        self.image_size = image_size
        self.vram = vram
        
        # Get list of images from directory
        source_images = self.__get_images_from_directory(source_image_dir)
        total_images = len(source_images)
        
        # Validate image count
        if total_images <= 1:
            raise ValueError("Please provide at least 2 images for training")
        if total_images > self.max_images:
            raise ValueError(f"Maximum allowed images is {self.max_images}")
        
        # Continue with existing flow
        output_folder = self.__setup_output_folder(output_name)
        captions = self.__process_captions(source_images, trigger_keyword, 
                                         auto_caption, manual_captions)
        self.__create_dataset_files(source_images, captions, output_folder, image_size)
        self.__generate_training_configs(output_folder, base_model, text_encoder, 
                                       clip_model, vae_model, trigger_keyword)
        
        # Calculate processing time
        total_time = time.time() - start_time
        
        # Generate operation summary
        self.logger.info("-" * 40)
        self.logger.info("Dataset Creation Summary:")
        self.logger.info("-" * 40)
        self.logger.info(f"Total images processed: {total_images}")
        self.logger.info(f"Output directory: {output_folder}")
        self.logger.info(f"Target image size: {image_size}x{image_size}")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Average time per image: {total_time/total_images:.2f} seconds")
        self.logger.info("-" * 40)
        self.logger.info("Configuration Details:")
        self.logger.info("-" * 40)
        self.logger.info(f"Auto-captioning: {'Enabled' if auto_caption else 'Disabled'}")
        self.logger.info(f"Trigger keyword: {trigger_keyword if trigger_keyword else 'None'}")
        self.logger.info(f"Dataset repeats: {num_repeats}")
        self.logger.info(f"Training epochs: {max_train_epochs}")
        self.logger.info(f"Save frequency: Every {save_every_n_epochs} epochs")
        self.logger.info(f"VRAM configuration: {vram}")
        self.logger.info("-" * 40)
        
        return output_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and prepare image datasets for Flux LoRA training")
    
    # Required arguments
    parser.add_argument("source_dir", help="Directory containing source images")
    parser.add_argument("output_name", help="Name for the output folder and LoRA model")
    
    # Model paths
    parser.add_argument("--base_model", required=True, help="Path to the base model")
    parser.add_argument("--text_encoder", required=True, help="Path to the text encoder model")
    parser.add_argument("--clip_model", required=True, help="Path to the CLIP model")
    parser.add_argument("--vae_model", required=True, help="Path to the VAE model")
    
    # Optional arguments
    parser.add_argument("--trigger_keyword", help="Optional trigger keyword for captions")
    parser.add_argument("--auto_caption", action="store_true", help="Enable automatic image captioning")
    parser.add_argument("--image_size", type=int, default=512, help="Target image size (default: 512)")
    parser.add_argument("--num_repeats", type=int, default=10, help="Number of times to repeat the dataset (default: 10)")
    parser.add_argument("--max_train_epochs", type=int, default=16, help="Maximum number of training epochs (default: 16)")
    parser.add_argument("--save_every_n_epochs", type=int, default=4, help="Save model every N epochs (default: 4)")
    parser.add_argument("--vram", choices=["12G", "16G", "24G"], default="12G", help="GPU VRAM size configuration (default: 12G)")
    parser.add_argument("--max_images", type=int, default=150, help="Maximum number of images to process (default: 150)")
    
    args = parser.parse_args()
    
    # Initialize and run dataset creation
    lora = SimpleFluxLora()
    try:
        output_folder = lora.create_dataset(
            source_image_dir=args.source_dir,
            output_name=args.output_name,
            base_model=args.base_model,
            text_encoder=args.text_encoder,
            clip_model=args.clip_model,
            vae_model=args.vae_model,
            trigger_keyword=args.trigger_keyword,
            auto_caption=args.auto_caption,
            image_size=args.image_size,
            num_repeats=args.num_repeats,
            max_train_epochs=args.max_train_epochs,
            save_every_n_epochs=args.save_every_n_epochs,
            vram=args.vram,
            max_images=args.max_images
        )
        print(f"\nDataset created successfully in: {output_folder}")
    
    except Exception as e:
        print(f"\nError creating dataset: {str(e)}")
