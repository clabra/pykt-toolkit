# Host Machine Installation Guide for GPU Training

This guide provides instructions on how to set up the environment to run the training on your host machine and leverage the GPUs.

### 1. Install Miniconda

Navigate to your project directory and run the installer script:

```bash
cd /path/to/your/pykt-toolkit # Replace with the actual path on your host
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts in the installer. When it asks if you want to initialize Miniconda, answer **yes**. After the installation is complete, **close and reopen your terminal**.

### 2. Create the Conda Environment

Now, create the `pykt` environment with the correct Python version:

```bash
conda create --name=pykt python=3.7.5
conda activate pykt
```

### 3. Install Dependencies

With the `pykt` environment active, install the necessary libraries:

```bash
# Navigate back to your project directory if you're not already there
cd /path/to/your/pykt-toolkit

# Install PyTorch with CUDA support
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install the project's dependencies
pip install -e .
```

### 4. Preprocess the Data

To avoid any potential data file compatibility issues, run the preprocessing script:

```bash
# Navigate to the examples directory
cd examples

# Run the preprocessing script
python data_preprocess.py --dataset_name assist2015
```

### 5. Run the Training on GPU

Finally, you can run the training script, specifying which GPU to use.

```bash
# From the examples directory
CUDA_VISIBLE_DEVICES=0 python wandb_simakt_train.py \
    --model_name="simakt" \
    --dataset_name="assist2015" \
    --emb_type="qid_cl" \
    --d_model=256 \
    --d_ff=256 \
    --dropout=0.1 \
    --learning_rate=0.001 \
    --num_attn_heads=8 \
    --n_blocks=3 \
    --n_know=16 \
    --seed=3407 \
    --fold=0 \
    --use_wandb=0
```
