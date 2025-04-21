# Windows Kohya_ss FLUX LoRA Training Complete Guide (12GB VRAM Focus)

This guide provides a recommended setup and workflow for training FLUX LoRA models using Kohya_ss, optimized for GPUs with approximately 12GB VRAM (like an RTX 3060 12GB).

## Prerequisites

Ensure you have downloaded the necessary FLUX model components:

1.  **FLUX Base Model:** e.g., `flux1-dev.safetensors`. Download [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors).
2.  **VAE:** e.g., `ae.safetensors`. Download [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors).
3.  **CLIP-L Text Encoder:** e.g., `clip_l.safetensors`. Download [here](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors).
4.  **T5-XXL Text Encoder:** e.g., `t5xxl_fp16.safetensors`. Download [here](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors). (fp16 version of t5xxl saves disk space and RAM/VRAM when loading the model compared to the full fp32 version, which is helpful during Kohya startup)

## Prerequisites (Software)

*   **Git:** You need Git installed to clone the repository. ([https://git-scm.com](https://git-scm.com))
*   **NVIDIA CUDA Toolkit:** Having the latest CUDA Toolkit installed (e.g., [version 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64) or higher), exclude the drivers during installation if you already have a more recent driver.

---

## Python Installation (Using uv on Windows)

These steps use `uv`, why you ask? Because you can install multiple lots of Python version and its each environment on same machine! uv is a fast rust-based Python package installer and virtual environment manager, for potentially quicker setup.

1.  **Install `uv`:**
    *   Open **PowerShell (Non-Administrator is fine)**.
    *   Run the following command:
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   Close and reopen your terminal (Command Prompt or PowerShell) to ensure `uv` is in the path.

2.  **Clone Kohya_ss Repository:**
    *   Navigate to the directory where you want to install Kohya_ss in your terminal.
    *   Run:
        ```bash
        git clone --recursive https://github.com/bmaltais/kohya_ss.git
        cd kohya_ss
        ```

3.  **Create and Activate Virtual Environment:**
    *   Still inside the `kohya_ss` directory, create the environment using Python 3.10 (Python 3.10 is recommended from kohya_ss gui repo):
        ```bash
        uv venv venv --python 3.10 --seed
        ```
    *   Activate the environment:
        ```bash
        venv\Scripts\activate
        ```
        *(Your terminal prompt should now show `(venv)` at the beginning)*

4.  **Install Python Requirements:**
    *   While the environment is active, install the necessary packages using `uv`:
        ```bash
        uv pip install -r requirements_pytorch_windows.txt
        ```
        *(This installs PyTorch (CUDA version) and other dependencies)*

5.  **Install Specific cuDNN Version (Optional but Recommended):**
    *   This step installs specific cuDNN binaries which can sometimes improve performance or resolve compatibility issues.
        ```bash
        python -m pip install nvidia-cudnn-cu12==8.9.6.50
        ```
        *(Note: This is cu12 version recommended on kohya_ss gui repo. Also this matches the CUDA version PyTorch will be used, which is CUDA 12.x)*

6.  **Manually Copy cuDNN DLLs (Potential Optimization):**
    *   Sometimes PyTorch doesn't automatically find the cuDNN DLLs installed by the previous step. Manually copying them can fix potential runtime errors.
    *   **Source:** `.\venv\Lib\site-packages\nvidia\cudnn\bin\`
    *   **Destination:** `.\venv\Lib\site-packages\torch\lib\`
    *   Copy **all `.dll` files** from the source directory to the destination directory. You can use File Explorer or the command line (`copy .\venv\Lib\site-packages\nvidia\cudnn\bin\*.dll .\venv\Lib\site-packages\torch\lib\`).
    *   *This step might not be strictly necessary for everyone, but it's a common fix if you encounter cuDNN-related errors later.*

7.  **Run Kohya Setup Script (Command Prompt, not PowerShell):**
    *   Ensure the virtual environment (`venv`) is still active.
        ```bash
        venv\Scripts\activate
        ```
    *   **Run this specific command in Command Prompt (`cmd.exe`)**, as `setup.bat` might rely on batch file specifics:
        ```bash
        setup.bat
        ```
    *   A menu will appear. Choose the following options:
        *   Select `1`: Setup gui (Installs any remaining packages needed specifically for the GUI).
        *   Wait for it to finish.
        *   Select `7`: Exit setup.

8.  **Verify Installation (Optional):**
    *   With the environment active, run:
        ```bash
        uv pip check
        ```
    *   This command checks for broken dependencies. Ideally, it should report no issues.

---

## Running Kohya_ss GUI

1.  Open your terminal (Command Prompt or PowerShell).
2.  Navigate to the `kohya_ss` directory:
    ```bash
    cd path\to\your\kohya_ss
    ```
3.  Activate the virtual environment:
    ```bash
    venv\Scripts\activate
    ```
4.  Launch the GUI:
    ```bash
    gui
    ```
    *   This should open the Kohya_ss interface in your web browser. If not, open the browser and use this URL [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## Initial Kohya_ss Setup & Configuration Saving

Configure the base paths and save a preset for easier setup later.

**`[Configuration]`**

*   `Load/Save Config file:` # Use this field to load/save `.json` configuration files. Save your initial paths here.

**`[Model]`**

*   `Pretrained model name or path:` `\path\to\your\flux1-dev.safetensors`
    *   Select the main FLUX base model file.

**`[Parameters] [Basic] [Flux.1]`**

*   `VAE Path:` `\path\to\your\ae.safetensors`
*   `CLIP-L Path:` `\path\to\your\clip_l.safetensors`
*   `T5-XXL Path:` `\path\to\your\t5xxl_fp16.safetensors`

**Saving the Base Configuration:**

1.  Navigate back to the **`[Configuration]`** tab.
2.  Enter a path and filename in `Load/Save Config file:` (e.g., `\path\to\configs\base_flux_settings.json`).
3.  Click the `Save` button next to the field.

---

## Core Training Parameter Recommendations (12GB VRAM)

Load your saved base configuration (`base_flux_settings.json`) and adjust these parameters.

**`[Model]`**

*   `LoRA type:` `Flux1`
    *   Ensure this is selected from the dropdown.
*   `Save precision:` `fp16`
    *   Good balance of final LoRA file size and quality. `bf16` is also viable if preferred.

**`[Folders]`**

*   `Output directory for trained model:`
    *   it will be filled automatically after clicking `Prepare training data` and `Copy info to respective fields` on `[Dataset Preparation]` section
*   `Logging directory:`
    *   it will be filled automatically after clicking `Prepare training data` and `Copy info to respective fields` on `[Dataset Preparation]` section

**`[Parameters] [Basic]`**

*   `Epoch:` `15`
    *   **STARTING POINT.** Total steps = (num_images \* repeats \* epochs). For 25 images \* 10 repeats, 15 epochs = 3750 steps. Aim for a reasonable starting step count (e.g., 3000-6000). Test checkpoints frequently rather than aiming for a specific final loss value initially. Increase epochs later via resuming if needed.
*   `Save every N epochs:` `1`
    *   **ESSENTIAL** For testing progress and finding the best epoch. Note: Checkpoints can be large (potentially 1-2GB each for Rank 64, fp16), so ensure sufficient disk space. I really mean it: ensure the disk space is enough!
*   `Cache latents:` `checked`
*   `Cache latents to disk:` `checked`
    *   Helps reduce VRAM usage by pre-calculating and optionally storing latents on disk.
*   `LR Scheduler:` `constant`
    *   Simplest option to start. Experiment with 'cosine' or others later if desired. Set LR warmup steps to 0 when using 'constant'.
*   `Optimizer:` `Adafactor`
    *   Memory-efficient choice, good for 12GB VRAM. `AdamW8bit` is a common alternative, potentially faster convergence but uses slightly more VRAM and may need different learning rates. If switching from Adafactor to AdamW8bit (or vice-versa), the optimal learning rate might need significant re-tuning, (also potentially slightly lower than Adafactor, requiring testing).
*   `Max resolution:` `768,768`
    *   Good compromise for FLUX on 12GB VRAM. FLUX's native resolution is 1024x1024, which requires significantly more VRAM for training.
*   `Enable Buckets:` `unchecked`
    *   Keep unchecked **ONLY** if **ALL** training images are **EXACTLY** 768x768 pixels. Check this box if images have different aspect ratios to train them more efficiently near their original shape.
*   `Learning rate:` `0.0001`
*   `Text Encoder learning rate:` `0.0001`
*   `Unet learning rate (Optional):` `0.0001`
    *   `0.0001` is a recommended **STARTING POINT** for Adafactor/AdamW8bit with LoRA. `0.00005` might be too slow. `0.00035` or higher increases risk of instability/poor results. Requires testing and potential adjustment based on results.
*   `Network Rank (Dimension):` `64`
*   `Network Alpha:` `64`
    *   Rank 64 is a good balance for quality/speed/VRAM on 12GB. Lower rank (e.g., 32) is faster but might capture less detail. Higher (e.g., 128) requires more VRAM/time. Alpha is often set equal to Rank.

**`[Parameters] [Basic] [Flux.1]`**

*   `Cache Text Encoder Outputs:` `checked`
*   `Cache Text Encoder Outputs to Disk:` `checked`
    *   More caching to potentially reduce VRAM and computation during training.
*   `Memory Efficient Save:` `checked`
    *   Recommended setting for FLUX training.

**`[Parameters] [Advanced] [Weights tab]`**

*   `fp8 base:` `checked`
*   `fp8 base unet:` `checked`
    *   **ABSOLUTELY ESSENTIAL** for FLUX training on 12GB VRAM to prevent Out-of-Memory errors and improve speed. Can potentially be disabled on high-VRAM GPUs (24GB+), but often still beneficial.
*   `Use bf16 training (experimental):` `unchecked`
    *   Keep unchecked when using `fp8` settings above. `bf16` is an alternative mixed precision, typically not combined with `fp8` base model training.
*   `highvram:` `unchecked`
    *   Keep unchecked for 12GB VRAM. Checking this disables VRAM-saving optimizations like model offloading.
*   `Gradient checkpointing:` `checked`
    *   Crucial VRAM saving technique. Keep checked.
*   `CrossAttention:` `xformers`
    *   Usually offers the best speed/memory balance on NVIDIA GPUs (especially pre-40 series). `sdpa` is an alternative.

---

## Dataset Configuration (Set Every Run/Load)

These fields under **`[Dataset Preparation]`** are **NOT SAVED** in the main configuration JSON and need to be set manually each time you prepare for training.

**`[Dataset Preparation]`**

*   `Instance prompt:` `your_instance_prompt` # e.g., ohwxohwx man
*   `Class prompt:` `your_class_prompt`       # e.g., style, man, woman, person, creature, object (describes the general category)
*   `Training images:` `\path\to\your\source\training\images` # Directory containing images for training
*   `Repeats:` `10` # A reasonable start for smaller datasets (~25 images). More repeats mean each image is shown more times per epoch. Adjust based on dataset size and desired focus.
*   `Destination training directory:` `\path\to\your\trained\directory` # Directory of source images, log, saved models, and saved state will be structurally put into

*   **Click These Two Buttons:**
    1.  Click `Prepare training data` (Creates the necessary folder structure and JSON files in your 'Training images' directory if not already done)
    2.  Click `Copy info to respective fields` (Copies prompts, paths, etc., to the other sections - **VERIFY** they copied correctly)

---

## Iterative Training Workflow (Example)

This workflow allows training in chunks and resuming, useful for long training times or testing intermediate results.

### 1. First Training Iteration (e.g., 15 Epochs)

1.  Load your base configuration JSON (`base_flux_settings.json`) using **`[Configuration] Load/Save Config file`**.
2.  Go to **`[Dataset Preparation]`** and set prompts/paths/repeats/destination, then click `Prepare training data` and `Copy info to respective fields`.
3.  Verify/Set Parameters:
    *   **`[Model]`**
        *   `Trained Model output name:` `ohwxohwx_e15` (Choose a descriptive name)
    *   **`[Parameters] [Basic]`**
        *   `Epoch:` `15` (Set the target epoch for this run)
4.  Enable State Saving:
    *   **`[Parameters] [Advanced] [Weights tab]`**
        *   `Save training state:` `checked`
        *   `Save training state at end of training:` `checked`
5.  Save a specific config for this run:
    *   **`[Configuration]`**
        *   `Load/Save Config file:` `\path\to\save\configs\ohwxohwx_e15_run.json` (Enter path)
        *   Click `Save` button.
6.  Click `Start training`.

### 2. Second Training Iteration (Resume, e.g., to 25 Epochs total)

Before deciding whether to proceed with another training iteration, it's important to test the saved safetensors checkpoints from various epochs (ohwxohwx_e15-0000xx.safetensors) to find the best balance (creativity/underfitting vs. likeness/overfitting). Overfitting means your image results will be more exactly like your training image with no variations or control from prompting, which isn't preferable.

1.  Load the config from the *previous* run (`ohwxohwx_e15_run.json`) using **`[Configuration] Load/Save Config file`**.
2.  Go to **`[Dataset Preparation]`** and set prompts/paths/repeats/destination, then click `Prepare training data` and `Copy info to respective fields` (if Kohya cleared them).
3.  Configure Resume Settings:
    *   **`[Model]`**
        *   `Trained Model output name:` `ohwxohwx_e25` (New name for the final output of this run)
    *   **`[Parameters] [Basic]`**
        *   `Epoch:` `25` (**Set the NEW TOTAL target epoch**, not the additional number of epochs)
        *   `Network weights:` `\path\to\your\output\models\ohwxohwx_e15.safetensors` (**Path to the LoRA file saved from the previous run**)
    *   **`[Parameters] [Advanced] [Weights tab]`**
        *   `Resume from saved training state:` `\path\to\your\output\models\ohwxohwx_e15-state` (**Path to the state folder saved from the previous run**)
    > Both `Network weights` and `Resume from saved training state` are needed when resuming: Network weights primarily tells Kohya the structure (rank/alpha etc.) of the LoRA file being continued, while Resume from saved training state loads the actual optimizer state, step count, and precise weight values to continue learning seamlessly. (Kohya might infer structure from state sometimes, but explicitly setting both is safest).
4.  Ensure State Saving is still enabled:
    *   **`[Parameters] [Advanced] [Weights tab]`**
        *   `Save training state:` `checked`
        *   `Save training state at end of training:` `checked`
5.  Save a specific config for this run:
    *   **`[Configuration]`**
        *   `Load/Save Config file:` `\path\to\save\configs\ohwxohwx_e25_run.json` (Enter path)
        *   Click `Save` button.
6.  Click `Start training`.

> **Note on Progress Bar/Epoch Counter:** When resuming, the console output should show it starting from the correct epoch (e.g., epoch 16). The graphical progress bar might reset its step count visually (starts from 0), but the underlying training resumes correctly based on the loaded state.

### 3. Third Training Iteration (Resume, e.g., to 30 Epochs total) - WARNING!!! There's bug [here](https://github.com/bmaltais/kohya_ss/issues/2771) for 3rd iteration

1.  Go to **`[Configuration]`**
    *   `Load/Save Config file:` Click the `folder button` to open `\path\to\save\configs\ohwxohwx_e25.json` (Load the config saved after the 2nd iteration finished).
2.  Go to **`[Dataset Preparation]`** and set prompts/paths/repeats/destination, then click `Prepare training data` and `Copy info to respective fields` (if Kohya cleared them).
3.  Configure Resume Settings:
    *   **`[Model]`**
        *   `Trained Model output name:` `ohwxohwx_e30` (New name for the final output of this run)
    *   **`[Parameters] [Basic]`**
        *   `Epoch:` `30` (**Set the NEW TOTAL target epoch**, not the additional number of epochs)
        *   `Network weights:` `\path\to\your\output\models\ohwxohwx_e25.safetensors` (**Path to the LoRA file saved from the 2nd run**)
    *   **`[Parameters] [Advanced] [Weights tab]`**
        *   `Resume from saved training state:` `\path\to\your\output\models\ohwxohwx_e25-state` (**Path to the state folder saved from the 2nd run**)
    > Both `Network weights` and `Resume from saved training state` are needed when resuming: Network weights primarily tells Kohya the structure (rank/alpha etc.) of the LoRA file being continued, while Resume from saved training state loads the actual optimizer state, step count, and precise weight values to continue learning seamlessly. (Kohya might infer structure from state sometimes, but explicitly setting both is safest).
4.  **Bug Fix Check (IMPORTANT - Do Before Starting Training):**
    *   3rd iteration training bug discussed here: https://github.com/bmaltais/kohya_ss/issues/2771
    *   Navigate to the state folder from the previous run: `\path\to\your\output\models\ohwxohwx_e25-state`
    *   Open the `train_state.json` file inside that folder in a text editor.
    *   Find the `"current_step"` value.
    *   **Verify/Correct the value:** Some Kohya versions might incorrectly record only the steps from the *last* run instead of the cumulative total. Calculate the *correct* total steps.
        *   *Example Calculation:* If Iteration 1 was 15 epochs (25 imgs * 10 repeats = 3750 steps) and Iteration 2 added 10 epochs (2500 steps), the `current_step` *should* be 3750 + 2500 = **6250**.
    *   If the value in `train_state.json` is wrong (e.g., it shows only `2500` from 2nd iteration), **manually edit it** to the correct cumulative total (`6250` in this example).
    *   **Save the changes** to `train_state.json`.
5.  Ensure State Saving is still enabled:
    *   **`[Parameters] [Advanced] [Weights tab]`**
        *   `Save training state:` `checked`
        *   `Save training state at end of training:` `checked`
6.  Save a specific config for this run:
    *   **`[Configuration]`**
        *   `Load/Save Config file:` `\path\to\save\configs\ohwxohwx_e30.json` (Enter path for the new config)
        *   Click `Save` button.
7.  Click `Start training`.

> **Note on Progress Bar/Epoch Counter:** When resuming, the console output should show it starting from the correct epoch (e.g., epoch 26). The graphical progress bar might reset its step count visually (starts from 0), but the underlying training resumes correctly based on the loaded state and corrected `train_state.json`.

### 4. Subsequent Iterations (e.g., 4th, 5th...)

*   Follow the same procedure as the "Third Training Iteration".
*   **Crucially, perform the `train_state.json` bug check and correction before starting each subsequent resumed run**, calculating the correct cumulative `current_step` based on *all* previous runs.
    *   *Example for starting 4th run (aiming for 35 epochs, adding 5 epochs = 1250 steps):* Check `ohwxohwx_e30-state\train_state.json`. You might find the `current_step` is incorrectly set to 1250 (only the steps from the last 3rd iteration), it should be **7500** = 3750 (1st) + 2500 (2nd) + 1250 (3rd). Edit this. This bug is kinda annoying, really.
*   Save a new configuration file for the current run.
*   Start training.

---

Remember to **test your saved LoRA checkpoints** at different epochs to find the one that best captures your desired concept without overfitting! Good luck!
