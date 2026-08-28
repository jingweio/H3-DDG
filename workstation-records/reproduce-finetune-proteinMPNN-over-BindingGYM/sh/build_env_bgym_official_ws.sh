#!/bin/bash
# Build the bgym-official env on the WORKSTATION, matching the Ibex env that actually ran
# folds 1 and 2 to completion (jobs 50922622 / 50922623).
#
# WHY NOT install.sh: upstream's install.sh installs a superset (torchvision, torchaudio,
# datasets, foldseek, an editable esm clone). The env that actually works has 44 packages and
# none of those. We reproduce the WORKING env, taken from its own pip list --format=freeze,
# not the aspirational install list. Two earlier workstation env builds died at import because
# they were assembled from "what the code looks like it needs" (easydict, then yaml); the rule
# now is to mirror a known-good env exactly and then diff to prove it.
#
# CUDA: workstation driver is 535 => runtime must be <= 12.2. cu117 wheels satisfy that. Do NOT
# use the system nvcc (CUDA 10.1). Skipping install.sh's 'conda install -c conda-forge
# cudatoolkit=11.7' on purpose -- the pip wheel bundles its own runtime, and skill section 10
# forbids mixing conda-forge into a defaults-channel install.
set -euo pipefail
source /data/guoj0f/miniconda3/etc/profile.d/conda.sh

conda create -y -n bgym-official python=3.8
conda activate bgym-official

pip install -q torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# All four PyG companion wheels, verbatim. Dropping any one of them breaks torch_geometric's
# import at runtime, not at install time.
pip install -q torch-scatter==2.1.0+pt113cu117 torch-sparse==0.6.16+pt113cu117 \
    torch-cluster==1.6.0+pt113cu117 torch-spline-conv==1.2.1+pt113cu117 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -q torch_geometric==2.2.0

pip install -q accelerate==1.0.1 biopython==1.83 certifi==2026.7.22 charset-normalizer==3.5.1 fair-esm==2.0.0 filelock==3.16.1 fsspec==2025.3.0 hf-xet==1.6.0 huggingface_hub==0.36.2 idna==3.15 Jinja2==3.1.6 joblib==1.4.2 MarkupSafe==2.1.5 numpy==1.24.4 packaging==26.2 pandas==2.0.3 peft==0.12.0 psutil==7.2.2 pyparsing==3.1.4 python-dateutil==2.9.0.post0 pytz==2026.3.post1 PyYAML==6.0.3 regex==2024.11.6 requests==2.32.4 safetensors==0.5.3 scikit-learn==1.3.2 scipy==1.10.1 six==1.17.0 threadpoolctl==3.5.0 tokenizers==0.20.3 tqdm==4.70.0 transformers==4.46.3 typing_extensions==4.13.2 tzdata==2026.3 urllib3==2.2.3 

echo "=== verify ==="
python -c "
import torch, numpy, sklearn, torch_geometric, esm
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('numpy', numpy.__version__, '| sklearn', sklearn.__version__)
n = torch.cuda.get_device_name(0); print('GPU:', n); assert 'A100' in n, n
assert (numpy.__version__, sklearn.__version__) == ('1.24.4','1.3.2'), 'fold assignment depends on these'
"
pip list --format=freeze | sort > /tmp/bgym_ws_pkgs.txt
echo "package count: $(wc -l < /tmp/bgym_ws_pkgs.txt)"
echo "=== BUILD DONE ==="
