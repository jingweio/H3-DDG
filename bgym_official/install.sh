conda create --name BindingGYM python=3.8

source activate BindingGYM

conda install -c conda-forge cudatoolkit=11.7

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install torch-scatter==2.1.0+pt113cu117 torch-sparse==0.6.16+pt113cu117 torch-cluster==1.6.0+pt113cu117 torch-spline-conv==1.2.1+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

pip install torch-geometric==2.2.0

python -m pip install PyYAML scipy "networkx[default]" biopython pandas click

pip install peft biotite numba ruamel.yaml billiard seaborn numba_progress

pip install datasets==2.14.4

conda install -c conda-forge -c bioconda foldseek=8.ef4e960

git clone https://github.com/facebookresearch/esm

cd esm

pip install -e .

cd ..
