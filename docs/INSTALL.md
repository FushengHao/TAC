## Installation

* Setup conda environment
```bash
# Create a conda environment
conda create -y -n TAC python=3.8

# Activate the environment
conda activate TAC

# Install torch, torchvision, and torchaudio
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
```

* Install dassl library.
```bash
# Clone Dassl
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install the library
python setup.py develop
cd ..
```

The directory structure should look like
```
|–– Dassl.pytorch/
|–– code/
```

* Install the requirements for TAC
```bash
cd code/
# Install requirements
pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0
```
