# im2typst
A formula screenshot to typst math converter

# goal
The goal is to create a image 2 typst converter which is able to take any formula found online and easily convert it into a typst formula.

# Python Environment setup

First create the python environment
```bash
gh repo clone peterhorauer/im2typst
cd im2typst

python -m venv .
```

Under Linux source the bin/activate with following command
```bash 
source ./bin/activate
```

To confirm run `which python`

Upgrade the pip installation
```bash
python -m pip install --upgrade pip
```

And install all dependencies
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install requirements.txt
```