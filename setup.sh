pyenv install 3.10.0
pyenv virtualenv 3.10.0 condense
pyenv activate condense
pip install -r requirements.txt
pip install -e .