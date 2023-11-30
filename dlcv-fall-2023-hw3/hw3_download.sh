#!/bin/bash

# TODO - run your inference Python3 code
wget -O hw3_model.zip 'https://www.dropbox.com/scl/fi/rhx72a81yf9jiqgur24fn/hw3_model.zip?rlkey=xmmugatkbroarucptpz5h4e2p&dl=1'

unzip ./hw3_model.zip

python -c "import clip; clip.load('ViT-B/16')"
python -c "import timm; timm.create_model('vit_huge_patch14_clip_378', pretrained=True)"