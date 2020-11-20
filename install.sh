#!/bin/bash

virtualenv poesIA
source poesIA/bin/activate
pip install -r requirements
cp nlputils_fastai2.py poesIA/lib/python3.8/site-packages/
