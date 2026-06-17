install:
	pip install -r requirements.txt

download:
	python scripts/download_data.py

train:
	python -m src.train --target accumulator

app:
	streamlit run app.py

all:
	python scripts/download_data.py
	python -m src.train --target accumulator
	streamlit run app.py
