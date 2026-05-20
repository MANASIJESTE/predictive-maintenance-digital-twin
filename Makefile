.PHONY: install download features train evaluate predict test run-all clean

install:
	pip install -r requirements.txt

download:
	python -m src.download_data

features:
	python -m src.cli features

train:
	python -m src.train

evaluate:
	python -m src.evaluate

predict:
	python -m src.predict --machine-id 1 --timestamp "2015-10-01 08:00:00"

test:
	pytest -q

run-all:
	python -m src.pipeline

clean:
	rm -rf artifacts/* reports/* mlruns .pytest_cache
	touch artifacts/.gitkeep reports/.gitkeep
