PYTHON ?= python

.PHONY: install help

install:
	$(PYTHON) -m pip install -e .

help:
	$(PYTHON) -m video_konspekt --help
