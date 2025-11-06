# Minimal makefile to run the integration steps
.PHONY: all base turnout features validate

all: base turnout features validate

base:
	python -m src.build_base

turnout:
	python -m src.build_turnout

features:
	python -m src.build_features

validate:
	python -m src.validate
