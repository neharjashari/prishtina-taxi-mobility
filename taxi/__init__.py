"""Prishtina taxi mobility: the preprocessing pipeline for the Kaggle taxi GPS dataset."""


class PipelineError(Exception):
    """A check failed and a human has to look at it. The pipeline stops and prints the message."""
