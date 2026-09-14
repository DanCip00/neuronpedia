from unittest.mock import patch

import pytest

from neuronpedia_inference.server import parse_env_and_args
from start import parse_args


@patch.dict("os.environ", {"SAE_SETS": '["res-jb", "att-kk"]'})
def test_multiple_sae_sets():
    parsed_args = parse_env_and_args()
    assert parsed_args.sae_sets == ["res-jb", "att-kk"]


@patch.dict("os.environ", {}, clear=True)
def test_activation_batch_size_defaults_to_four():
    parsed_args = parse_env_and_args()
    assert parsed_args.activation_batch_size == 4


@patch.dict("os.environ", {"ACTIVATION_BATCH_SIZE": "8"})
def test_activation_batch_size_env_accepts_positive_integer():
    parsed_args = parse_env_and_args()
    assert parsed_args.activation_batch_size == 8


@pytest.mark.parametrize("value", ["0", "-1", "lots"])
def test_activation_batch_size_env_rejects_invalid_values(value: str):
    with (
        patch.dict("os.environ", {"ACTIVATION_BATCH_SIZE": value}),
        pytest.raises(ValueError, match="ACTIVATION_BATCH_SIZE"),
    ):
        parse_env_and_args()


def test_activation_batch_size_cli_accepts_positive_integer(monkeypatch):
    monkeypatch.setattr("sys.argv", ["start.py", "--activation_batch_size", "8"])
    assert parse_args().activation_batch_size == 8


@pytest.mark.parametrize("value", ["0", "-1", "lots"])
def test_activation_batch_size_cli_rejects_invalid_values(monkeypatch, value: str):
    monkeypatch.setattr("sys.argv", ["start.py", "--activation_batch_size", value])
    with pytest.raises(SystemExit):
        parse_args()
