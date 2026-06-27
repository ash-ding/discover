"""Configuration loader for TTT-Discover experiments."""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary of configuration parameters
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    Override config takes precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def load_config_with_overrides(
    config_path: str,
    cli_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Load config file and apply command-line overrides.

    Args:
        config_path: Path to YAML config file
        cli_overrides: Dictionary of CLI overrides (optional)

    Returns:
        Final configuration dictionary
    """
    config = load_config(config_path)

    if cli_overrides:
        config = merge_configs(config, cli_overrides)

    return config
