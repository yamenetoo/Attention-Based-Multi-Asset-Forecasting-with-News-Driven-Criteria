"""
Configuration management module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path


@dataclass
class Config:
    """Main configuration class loaded from YAML."""
    
    # Data configuration
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Model configuration
    model: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    evaluation: Dict[str, Any] = field(default_factory=dict)
    
    # Interpretability configuration
    interpretability: Dict[str, Any] = field(default_factory=dict)
    
    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=dict)
    
    # Global paths
    paths: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: str = "src/config/default.yaml") -> Config:
    """Convenience function to load config."""
    return Config.from_yaml(config_path)


__all__ = ["Config", "load_config"]
