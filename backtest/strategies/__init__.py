#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Strategy Base Framework
Provides abstract base class and registry for all backtesting strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import date
import logging

logger = logging.getLogger(__name__)


class StrategyBase(ABC):
    """Abstract base class for all backtesting strategies."""

    _registry: Dict[str, type] = {}

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}

    def __init_subclass__(cls, **kwargs):
        """Register strategy class automatically."""
        super().__init_subclass__(**kwargs)
        if cls.__name__ != "StrategyBase":
            strategy_id = cls.get_strategy_id()
            if strategy_id:
                StrategyBase._registry[strategy_id] = cls
                logger.debug(f"Registered strategy: {strategy_id}")

    @classmethod
    @abstractmethod
    def get_strategy_id(cls) -> str:
        """Return unique strategy identifier."""
        pass

    @classmethod
    @abstractmethod
    def get_strategy_name(cls) -> str:
        """Return human-readable strategy name."""
        pass

    @classmethod
    @abstractmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        """Return parameters schema for validation."""
        pass

    @abstractmethod
    def generate_signals(
        self, as_of_date: date, universe: List[str]
    ) -> Dict[str, float]:
        """
        Generate signal values for given universe as of specific date.
        Must use only data available on or before as_of_date (PIT constraint).

        Args:
            as_of_date: Date to query data as of (PIT)
            universe: List of tickers to evaluate

        Returns:
            Dict mapping ticker -> signal value (higher = more attractive)
        """
        pass

    @abstractmethod
    def build_portfolio(
        self, signals: Dict[str, float], top_n: int
    ) -> Dict[str, float]:
        """
        Build portfolio weights from signals.

        Args:
            signals: Ticker -> signal value
            top_n: Number of top tickers to select

        Returns:
            Dict mapping ticker -> weight (sum to 1.0)
        """
        pass

    def validate_params(self) -> bool:
        """Validate parameters against schema."""
        schema = self.get_params_schema()
        for key, param_def in schema.items():
            if key not in self.params:
                if param_def.get("required", False):
                    logger.error(f"Missing required parameter: {key}")
                    return False
            else:
                value = self.params[key]
                expected_type = param_def.get("type")
                if expected_type and not isinstance(value, expected_type):
                    logger.error(
                        f"Parameter {key} must be {expected_type}, got {type(value)}"
                    )
                    return False
        return True

    @classmethod
    def get_strategy(
        cls, strategy_id: str, params: Dict[str, Any] = None
    ) -> "StrategyBase":
        """Get strategy instance by ID."""
        if strategy_id not in cls._registry:
            raise ValueError(f"Strategy not found: {strategy_id}")
        strategy_class = cls._registry[strategy_id]
        return strategy_class(params=params)

    @classmethod
    def list_strategies(cls) -> List[Dict[str, Any]]:
        """List all registered strategies."""
        return [
            {
                "id": strategy_id,
                "name": cls._registry[strategy_id].get_strategy_name(),
                "params_schema": cls._registry[strategy_id].get_params_schema(),
            }
            for strategy_id in sorted(cls._registry.keys())
        ]
