#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Risk Analyzer
Analyzes correlation and volatility risk for a portfolio
"""

import os
import sys
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.fmp_client import get_fmp_client
from utils.symbols import to_fmp_symbol
from utils.db_writer import write_market_documents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioRiskAnalyzer:
    """Analyze portfolio risk metrics"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.output_file = os.path.join(data_dir, 'portfolio_risk.json')
        self.fmp = get_fmp_client()
        self.lookback_days = 182

    def _fetch_close_series(self, ticker: str) -> pd.Series:
        end_date = datetime.utcnow().date()
        from_date = (end_date - timedelta(days=self.lookback_days)).isoformat()
        to_date = end_date.isoformat()
        data = self.fmp.historical_price_full(
            to_fmp_symbol(ticker),
            from_date=from_date,
            to_date=to_date,
        )
        hist_list = data.get("historical", []) if isinstance(data, dict) else []
        if not hist_list:
            return pd.Series(dtype=float)
        df = pd.DataFrame(hist_list)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df.set_index('date')['close'].astype(float)
    
    def analyze_portfolio(self, tickers: List[str], weights: List[float] = None) -> Dict:
        """
        Analyze portfolio risk
        
        Args:
            tickers: List of ticker symbols
            weights: Optional list of weights (default: equal weight)
        """
        logger.info(f"📊 Analyzing portfolio risk for {len(tickers)} stocks...")
        
        if len(tickers) < 2:
            return {'error': 'Need at least 2 tickers for portfolio analysis'}
        
        try:
            series_map = {}
            for ticker in tickers:
                series = self._fetch_close_series(ticker)
                if not series.empty:
                    series_map[ticker] = series

            if not series_map:
                return {'error': 'No price data available'}

            data = pd.DataFrame(series_map)
            
            # Drop any tickers with missing data
            data = data.dropna(axis=1, how='all')
            available_tickers = list(data.columns)
            
            if len(available_tickers) < 2:
                return {'error': 'Need at least 2 tickers with valid data'}
            
            # Calculate daily returns
            returns = data.pct_change().dropna()
            
            # Default to equal weights
            if weights is None:
                weights = np.array([1/len(available_tickers)] * len(available_tickers))
            else:
                # Normalize weights
                weights = np.array(weights[:len(available_tickers)])
                weights = weights / weights.sum()
            
            # === Correlation Analysis ===
            corr_matrix = returns.corr()
            
            # Find high correlations
            high_correlations = []
            cols = corr_matrix.columns
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_correlations.append({
                            'pair': [cols[i], cols[j]],
                            'correlation': round(corr_val, 3),
                            'warning': 'High' if corr_val > 0.8 else 'Moderate'
                        })
            
            # Sort by absolute correlation
            high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # === Volatility Analysis ===
            # Annualized covariance matrix
            cov_matrix = returns.cov() * 252
            
            # Portfolio variance and volatility
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Individual volatilities
            individual_vol = returns.std() * np.sqrt(252)
            
            # === Value at Risk (VaR) ===
            # 95% VaR (assuming normal distribution)
            var_95 = portfolio_volatility * 1.645  # 95% confidence
            var_99 = portfolio_volatility * 2.326  # 99% confidence
            
            # === Expected Return (historical) ===
            portfolio_returns = returns.dot(weights)
            expected_return = portfolio_returns.mean() * 252
            
            # === Sharpe Ratio (assuming 5% risk-free rate) ===
            risk_free_rate = 0.05
            sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # === Beta (vs SPY) ===
            try:
                spy_series = self._fetch_close_series('SPY')
                spy_returns = spy_series.pct_change().dropna()
                
                # Align dates
                common_dates = portfolio_returns.index.intersection(spy_returns.index)
                if len(common_dates) > 20:
                    port_aligned = portfolio_returns.loc[common_dates]
                    spy_aligned = spy_returns.loc[common_dates]
                    
                    covariance = np.cov(port_aligned, spy_aligned)[0, 1]
                    spy_variance = spy_aligned.var()
                    beta = covariance / spy_variance if spy_variance > 0 else 1
                else:
                    beta = 1
            except:
                beta = 1
            
            # === Risk Rating ===
            risk_score = 0
            if portfolio_volatility > 0.4:
                risk_score += 3
            elif portfolio_volatility > 0.25:
                risk_score += 2
            elif portfolio_volatility > 0.15:
                risk_score += 1
            
            if len(high_correlations) > 5:
                risk_score += 2
            elif len(high_correlations) > 2:
                risk_score += 1
            
            if beta > 1.5:
                risk_score += 2
            elif beta > 1.2:
                risk_score += 1
            
            if risk_score >= 5:
                risk_level = "High"
            elif risk_score >= 3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'portfolio': {
                    'tickers': available_tickers,
                    'weights': {t: round(w, 4) for t, w in zip(available_tickers, weights)}
                },
                'metrics': {
                    'volatility_annual': round(portfolio_volatility * 100, 2),
                    'expected_return_annual': round(expected_return * 100, 2),
                    'sharpe_ratio': round(sharpe_ratio, 2),
                    'beta': round(beta, 2),
                    'var_95_daily': round(var_95 / np.sqrt(252) * 100, 2),
                    'var_99_daily': round(var_99 / np.sqrt(252) * 100, 2)
                },
                'individual_volatility': {
                    t: round(v * 100, 2) for t, v in individual_vol.items()
                },
                'correlation': {
                    'high_correlations': high_correlations[:10],
                    'avg_correlation': round(corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean(), 3),
                    'matrix': corr_matrix.round(3).to_dict()
                },
                'risk_assessment': {
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'warnings': []
                }
            }
            
            # Add warnings
            if portfolio_volatility > 0.3:
                result['risk_assessment']['warnings'].append("High portfolio volatility")
            if len(high_correlations) > 3:
                result['risk_assessment']['warnings'].append(f"{len(high_correlations)} highly correlated pairs - consider diversification")
            if beta > 1.3:
                result['risk_assessment']['warnings'].append("Portfolio is more volatile than market (high beta)")
            
            # Save to file
            with open(self.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"✅ Saved to {self.output_file}")
            write_market_documents(
                "portfolio_risk",
                result,
                as_of_date=result.get("as_of_date") if isinstance(result, dict) else None,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {'error': str(e)}
    
    def print_summary(self, data: Dict):
        """Print analysis summary"""
        if 'error' in data:
            print(f"❌ Error: {data['error']}")
            return
        
        print("\n📊 Portfolio Risk Analysis")
        print("=" * 50)
        
        metrics = data['metrics']
        print(f"\n📈 Key Metrics:")
        print(f"   Annualized Volatility: {metrics['volatility_annual']:.1f}%")
        print(f"   Expected Return: {metrics['expected_return_annual']:.1f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Beta: {metrics['beta']:.2f}")
        print(f"   95% Daily VaR: {metrics['var_95_daily']:.2f}%")
        
        risk = data['risk_assessment']
        emoji = "🟢" if risk['risk_level'] == "Low" else ("🟡" if risk['risk_level'] == "Medium" else "🔴")
        print(f"\n{emoji} Risk Level: {risk['risk_level']}")
        
        if risk['warnings']:
            print("\n⚠️ Warnings:")
            for w in risk['warnings']:
                print(f"   - {w}")
        
        if data['correlation']['high_correlations']:
            print("\n🔗 High Correlations:")
            for c in data['correlation']['high_correlations'][:3]:
                print(f"   {c['pair'][0]} - {c['pair'][1]}: {c['correlation']:.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio Risk Analyzer')
    parser.add_argument('--dir', default='.', help='Data directory')
    parser.add_argument('--tickers', nargs='+', 
                       default=['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN'],
                       help='Tickers to analyze')
    parser.add_argument('--weights', nargs='+', type=float, help='Portfolio weights')
    args = parser.parse_args()
    
    analyzer = PortfolioRiskAnalyzer(data_dir=args.dir)
    result = analyzer.analyze_portfolio(args.tickers, args.weights)
    analyzer.print_summary(result)


if __name__ == "__main__":
    main()
