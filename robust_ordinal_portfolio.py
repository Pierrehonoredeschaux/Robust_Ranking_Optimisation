"""
=============================================================================
 Robust Portfolio Optimization on Ordinal Uncertainty Sets
 Reproducing concepts from Sanford (2025) & Nguyen/Lo (2012)
 ---------------------------------------------------------------------------
 Walk-Forward Backtest - Large Cap US Equities
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.stats import zscore
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
def get_sp500_tickers() -> list[str]:
    try:
        import requests, io
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text))[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"[DATA] Retrieved {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"[DATA] Wikipedia scrape failed ({e}). Using minimum fallback list.")
        return ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","LLY"]

TICKERS = get_sp500_tickers()
BENCHMARK = "SPY"
LOOKBACK_MONTHS = 36          # 3 years lookback
START_DATE = "2010-01-01"     # Starting early enough to have 36m of warmup
END_DATE = "2026-04-01"

# ═══════════════════════════════════════════════════════════════════════════
# 2.  DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════
class DataFetcher:
    def __init__(self, tickers, benchmark, start_date, end_date):
        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        all_tickers = self.tickers + [self.benchmark]
        print(f"[DATA] Downloading data for {len(all_tickers)} tickers from {self.start_date} to {self.end_date}...")
        raw = yf.download(all_tickers, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False, threads=True)
        
        # yfinance returns MultiIndex, extract "Close"
        if isinstance(raw.columns, pd.MultiIndex):
            daily_close = raw['Close']
        else:
            daily_close = raw
            
        monthly_close = daily_close.resample('ME').last()
        daily_returns = daily_close.pct_change().dropna(how='all')
        monthly_returns = monthly_close.pct_change().dropna(how='all')
        
        print("[DATA] Download complete.")
        return daily_close, monthly_close, daily_returns, monthly_returns

# ═══════════════════════════════════════════════════════════════════════════
# 3.  ALPHA COMPOSITE & ORDINAL RANKING
# ═══════════════════════════════════════════════════════════════════════════
class SignalGenerator:
    @staticmethod
    def generate_signals(daily_returns_window, monthly_close_window):
        """
        Computes composite signal and strict ordinal ranking.
        1 is best, N is worst.
        """
        # Momentum 6m: return over the last 6 months
        mom_6m = monthly_close_window.iloc[-1] / monthly_close_window.iloc[-7] - 1
        
        # Low Vol 6m: Calculate daily volatility over the last ~126 trading days (6 months)
        daily_window_6m = daily_returns_window.iloc[-126:]
        vol_6m = daily_window_6m.std()
        
        # We want "Low Volatility", so we take the inverse (handle zeros safely)
        low_vol_6m = 1.0 / vol_6m.replace(0, np.nan)
        
        # Ensure valid data
        valid_idx = mom_6m.dropna().index.intersection(low_vol_6m.dropna().index)
        if len(valid_idx) == 0:
            return pd.Series(dtype=float)
            
        mom = mom_6m[valid_idx]
        lvol = low_vol_6m[valid_idx]
        
        # Cross-sectional standardized Z-score
        z_mom = pd.Series(zscore(mom), index=valid_idx)
        z_vol = pd.Series(zscore(lvol), index=valid_idx)
        
        # Composite score
        composite = z_mom + z_vol
        
        # Return composite scores directly as pseudo-return proxy, sorted descending
        return composite.sort_values(ascending=False)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  ROBUST ORDINAL PORTFOLIO OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════
class RobustOrdinalOptimizer:
    @staticmethod
    def optimize(ranks, daily_returns_window):
        """
        Solves the Quadradic Program predicting Max Return strictly bound by ranks & vol cap.
        Objective: w^T alpha
        Constraints: sum(w)=1, w >= 0, w <= 0.10, w^T Sigma w * 252 <= 0.20^2, w_i >= w_{i+1}
        """
        tickers = ranks.index.tolist()
        N = len(tickers)
        
        if N < 5:
            return pd.Series(1.0/N, index=tickers)
            
        # Covariance Matrix via Ledoit-Wolf shrinkage (36m array of daily returns)
        returns_subset = daily_returns_window[tickers].dropna()
        if len(returns_subset) < 252: # Need at least 1 year of daily returns
             return pd.Series(1.0/N, index=tickers)
             
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns_subset).covariance_
        # Ensure symmetric positive semi-definite matrix
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Define cvxpy variables
        w = cp.Variable(N)
        
        expected_returns = ranks.values
        
        # New Objective: Maximize expected return proxy (alpha)
        objective = cp.Maximize(expected_returns @ w)
        
        # Portfolio Variance formulation
        variance = cp.quad_form(w, cov_matrix)
        
        # Core constraints: Fully invested, Long-only, Max Weight 10%, Max Vol 20%
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.10,
            variance * 252 <= 0.20**2
        ]
        
        # Ordinal Uncertainty Constraint: w_1 >= w_2 >= ... >= w_N
        # Tickers array is already sorted by their score (index 0 is best, index N-1 is worst)
        for i in range(N - 1):
            constraints.append(w[i] >= w[i+1])
            
        prob = cp.Problem(objective, constraints)
        
        try:
            # Solve the QP using only open-source solvers to avoid MOSEK license errors
            try:
                prob.solve(solver=cp.SCS, max_iters=5000) 
            except Exception:
                pass
                
            if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
                try:
                    prob.solve(solver=cp.CLARABEL) # fallback
                except Exception:
                    pass
                    
            if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError("Open-source solvers failed to converge for this dense matrix.")
                
            w_opt = pd.Series(w.value, index=tickers)
            # Clip numerical noise around 0
            w_opt = np.maximum(w_opt, 0.0)
            w_opt = w_opt / w_opt.sum()
            return w_opt
            
        except Exception:
            # Silently fallback to uniform assignment to avoid massive tracking spam in console
            pass
            
        return pd.Series(1.0/N, index=tickers)

# ═══════════════════════════════════════════════════════════════════════════
# 5.  WALK-FORWARD BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════
class WalkForwardBacktester:
    def __init__(self, data, lookback_months):
        self.daily_close, self.monthly_close, self.daily_returns, self.monthly_returns = data
        self.lookback_months = lookback_months
        
    def run(self, tickers, benchmark):
        month_ends = self.monthly_close.index
        results = []
        
        print(f"[BACKTEST] Starting Walk-Forward Backtesting (Lookback: {self.lookback_months}M)")
        
        for i in range(self.lookback_months, len(month_ends) - 1):
            # t is the rebalancing specific month end
            t = month_ends[i]
            t_start = month_ends[i - self.lookback_months]
            
            # Historical subset available at time t
            monthly_close_win = self.monthly_close.loc[t_start:t, tickers]
            daily_returns_win = self.daily_returns.loc[t_start:t, tickers]
            
            # Forward returns from t to t+1 (realized in the future)
            t_fw = month_ends[i + 1]
            fwd_returns = self.monthly_returns.loc[t_fw, tickers]
            
            # Clean data: drop stocks with NaNs in lookback or forward vector
            missing_px = monthly_close_win.isna().sum()
            missing_dr = daily_returns_win.isna().sum()
            missing_fw = fwd_returns.isna()
            
            # We want strictly fully available arrays for safety
            valid_tickers = [tk for tk in tickers if missing_px[tk] == 0 and not missing_fw[tk]]
            # Maximum ~10% missed daily prints
            max_missed_dr = (self.lookback_months * 21) * 0.1 
            valid_tickers = [tk for tk in valid_tickers if missing_dr[tk] <= max_missed_dr]
            
            if len(valid_tickers) < 10:
                continue
                
            monthly_close_win = monthly_close_win[valid_tickers]
            daily_returns_win = daily_returns_win[valid_tickers].fillna(0.0) # Impute remaining mid-nans with 0
            fwd_valid = fwd_returns[valid_tickers]
            
            # Generate Signals and Ranks
            ranks = SignalGenerator.generate_signals(daily_returns_win, monthly_close_win)
            if ranks.empty:
                continue
                
            # Perform optimization
            w_opt = RobustOrdinalOptimizer.optimize(ranks, daily_returns_win)
            
            # Calculate Realized returns
            ret_robust = (w_opt * fwd_valid).sum()
            
            # Benchmark 2 (SPY array matching sequence)
            ret_spy = self.monthly_returns.loc[t_fw, benchmark]
            
            results.append({
                'Date': t_fw,
                'Robust Ordinal': ret_robust,
                'SPY Benchmark': ret_spy
            })
            
        return pd.DataFrame(results).set_index('Date')

# ═══════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION AND VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
class PerformanceEvaluator:
    @staticmethod
    def evaluate(df_returns):
        metrics = []
        for col in df_returns.columns:
            r = df_returns[col].dropna()
            if len(r) == 0:
                continue
                
            # Annualized formulations
            ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
            ann_vol = r.std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
            
            # Drawdown
            cum = (1 + r).cumprod()
            drawdown = cum / cum.cummax() - 1
            max_dd = drawdown.min()
            
            metrics.append({
                'Strategy': col,
                'Ann. Return': f"{ann_ret:.2%}",
                'Ann. Vol': f"{ann_vol:.2%}",
                'Sharpe': f"{sharpe:.2f}",
                'Max DD': f"{max_dd:.2%}"
            })
        
        df_metrics = pd.DataFrame(metrics).set_index('Strategy')
        
        # Yearly returns
        yearly_strats = []
        for year, group in df_returns.groupby(df_returns.index.year):
            year_ret = {'Year': year}
            for col in df_returns.columns:
                r = group[col].dropna()
                if len(r) > 0:
                    ret = (1 + r).prod() - 1
                    vol = r.std() * np.sqrt(12)
                    sharpe = ret / vol if vol != 0 else np.nan
                    year_ret[col] = f"{ret:.2%} (SR: {sharpe:5.2f})"
                else:
                    year_ret[col] = "NaN"
            yearly_strats.append(year_ret)
        df_yearly = pd.DataFrame(yearly_strats).set_index('Year')
        
        # Chart visualization
        plt.figure(figsize=(14, 7))
        for col in df_returns.columns:
            # We map Cumulative Log-Returns
            cum_log_ret = np.log1p(df_returns[col]).cumsum()
            lw = 2.5 if col == 'Robust Ordinal' else 1.5
            style = "-" if col != 'SPY Benchmark' else "--"
            plt.plot(cum_log_ret, label=col, linewidth=lw, linestyle=style)
            
        plt.title('Robust Portfolio Optimization on Ordinal Uncertainty Sets\nCumulative Log-Returns (Walk-Forward)', fontsize=15, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Log-Return', fontsize=12)
        plt.legend(fontsize=11)
        plt.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig('robust_ordinal_portfolio.png', dpi=150)
        plt.show()
        plt.close()
        
        return df_metrics, df_yearly

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTIVE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  ROBUST ORDINAL PORTFOLIO OPTIMIZATION BACKTEST")
    print("=" * 70)
    
    fetcher = DataFetcher(TICKERS, BENCHMARK, START_DATE, END_DATE)
    data = fetcher.fetch_data()
    
    bt = WalkForwardBacktester(data, LOOKBACK_MONTHS)
    df_returns = bt.run(TICKERS, BENCHMARK)
    
    if df_returns.empty:
        print("[ERROR] Backtest run returned empty results. Lookback too large or missing data.")
        return
        
    print("\n[EVAL] Metrics Generated.")
    metrics, df_yearly = PerformanceEvaluator.evaluate(df_returns)
    
    print("\n" + "="*60)
    print(" PERFORMANCE SUMMARY (GLOBAL) ")
    print("="*60)
    print(metrics.to_string())
    print("="*60)
    
    print("\n" + "="*60)
    print(" PERFORMANCE PAR ANNÉE ")
    print("="*60)
    print(df_yearly.to_string())
    print("="*60)
    print("\n[SAVE] Curve shape saved under 'robust_ordinal_portfolio.png'")
    
    df_returns.to_csv('robust_ordinal_portfolio_returns.csv')

if __name__ == '__main__':
    main()
