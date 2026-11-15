"""Calibrate the model on real Euro Stoxx 50 prices -> data/params.npz

Run this ONCE on a laptop or Google Colab (needs internet):
    pip install yfinance
    python get_data.py
then commit data/params.npz (it is tiny). If the file is missing, run.py falls
back to synthetic parameters, which is fine for timing experiments.
"""
from pathlib import Path

import numpy as np
import yfinance as yf

TICKERS = [
    "ASML.AS", "SAP.DE", "SIE.DE", "MC.PA", "TTE.PA", "ALV.DE", "SAN.PA",
    "OR.PA", "AIR.PA", "SU.PA", "DTE.DE", "IBE.MC", "SAN.MC", "BNP.PA",
    "ITX.MC", "ENEL.MI", "CS.PA", "ISP.MI", "BBVA.MC", "AI.PA", "MUV2.DE",
    "DG.PA", "SAF.PA", "UCG.MI", "RMS.PA", "EL.PA", "ABI.BR", "BAS.DE",
    "DB1.DE", "BMW.DE", "MBG.DE", "IFX.DE", "ADS.DE", "KER.PA", "ENI.MI",
    "PRX.AS", "AD.AS", "INGA.AS", "BN.PA", "NOKIA.HE", "RI.PA", "VOW3.DE",
    "BAYN.DE", "SGO.PA", "RACE.MI", "DHL.DE",
]
START, END = "2016-01-01", "2026-01-01"

prices = yf.download(TICKERS, start=START, end=END, auto_adjust=True,
                     progress=False)["Close"]
prices = prices.dropna(axis=1, thresh=int(0.95 * len(prices))).ffill().dropna()
rets = np.log(prices).diff().dropna()          # daily log returns

out = Path(__file__).resolve().parent / "data" / "params.npz"
out.parent.mkdir(exist_ok=True)
np.savez(out, mu=rets.mean().values, cov=rets.cov().values,
         tickers=np.array(rets.columns), returns=rets.values)  # returns: for bootstrap later
print(f"{rets.shape[1]} stocks, {rets.shape[0]} days ({rets.index[0].date()} "
      f"to {rets.index[-1].date()}) -> {out}")
