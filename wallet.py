import numpy as np
import pandas as pd

class Wallet:
  def __init__(self, data, **kwargs):
    self.current_step=0
    self.data=data['close'].values
    self.base_precision: int = kwargs.get('base_precision', 2)
    self.asset_precision: int = kwargs.get('asset_precision', 8)
    self.exchange_rate=self.data[self.current_step]
    self.default_usd: float = kwargs.get('usd', 700)
    self.default_btc: float = kwargs.get('btc', 0)
    self.usd: float = kwargs.get('usd', 700)
    self.btc: float = kwargs.get('btc', 0)
    self.commission: float=kwargs.get('commision', 0.01)

  def reset(self):
    self.current_step=0
    self.usd= self.default_usd
    self.btc= self.default_btc
  def step(self, step=1):
    self.current_step+=step
    if self.current_step==len(self.data):
      self.exchange_rate=self.data[len(self.data)-1]
    else:
      self.exchange_rate=self.data[self.current_step]

  def to_usd(self, btc):
    return round(btc*self.exchange_rate, self.base_precision)

  def to_btc(self, usd):
    return round(usd / self.exchange_rate, self.asset_precision)

  def total(self):
    return round((self.usd+self.to_usd(self.btc)), self.base_precision)

  def buy(self, percents):
    result=False
    amountUSD=self.usd*percents
    if self.usd>round((amountUSD+(amountUSD*self.commission)), self.base_precision):
      self.usd-=round((amountUSD+(amountUSD*self.commission)), self.base_precision)
      self.btc+=self.to_btc(amountUSD)
      result=True
    return result

  def sell(self, percents):
    result=False
    if self.btc>0:
      amountBTC=round((self.btc*percents), self.asset_precision)
      self.btc-=amountBTC
      amountUSD=self.to_usd(amountBTC)
      self.usd+=amountUSD
      self.usd-=(amountUSD*self.commission)
      result=True
    return result
