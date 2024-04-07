import pandas as pd
import math

class portfolio:
    def __init__(self,initial_amount,hold_period, buy_cost_pct,sell_cost_pct):
        self.hold_period = hold_period
        dtypes = {'tic': 'str', 'price': 'float','buy_price':'float','amount':'int','weight':'float','hold_on': 'int'}
        columns = ['tic','price','buy_price','amount','weight','hold_on']
        self.portfolio = pd.DataFrame([['cap',initial_amount,initial_amount,1,1,0]])
        self.portfolio.columns = columns
        self.portfolio = self.portfolio.astype(dtype=dtypes)
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct

    def add_buy_stock(self,tic,buy_price,prop):
        if buy_price > self.portfolio.loc[0].price:
            return 0,0
        
        cost = self.portfolio.loc[0].price*prop
        buy_amount = math.ceil(abs(cost/buy_price))
        buy_fee = buy_amount*buy_price*self.buy_cost_pct
        self.portfolio.loc[len(self.portfolio)] = [tic,buy_price,buy_price,buy_amount,0,self.hold_period]
        self.portfolio.loc[0,'price'] -= (buy_amount*buy_price + buy_fee)
        self._compute_weight()
        return buy_amount, buy_fee

    def update_new_state(self,tic,price):
        if self._check_empty(tic):
            return

        selected_tic = self.portfolio[self.portfolio.tic == tic].index
        for i in selected_tic:
            self.portfolio.loc[i,'price'] = price
            self.portfolio.loc[i,'hold_on'] -= 1 if self.portfolio.loc[i,'hold_on']>0 else 0
            
        self._compute_weight()

    def minus_sell_stock(self,tic,sell_price,prop):
        if self._check_empty(tic):
            return 0,0,0
            
        selected_tic = self.portfolio[self.portfolio.tic == tic].index
        sell_amount = sum(self.portfolio[(self.portfolio.tic == tic) & (self.portfolio.hold_on == 0)].amount)*abs(prop)
        sell_amount = math.ceil(sell_amount)
        sell_fee = sell_amount*sell_price*self.sell_cost_pct
        remain_sell_amount = sell_amount
        surplus = 0
        income = 0
        for i in selected_tic:
            if self.portfolio.loc[i,'hold_on'] > 0:
                continue
            if remain_sell_amount <= 0:
                break
            minus_amount = remain_sell_amount if self.portfolio.loc[i].amount > remain_sell_amount else self.portfolio.loc[i].amount
            surplus += minus_amount*(sell_price - self.portfolio.loc[i].buy_price)
            self.portfolio.loc[i,'amount'] -= minus_amount
            income += minus_amount*sell_price
            remain_sell_amount -= minus_amount
        self.portfolio = self.portfolio.drop(self.portfolio[(self.portfolio.amount <= 0) & (self.portfolio.tic == tic)].index)
        self.portfolio.reset_index(inplace=True,drop=True)
        self.portfolio.loc[0,'price'] += (income - sell_fee)
        self._compute_weight()
        return sell_amount,surplus,sell_fee

    def get_asset_value(self):
        return sum(self.portfolio.price * self.portfolio.amount)

    def get_remain_capital(self):
        return self.portfolio.loc[0].price

    def get_stock_amount(self,tic):
        if self._check_empty(tic):
            return 0
        return sum(self.portfolio[(self.portfolio.tic == tic) & (self.portfolio.hold_on == 0)].amount)

    def get_stock_profit(self,tic):
        if self._check_empty(tic):
            return 0
        selected_tic = self.portfolio[self.portfolio.tic == tic]
        return sum(selected_tic.price * selected_tic.amount) / sum(selected_tic.buy_price * selected_tic.amount)

    def _check_empty(self,tic):
        selected_tic = self.portfolio[self.portfolio.tic == tic].index
        return selected_tic.empty
    
    def _compute_weight(self):
        nav = sum(self.portfolio.price*self.portfolio.amount)
        self.portfolio['weight'] = self.portfolio.apply(lambda x: x.price * x.amount / nav, axis=1)