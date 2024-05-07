import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from portfolio import portfolio

matplotlib.use("Agg")

# from stable_baselines3.common import logger


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        stop_loss,
        hold_period,
        make_plots=False,
        print_verbosity=10,
        row=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        # self.row = row
        self.df = df
        # self.stock_dim = stock_dim
        self.hmax = hmax
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list        
        self.initial_amount = initial_amount
        self.hold_period = hold_period
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.stop_loss = stop_loss # the game stops when the asset loses more than stop_loss percent
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        # self.turbulence_threshold = turbulence_threshold
        # self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.tic_list = self.df.tic.unique()
        self.original_df = self.df.copy()
        self.row = 0
        
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        # self.reset()
        self._seed()

    def _buy_stock(self, action):
        def _do_buy():
            if self.data.close > 0: # Buy only if the price is > 0 (no missing data in this particular date)
                buy_num_shares, buy_fee = self.portfolio.add_buy_stock(self.data.tic,self.data.close,action)
                # print(f'Buy amount: {buy_num_shares}')
                self.cost += buy_fee
            else:
                buy_num_shares = 0

            return buy_num_shares

        buy_num_shares = _do_buy()
        return buy_num_shares
    
    def _sell_stock(self, action):
        def _do_sell_normal():
            if self.data.close > 0: # Sell only if the price is > 0 (no missing data in this particular date)
                sell_amount,surplus,sell_fee = self.portfolio.minus_sell_stock(self.data.tic,self.data.close,action)
                self.cost += sell_fee
                if sell_amount > 0:
                    self.win_trade += 1 if surplus > 0 else 0
                    self.trades += 1
            else:
                sell_amount = 0

            return sell_amount

        sell_amount = _do_sell_normal()
        return sell_amount

    def step(self, actions):

        self.terminal = (self.row >= len(self.df.index.unique()) - 1) | (self.portfolio.get_asset_value() < self.initial_amount*(1-self.stop_loss))

        # Reset reward to zero
        # self.reward = 0
        
        # --> IN CASE THE STEP IS A TERMINATED STEP
        if self.terminal:
            
            # Summary the training performance after an episode
            end_total_asset = self.portfolio.get_asset_value()
            surplus_from_buy_hold = end_total_asset - self.initial_amount * (self.df.iloc[-1].close / self.df.iloc[0].close) # compare with buy-and-hold strategy
            tot_reward = end_total_asset - self.initial_amount
            
            # Show at each episode
            print(f"Episode: {self.episode}, com: {self.df.iloc[0].tic}, win trade: {self.win_trade}/{self.trades},"+
                f" Total profit: {tot_reward} ,Surplus from buy-hold: {surplus_from_buy_hold}")

            # Print out training results after a certain amount of episodes
            if self.episode % self.print_verbosity == 0:
                print(f"Current company: {self.df.iloc[0].tic}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total profit: {tot_reward:0.2f}")
                print(f"surplus from buy-and-hold: {surplus_from_buy_hold:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                # if df_total_value["daily_return"].std() != 0:
                #     print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")
            
            truncated = False  # we do not limit the number of steps here
            # Optionally we can pass additional info, we are not using that for now
            info = {}


            return (
                np.array(self.state).astype(np.float32),
                self.reward,
                self.terminal,
                truncated,
                info,
            )

        # --> IN A NORMAL STEP
        else: 

            begin_total_asset = self.portfolio.get_asset_value()
            
            # Act according to actions
            action = actions[0]
                
            if action > 0:
                self._buy_stock(action)
            elif action < 0:
                self._sell_stock(action)

            self.current_actions = actions
            self.actions_memory.append(actions)

            # self.accumulated_reward += self.reward

            # Update selected row in the dataset based on state: s -> s+1
            self.row += 1
            self.data = self.df.loc[self.row]
            self.state = self._update_state()

            end_total_asset = self.portfolio.get_asset_value()

            # Update asset memory
            # self.current_asset = end_total_asset
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling
            self.rewards_memory.append(self.reward)

        truncated = False  # we do not limit the number of steps here
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        # return self.state, self.reward, self.terminal, {}
    
        return (
            np.array(self.state).astype(np.float32),
            self.reward,
            self.terminal,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        # initiate state
        self.state = self._initiate_state()

        # Reset asset_memory
        self.asset_memory = [self.initial_amount]

        # Reset support variables
        self.cost = 0
        self.trades = 0
        self.win_trade = 0
        self.terminal = False
        # self.accumulated_reward = 0
        self.block_remain = 0
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1

        return np.array(self.state).astype(np.float32), {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        
        # Reset portfolio & previous_portfolio
        self.portfolio = portfolio(initial_amount=self.initial_amount,hold_period=self.hold_period,
                                   buy_cost_pct=self.buy_cost_pct,sell_cost_pct=self.sell_cost_pct)

        # Select a random ticker from df
        self.df = self.original_df[self.original_df.tic == random.choice(self.tic_list)].reset_index(drop=True)
        self.punishment_rate = 1/(len(self.df)*10)
        
        # Reset data
        self.row = 0
        self.data = self.df.loc[self.row]
        
         # Reset state
        state = ([self.portfolio.get_remain_capital()]
                    + [self.portfolio.get_stock_amount(self.data.tic)]
                    + [self.portfolio.get_stock_weight(self.data.tic)]
                    + [self.portfolio.get_stock_profit(self.data.tic)]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], []))
        
        return state

    def _update_state(self):

        self.portfolio.update_new_state(self.data.tic,self.data.close)
        state = ([self.portfolio.get_remain_capital()]
                    + [self.portfolio.get_stock_amount(self.data.tic)]
                    + [self.portfolio.get_stock_weight(self.data.tic)]
                    + [self.portfolio.get_stock_profit(self.data.tic)]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], []))

        return state

    def _get_date(self):
        return self.row

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({"date": date_list, "account_value": asset_list})
        return df_account_value

    def save_action_memory(self):
        
        date_list = self.date_memory[:-1]
        action_list = self.actions_memory
        df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs