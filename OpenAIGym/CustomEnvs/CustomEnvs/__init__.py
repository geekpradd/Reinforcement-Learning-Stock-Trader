from gym.envs.registration import register

register(
    id='stock-v0',
    entry_point='CustomEnvs.envs:StockTradingEnv',
)