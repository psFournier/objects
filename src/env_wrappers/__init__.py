from env_wrappers.registration import register

register(
    id='Objects-v0',
    entry_point='environments:Objects',
    wrapper_entry_point='env_wrappers.objects:Objects'
)

register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    wrapper_entry_point='env_wrappers.base:Base'
)