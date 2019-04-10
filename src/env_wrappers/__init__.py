from env_wrappers.registration import register

for nbObjects in [1, 2, 4, 8, 16, 32, 64]:
    register(
        id='{}_Objects-v0'.format(nbObjects),
        entry_point='environments:Objects',
        kwargs={'nbObjects': nbObjects},
        wrapper_entry_point='env_wrappers.objects:Objects'
    )

register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    wrapper_entry_point='env_wrappers.base:Base'
)