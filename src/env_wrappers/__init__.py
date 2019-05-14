from env_wrappers.registration import register

# for nbObjects in [1, 2, 4, 8, 16, 32, 64]:
#     register(
#         id='{}_Objects-v0'.format(nbObjects),
#         entry_point='environments:Objects',
#         kwargs={'nbObjects': nbObjects},
#         wrapper_entry_point='env_wrappers.objects:Objects'
#     )

register(
    id='MountainCar-v0',
    entry_point='environments.mountainCar:MountainCarEnv',
    wrapper_entry_point='env_wrappers.mountaincar:MountainCar'
)

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    wrapper_entry_point='env_wrappers.base:Base'
)

register(
        id='Objects-v0',
        entry_point='environments:ObjectsPlayroom',
        kwargs={'nbObjects': 10},
        wrapper_entry_point='env_wrappers.objects:Objects'
    )