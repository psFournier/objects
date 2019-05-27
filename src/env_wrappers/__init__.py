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
        id='Objects3-v0',
        entry_point='environments.objects3:Objects3',
        kwargs={'seed': 1},
        wrapper_entry_point='env_wrappers.objects2:Objects2'
    )

register(
        id='Objects4-v0',
        entry_point='environments.objects4:Objects4',
        kwargs={'seed': 1},
        wrapper_entry_point='env_wrappers.objects1:Objects1'
    )

register(
        id='Objects5-v0',
        entry_point='environments.objects5:Objects5',
        kwargs={'seed': 1},
        wrapper_entry_point='env_wrappers.objects1:Objects1'
    )