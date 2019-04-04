from env_wrappers.registration import register

register(
    id='Objects-v0',
    entry_point='environments:Objects',
    wrapper_entry_point='env_wrappers.objects:Objects'
)