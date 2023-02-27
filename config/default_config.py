AE_DEFAULT_CONFIG = {
    'embed_dim': 3,
    'z_channel': 3,
    'resolution': 256,
    'in_channels': 3,
    'out_ch': 3,
    'ch': 128,
    'ch_mult': '1,2,4',
    'num_res_blocks': 2,
    'droutput': 0.0,
}

# AE_DEFAULT_CONFIG = {
#     'embed_dim': 16,
#     'z_channel': 16,
#     'resolution': 256,
#     'in_channels': 3,
#     'out_ch': 3,
#     'ch': 128,
#     'ch_mult': '1,1,2,2,4',
#     'num_res_blocks': 2,
#     'attn_resolutions':'16'
#     'droutput': 0.0,
# }

UNET_DEFAULT_CONFIG = {
    'image_size': 256,
    'in_channels': 3,
    'out_channels': 3,
    'model_channels': 224,
    'attention_resolutions': '8,4,2',
    'unet_num_res_blocks': 2,
    'channel_mult': '1,2,3,4',
    'num_head_channels': 32
}

LDM_DEFAULT_CONFIG = {
    'linear_start': 0.0015,
    'linear_end': 0.0195,
    'log_every_t': 200,
    'timesteps': 1000,
    'first_stage_key': 'image',
    'image_size': 256,
    'channels': 3,
    'monitor': 'val/loss_simple_ema',
}