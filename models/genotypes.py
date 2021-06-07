from collections import namedtuple


Genotype_all = namedtuple('Genotype', 'fusion1 fusion1_concat fusion2 fusion2_concat fusion3 fusion3_concat aggregation aggregation_concat final_agg final_aggregation_concat low_high_agg low_high_agg_concat')
"""
Operation sets
"""
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'dil_conv_3x3',
    'conv_1x1',
    'conv_3x3',
    'spatial_attention',
    'channel_attention'
]


attention_snas_3_4_1 = Genotype_all(fusion1=[('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('spatial_attention', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 4), ('conv_1x1', 2), ('channel_attention', 3), ('conv_1x1', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 5), ('conv_1x1', 0), ('conv_1x1', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2), ('sep_conv_3x3', 6), ('conv_1x1', 0), ('conv_1x1', 4), ('dil_conv_3x3', 7), ('dil_conv_3x3', 1), ('skip_connect', 5), ('dil_conv_3x3', 0), ('conv_1x1', 1), ('conv_3x3', 4), ('conv_3x3', 8), ('spatial_attention', 3), ('sep_conv_3x3', 0), ('channel_attention', 4), ('conv_1x1', 5), ('conv_1x1', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 8), ('dil_conv_3x3', 9)], fusion1_concat=range(6, 12), fusion2=[('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('spatial_attention', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 4), ('conv_1x1', 2), ('channel_attention', 3), ('conv_1x1', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 5), ('conv_1x1', 0), ('conv_1x1', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2), ('sep_conv_3x3', 6), ('conv_1x1', 0), ('conv_1x1', 4), ('dil_conv_3x3', 7), ('dil_conv_3x3', 1), ('skip_connect', 5), ('dil_conv_3x3', 0), ('conv_1x1', 1), ('conv_3x3', 4), ('conv_3x3', 8), ('spatial_attention', 3), ('sep_conv_3x3', 0), ('channel_attention', 4), ('conv_1x1', 5), ('conv_1x1', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 8), ('dil_conv_3x3', 9)], fusion2_concat=range(6, 12), fusion3=[('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('spatial_attention', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 4), ('conv_1x1', 2), ('channel_attention', 3), ('conv_1x1', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 5), ('conv_1x1', 0), ('conv_1x1', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2), ('sep_conv_3x3', 6), ('conv_1x1', 0), ('conv_1x1', 4), ('dil_conv_3x3', 7), ('dil_conv_3x3', 1), ('skip_connect', 5), ('dil_conv_3x3', 0), ('conv_1x1', 1), ('conv_3x3', 4), ('conv_3x3', 8), ('spatial_attention', 3), ('sep_conv_3x3', 0), ('channel_attention', 4), ('conv_1x1', 5), ('conv_1x1', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 8), ('dil_conv_3x3', 9)], fusion3_concat=range(6, 12), aggregation=[('spatial_attention', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('spatial_attention', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 3), ('spatial_attention', 1), ('conv_3x3', 4), ('conv_1x1', 2), ('conv_3x3', 0), ('sep_conv_3x3', 5), ('dil_conv_3x3', 3), ('conv_3x3', 1), ('conv_1x1', 5), ('dil_conv_3x3', 0), ('channel_attention', 3), ('spatial_attention', 4), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 3), ('conv_3x3', 4), ('channel_attention', 3), ('skip_connect', 1), ('sep_conv_3x3', 6)], aggregation_concat=range(5, 11), final_agg=[('conv_1x1', 1), ('conv_1x1', 0), ('max_pool_3x3', 2), ('conv_1x1', 3), ('channel_attention', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('conv_1x1', 0), ('spatial_attention', 2), ('sep_conv_3x3', 4), ('conv_1x1', 5), ('conv_3x3', 0), ('dil_conv_3x3', 4), ('conv_1x1', 2), ('conv_1x1', 1), ('dil_conv_3x3', 6), ('conv_1x1', 4), ('skip_connect', 2), ('conv_1x1', 7), ('max_pool_3x3', 6), ('conv_3x3', 5), ('channel_attention', 7), ('max_pool_3x3', 2), ('conv_3x3', 4), ('spatial_attention', 7), ('max_pool_3x3', 3), ('sep_conv_3x3', 0), ('spatial_attention', 8), ('max_pool_3x3', 2), ('conv_1x1', 4), ('sep_conv_3x3', 5), ('conv_1x1', 3)], final_aggregation_concat=range(6, 12), low_high_agg=[('max_pool_3x3', 2), ('spatial_attention', 1), ('conv_3x3', 0), ('channel_attention', 1), ('max_pool_3x3', 2), ('conv_1x1', 3), ('max_pool_3x3', 3), ('channel_attention', 1), ('conv_3x3', 4), ('max_pool_3x3', 3), ('skip_connect', 1), ('conv_1x1', 2)], low_high_agg_concat=range(3, 7))
