"""Build official HDiT with six input and three output channels.

No thesis-missing architecture values are supplied by this factory.
The generic upstream make_model ties input/output channel counts together,
so construct ImageTransformerDenoiserModelV2 directly instead.
"""
import importlib
import math


def validate_spec(spec):
    required = {'levels', 'mapping', 'patch_size'}
    if set(spec) != required:
        raise ValueError('Specify exactly levels, mapping and patch_size')
    def positive(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if len(spec['patch_size']) != 2 or not all(positive(v) for v in spec['patch_size']):
        raise ValueError('patch_size needs two positive integers')
    if not spec['levels']:
        raise ValueError('At least one level is required')
    for level in spec['levels']:
        if set(level) != {'depth', 'width', 'd_ff', 'dropout', 'attention'}:
            raise ValueError('Each level needs explicit depth, width, d_ff, dropout, attention')
        if not all(positive(level[k]) for k in ('depth', 'width', 'd_ff')):
            raise ValueError('Invalid level dimensions')
        if not math.isfinite(level['dropout']) or not 0 <= level['dropout'] < 1:
            raise ValueError('Invalid dropout')
        attention = level['attention']
        kind = attention.get('type')
        expected = {'type', 'd_head'} | ({'kernel_size'} if kind == 'neighborhood' else set())
        if kind not in ('global', 'neighborhood') or set(attention) != expected:
            raise ValueError('Choose explicit global or neighborhood attention')
        if not positive(attention['d_head']) or level['width'] % attention['d_head']:
            raise ValueError('width must be divisible by d_head')
        if kind == 'neighborhood' and (not positive(attention['kernel_size']) or attention['kernel_size'] % 2 == 0):
            raise ValueError('Neighborhood kernel must be positive and odd')
    mapping = spec['mapping']
    if set(mapping) != {'depth', 'width', 'd_ff', 'dropout'}:
        raise ValueError('Mapping parameters must be explicit')
    if not all(positive(mapping[k]) for k in ('depth', 'width', 'd_ff')):
        raise ValueError('Invalid mapping dimensions')
    if not math.isfinite(mapping['dropout']) or not 0 <= mapping['dropout'] < 1:
        raise ValueError('Invalid mapping dropout')


def build_hdit(spec):
    validate_spec(spec)
    try:
        upstream = importlib.import_module('k_diffusion.models.image_transformer_v2')
    except ImportError as error:
        raise RuntimeError('Install the verified official k-diffusion environment first') from error
    levels = []
    for level in spec['levels']:
        attention = level['attention']
        if attention['type'] == 'neighborhood':
            if getattr(upstream, 'natten', None) is None:
                raise RuntimeError('Neighborhood attention requires NATTEN; no automatic fallback')
            attn = upstream.NeighborhoodAttentionSpec(attention['d_head'], attention['kernel_size'])
        else:
            attn = upstream.GlobalAttentionSpec(attention['d_head'])
        levels.append(upstream.LevelSpec(level['depth'], level['width'], level['d_ff'], attn, level['dropout']))
    m = spec['mapping']
    mapping = upstream.MappingSpec(m['depth'], m['width'], m['d_ff'], m['dropout'])
    return upstream.ImageTransformerDenoiserModelV2(
        levels=levels, mapping=mapping, in_channels=6, out_channels=3,
        patch_size=spec['patch_size'], num_classes=0, mapping_cond_dim=0)
