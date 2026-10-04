"""Teacher denoised-image loss weights and a strictly scoped fork transition."""
from .teacher_reconstruction import reconstruction_weight


def weighting_mode(config):
    mode = config['loss']['weighting']
    if mode not in ('edm', 'soft_min_snr'):
        raise ValueError('unsupported teacher loss weighting')
    return mode


def denoising_weights(sigmas, sigma_data, mode):
    if mode == 'edm':
        return (sigmas.square() + sigma_data**2) / (sigmas * sigma_data).square()
    if mode == 'soft_min_snr':
        # Official preconditioned-output soft weight divided by c_out squared.
        return 1.0 / (sigmas.square() + sigma_data**2)
    raise ValueError('unsupported teacher loss weighting')


def validate_soft_min_snr_transition(saved, current):
    if weighting_mode(saved) != 'edm' or weighting_mode(current) != 'soft_min_snr':
        raise ValueError('approved transition is EDM to soft_min_snr only')
    if reconstruction_weight(saved) != 0 or reconstruction_weight(current) != 0:
        raise ValueError('soft_min_snr pilot cannot include extra reconstruction loss')
    maximum = current['maximum_epochs']
    if maximum != saved['maximum_epochs'] + 10:
        raise ValueError('soft_min_snr pilot must extend exactly ten epochs')
    expected = {**saved, 'maximum_epochs': maximum,
                'stage_epochs': [*saved['stage_epochs'], maximum],
                'loss': {**saved['loss'], 'weighting': 'soft_min_snr'}}
    if current != expected:
        raise ValueError('soft_min_snr transition changed unrelated configuration')
