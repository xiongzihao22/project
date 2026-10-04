"""Stratified draws from the unchanged clamped lognormal sigma distribution."""


def sampling_mode(config):
    mode = config.get('sigma_sampling', 'independent')
    if mode not in ('independent', 'stratified_lognormal'):
        raise ValueError('unsupported teacher sigma sampler')
    if mode == 'stratified_lognormal' and 'high_noise_mixture' in config['sigma']:
        raise ValueError('stratified pilot cannot include a high-noise mixture')
    return mode


def validate_stratified_transition(saved, current):
    if sampling_mode(saved) != 'independent' or sampling_mode(current) != 'stratified_lognormal':
        raise ValueError('approved sampler transition is independent to stratified_lognormal')
    if saved['loss'] != {'reduction': 'pixel_mean', 'weighting': 'edm'}:
        raise ValueError('stratified pilot requires the original EDM loss without additions')
    maximum = current['maximum_epochs']
    if maximum != saved['maximum_epochs'] + 10:
        raise ValueError('stratified pilot must extend exactly ten epochs')
    expected = {**saved, 'maximum_epochs': maximum,
                'stage_epochs': [*saved['stage_epochs'], maximum],
                'sigma_sampling': 'stratified_lognormal'}
    if current != expected:
        raise ValueError('sampler transition changed unrelated configuration')


def stratified_from_normals(normals, sigma_config, generator):
    import torch
    if normals.ndim != 1 or normals.numel() < 1:
        raise ValueError('one nonempty batch of standard normals is required')
    if generator is None or generator.device.type != 'cpu':
        raise ValueError('stratum permutations require a separate CPU generator')
    n = normals.numel()
    strata = torch.randperm(n, generator=generator)
    # Transform the existing normal draws, preserving the main RNG consumption.
    jitter = torch.special.ndtr(normals.double()).clamp(1e-12, 1 - 1e-12)
    u = (strata.to(normals.device) + jitter) / n
    z = torch.special.ndtri(u)
    sigmas = (z * sigma_config['log_normal_std'] + sigma_config['log_normal_mean']).exp()
    sigmas = sigmas.to(normals.dtype).clamp(sigma_config['minimum'], sigma_config['maximum'])
    return sigmas, strata.tolist()


def sample_training_sigmas(batch_size, config, device, generator):
    import torch
    from .train_teacher_short import sample_sigmas
    if sampling_mode(config) == 'independent':
        return sample_sigmas(batch_size, config['sigma'], device), None
    normals = torch.randn(batch_size, device=device)
    return stratified_from_normals(normals, config['sigma'], generator)


def restore_sampler_rng(generator, checkpoint):
    if 'stratified_rng_state' in checkpoint:
        generator.set_state(checkpoint['stratified_rng_state'])
    elif sampling_mode(checkpoint['config']) == 'stratified_lognormal':
        raise ValueError('stratified checkpoint lacks its permutation RNG state')
