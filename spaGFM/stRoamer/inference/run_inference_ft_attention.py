import torch
from spaGFM.stRoamer.inference.inference_ft_attention import inference_attention
from spaGFM.stRoamer.utils.rw_sampler import get_patterns
from spaGFM.stRoamer.utils.helper import set_seed

def run_inference_ft_attention(
    graph,
    model_file,
    device,
    batch_size=256,
    use_amp=False,
    amp_dtype="bfloat16",
):

    model_files = torch.load(model_file, weights_only=True, map_location=device)

    set_seed(12345)
    patterns = get_patterns(graph, model_files['ft_params'], batch_size=8192)

    params = model_files['params']
    ft_params = model_files['ft_params']
    ft_params['device'] = device
    ft_params['use_amp'] = use_amp
    ft_params['amp_dtype'] = amp_dtype
    
    correlations = inference_attention(graph, patterns, model_file, params, ft_params, batch_size=batch_size)

    return correlations, patterns
