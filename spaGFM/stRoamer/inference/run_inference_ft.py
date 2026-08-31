from spaGFM.stRoamer.inference.inference_ft import inference_ft
import torch
from spaGFM.stRoamer.utils.rw_sampler import get_patterns
from spaGFM.stRoamer.utils.helper import set_seed

def run_inference_ft(graph, model_file, device, batch_size=256, use_amp=False, amp_dtype="bfloat16"):

    model_files = torch.load(model_file, weights_only=True, map_location=device)
    params = model_files['ft_params']
    set_seed(2025)
    patterns = get_patterns(graph, params, batch_size=8192)
    params['device'] = device
    params['use_amp'] = use_amp
    params['amp_dtype'] = amp_dtype
    
    attn_weights_output, logits_output = inference_ft(graph, patterns, model_file, params, batch_size=batch_size)
    return attn_weights_output, logits_output
