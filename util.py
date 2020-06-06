import torch
from skimage.color import rgb2lab, lab2rgb

def convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)

def generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = convert(output, out_type)
        return output.to(device)
    return apply_transform

rgb_to_lab = generic_transform_sk_4d(rgb2lab)
lab_to_rgb = generic_transform_sk_4d(lab2rgb)
