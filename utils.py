import torch

def mmd_loss_laplacian(samples1, samples2, sigma=0.2):
    """MMD constraint with Laplacian kernel for support matching"""
    # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
    diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
    diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
    diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss


def mmd_loss_gaussian(samples1, samples2, sigma=0.2):
    """MMD constraint with Gaussian Kernel support matching"""
    # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
    diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
    diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
    diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss


