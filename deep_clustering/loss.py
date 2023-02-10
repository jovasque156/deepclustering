import torch
import torch.nn as nn
import math

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(points, sensitive_attribute, t):
        '''
        Compute the contrastive loss

        Input:
            points: points to compute the loss on
            sensitive_attribute: sensitive attribute of the points
            m: margin

        Output:
            loss_c: contrastive loss
        '''
        # Compute the similarity matrix
        
        # ipdb.set_trace()

        # sensitive_attribute = sensitive_attribute.contiguous().view(-1, 1)
        # mask = torch.eq(sensitive_attribute, sensitive_attribute.T).float().to(DEVICE)
        
        device = (torch.device('cuda') if points.is_cuda else torch.device('cpu'))
        
        contrast_count = points.shape[0]

        loss = torch.zeros(points.shape[0]).to(device)
        
        for i, (p, s) in enumerate(zip(points, sensitive_attribute)):
            positives = points[sensitive_attribute==s]
            anchor_dot_contrast = p*positives/t
            logit_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logit_max.detach()
            
            # phi_p = torch.exp(torch.sum(logits, dim=-1))
            # phi_p /= phi_p.sum()
            # phi_p = torch.log(phi_p).sum()
            
            exp_logits_fair = torch.exp(logits)
            exp_logits_sum = exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))
            
            
            mean_log_prob = log_prob.sum(1)/positives.shape[0]
            
            loss[i] = -mean_log_prob.mean()

        return loss.mean()