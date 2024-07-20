import torch

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        # The shape of the pred: torch.Size([16, 12, 64, 64])
        num_parts = 12
        eyeIdx = 10
        gt_1 = gt[:, :num_parts]
        gt_2 = gt[:, num_parts:]
        
        pred_1 = pred[:,:num_parts].clone()
        pred_2 = pred[:,num_parts:].clone()

        # Lets get their backbone right
        regLoss  = ((pred_1[:,:eyeIdx] - gt_1[:,:eyeIdx])**2).mean(dim = 3).mean(dim = 2).mean(dim = 1)
        #flipLoss = ((pred_1[:,:eyeIdx] - gt_1[:,:eyeIdx])**2).mean(dim = 3).mean(dim = 2).mean(dim = 1)
        flipLoss = ((pred_2[:,:eyeIdx] - gt_1[:,:eyeIdx])**2).mean(dim = 3).mean(dim = 2).mean(dim = 1)
        mask = flipLoss < regLoss
        pred_1[mask] = pred_2[mask].clone()
        pred_2[mask] = pred[mask,:num_parts].clone()

        # Lets get their eyes right
        pred_1_eyeSwitched = pred_1.clone()
        pred_1_eyeSwitched[:, eyeIdx:] = torch.flip( pred_1_eyeSwitched[:, eyeIdx:], [1])
        regLoss = ((pred_1[:,eyeIdx:] - gt_1[:,eyeIdx:])**2).mean(dim = 3).mean(dim = 2).mean(dim = 1)
        flipLoss = ((pred_1_eyeSwitched[:, eyeIdx:] - gt_1[:, eyeIdx:])**2).mean(dim = 3).mean(dim = 2).mean(dim = 1)
        mask = flipLoss < regLoss
        
        pred_1[mask] = pred_1_eyeSwitched[mask]


        pred_2_eyeSwitched = pred_2.clone()
        pred_2_eyeSwitched[:, eyeIdx:] = torch.flip( pred_2_eyeSwitched[:, eyeIdx:], [1])
        regLoss = ((pred_2[:,eyeIdx:] - gt_2[:,eyeIdx:])**2).mean(dim = 3).mean(dim = 2).mean(dim = 1)
        flipLoss = ((pred_2_eyeSwitched[:, eyeIdx:] - gt_2[:, eyeIdx:])**2).mean(dim = 3).mean(dim = 2).mean(dim = 1)
        mask = flipLoss < regLoss

        pred_2[mask] = pred_2_eyeSwitched[mask]        
        
        pred[:,:num_parts] = pred_1
        pred[:,num_parts:] = pred_2

        #t2[:, 10:] = torch.flip(t2[:,10:], [1]  )

        
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize
