import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kinetic_chains import get_kinetic_chain_matrix

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, 1, 1))
        nn.init.kaiming_normal_(self.weight, mode='fan_in')

    def forward(self, x):
        # N, C, T, V = x.size()
        x = torch.einsum('nctv,cvw->nctw', (x, self.A.to(x.device)))
        x = F.conv2d(x, self.weight, padding=(0, 0))
        return x

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class st_gcn_block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(st_gcn_block, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x += res
        return self.relu(x)

class ScienceInformedAnalyzer(nn.Module):
    def __init__(self, num_classes_correctness, num_classes_quality, num_points=25, in_channels=3):
        super(ScienceInformedAnalyzer, self).__init__()
        
        self.num_points = num_points
        
        # Define the 5PKC partitioning strategy
        self.A = self.get_5pkc_adjacency_matrix()
        
        # Define the ST-GCN model architecture
        self.st_gcn_networks = nn.Sequential(
            st_gcn_block(in_channels, 64, self.A, residual=False),
            st_gcn_block(64, 64, self.A),
            st_gcn_block(64, 64, self.A),
            st_gcn_block(64, 128, self.A, stride=2),
            st_gcn_block(128, 128, self.A),
            st_gcn_block(128, 128, self.A),
            st_gcn_block(128, 256, self.A, stride=2),
            st_gcn_block(256, 256, self.A),
            st_gcn_block(256, 256, self.A)
        )
        
        # Define the two heads for the multi-task loss
        self.correctness_head = nn.Linear(256, num_classes_correctness)
        self.quality_head = nn.Linear(256, num_classes_quality)

    def get_5pkc_adjacency_matrix(self):
        """
        Gets the 5PKC adjacency matrix from the kinetic_chains helper.
        """
        A = get_kinetic_chain_matrix(self.num_points)
        return torch.tensor(A, dtype=torch.float32, requires_grad=False)

    def forward(self, x):
        # Pass input through the ST-GCN model
        x = self.st_gcn_networks(x)
        
        # Global Average Pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        # Get the output from each head
        correctness_output = self.correctness_head(x)
        quality_output = self.quality_head(x)
        
        return correctness_output, quality_output

def multi_task_loss(correctness_pred, correctness_true, quality_pred, quality_true, alpha=0.3):
    """
    Multi-task loss function as described in the paper.
    alpha = 0.3 was found to be the best value in the paper's experiments.
    """
    loss_correctness = nn.CrossEntropyLoss()(correctness_pred, correctness_true)
    loss_quality = nn.CrossEntropyLoss()(quality_pred, quality_true)
    
    return alpha * loss_correctness + (1 - alpha) * loss_quality

if __name__ == '__main__':
    # Example usage for NTU-RGB+D. It has 60 action classes.
    # The paper doesn't use the quality evaluation for this dataset,
    # so we'll set num_classes_quality to a placeholder value (e.g., 1)
    # and use a single loss for the main task.
    
    num_action_classes = 60
    model = ScienceInformedAnalyzer(num_classes_correctness=num_action_classes, num_classes_quality=1)
    
    # Example input tensor (Batch, Channels, Frames, Vertices)
    # This would come from a pre-processed video from the NTU-RGB+D dataset
    example_input = torch.randn(64, 3, 300, 25) # Batch size 64, 3 coords, 300 frames, 25 joints
    
    correctness, quality = model(example_input)
    
    print("Model created successfully for NTU-RGB+D.")
    print("Correctness output shape:", correctness.shape)
    print("Quality output shape (placeholder):", quality.shape)
