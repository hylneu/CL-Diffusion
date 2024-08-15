import torch
import torch.nn as nn
import torch.optim as optim

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = nn.Parameter(torch.randn(self.head_dim))

        # Ensuring the hidden size is divisible by the number of heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        keys = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim)
        queries = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim)
        values = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim)

        # Scale the query
        queries = queries * self.scale

        # Dot product of keys and queries
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_dim ** 0.5
        attention = self.softmax(attention_scores)

        # Weighted sum of values
        weighted_sum = torch.matmul(attention, values).view(batch_size, -1, self.head_dim * self.num_heads)

        # Final linear transformation
        transformed = self.output(weighted_sum)
        return transformed


class Image_adapter(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=8):
        super().__init__()
        self.adapter = MultiHeadAttention(hidden_size, num_heads=num_heads)

        # Neural network to generate the mask
        self.mask_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()  # Ensure the mask values are between 0 and 1
        )

    def forward(self, feature):
        # Generate feature-level mask using the neural network
        mask = self.mask_generator(feature)

        # Apply feature-level mask
        masked_feature = mask * feature
        adapted_feature = self.adapter(masked_feature)
        out_feature = adapted_feature + masked_feature
        return out_feature



# Example usage
if __name__ == "__main__":
    batch_size = 2
    hidden_size = 1024
    num_heads = 8

    # Dummy input
    feature = torch.randn(batch_size, hidden_size)

    # Initialize and test Image_adapter
    adapter = Image_adapter(hidden_size, num_heads)
    output = adapter(feature)
    print(output.shape)



    # Initialize model
    hidden_size = 1024
    num_heads = 8
    adapter = Image_adapter(hidden_size, num_heads)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(adapter.parameters(), lr=0.001)

    # Regularization strength
    l2_lambda = 0.01

    # Dummy data for demonstration
    batch_size = 2
    features = torch.randn(batch_size, hidden_size)
    labels = torch.randn(batch_size, hidden_size)

    # Training loop
    for epoch in range(10):  # Example number of epochs
        adapter.train()

        # Forward pass
        outputs = adapter(features)
        loss = criterion(outputs, labels)

        # L2 regularization
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in adapter.mask_generator.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)

        # Add L2 regularization to the loss
        loss = loss + l2_lambda * l2_reg

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/10], Loss: {loss.item()}')


def cal_cos(text, img, cos):
    a = text.mean(dim=1)
    b = img.squeeze(0)
    sim = cos(a, b).mean()
    return sim