import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

X = torch.tensor([
    [0.0, 0.0],  # Cold & dry  -> Stay (0)
    [0.0, 1.0],  # Cold & rain -> Stay (0)
    [1.0, 0.0],  # Warm & dry  -> RUN  (1)
    [1.0, 1.0],  # Warm & rain -> Stay (0)
], device=device)

y = torch.tensor([[0.0], [0.0], [1.0], [0.0]], device=device)

class RunAdvisor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden  = nn.Linear(2, 4)   # 2 inputs -> 4 hidden neurons
        self.relu    = nn.ReLU()
        self.output  = nn.Linear(4, 1)   # 4 hidden neurons -> 1 output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

model = RunAdvisor().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)
for epoch in range(500):
    predictions = model(X)              # Forward pass (runs forward())
    loss = criterion(predictions, y)    # Measure mistake

    optimizer.zero_grad()               # Clear old gradients
    loss.backward()                     # Compute new gradients
    optimizer.step()                    # Update weights & biases

# 5. Test predictions without gradient tracking
with torch.no_grad():
    probs = model(X)
    for clues, prob in zip(X, probs):
        verdict = "GO RUN" if prob.item() >= 0.5 else "STAY"
        print(f"Clues {clues.tolist()} -> Confidence: {prob.item()*100:5.1f}% -> {verdict}")