import torch
import torch.nn as nn
import torch.optim as optim

# Setup GPU/CPU and lock random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# =========================================================
# MILESTONE 1: DATASET (Split into Train and Unseen Test)
# Columns: [Kills, Deaths, Assists, Objectives]
# =========================================================

# 10 matches for the model to learn from
X_train = torch.tensor([
    [22.0,  8.0, 15.0, 4.0],  # High kills, low deaths, good objectives -> Win (1)
    [ 6.0, 20.0,  5.0, 0.0],  # Few kills, many deaths, no objectives   -> Loss (0)
    [18.0, 10.0, 12.0, 3.0],  # Solid all-around game                   -> Win (1)
    [10.0, 19.0,  8.0, 1.0],  # Behind on kills & objectives            -> Loss (0)
    [25.0, 12.0, 18.0, 5.0],  # Dominant game                           -> Win (1)
    [ 8.0, 15.0,  7.0, 1.0],  # Losing game                             -> Loss (0)
    [14.0, 14.0, 16.0, 4.0],  # Even kills, won via assists/objectives  -> Win (1)
    [19.0, 18.0,  9.0, 0.0],  # High kills, but died too much & 0 obj   -> Loss (0)
    [21.0,  9.0, 14.0, 3.0],  # Strong win                              -> Win (1)
    [ 7.0, 17.0,  6.0, 0.0],  # Heavy loss                              -> Loss (0)
], device=device)

y_train = torch.tensor([
    [1.0], [0.0], [1.0], [0.0], [1.0],
    [0.0], [1.0], [0.0], [1.0], [0.0]
], device=device)

# 4 brand-new matches kept locked away for the final exam
X_test = torch.tensor([
    [20.0,  7.0, 16.0, 4.0],  # Should predict WIN  (1)
    [ 5.0, 21.0,  4.0, 0.0],  # Should predict LOSS (0)
    [15.0, 13.0, 15.0, 4.0],  # Should predict WIN  (1)
    [16.0, 19.0,  7.0, 0.0],  # Should predict LOSS (0)
], device=device)

y_test = torch.tensor([[1.0], [0.0], [1.0], [0.0]], device=device)


# =========================================================
# MILESTONE 2: BUILD THE NETWORK (Fill in the 5 TODOs!)
# Architecture: 4 inputs -> 8 hidden -> 4 hidden -> 1 output
# =========================================================
class WinPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 1: Create first hidden layer (4 inputs -> 8 neurons)
        self.layer1 = nn.Linear(4,8)

        # TODO 2: Create second hidden layer (8 neurons -> 4 neurons)
        self.layer2 = nn.Linear(8,4)

        # TODO 3: Create output layer (4 neurons -> 1 final output)
        self.output = nn.Linear(4,1)

        # TODO 4: Create ReLU and Sigmoid activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # TODO 5: Connect the layers in order:
        # layer1 -> relu -> layer2 -> relu -> output -> sigmoid
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# Quick test to verify your network runs without shape errors
model = WinPredictor().to(device)
sample_guess = model(X_train)
print("Output shape:", sample_guess.shape)  # Should print: torch.Size([10, 1])

# =========================================================
# MILESTONE 3: TRAIN THE MODEL (Only on X_train!)
# =========================================================
# TODO 6: Create the Binary Cross-Entropy loss function
criterion = nn.BCELoss()

# TODO 7: Create the Adam optimizer for model.parameters() with lr=0.01
optimizer = optim.Adam(model.parameters(),lr=0.05)

epochs = 1000
for epoch in range(epochs):
    # TODO 8: Run the 5-step training heartbeat on X_train and y_train
    # (1. forward pass -> 2. calculate loss -> 3. zero_grad -> 4. backward -> 5. step)
    predictions = model(X_train)
    loss = criterion(predictions,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.4f}")


# =========================================================
# MILESTONE 4: FINAL EXAM ON UNSEEN MATCHES (X_test)
# =========================================================
print("\n--- Final Exam: Brand-New Unseen Matches ---")

# TODO 9: Turn off gradient tracking for testing
with torch.no_grad():
    test_probs = model(X_test)

    for stats, actual, prob in zip(X_test, y_test, test_probs):
        # TODO 10: Set verdict to "WIN" if prob.item() >= 0.5 else "LOSS"
        verdict = "WIN" if prob.item() >= 0.5 else "LOSS"
        expected = "WIN" if actual.item() == 1.0 else "LOSS"
        print(f"Stats {stats.tolist()} | Actual: {expected:<4} | Predicted: {verdict:<4} ({prob.item()*100:5.1f}%)")