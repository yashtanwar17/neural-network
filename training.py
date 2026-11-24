import numpy as np

# 1. ACTIVATION FUNCTIONS
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2. FULL TRAINING DATA (All 8 combinations of [Sunny, Raining, Weekend])
#    Rule to discover: Run (1) if (Sunny & Dry) OR (Rainy Weekend)
X = np.array([
    [0, 0, 0],  # Cloudy, dry, weekday   -> Stay (0)
    [0, 0, 1],  # Cloudy, dry, weekend   -> Stay (0)
    [0, 1, 0],  # Raining, weekday       -> Stay (0)
    [0, 1, 1],  # Raining, weekend       -> RUN  (1)
    [1, 0, 0],  # Sunny, dry, weekday    -> RUN  (1)
    [1, 0, 1],  # Sunny, dry, weekend    -> RUN  (1)
    [1, 1, 0],  # Sunny, raining, wkday  -> Stay (0)
    [1, 1, 1],  # Sunny, raining, wkend  -> RUN  (1)
], dtype=float)

# Answer key (y) matching each row of X above
y = np.array([[0], [0], [0], [1], [1], [1], [0], [1]], dtype=float)

# 3. RANDOM INITIALIZATION (3 inputs -> 4 hidden neurons -> 1 output)
np.random.seed(42)

W1 = np.random.randn(3, 4) * 0.5   # Random hidden weights
b1 = np.zeros((1, 4))              # Zero hidden biases

W2 = np.random.randn(4, 1) * 0.5   # Random output weights
b2 = np.zeros((1, 1))              # Zero output bias

# Training settings (hyperparameters)
learning_rate = 0.2
epochs = 4000

# 4. TRAINING LOOP
for epoch in range(epochs):
    # --- STEP A: FORWARD PASS (Make guesses) ---
    hidden_scores = np.dot(X, W1) + b1
    hidden_output = relu(hidden_scores)

    final_scores = np.dot(hidden_output, W2) + b2
    y_pred = sigmoid(final_scores)

    # --- STEP B: MEASURE MISTAKE (Binary Cross-Entropy Loss) ---
    loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))

    # --- STEP C: BACKPROPAGATION (Assign blame right-to-left) ---
    # Output layer gradients
    dz2 = y_pred - y
    dW2 = np.dot(hidden_output.T, dz2) / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)

    # Pass blame backward to hidden layer (only neurons with score > 0 take blame)
    d_hidden = np.dot(dz2, W2.T)
    dz1 = d_hidden * (hidden_scores > 0)

    # Hidden layer gradients
    dW1 = np.dot(X.T, dz1) / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # --- STEP D: UPDATE WEIGHTS & BIASES ---
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print progress every 800 epochs
    if epoch % 800 == 0:
        print(f"Epoch {epoch:4d} | Loss (Mistake Score): {loss:.4f}")

# 5. TEST THE TRAINED NETWORK
print("\n--- Final Decisions After Training ---")
days = [
    "Cloudy, dry, weekday",
    "Cloudy, dry, weekend",
    "Raining, weekday",
    "Raining, weekend",
    "Sunny, dry, weekday",
    "Sunny, dry, weekend",
    "Sunny, raining, wkday",
    "Sunny, raining, wkend",
]

final_probs = sigmoid(np.dot(relu(np.dot(X, W1) + b1), W2) + b2)

for day, target, prob in zip(days, y, final_probs):
    verdict = "GO RUN" if prob[0] >= 0.5 else "STAY INSIDE"
    print(f"{day:<22} | Target: {int(target[0])} | Confidence: {prob[0]*100:5.1f}% -> {verdict}")