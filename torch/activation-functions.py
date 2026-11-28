import numpy as np

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.array([
    [1, 0, 0],  # Day 1: Sunny weekday   -> Should RUN (1)
    [0, 1, 1],  # Day 2: Rainy weekend   -> Should RUN (1)
    [0, 1, 0],  # Day 3: Rainy weekday   -> Should STAY (0)
    [0, 0, 1],  # Day 4: Cloudy weekend  -> Should STAY (0)
])

# Column 0 checks "Sunny & Dry"; Column 1 checks "Rainy Weekend"
W1 = np.array([
    [ 5,  0],   # Sunny weight
    [-5,  4],   # Raining weight
    [ 0,  4],   # Weekend weight
])
b1 = np.array([-2, -5])

W2 = np.array([
    [2],        # Weight for Specialist 1 ("Sunny & Dry")
    [2],        # Weight for Specialist 2 ("Rainy Weekend")
])
b2 = np.array([-3])

hidden_scores = np.dot(X, W1) + b1
hidden_output = relu(hidden_scores)

print("Hidden Specialist Reports (Rows = Days, Cols = [Sunny&Dry, RainyWeekend]):")
print(hidden_output, "\n")

final_scores = np.dot(hidden_output, W2) + b2
probabilities = sigmoid(final_scores)

days = ["Sunny weekday", "Rainy weekend", "Rainy weekday", "Cloudy weekend"]
for day, prob in zip(days, probabilities):
    verdict = "GO RUN" if prob[0] >= 0.5 else "STAY INSIDE"
    print(f"{day:<16} -> Confidence: {prob[0]*100:5.1f}% -> {verdict}")