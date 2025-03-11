from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.optim as optim

# Load data
text_inputs = torch.rand((32, 768))  # Simulating text embeddings
features = torch.rand((32, 10))  # Simulating linguistic feature vectors
labels = torch.randint(0, 3, (32,))  # MCI category (0, 1, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(text_inputs, labels, test_size=0.2, random_state=42)

# Model initialization
model = CCDA()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    output = model(X_train, features)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Evaluation
model.eval()
y_pred = torch.argmax(model(X_test, features), dim=1)
print(classification_report(y_test.numpy(), y_pred.numpy()))
