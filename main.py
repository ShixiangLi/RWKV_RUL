import torch
from torch import nn
from torch.utils.data import DataLoader

from transformer import Bert
from cmapss import get_data

def get_score(out, labels):
    score = 0
    for i in range(out.shape[0]):
        if out[i] >= labels[i]:
            score += ((out[i]-labels[i])*125/10).exp() - 1
        else:
            score += ((labels[i]-out[i])*125/13).exp() - 1
    return score

TASK = 'train'
FD = '4'
num_test = 100

batch_size = 1024

if FD == '1':
    sequence_length = 31
    FD_feature_columns = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    head = 3

if FD == '2':
    sequence_length = 21
    FD_feature_columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']
    head = 3
if FD == '3':
    sequence_length = 38
    FD_feature_columns = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's17',
                            's20', 's21']
    head = 4
if FD == '4':
    sequence_length = 19
    FD_feature_columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's20', 's21']
    head = 4

# Load the dataset
train_feature, train_labels = get_data(
    FD=FD, 
    sequence_length=sequence_length, 
    batch_size=batch_size, 
    feature_columns=FD_feature_columns,
    label='train'
)

test_feature, test_labels = get_data(
    FD=FD,
    sequence_length=sequence_length,
    batch_size=batch_size,
    feature_columns=FD_feature_columns,
    label='test'
)

# Model
model = Bert(
    embed_size=len(FD_feature_columns),
    seq_len=sequence_length,
    heads=head,
    ff_hidden_size=64,
    dropout=0.1,
    num_layers=3,
    num_classes=1
)

# Training setting
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
best_loss = 1000

# Dataloader
train_loader = DataLoader(
    dataset=list(zip(train_feature, train_labels)),
    batch_size=batch_size,
    shuffle=True
)

test_feature = torch.Tensor(test_feature).to(device)
test_labels = torch.Tensor(test_labels).to(device)

# test_loader = DataLoader(
#     dataset=list(zip(test_feature, test_labels)),
#     batch_size=batch_size,
#     shuffle=True
# )

if TASK == 'train':
    # Training
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = torch.ones(features.shape[0], 1, 1, sequence_length).to(device)

            # Forward pass
            outputs = model(features, mask)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            mask = torch.ones(test_feature.shape[0], 1, 1, sequence_length).to(device)
            out = model(test_feature, mask)
            out = torch.clamp(out, 0, 1)
            rmse = torch.sqrt(criterion(out, test_labels)) * 125

            score = get_score(out, test_labels)

            if rmse < best_loss:
                print(f'Best model found at epoch {epoch+1}, RMSE: {rmse.item()}, Score: {score.item()}')
                best_loss = rmse
                torch.save(model.state_dict(), f'./checkpoint/best_model_FD{FD}.pt')
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, RMSE: {rmse.item()}')
else:
    model.load_state_dict(torch.load(f'./checkpoint/best_model_FD{FD}.pt'))
    model.to(device)
    model.eval()
    with torch.no_grad():
        mask = torch.ones(test_feature.shape[0], 1, 1, sequence_length).to(device)
        out = model(test_feature, mask)
        out = torch.clamp(out, 0, 1)
        rmse = torch.sqrt(criterion(out, test_labels)) * 125

        score = get_score(out, test_labels)

        print(f'RMSE: {rmse.item()}, Score: {score.item()}')
        

    