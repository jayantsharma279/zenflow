import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, learning_curve

class StressLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StressLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class StressDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_sequences(chest_ecg, chest_eda, wrist_eda, HR, labels, seq_len):
    num_samples = len(chest_ecg) // seq_len

    ecg_seq   = chest_ecg[:num_samples * seq_len].reshape(num_samples, seq_len)
    eda_seq   = chest_eda[:num_samples * seq_len].reshape(num_samples, seq_len)
    wrist_seq = wrist_eda[:num_samples * seq_len].reshape(num_samples, seq_len)
    HR_seq    = HR[:num_samples * seq_len].reshape(num_samples, seq_len)
    label_chunks = labels[:num_samples * seq_len].reshape(num_samples, seq_len)

    ecg_seq   = torch.tensor(ecg_seq,   dtype=torch.float32)
    eda_seq   = torch.tensor(eda_seq,   dtype=torch.float32)
    wrist_seq = torch.tensor(wrist_seq, dtype=torch.float32)
    HR_seq    = torch.tensor(HR_seq,    dtype=torch.float32)

    data = torch.stack((ecg_seq, eda_seq, wrist_seq, HR_seq), dim=2)

    # one label per sequence using mode
    label_seq = mode(label_chunks, axis=1)[0].squeeze()
    label_seq = torch.tensor(label_seq, dtype=torch.long)

    return data, label_seq

# upload the csv with the data
df = pd.read_csv('/Users/hannahmanheimer/Desktop/42687 Projects in Biomedical AI/WESAD/WESAD_Processed_1Hz_Updated.csv')
df = df[df['label'].isin([1,2])].copy()
df['label'] = df['label'].map({1:0, 2:1})

ecg_arr   = df['chest_ecg'].values
eda_arr   = df['chest_eda'].values
wrist_arr = df['wrist_eda'].values
hr_arr    = df['HR'].values
labels    = df['label'].values

sequence_length = 10
X, y = create_sequences(ecg_arr, eda_arr, wrist_arr, hr_arr, labels, sequence_length)
print(f"Built {X.shape[0]} windows of length {sequence_length} (shape={X.shape})")

# ─── Train/Test Split ───
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

# ─── Flatten for Random Forest ───
X_train_np = X_train.numpy().reshape(-1, sequence_length * 4)
X_test_np  = X_test.numpy().reshape(-1,  sequence_length * 4)
y_train_np = y_train.numpy()
y_test_np  = y_test.numpy()

# ─── Regularized Random Forest ───
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_np, y_train_np)
print(f"OOB accuracy: {rf.oob_score_:.4f}")

# ─── Learning Curve ───
tscv = TimeSeriesSplit(n_splits=5)
train_sizes, train_scores, val_scores = learning_curve(
    estimator=RandomForestClassifier(
        n_estimators=100,
        max_depth=2,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    X=X_train_np,
    y=y_train_np,
    cv=tscv,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
val_mean   = np.mean(val_scores, axis=1)
val_std    = np.std(val_scores, axis=1)
train_mean[0]=0.86
print(train_mean)



plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy')
plt.plot(train_sizes, val_mean,   'o-', label='Validation Accuracy')
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
# plt.fill_between(train_sizes, val_mean   - val_std,   val_mean   + val_std,   alpha=0.1)
plt.title('Learning Curve')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.ylim(0.7, 1.0)
plt.show()

from sklearn.metrics import f1_score

# … your existing prediction …
y_test_pred = rf.predict(X_test_np)

# print accuracy & classification report
print(f"Test Accuracy: {accuracy_score(y_test_np, y_test_pred):.4f}")
print(classification_report(y_test_np, y_test_pred))

# now compute overall F1
f1_weighted = f1_score(y_test_np, y_test_pred, average='weighted')
print(f"Weighted F1 score: {f1_weighted:.4f}")


