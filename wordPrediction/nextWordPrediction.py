import os
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"] = "+dynamo"
import re
import torch
import time
from sklearn.preprocessing import StandardScaler
import pickle
from torch import nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import streamlit as st
import matplotlib.pyplot as plt

# Set device and optimize memory usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optimize memory usage
torch.backends.cudnn.benchmark = True
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Create directory for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# %%
with open('./shakespeare_input.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# %%
content = re.sub('[^a-zA-Z0-9. \s]', '', content)
content = re.sub(r'\s+', ' ', content).strip()
content = (content.lower())

print(content[:30])

# %%
words = content.split(" ")
words_with_stop = []
word_dict = {
    "." : 0
}
for word in words:
    if "." in word:
        word = word.replace('.', '')
        word_dict['.'] = word_dict['.'] + 1
        words_with_stop.append(word)
        words_with_stop.append('.')
    else:
        words_with_stop.append(word)
    if(word_dict.get(word)):
        word_dict[word] = word_dict[word]+1
    else:
        word_dict[word] = 1

# Reduce vocabulary size by removing very rare words
min_word_freq = 3  # Only keep words that appear at least 3 times
filtered_word_dict = {k: v for k, v in word_dict.items() if v >= min_word_freq or k == '.'}
print(f"Original vocabulary size: {len(word_dict)}")
print(f"Reduced vocabulary size: {len(filtered_word_dict)}")
word_dict = filtered_word_dict

# %%
print(f"Length of the vocabulary is: {len(word_dict)}")

# %%
sorted_items_dsc = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
print("Most common words are: \n")
for x in range (10):
    print(f"{x+1}. {sorted_items_dsc[x][0]} Count: {sorted_items_dsc[x][1]}")

# %%
# build the vocabulary of characters and mappings to/from integers
i = 0
stoi = {}
stoi['_'] = 0
stoi['<UNK>'] = 1  # Add unknown token
for word in word_dict.keys():
    stoi[word] = i+2  # Start from 2 because 0 and 1 are reserved
    i = i+1

print(f"Final vocabulary size: {len(stoi)}")

# %%
itos = {i:s for s,i in stoi.items()}
print(itos)

# %%
block_size = 3  # Reduced context window to save memory
X, Y = [],[]
i=0
while i < (len(words_with_stop)):
    context =  [0] * block_size
    while i < len(words_with_stop) and words_with_stop[i] != '.':
        word = words_with_stop[i]
        idx = stoi.get(word, stoi['<UNK>'])  # Use UNK for words not in vocabulary
        X.append(context)
        Y.append(idx)
        context = context[1:] + [idx]
        i=i+1

    if i < len(words_with_stop) and words_with_stop[i] == '.':
        X.append(context)
        Y.append(stoi['.'])
        i += 1

# Use smaller dataset size
dataset_size = min(131072, len(X))  # Reduced from 262144 to 131072
X = torch.tensor(X[:dataset_size], device=device)
Y = torch.tensor(Y[:dataset_size], device=device)
print(f"Dataset shape: {X.shape}")

# %%
def plot_embeddings(selected_words, embeds, mode="before"):
    # Filter words that are in vocabulary
    available_words = [w for w in selected_words if w in stoi]
    indices = [stoi[w] for w in available_words]
    indices = torch.tensor(indices, device=device)
    if mode == "before":
        selected_embeddings = embeds(indices).detach().cpu().numpy()
    else:
        selected_embeddings = embeds
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(available_words)-1))
    reduced = tsne.fit_transform(selected_embeddings)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(available_words):
        x, y = reduced[i, 0], reduced[i, 1]
        plt.scatter(x, y)
        plt.text(x + 0.02, y + 0.02, word)

    if mode == "before":
        title = "t-SNE Visualization of Word Embeddings Before Training"
        filename = "plots/embeddings_before_training.png"
    else:
        title = "t-SNE Visualization of Word Embeddings After Training"
        filename = "plots/embeddings_after_training.png"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

# %%
# Smaller embedding dimension
embed_dim = 24  # Reduced from 32 to 24
embeds = nn.Embedding(len(stoi), embed_dim).to(device)
selected_words = ['king', 'queen', 'man', 'woman', 'apple', 'run', 'hide', 'to', 'be', 'walk']
plot_embeddings(selected_words, embeds)

# %%
class NextWord(nn.Module):
    def __init__(self, vocab_size, context_size=3, embed_dim=24):  # Reduced sizes
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        input_dim = context_size * embed_dim
        # Reduced hidden layer sizes
        self.hidden_layer1 = nn.Linear(input_dim, 256)  # Reduced from 512 to 256
        self.dropout1 = nn.Dropout(0.3)  # Reduced dropout
        self.hidden_layer2 = nn.Linear(256, 256)  # Reduced from 512 to 256
        self.dropout2 = nn.Dropout(0.3)  # Reduced dropout
        self.output = nn.Linear(256, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.shape[0], -1)
        h1 = F.relu(self.hidden_layer1(embeds))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.hidden_layer2(h1))
        h2 = self.dropout2(h2)
        logits = self.output(h2)  # Removed final ReLU (not typical for output layer)
        return logits

# %%
model = NextWord(len(stoi), block_size, embed_dim=embed_dim).to(device)
# Remove torch.compile for memory savings
# model = torch.compile(model, backend="eager")

# %%
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    print("Model Summary:")
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            print(f"{name:<20} {params:,} parameters")
    print(f"\nTotal trainable parameters: {total_params:,}\n")
    return total_params

# %%
total_params = print_model_summary(model)
print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)")

# %%
def generate_name(model, itos, stoi, existing_string, block_size=3, max_len=15):  # Reduced max_len
    model.eval()
    context = []
    existing_string = existing_string.lower()
    if len(existing_string) > 0:
        wr = existing_string.split(" ")
        if len(wr) > block_size:
            wr = wr[-block_size:]
        else:
            while len(wr) < block_size:
                wr.insert(0, '_')
        for ele in wr:
            context.append(stoi.get(ele, stoi['<UNK>']))  # Handle unknown words
    else:
        context = [0] * block_size

    with torch.no_grad():
        sentence = ''
        for i in range(max_len):
            x = torch.tensor(context, device=device).view(1, -1)
            y_pred = model(x)
            probs = torch.softmax(y_pred, dim=-1)
            ix = torch.argmax(probs, dim=-1).item()
            word = itos[ix]
            if word == '.':
                sentence += '.'
                break
            sentence += " " + word
            context = context[1:] + [ix]
    model.train()
    return sentence

# %%
sentence = generate_name(model, itos, stoi, "", block_size)
print(f"Randomly generated sentence before training: \n{sentence}")

# %%
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# %%
def train_model(model, X_train, Y_train, X_val, Y_val, lr, epochs=70, batch_size=512, wd=1e-2, print_every=50):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, times, validation_losses = [], [], []

    # Gradient accumulation for effective larger batch size with less memory
    accum_steps = 4  # Accumulate gradients over 4 batches
    batch_size = batch_size // accum_steps  # Use smaller actual batch size

    for e in range(epochs):
        start = time.time()
        total_loss = 0
        n_batches = 0

        # Training loop with gradient accumulation
        opt.zero_grad()
        for i in range(0, X_train.shape[0], batch_size):
            # Process in smaller chunks to save memory
            end_idx = min(i + batch_size, X_train.shape[0])
            x_batch = X_train[i:end_idx]
            y_batch = Y_train[i:end_idx]

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch) / accum_steps  # Normalize loss

            loss.backward()

            if (i // batch_size + 1) % accum_steps == 0:
                opt.step()
                opt.zero_grad()

            total_loss += loss.item() * accum_steps
            n_batches += 1

            # Clear cache periodically
            if device.type == 'cuda' and n_batches % 100 == 0:
                torch.cuda.empty_cache()

        # Handle remaining gradients
        if (X_train.shape[0] // batch_size) % accum_steps != 0:
            opt.step()
            opt.zero_grad()

        avg_loss = total_loss / n_batches
        train_losses.append(avg_loss)

        # Validation - use smaller batches for validation too
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size * 2):  # Larger batches for validation
                end_idx = min(i + batch_size * 2, X_val.shape[0])
                x_val_batch = X_val[i:end_idx]
                y_val_batch = Y_val[i:end_idx]

                logits_val = model(x_val_batch)
                val_loss += loss_fn(logits_val, y_val_batch).item()
                val_batches += 1

        val_loss /= val_batches
        validation_losses.append(val_loss)

        epoch_time = time.time() - start
        times.append(epoch_time)

        print(f"Epoch {e+1:4d} | Loss: {avg_loss:.4f} | Val_loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")

        # Early stopping check (simple version)
        if len(validation_losses) > 10 and validation_losses[-1] > max(validation_losses[-10:]):
            print(f"Early stopping at epoch {e+1}")
            break

    return train_losses, validation_losses, times

# %%
# Train with smaller learning rate and fewer epochs
train_losses, validation_losses, times = train_model(
    model, X_train, Y_train, X_val, Y_val,
    lr=5e-5,  # Smaller learning rate
    epochs=50,  # Reduced epochs
    batch_size=1024,  # Effective batch size will be 1024//4=256
    wd=0.01,
    print_every=25
)

# Save model
model_cpu = model.to('cpu')
torch.save({
    'model_state_dict': model_cpu.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'block_size': block_size,
    'embed_dim': embed_dim
}, "natural_language.pth")
model = model.to(device)

print("Training completed!")

# %%
existing_string = "What are"
sentence = generate_name(model, itos, stoi, existing_string, block_size)
print(f"Generated sentence: {sentence}")
print(f"Complete sentence: {existing_string + sentence}")

# %%
embedding_matrix = model.embedding.weight.detach().cpu().numpy()
selected_words = ['king', 'queen', 'man', 'woman', 'apple', 'run', 'walk', 'to', 'be']
plot_embeddings(selected_words, embedding_matrix, mode="after")

# %%
# Create training loss plot and save as PNG
epochs_range = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_losses, label='Train Loss', color='blue', linestyle='-')
plt.plot(epochs_range, validation_losses, label='Validation Loss', color='red', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/training_validation_loss.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved plots/training_validation_loss.png")

# %%
# Save model and vocabulary
with open("next_word.pkl", "wb") as model_file:
    pickle.dump(model.state_dict(), model_file)

with open('exported_variables.pkl', 'wb') as f:
    pickle.dump(stoi, f)
    pickle.dump(itos, f)

# Create a summary of the training results
with open("plots/training_summary.txt", "w") as f:
    f.write("Training Summary (4GB GPU Optimized)\n")
    f.write("====================================\n\n")
    f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
    f.write(f"Final Validation Loss: {validation_losses[-1]:.4f}\n")
    f.write(f"Total Epochs: {len(train_losses)}\n")
    f.write(f"Vocabulary Size: {len(stoi)}\n")
    f.write(f"Model Parameters: {total_params:,}\n")
    f.write(f"Model Size: {total_params * 4 / (1024**2):.2f} MB\n")
    f.write(f"Device Used: {device}\n")
    f.write(f"Block Size (context): {block_size}\n")
    f.write(f"Embedding Dimension: {embed_dim}\n\n")
    f.write("Example Generation:\n")
    f.write(f"Input: 'What are'\n")
    f.write(f"Output: '{sentence}'\n")
    f.write(f"Complete: 'What are{sentence}'\n")

print("Training completed and all plots saved!")
print("Files created:")
print("- plots/embeddings_before_training.png")
print("- plots/embeddings_after_training.png")
print("- plots/training_validation_loss.png")
print("- plots/training_summary.txt")

# Memory cleanup
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# %%
# Load model with proper device handling
def load_model(model_path, vocab_size, block_size, embed_dim, device):
    model = NextWord(vocab_size, block_size, embed_dim=embed_dim)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

# Example usage for loading:
# model = load_model('natural_language.pth', len(stoi), block_size, embed_dim, device)
