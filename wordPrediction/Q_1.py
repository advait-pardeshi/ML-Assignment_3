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

# %%
print(words_with_stop[:20])

# %%
print(f"Length of the vocabulary is: {len(word_dict)}")


# %%
sorted_items_asc = sorted(word_dict.items(), key=lambda item: item[1])
print("Least common words are: \n")
for x in range (10):
    print(f"{x+1}. {sorted_items_asc[x][0]} Count: {sorted_items_asc[x][1]}")

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
for word in word_dict.keys():
    stoi[word] = i+1
    i = i+1

print(stoi)

# %%
itos = {i:s for s,i in stoi.items()}
print(itos)

# %%
block_size = 5
X, Y = [],[]
i=0
while i < (len(words_with_stop)):
    context =  [0] * block_size
    while words_with_stop[i] != '.':
        idx = stoi[words_with_stop[i]]
        X.append(context)
        Y.append(idx)
        #print(' '.join(itos[i] for i in context), '--->', itos[idx])
        context = context[1:] + [idx]
        i=i+1

    if i < len(words_with_stop) and words_with_stop[i] == '.':
        X.append(context)
        Y.append(stoi['.'])
        #print(' '.join(itos[j] for j in context), '--->', '.')
        i += 1



# %%
X = torch.tensor(X[:262144])
Y = torch.tensor(Y[:262144])
print(X.shape)

# %%
print(X.shape)

# %%
def plot_embeddings(selected_words, embeds, mode ="before"):
    indices = [stoi[w] for w in selected_words]
    indices = torch.tensor(indices)
    if mode == "before":
        selected_embeddings = embeds(indices).detach().cpu().numpy()
    else:
        selected_embeddings = embeds
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    reduced = tsne.fit_transform(selected_embeddings)
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(selected_words):
        x, y = reduced[i, 0], reduced[i, 1]
        plt.scatter(x, y)
        plt.text(x + 0.02, y + 0.02, word)
    plt.title("t-SNE Visualization of Learned Word Embeddings before training")
    plt.show()


# %%
embeds = nn.Embedding(len(stoi), 32)
selected_words = ['king', 'queen', 'man', 'woman', 'apple', 'run', 'hide', 'salute', 'to', 'be', 'walk', 'quickly', 'slowly']
plot_embeddings(selected_words, embeds)

# %%
class NextWord(nn.Module):
    def __init__(self, vocab_size, context_size=5, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        input_dim = context_size * embed_dim
        self.hidden_layer1 = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.4)
        self.hidden_layer2 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(512, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.shape[0], -1)
        h2 = F.relu(self.hidden_layer1(embeds))
        h2 = self.dropout(h2)
        h2=F.relu(self.hidden_layer2(h2))
        h2 = self.dropout(h2)
        logits = F.relu(self.output(h2))
        return logits

# %%
model = NextWord(len(stoi), 5, embed_dim=32)
model = torch.compile(model,  backend="eager")

# %%
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    print("Model Summary:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            print(f"{name:<20} {params:,} parameters")
    print(f"\nTotal trainable parameters: {count_params(model):,}\n")

# %%
g = torch.Generator()
g.manual_seed(4000002)
def generate_name(model, itos, stoi,existing_string, block_size=5, max_len=20, ):
    model.eval()
    context = []
    existing_string = existing_string.lower()
    if(len(existing_string)>0):
        wr = existing_string.split(" ")
        if(len(wr)>block_size):
            wr=wr[-block_size:]
        else:
            while len(wr) < block_size:
                wr.insert(0,'_')
        for ele in wr:
            context.append(stoi[ele])
    else:
        context = [0] * block_size
    with torch.no_grad():
        sentence = ''
        for i in range(max_len):
            x = torch.tensor(context).view(1,-1)
            y_pred = model(x)
            probs = torch.softmax(y_pred, dim=-1)
            ix = torch.argmax(probs, dim=-1).item()
            word = itos[ix]
            if word == '.':
                sentence+='.'
                break
            sentence+=" "+ word
            context = context[1:] + [ix]
            i=i+1
    model.train()
    return sentence


# %%
print_model_summary(model)

# %%
sentence = generate_name(model, itos, stoi,"", block_size)
print(f"Randomly generated sentence before training: \n{sentence}")

# %%
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

# %%
def train_model(model, X_train,Y_train,X_val, Y_val, lr, epochs =70, batch_size = 512,  wd = 1e-2, print_every = 100):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, times, validation_losses = [], [], []
    for e in range(epochs):
        start = time.time()
        total_loss = 0
        n_batches = 0
        for i in range(0, X_train.shape[0], batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = Y_train[i:i+batch_size]
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        train_losses.append(avg_loss)
        times.append(time.time() - start)
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val)
            val_loss = loss_fn(logits_val, Y_val).item()

        validation_losses.append(val_loss)
        times.append(time.time() - start)
        print(f"Epoch {e:4d} | Loss: {avg_loss:.4f} | Validation_loss: {val_loss:.4f} Time: {times[-1]:.2f}s")
    return train_losses,validation_losses, times



# %%
train_losses,validation_losses, times = train_model(model, X_train,Y_train, X_val, Y_val,lr=1e-5, epochs=700, batch_size=2048,  wd=0.01, print_every=50)
torch.save(model, "natural_language.pth")

# %%
existing_string = "What are"
sentence = generate_name(model, itos, stoi,existing_string, block_size, )
print(f"Generated sentence: {sentence}")
print(f"Complete sentence: {existing_string+sentence}")

# %%
embedding_matrix = model.embedding.weight.detach().cpu().numpy()
selected_words = ['king', 'queen', 'man', 'woman', 'apple', 'run', 'walk', 'quickly', 'slowly']
plot_embeddings(selected_words, embedding_matrix, mode="after")

# %%
epochs = []
print(len(train_losses))
print(len(validation_losses))
for i in range(1, 701):
    epochs.append(i)



# %%
plt.plot(epochs, train_losses, label='Train Loss', color='blue', linestyle='-')
plt.plot(epochs, validation_losses, label='Validation Loss', color='red', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# %%
import pickle

with open("next_word.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# %%
model = torch.load('natural_language.pth', weights_only=False)

# %%
with open('exported_variables.pkl', 'wb') as f:
    pickle.dump(stoi, f)
    pickle.dump(itos, f)
