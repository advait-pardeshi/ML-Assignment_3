import streamlit as st
import pickle
import numpy as np
import torch
from torch import nn
from torch.functional import F

st.title('Next word predictor')
st.write('Enter the context:')
context = st.text_input("Enter the context here")

options = ['Greedy', 'Sampling']
st.sidebar.title("Settings")
approach = st.sidebar.selectbox(
    "Output generation approach",
    options
)

temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)
max_k_words = st.sidebar.slider("Max k words", min_value=5, max_value=20, value=10)
random_seed = st.sidebar.text_input("Random seed", value="42")

class NextWord(nn.Module):
    def __init__(self, vocab_size, context_size=5, embed_dim=32, hidden_dim=256, activation_fn=F.relu):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        input_dim = context_size * embed_dim
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.4)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.4)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.shape[0], -1)
        h2 = self.activation_fn(self.hidden_layer1(embeds))
        h2 = self.dropout1(h2)
        h2 = self.activation_fn(self.hidden_layer2(h2))
        h2 = self.dropout2(h2)
        logits = self.output(h2)
        return logits

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {device}")

try:
    # Load the complete model file
    checkpoint = torch.load('natural_language.pth', map_location='cpu', weights_only=False)

    # Extract components from the checkpoint
    state_dict = checkpoint['model_state_dict']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    saved_block_size = checkpoint['block_size']
    saved_embed_dim = checkpoint['embed_dim']

    vocab_size = len(stoi)

    # Determine hidden dimension from the state dict
    # Look at the hidden_layer1 weight shape: it should be [hidden_dim, input_dim]
    hidden_layer1_weight_shape = state_dict['hidden_layer1.weight'].shape
    hidden_dim = hidden_layer1_weight_shape[0]  # This should be 256 based on the error

    st.sidebar.write(f"**Model Info:**")
    st.sidebar.write(f"Trained block size: {saved_block_size}")
    st.sidebar.write(f"Trained embed dim: {saved_embed_dim}")
    st.sidebar.write(f"Hidden dimension: {hidden_dim}")
    st.sidebar.write(f"Vocabulary size: {vocab_size}")

    # Use the saved parameters from training
    block_size = saved_block_size
    embed_dim = saved_embed_dim

    # Initialize model with the exact architecture from training
    model = NextWord(
        vocab_size=vocab_size,
        context_size=block_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        activation_fn=F.relu
    )

    # Load the state dict
    model.load_state_dict(state_dict)

    # Move model to GPU
    model = model.to(device)
    model.eval()

    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize generator
g = torch.Generator(device=device)
try:
    g.manual_seed(int(random_seed))
except:
    g.manual_seed(42)

def generate_name(model, itos, stoi, existing_string, approach, block_size=5, max_len=20, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    # Prepare context
    context = []
    existing_string = existing_string.lower().strip()

    if len(existing_string) > 0:
        words = existing_string.split(" ")
        # Take only the last 'block_size' words
        words = words[-block_size:]
        # Pad with '_' if needed
        while len(words) < block_size:
            words.insert(0, '_')

        for word in words:
            if word in stoi:
                context.append(stoi[word])
            else:
                # Use underscore for unknown words
                context.append(stoi.get('_', 0))
    else:
        # Start with all underscores if no context
        context = [stoi.get('_', 0)] * block_size

    with torch.no_grad():
        generated_words = []

        for i in range(max_len):
            # Convert context to tensor and move to device
            x = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)

            # Get predictions
            logits = model(x)
            logits = logits / temperature

            if approach == "Greedy":
                probs = F.softmax(logits, dim=-1)
                next_word_idx = torch.argmax(probs, dim=-1).item()
            else:  # Sampling
                probs = F.softmax(logits, dim=-1)
                next_word_idx = torch.multinomial(probs[0], num_samples=1).item()

            next_word = itos[next_word_idx]

            # Stop conditions
            if next_word == '.' or next_word == '<END>' or next_word == '<end>':
                break
            if next_word == '_':  # Skip padding tokens
                continue

            generated_words.append(next_word)

            # Update context: remove first word, add new word
            context = context[1:] + [next_word_idx]

    return " ".join(generated_words)

if st.button("Predict"):
    if context.strip():
        try:
            with st.spinner("Generating text..."):
                sentence = generate_name(
                    model, itos, stoi, context,
                    approach, block_size, max_k_words, temperature
                )
                st.write(f"**Complete sentence:** {context} {sentence}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some context text.")

# Display additional info
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("First 10 vocabulary items:")
    for i, (k, v) in enumerate(list(stoi.items())[:10]):
        st.sidebar.write(f"  {k}: {v}")
