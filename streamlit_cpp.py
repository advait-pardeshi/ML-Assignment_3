import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re

# Model definition (must match training)
class NextToken(nn.Module):
    def __init__(self, vocab_size, context_size=4, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        input_dim = context_size * embed_dim
        self.hidden_layer1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.hidden_layer2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.4)
        self.output = nn.Linear(256, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.shape[0], -1)
        h1 = F.relu(self.hidden_layer1(embeds))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.hidden_layer2(h1))
        h2 = self.dropout2(h2)
        logits = self.output(h2)
        return logits

# C++ preprocessing function
def preprocess_cpp_code(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code)
    code = re.sub(r'([{}();,<>=+\-*/&|!])', r' \1 ', code)
    code = re.sub(r'\s+', ' ', code)
    return code.strip()

# Load model and vocabulary
@st.cache_resource
def load_model_and_vocab():
    try:
        with open('cpp_vocabulary.pkl', 'rb') as f:
            stoi = pickle.load(f)
            itos = pickle.load(f)

        vocab_size = len(stoi)
        model = NextToken(vocab_size, context_size=4, embed_dim=32)
        model.load_state_dict(torch.load('cpp_code_model_gpu.pth', map_location='cpu', weights_only=True))
        model.eval()

        return model, stoi, itos
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Generation function
def generate_cpp_code(model, itos, stoi, existing_code, approach="Greedy", block_size=4, max_len=50, temperature=1.0):
    model.eval()

    existing_code = preprocess_cpp_code(existing_code)
    code_tokens = existing_code.split(" ")
    code_tokens = [token for token in code_tokens if token.strip()]

    if len(code_tokens) >= block_size:
        context_tokens = code_tokens[-block_size:]
    else:
        context_tokens = ['_'] * (block_size - len(code_tokens)) + code_tokens

    context = [stoi.get(token, 0) for token in context_tokens]

    generated_tokens = []
    with torch.no_grad():
        for i in range(max_len):
            x = torch.tensor(context).view(1, -1)
            y_pred = model(x)
            y_pred = y_pred / temperature

            if approach == "Greedy":
                probs = F.softmax(y_pred, dim=-1)
                next_token_idx = torch.argmax(probs, dim=-1).item()
            else:
                probs = F.softmax(y_pred, dim=-1)
                next_token_idx = torch.multinomial(probs[0], num_samples=1).item()

            next_token = itos.get(next_token_idx, '<?>')
            generated_tokens.append(next_token)

            if next_token in [';', '}', '{'] and len(generated_tokens) > 10:
                break

            context = context[1:] + [next_token_idx]

    result = ' '.join(generated_tokens)
    result = re.sub(r'\s+([,;{}()])', r'\1', result)
    result = re.sub(r'([,;{}()])\s+', r'\1 ', result)

    return result

# Main app
def main():
    st.title('C++ Code Predictor')

    # Load model
    model, stoi, itos = load_model_and_vocab()

    if model is None:
        st.error("Failed to load model. Please make sure the model files are available.")
        return

    # Sidebar settings
    st.sidebar.title("Settings")

    approach = st.sidebar.selectbox(
        "Generation Approach",
        ["Greedy", "Sampling"]
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0
    )

    max_length = st.sidebar.slider(
        "Max Tokens to Generate",
        min_value=10,
        max_value=100,
        value=50
    )

    block_size = st.sidebar.slider(
        "Context Size",
        min_value=2,
        max_value=8,
        value=4
    )

    # Main content
    st.write("Enter C++ code context:")

    example_contexts = {
        "Empty": "",
        "Main function": "int main() {",
        "For loop": "for (int i = 0;",
        "While loop": "while (condition) {",
        "If statement": "if (x > 0) {",
        "Function declaration": "void myFunction(",
        "Variable declaration": "int counter ="
    }

    selected_example = st.selectbox("Choose an example:", list(example_contexts.keys()))

    context = st.text_area(
        "Code context:",
        value=example_contexts[selected_example],
        height=100
    )

    if st.button("Generate Code"):
        if context.strip():
            with st.spinner("Generating..."):
                try:
                    generated = generate_cpp_code(
                        model, itos, stoi,
                        context, approach,
                        block_size, max_length,
                        temperature
                    )

                    st.subheader("Results")
                    st.write("Original context:")
                    st.code(context)

                    st.write("Generated continuation:")
                    st.code(generated)

                    st.write("Complete code:")
                    complete_code = context + " " + generated
                    st.code(complete_code)

                except Exception as e:
                    st.error(f"Error during generation: {e}")
        else:
            st.warning("Please enter some code context.")

if __name__ == "__main__":
    main()
