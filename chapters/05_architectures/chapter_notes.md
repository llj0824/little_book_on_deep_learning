# Chapter 5 Notes: Architectures

## 5.1 Multi-Layer Perceptrons (MLP)
*   **Definition:** The simplest deep architecture. A sequence of fully connected layers separated by activation functions.
*   **Structure:**
    *   Input $\to$ Linear Layer $\to$ Activation $\to$ ... $\to$ Linear Layer $\to$ Output.
    *   **Hidden Layers:** The count of linear layers *excluding* the final output layer.
*   **Universal Approximation Theorem [Cybenko, 1989]:**
    *   **Statement:** An MLP with just **one** hidden layer can approximate *any* continuous function arbitrarily well on a bounded domain.
    *   **Conditions:**
        *   Activation function ($\sigma$) must be continuous and non-polynomial.
        *   The width of the hidden layer may need to be **arbitrarily large**.
    *   **Implications (The "Holy Grail"):**
        *   **The Promise:** It proves that Neural Networks are capable of solving *any* problem that can be framed as a continuous mathematical function (Translation, Driving, Recognition). The architecture is not the bottleneck.
        *   **The "Squiggle" (Non-Linearity):** The theorem relies entirely on the activation function. Stacking linear layers ($y=mx+b$) only creates lines/planes. Adding non-linearities allows the network to construct complex curves and surfaces.
    *   **The "In Passing" Catch (Theory vs. Practice):** The theorem is an **existence proof**, not a construction manual.
        1.  **The Width Problem (Efficiency):** While one layer *can* do it, it might need billions of neurons. **Deep** Learning (stacking layers) allows us to approximate the same functions exponentially more efficiently (fewer parameters) by building hierarchical features.
        2.  **The Optimization Problem (Training):** The theorem says the weights *exist*, but doesn't guarantee our training algorithms (Gradient Descent) can *find* them without getting stuck.
        3.  **The Generalization Problem:** The theorem ensures we can match the training data, but says nothing about how well the model will perform on *new, unseen* data.

---
## 5.2 Convolutional Networks (ConvNets)
*   **Purpose:** The standard architecture for processing images.
*   **Core Mechanism:** Uses convolutional layers to exploit local structure (pixels near each other matter) and shared weights (a "cat" ear detector is useful everywhere).

### LeNet-Like Architecture (The Classic Blueprint)
*   **Structure:** Two distinct halves.
    1.  **Feature Extractor:** Alternates Convolutional Layers (detect patterns) and Max Pooling Layers (shrink size).
        *   *Goal:* Turn a large raw image ($28 \times 28$ pixels) into a compact, high-level feature vector ($256$ values).
    2.  **Classifier:** A Multi-Layer Perceptron (MLP) / Fully Connected layers.
        *   *Goal:* Turn the feature vector into Class Logits (Probabilities).
*   **Evolution:** This blueprint (Conv/Pool $\to$ FC) was used for AlexNet and VGG.

### Residual Networks (ResNets)
*   **Problem:** Standard LeNet-like models stop learning if you make them too deep (Vanishing Gradient).
*   **Solution:** **Residual Connections** (See § 4.7).
    *   Allows gradients to flow through the "shortcut" path unimpeded.
    *   Enables networks with hundreds of layers (e.g., ResNet-50, ResNet-101).

### Key Concept: Channels vs. Layers
*   **Confusion:** Channels are often confused with Layers, but they are orthogonal dimensions.
*   **Layers (Depth):** The sequential steps of processing (Vertical).
    *   *Analogy:* The number of steps in an assembly line. ResNet-50 has 50 of these steps.
*   **Channels (Width/Thickness):** The number of parallel features existing *within* a single layer.
    *   *Analogy:* The number of workers at one specific station on the assembly line.
    *   *Example:* An image starts with 3 Channels (RGB). Deep in the network, a single layer might process 2048 Channels (2048 different feature maps) simultaneously.

### The "Quadratic Cost" of Convolutions
*   **The Math:** Parameters $\propto C_{in} \times C_{out}$.
*   **Why?** Every single Output Channel is a weighted sum of **ALL** Input Channels.
    *   To produce *one* new feature map (e.g., "Cat Ear"), the filter must look at *all* previous feature maps ("Edges", "Curves", "Colors").
*   **Implication:** If you double the number of input channels ($2 \times$) and double the number of output channels ($2 \times$), the computational cost and parameter count increase by **$4 \times$**.
*   **Solution:** This is why ResNet uses the **Bottleneck Block** (1x1 Conv) to temporarily reduce the number of channels before doing expensive spatial convolutions.

#### The "Bottleneck" Block (Efficiency Hack)
*   **Challenge:** We want many channels (rich representation), but convolutions are expensive (Cost $\propto \text{Channels}^2$).
*   **The Fix:** The $1 \times 1$ Sandwich.
    1.  **Squeeze ($1 \times 1$ Conv):** Reduce channels (e.g., $256 \to 64$).
    2.  **Process ($3 \times 3$ Conv):** Do the expensive spatial work on fewer channels.
    3.  **Expand ($1 \times 1$ Conv):** Restore channels (e.g., $64 \to 256$).
*   **Result:** Deep processing with a fraction of the compute cost.

#### ResNet-50 Structure
*   **Stem:** Starts with a $7 \times 7$ conv to aggressively shrink the image.
*   **Sections:** Four main stages. Each stage has multiple Residual Blocks.
*   **Downscaling:** Done by the *first* block in a section (using Stride=2). It halves the image size (Height/Width) and doubles the Channel count (preserving information).
*   **Head:**
    *   Ends with a $2048 \times 7 \times 7$ feature map.
    *   **Global Average Pooling:** Smashes the $7 \times 7$ spatial area into a single vector ($2048$).
    *   **FC Layer:** Maps 2048 $\to$ 1000 Classes (Logits).
*   **Use Case:** Best for processing signals where the dimension is **not too large**. (Inefficient for high-dimensional data like images, as per Chapter 4).

---
## 5.3 Attention Models
*   **Context:** The dominant architecture for Natural Language Processing (NLP) and increasingly for Vision.
*   **Key Innovation:** Allows the model to focus on (attend to) different parts of the input sequence dynamically, rather than processing fixed local windows (like CNNs) or fixed sequential steps (like RNNs).

### The Transformer [Vaswani et al., 2017]
*   **Origin:** Designed for Sequence-to-Sequence translation (e.g., English to French).
*   **Architecture Parts:**
    1.  **Encoder:** reads the input.
        *   Uses **Self-Attention** to relate every word to every other word (understanding context).
    2.  **Decoder:** generates the output.
        *   Uses **Causal Self-Attention** (looks at what it has written so far).
        *   Uses **Cross-Attention** (looks back at the Encoder's understanding of the input).
*   **Building Blocks:**
    *   **Feed-Forward Block:** A standard MLP (processed per-position).
    *   **Attention Block:** Recombines information globally.

### GPT (Generative Pre-trained Transformer)
*   **Structure:** Basically just the **Decoder** part of the Transformer.
*   **Mechanism:** "Autoregressive" — it simply predicts the next word in the sequence based on all previous words.
*   **Scaling:** This architecture proved to scale incredibly well (GPT-3, GPT-4), learning complex reasoning just by learning to predict the next token.

### Vision Transformer (ViT) [Dosovitskiy et al., 2020]
*   **Concept:** Treating images like text.
*   **Process:**
    1.  **Cut:** Slice the image into small square patches (e.g., $16 \times 16$ pixels).
    2.  **Flatten:** Turn each patch into a flat vector (like a "word" embedding).
    3.  **Add CLS:** Add a special "Class Token" ($E_0$) to the start of the sequence.
    4.  **Process:** Run the sequence of patches through a standard Transformer Encoder.
    5.  **Predict:** Use the final state of the **CLS Token** to classify the image.
*   **Significance:** Proved that Convolutions (CNNs) aren't strictly necessary for Computer Vision if you have enough data and compute; Attention can learn to see.

### Understanding Transformer Components: "Buildings vs. Furniture"
*   **The Furniture (The Blocks):**
    *   **Feed-Forward Block (The Desk):** Where the processing happens (reasoning/computation).
    *   **Self-Attention Block (The Intercom):** Communication *within* the same sequence (building).
    *   **Cross-Attention Block (The Telephone):** Communication *between* two sequences (Encoder $\leftrightarrow$ Decoder).
*   **The Buildings (The Architectures):**
    *   **Encoder (The Reader):** Uses Self-Attention + Feed-Forward. No Cross-Attention. (e.g., BERT).
    *   **Decoder (The Writer):** Uses Self-Attention + Cross-Attention + Feed-Forward. (e.g., GPT uses a modified Decoder without Cross-Attention).

### Example Flow: Translating "The cat" (Encoder-Decoder)
1.  **Input:** "The cat" enters the **Encoder**.
2.  **Encoder Action:**
    *   **Self-Attention:** Connects "The" to "cat".
    *   **Feed-Forward:** Processes the meaning.
    *   **Output:** Creates a "Memory Bank" (Keys/Values) representing the concept of a specific cat.
3.  **Decoder Action:**
    *   **Start:** Tries to generate the first word.
    *   **Cross-Attention:** Asks the Encoder's Memory Bank: "What is the subject?"
    *   **Result:** Retrieves "Cat" concept.
    *   **Feed-Forward:** Decides the French word is "Le".
    *   **Next Step:** Uses **Self-Attention** to remember it just said "Le", then asks Encoder for the noun, generates "chat".

### The "Soft Hashmap" Analogy
*   **Data Structure:** Attention is closer to a **Hash Map** or **Database Query** than a List or Tree.
*   **Components:**
    *   **Query ($Q$):** "What am I looking for?"
    *   **Key ($K$):** "What defines this piece of data?"
    *   **Value ($V$):** "What is the actual content?"
*   **The Difference (Soft Match):**
    *   **Standard Hash Map:** Uses Exact Match ($Q == K$). Returns one value or nothing.
    *   **Attention:** Uses **Dot Product Similarity** (Soft Match). Returns a weighted blend of values based on how well $Q$ matches $K$ (e.g., "50% Apple, 50% Pear").

### Model Weights & Parameter Count (Napkin Math)
*   **Concept:** "1 Trillion Parameters" is the sum of all entries in the weight matrices ($W_Q, W_K, W_V, W_{FF}$) across all layers.
*   **Example: "GPT-Model-X" (6B Params)**
    *   **Specs:** Embedding Dim ($D$) = 4,096, Layers ($L$) = 32.
    *   **Attention Block (1/3 of params):**
        *   4 Matrices ($Q, K, V, Out$). Each is $D \times D$.
        *   $4096 \times 4096 \approx 16\text{M}$. Total $16\text{M} \times 4 = 64\text{M}$.
    *   **Feed-Forward Block (2/3 of params):**
        *   Expands dimension by $4\times$ (to 16,384) to "think", then shrinks back.
        *   Up-Projection: $4096 \times 16384 \approx 67\text{M}$.
        *   Down-Projection: $16384 \times 4096 \approx 67\text{M}$.
        *   Total $\approx 134\text{M}$.
    *   **Per Layer Total:** $64\text{M} + 134\text{M} \approx 200\text{M}$.
    *   **Total Model:** $200\text{M} \times 32 \text{ Layers} \approx 6.4 \text{ Billion Parameters}$.

### Why GPT Scales (The "Bittersweet Lesson" & Parallelism)
*   **Causal Self-Attention:**
    *   **Definition:** "Attention that is blind to the future."
    *   **Implementation:** A mask ($-\infty$) in the attention matrix prevents Position $T$ from looking at Position $T+1$.
    *   **Why?** Ensures the model learns to *predict* based on history, not *copy* the answer from the future.
*   **The "Bittersweet Lesson" (Rich Sutton):**
    *   Human-designed rules (grammar, logic) always lose to **Generic Methods + Massive Compute**.
    *   GPT uses a "dumb" objective (Next Token Prediction) on "dumb" data (The Internet), but because it can scale, it learns better representations than any linguistic theory ever produced.
    *   **Scaling: RNNs vs. Transformers:**
    *   **RNNs (Sequential):** Must process Word 1 $\to$ Word 2 $\to$ Word 3. Cannot parallelize. GPUs sit idle.
    *   **Transformers (Parallel Training):**
        *   Uses **Teacher Forcing**. We know the ground truth ("The cat sat").
        *   We can calculate the error for "The $\to$ cat", "cat $\to$ sat", and "sat $\to$ on" **simultaneously** in one massive matrix operation.
        *   The mask ensures causality, but the *calculation* happens in parallel. This allows training on massive datasets efficiently.

### Emerging Architectures: Mamba & Simba
*   **The Problem:** Transformers are **$O(N^2)$**.
    *   Attention requires every token to look at every other token.
    *   10x sequence length = 100x compute cost. Limits context window (e.g., processing whole books or DNA).
*   **Mamba (State Space Models - SSMs):**
    *   **Concept:** Performance of a Transformer, Efficiency of an RNN.
    *   **Complexity:** **$O(N)$ (Linear)**.
        *   Processes sequence step-by-step, compressing history into a "State".
        *   10x sequence length = 10x cost. Allows for massive context.
    *   **Key Innovations:**
        1.  **Selection Mechanism:** A "Valve" that dynamically decides what to remember/forget based on the input (fixing the weakness of old RNNs).
        2.  **Parallel Scan (The Math Trick):** Uses the property of **Associativity** (like $(a+b)+c = a+(b+c)$) to calculate sequential updates in parallel (Tree structure) rather than waiting for the previous step.
            *   *Result:* Trains as fast as a Transformer on GPUs.
*   **Simba (Mamba for Vision):**
    *   Applies Mamba to image patches.
    *   Allows processing high-resolution images/video efficiently, avoiding the quadratic cost of Vision Transformers (ViT).


