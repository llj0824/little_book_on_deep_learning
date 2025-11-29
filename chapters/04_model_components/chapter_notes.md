# Chapter 4 Notes: Model Components

## 4.1 The Notion of Layer
*   **Concept:** Layers are the "LEGO blocks" of deep learning. Instead of writing raw matrix math, we compose complex models using these standardized, high-level operations.
*   **Composition:**
    *   **Trainable Parameters:** Weights that change during learning (e.g., the values in a filter).
    *   **Hyper-parameters:** Settings chosen before training that don't change (e.g., Kernel size, Stride).

## 4.2 Linear Layers
"Linear" in deep learning usually refers to **Affine** transformations ($Ax + b$). 

### Fully Connected (FC) Layers
*   **Operation:** Connects every input element to every output element.
*   **Scope:** **Global**. It looks at the entire input at once.
*   **Math:** $Y = WX + b$.
*   **Pitfall:** Extremely inefficient for high-dimensional data like images. It ignores spatial structure (pixels close to each other are related).

### Convolutional Layers (Conv1d / Conv2d)
*   **Operation:** Slides a small kernel (filter) over the input.
*   **Scope:** **Local**. Computes one output value based on a small local patch (window).
*   **Inductive Bias:** Assumes local neighbors matter and that features (like edges) are useful anywhere in the image (Translation Equivariance).
*   **Mechanism:** "Condensing." The kernel takes a patch ($D \times K$) and sums it up into a single vector.

### Key Hyper-parameters
1.  **Padding ($p$):** Adding zeros around the border. Keeps the output size the same as the input.
2.  **Stride ($s$):** The step size the window moves.
    *   $s=1$: Dense coverage.
    *   $s>1$: Skips pixels, reducing the output size (Downsampling).
3.  **Dilation ($d$):** "Spreading your fingers."
    *   Inserts gaps between kernel weights.
    *   **Benefit:** Drastically increases the area the kernel "sees" without adding extra cost or parameters.

### Advanced Concepts
#### Receptive Field
*   **Definition:** The specific area of the original input image that influences a specific pixel deep in the network.
*   **Effect of Depth:**
    *   Layer 1 sees a $3 \times 3$ patch.
    *   Layer 2 sees a $3 \times 3$ patch of Layer 1, which means it effectively sees a $5 \times 5$ patch of the Input.
    *   **Takeaway:** Deep networks understand global context ("this is a cat") because stacking many local layers expands the receptive field to cover the whole image.

#### Transposed Convolution ("Deconvolution")
*   **Purpose:** Upsampling. Making an image *bigger* (e.g., for Generative AI or Semantic Segmentation).
*   **Mechanism:** "Stamping."
    *   Standard Conv: Takes a patch $\to$ Calculates one number.
    *   Transposed Conv: Takes one number $\to$ Pastes (stamps) a patch of weights onto the output.
    *   Where stamps overlap, values are summed.

---

## 4.3 Activation Functions (The "Squiggle" Problem)

### Why do we need them?
*   **The Trap:** If you stack linear layers ($y = 2x$, then $z = 3y$), you just get another linear layer ($z = 6x$). You can't learn complex shapes.
*   **The Fix:** Activation functions inject **non-linearity**. They "bend" the line.
*   **Analogy:** A linear model draws a straight line/plane. To separate complex data (like a donut shape), you need to fold the space. Activations act like the "conditions" on a graphing calculator (e.g., "Draw this line ONLY if $x > 0$") that allow you to draw complex art.

### The Main Functions
1.  **ReLU (Rectified Linear Unit):**
    *   Formula: $\max(0, x)$.
    *   Graph: A hard corner at 0.
    *   **Pros:** Fast, solves Vanishing Gradient for positive numbers.
    *   **Cons:** "Dead Neurons" (if negative, gradient is 0 and it stops learning).
2.  **Tanh:**
    *   Graph: S-shape (-1 to 1).
    *   **Cons:** Saturates (flattens) at edges $\to$ Gradient dies $\to$ Network stops learning.
3.  **GELU (Gaussian Error Linear Unit):**
    *   **Concept:** A "Probabilistic Switch" rather than a hard switch.
    *   Formula: $x \times P(Z \le x)$, where $Z$ is a Standard Normal Distribution.
    *   **Graph:** Similar to ReLU but curved. Has a slight "dip" below zero because of the probability multiplication.
    *   **Note:** It does **not** normalize your data. It just uses the standard bell curve equation as a tool to draw a smooth curve.

---

## 4.4 Pooling

### The Goal
To shrink the image/tensor size to reduce cost and abstract away from specific pixels to general features.

### Types
1.  **Max Pooling (The Standard):**
    *   **Operation:** Pick the biggest number in the window.
    *   **Logic:** "Is there an edge here? (0.1, 0.9, 0.2, 0.3) -> YES (0.9)."
    *   **Benefit:** **Invariance**. If the feature moves slightly, the Max value stays the same.
2.  **Average Pooling:**
    *   **Operation:** Average the window.
    *   **Difference:** This is a **Linear** operation. Max Pooling is non-linear.

---

## Q&A: Conceptualizing "Training" & "Parameters"

### Where do the numbers come from?
*   **Initialization:** We start with a "giant spreadsheet" of empty matrices. We fill them with **Random Noise**.
*   **Parameters:** The count of these random numbers. "100 Billion Parameters" = 100 Billion cells in the matrices.
*   **Scale:** "Larger Training Runs" means defining bigger matrices from the start and showing them more data. We rarely "add columns" to an existing trained model.

### What is Training?
It is **not** just summing. It is a loop:
1.  **Forward:** Input * Random Weights = Guess ("Toaster").
2.  **Loss:** Compare Guess vs. Truth ("Cat").
3.  **Backward (The Key):** Calculate **Gradient**. Find out exactly which of the 100 Billion numbers caused the error.
4.  **Update:** Nudge those numbers slightly so the guess is better next time.

Repeat billions of times until the Random Noise becomes "Smart."