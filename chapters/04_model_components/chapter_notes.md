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
