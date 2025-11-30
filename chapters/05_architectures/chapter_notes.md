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
