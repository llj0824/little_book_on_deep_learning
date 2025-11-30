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
*   **Use Case:** Best for processing signals where the dimension is **not too large**. (Inefficient for high-dimensional data like images, as per Chapter 4).
