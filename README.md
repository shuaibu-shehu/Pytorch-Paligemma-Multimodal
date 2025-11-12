## Pytorch Paligemma Multimodal (Vision) Language Model Implementation

This project implements the Paligemma (Polygama) Visual Language Model (VLM), originally developed by Google, entirely from scratch in PyTorch. The methodology emphasizes coding every component, from tensor operations to complex architectural features, to ensure deep learning and comprehension.

> "Write [the code] line by line, character by character, because that's the best way to learn."

---

## 1. Architectural Overview
The Paligemma architecture is designed for conditional generation, using an image and a text prompt as inputs to generate a relevant textual response.

| Component | Function | Implementation Details |
| :--- | :--- | :--- |
| **Vision Encoder** | Extracts contextualized visual embeddings (image tokens) from the input image. | Implemented as a Vision Transformer (ViT) using the SigLIP (Sigmoid Loss for Language-Image Pretraining) approach. |
| **Multimodal Projector** | Connects the vision and language components by resizing the vision embeddings. | A simple linear layer that projects the hidden size of the Vision Encoder output to match the hidden size of the Language Model. |
| **Language Decoder** | Generates text tokens based on the combined sequence of image and text features. | Implemented as a decoder-only Gemma Transformer model. |

---

## 2. Implementation Structure
The source code is organized to reflect the modular nature of the architecture, adhering to the naming conventions used in the Hugging Face implementation for weight compatibility.

| File Name | Primary Components | Purpose |
| :--- | :--- | :--- |
| `modeling_siglip.py` | Vision Transformer, SigLIP Attention Block, Layer Normalization. | Contains the full implementation of the Vision Encoder. |
| `modeling_gamma.py` | Gemma Decoder Layers, RMS Normalization, Gemma Attention, Rotary Positional Encoding. | Contains the full implementation of the Language Model components. |
| `processing_polygama.py` | Image processing, Text tokenization, Feature merging logic. | Handles VLM data preparation, including placeholder insertion and resizing images. |
| `inference.py` | Generation loops, Sampling logic (Top-P, Temperature). | Manages the inference workflow and loads pre-trained weights. |

---

## 3. Core Technical Optimizations and Design Choices
The implementation incorporates several modern techniques crucial for the efficiency and performance of large language models:

### 3.1. Contrastive Vision Encoder (SigLIP)
The Vision Encoder uses Sigmoid Loss instead of the traditional Cross-Entropy Loss (found in CLIP).

* **Rationale:** Sigmoid Loss treats each dot product between an image embedding and a text embedding as an independent binary classification task (positive or negative match).
* **Advantage:** This eliminates the need to compute the normalization constant (softmax) across entire rows or columns of the attention matrix, making the computation easier to parallelize and enabling the use of much larger batch sizes during training (up to millions of items). The goal is to produce image embeddings that are highly compatible with text embeddings for VLM use.

### 3.2. Attention Mechanism Enhancements (Gemma Decoder)
The Language Model's attention layers feature key efficiency and stability enhancements:

* **Grouped Query Attention (GQA):** Used to optimize GPU memory and speed.
    * **Mechanism:** Instead of having an equal number of Query (Q), Key (K), and Value (V) heads (Multi-Head Attention), GQA uses fewer K and V heads.
    * **Benefit:** Multiple Query heads share the same K and V projections. This significantly reduces the data transfer bottleneck on the GPU (copying data from High Bandwidth Memory to local memory) and reduces the overall size of the Key-Value (KV) cache.
* **Rotary Positional Embeddings (RoPE):** Applied directly to the Query and Key vectors within the attention calculation.
    * **Mechanism:** RoPE modifies the dot product to incorporate relative positional information between tokens.
    * **Effect:** The attention score (dot product) between two tokens naturally decays as their relative distance increases.

### 3.3. Inference Optimization (Key-Value Cache)
The KV Cache is essential for high-speed, sequential text generation. It avoids redundant computation during the token-by-token decoding process by caching previously computed Key ($\text{K}$) and Value ($\text{V}$) vectors.

The KV Cache operates in two phases during inference:

* **Prefilling:** The entire input prompt (including image tokens and text prompt) is processed in one pass. The $\text{K}$ and $\text{V}$ vectors generated for all these tokens are calculated in parallel and stored in the cache.
* **Token Generation:** For subsequent steps, only the single newly generated token is passed as the Query ($\text{Q}$). The attention mechanism then uses this single Query against the entire accumulated Key/Value cache to compute only the necessary last contextualized embedding, which is used to predict the next token.

### 3.4. Normalization and Stability (RMS Norm)
The Gemma decoder employs Root Mean Square Normalization (RMS Norm).

* **Difference from Layer Norm:** Unlike Layer Normalization, RMS Norm only computes the Root Mean Square statistic and performs rescaling; it does not calculate or subtract the mean (recentering).
* **Advantage:** This reduces the number of required computations, speeding up the forward and backward passes while maintaining the necessary stability for training and inference.

---

## 4. Multimodal Data Processing and Input Preparation
The `PolygamaProcessor` handles the complex task of preparing both image and text inputs for the Language Model.

### 4.1. Image Preparation
Input images are processed into a tensor format ready for the Vision Encoder:

* **Resizing:** Images are resized to the expected input dimension (e.g., 224x224).
* **Scaling & Normalization:** Pixel values are scaled to a range of 0 to 1 and then normalized by subtracting the mean and dividing by the standard deviation (similar to ImageNet statistics).

### 4.2. Tokenization and Merging
The text prompt and image features are merged into a single sequence:

* **Placeholder Insertion:** The text prompt is tokenized after inserting `image_token` placeholders (e.g., 256 tokens for the base model) at the beginning of the sequence, followed by the Beginning-of-Sentence (BOS) token and the user's prompt.
* **Feature Replacement:** After the Vision Encoder processes the image and the Multimodal Projector resizes the features, these image features are used to replace the placeholder embeddings in the token sequence.

### 4.3. Attention Masking for Conditional Generation
A critical design choice in Paligemma is the application of causality in the attention mask:

* **Prompt (Condition):** The sequence containing image tokens and the user's textual prompt (the "prefix") is treated as a non-causal context. This means all prompt tokens can attend to each other, including future tokens within the prefix, maximizing the information available to condition the model's response.
* **Generated Output:** Causality (masking future tokens) is only strictly applied to the tokens generated by the model (the "suffix") during sequential decoding.

---

## 5. Inference and Generation Parameters
The `inference.py` script manages the generation process, which uses parameters to control the output quality and diversity:

* **Greedy Strategy:** By default, the model selects the single token with the highest probability score (no sampling).
* **Top-P Sampling:** If sampling is enabled, the model sorts all tokens by probability and restricts the selection pool to the top tokens whose cumulative probability sum reaches the threshold *P* (e.g., 0.9).
* **Temperature:** Applied to the logits before softmax, the temperature parameter adjusts the probability distribution. A lower temperature (closer to 0) makes the distribution sharper (favoring high-confidence tokens), while a higher temperature smooths the distribution, encouraging the selection of more diverse tokens.
* **Weight Tying:** The weights of the initial embedding layer are shared with the final language modeling head (logits projection) to reduce the total number of parameters.
