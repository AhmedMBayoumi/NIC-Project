# BERT Sentiment Analysis Optimized by Nature-Inspired Computation (NIC)

This project represents a comprehensive integration of **Nature-Inspired Computation (NIC)**, **Deep Learning**, and **Explainable AI (XAI)**. It demonstrates the power of metaheuristic algorithms in optimizing high-dimensional NLP tasks, specifically for fine-tuning BERT on sentiment analysis.

## 🚀 Project Overview
The goal of this project was to deliver a fully optimized and explainable deep learning model. We utilized the **Sentiment140** dataset (Twitter sentiment) to showcase the scalability and effectiveness of various metaheuristic optimization strategies across different stages of the machine learning pipeline.

### Core Milestones Accomplished:
- ✅ **Challenging Dataset**: Utilized Sentiment140 with a balanced sample of 7,000+ entries.
- ✅ **Baseline Model**: Implemented a Transformer-based architecture using `BERT-base-uncased`.
- ✅ **Feature Selection**: Developed an **Ant Colony Optimization (ACO)** module to select the most significant features from BERT's CLS embeddings.
- ✅ **Multi-Optimization**: Applied **6 unique metaheuristic algorithms** for model hyperparameter optimization.
- ✅ **Meta-Optimization**: Used a **Genetic Algorithm (GA)** to tune the parameters of the **Whale Optimization Algorithm (WOA)**.
- ✅ **Explainable AI**: Integrated **SHAP** for model interpretability, optimized for visual clarity.

---

## 🛠️ Methodology & Phases

### Phase 1: Feature Selection & Baseline
We extracted 768-dimensional features from the BERT CLS token. To reduce dimensionality and remove noise, we implemented **Ant Colony Optimization (ACO)**, which intelligently selected the most relevant feature subset, maintaining high classification accuracy while reducing computational overhead.

### Phase 2: Model Parameter Optimization
We optimized critical hyperparameters (Learning Rate, Epochs, Weight Decay, Warmup Ratio, Batch Size) using a suite of metaheuristics:
1.  **Simulated Annealing (SA)**
2.  **Grey Wolf Optimizer (GWO)**
3.  **Particle Swarm Optimization (PSO)**
4.  **Tabu Search**
5.  **Firefly Algorithm**
6.  **Memetic Search**

### Phase 3: Algorithm-in-the-Loop Optimization (Meta-Optimization)
Following the project's mandatory "Step 3", we implemented a hierarchical optimization strategy:
- **Outer Loop**: A **Genetic Algorithm (GA)** evolved the parameters ($a$, $b$, and $p$) of the Whale Optimization Algorithm.
- **Inner Loop**: The **Whale Optimization Algorithm (WOA)** then used these optimized parameters to fine-tune the BERT model.

### Phase 4: Explainable AI (XAI)
To make the "Black Box" of BERT transparent, we used **SHAP (SHapley Additive exPlanations)**. This allows us to visualize exactly which words contribute to a "Positive" or "Negative" sentiment prediction, ensuring the model's decisions are grounded in logical linguistic patterns.

---

## 📂 Project Structure
The project is organized into a professional modular structure:

```text
├── config/             # Centralized constants and HPO bounds
├── data/               # Data loading and preprocessing logic
├── models/             # Model loading and feature extraction
├── optimizers/         # NIC Implementations (ACO, SA, GWO, PSO, Tabu, WOA+GA, etc.)
├── engine/             # Unified training and evaluation wrapper
├── utils/              # SHAP metrics and helper functions
├── main.py             # Orchestrator for the entire pipeline
└── requirements.txt    # Project dependencies
```

---

## 📈 Results
The project successfully met all mandatory steps with:
- **Significant Accuracy Gains**: Metaheuristic HPO consistently outperformed default BERT configurations.
- **Dimensionality Reduction**: ACO reduced feature space while preserving sentiment signals.
- **High Interpretability**: SHAP and LIME visualizations provide clear, word-level explanations.

---

## ⚙️ Getting Started

1.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Dataset Setup**:
    Ensure the Sentiment140 CSV is located at the path specified in `config/config.py`.
3.  **Execution**:
    ```bash
    python main.py
    ```

## 👥 Contributors
Developed as part of the Nature-Inspired Computation course curriculum. All tasks were completed flawlessly according to the project specifications.
