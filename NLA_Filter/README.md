# NLA Proxy Filter Pipeline

This repository contains the pipeline for building a high-speed, serverless-friendly Natural Language Autoencoder (NLA) proxy filter. 

The core architecture relies on a two-step process to filter massive chat datasets (like OpenHermes, OpenOrca, and WizardLM) for fine-tuning custom MoE language models. It uses a heavy autoregressive (AR) critic model to establish a "golden" logic threshold, and then trains a lightweight, high-speed proxy model (LightGBM + MiniLM) to replicate those filtering decisions at scale.

## 📂 Repository Contents

* **`pull_wizard_sample.py`**
    * **Purpose:** Data extraction. Pulls a randomized subset of conversational data from the larger WizardLM dataset to be used for threshold discovery and proxy training.
* **`discover_wizard_threshold.py`**
    * **Purpose:** The "Heavy Lifter." Runs the extracted sample through a large AR value critic (e.g., Qwen AR). It extracts the final token's hidden state, projects it through a custom value head, and calculates the raw L2-normalized magnitude. This identifies the quality "cliff" and establishes the threshold for what constitutes high-quality, logically dense data.
* **`train_proxy_embeddings.py`**
    * **Purpose:** The "Fast Proxy" builder. Uses `all-MiniLM-L6-v2` to extract semantic embeddings from the text and combines them with structural heuristics (character length, word count). It then trains a LightGBM classifier using 5-fold cross-validation to predict the AR model's scores instantly.
* **`nla_45k_proxy_embeddings.lgb`**
    * **Purpose:** The compiled LightGBM model artifact. This is the production-ready proxy filter capable of blazing-fast inference over millions of rows of data without the compute bottleneck of a large transformer.

## 🚀 Pipeline Workflow

To replicate the filter generation process:

1.  **Extract Data:** Run `pull_wizard_sample.py` to generate your initial dataset slice.
2.  **Score & Label:** Run `discover_wizard_threshold.py` to calculate the true NLA magnitude scores using the AR critic model. This will output a JSONL file with normalized scores.
3.  **Train the Proxy:** Run `train_proxy_embeddings.py` on the scored data. This script handles feature extraction (semantic vectors + structural heuristics) and trains the classifier.
4.  **Deploy:** Use the resulting `nla_45k_proxy_embeddings.lgb` file to filter your broader training datasets at scale prior to Supervised Fine-Tuning (SFT).

## 🧠 Why a Proxy?

Running a large AR model to evaluate every single row of massive datasets like OpenHermes or WizardLM is computationally expensive and incredibly slow. 

By calculating the quality threshold on a representative sample and training a lightweight gradient-boosted tree (LightGBM) to mimic those decisions based on semantic embeddings, we achieve a highly accurate dataset curation filter that can be executed rapidly, even in serverless environments.
