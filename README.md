## 1. 📌 Project Title
🔐 CyberGraphAI — A Hybrid ML + GNN Based Network Anomaly Detection System

## 2. 🎯 Short Description / Purpose
CyberGraphAI is designed to detect network anomalies using both classic Machine Learning models and Graph Neural Networks by leveraging graph structure, node similarity, and feature interactions.
The system incorporates Explainable AI techniques to interpret model behavior and K-Means clustering to group attack types after detection—enabling cybersecurity analysts and researchers to uncover hidden threat patterns.

## 3. 🛠️ Tech Stack
Core Technologies Used:
🐍 Python — Main programming language
📊 Scikit-Learn — Machine Learning Models
🔥 PyTorch & PyTorch Geometric — Graph Neural Networks (GraphSAGE, GCN)
⚙️ NetworkX — Graph construction
🧮 SMOTE (Imbalanced-Learn) — Data balancing
🧠 SHAP & Permutation Importance — Explainable AI
📈 Matplotlib & Seaborn — Visualization

📁 Pandas / NumPy — Data handling
🧪 KMeans — Clustering

File Types:
.py for scripts
.csv for datasets
.png for visualizations

## 4. 📂 Data Source
Primary Dataset:
📌 Cybersecurity Threat and Awareness Program Dataset (Kaggle, 2024)

📊 Total Samples: 54,768
🔸 Normal: 46,589
🔸 Attack: 8,179
🧬 Features: 30

🎛️ Attributes include IPs, ports, protocols, flow duration, packet stats, anomaly scores, severity levels, IDS alerts.
Preprocessing:
Missing values handled (mean/mode)
Label encoding
Standardization
SMOTE applied → balanced dataset = 93,178 rows

## 5. ⭐ Features / Highlights
📌 Business Problem
The rise in cyberattacks demands accurate, interpretable, and scalable network anomaly detection systems. Traditional ML fails to capture relational/graph dependencies—making advanced GNN methods necessary.

📌 Project Goals
Build a multi-model anomaly detection pipeline.
Compare ML vs GNN performance.
Add Explainable AI for interpretability.
Group attacks using clustering.
Build a knowledge graph to visualize cybersecurity threats.

### 📌 Walkthrough of Key Components
1️⃣ Data Preprocessing

Missing value imputation

Scaling & Encoding

SMOTE oversampling

Outlier analysis

Correlation heatmap

2️⃣ Machine Learning Models
Evaluated models:
CatBoost
XGBoost
Random Forest
LightGBM
Decision Tree
Naive Bayes
MLP

📈 Best ML Model → CatBoost (Accuracy 87%, F1: 0.88)

3️⃣ Graph Neural Networks
Two GNN models applied:
GraphSAGE
GCN
Graph constructed using KNN similarity (3 neighbors).
Nodes = network flows
Edges = similarity links

🔥 Best Overall Model → GraphSAGE (Accuracy 95%, AUC 0.99)

4️⃣ Explainable AI
CatBoost → SHAP Feature Importance
GraphSAGE → Global Feature Permutation
Top factors:
Anomaly Severity Index
Flow Duration
Normalized Packet Flow
IDS Alert Count

5️⃣ Clustering Analysis (K-Means)

Used to identify subgroups of attack behavior after detection.
✔️ CatBoost → 2 clusters
✔️ GraphSAGE → Clearer attack segmentation

Cluster 1 = highly anomalous
Cluster 0 = suspicious/low-confidence

6️⃣ Cybersecurity Knowledge Graph

Nodes:
Internal IPs
External services
Malware families
Botnets
Attack types
Edges:
INFECTED_BY
USES_ATTACK
PART_OF_BOTNET
CONNECTS_TO
Visualizes attack propagation paths.
### 6.	Screenshots / Demos
Show what the graph looks like. - ![Alt text][(https://github.com/username/repo/assets/image.png](https://github.com/xahido7/Network-Anomaly-Detection-Using-Machine-Learning-andral-Networks-with-Explainable-AI-and-Clustering/blob/main/knowledge_graph.png))
Show what the graph looks like. - ![Alt text][(https://github.com/username/repo/assets/image.png](https://github.com/xahido7/Network-Anomaly-Detection-Using-Machine-Learning-andral-Networks-with-Explainable-AI-and-Clustering/blob/main/knowledge_graph.png))
