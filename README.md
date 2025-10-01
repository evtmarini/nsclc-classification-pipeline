# 🩺 NSCLC Classification Pipeline 🚀  
**Development of a Machine Learning Model for Non-Small Cell Lung Cancer Detection**

---

## 📘 Project Overview

This repository contains the complete machine learning pipeline developed as part of the MSc thesis project *"Development of ML model for non-small cell lung cancer detection"*, within the MSc Program **Bioinformatics and Neuroinformatics** at the **Ionian University**.

The main objective of this project is to design and implement a **modular and explainable ML pipeline** for classifying **non-small cell lung cancer (NSCLC)** subtypes using **radiomic features** extracted from **CT scans** — with future extension to combine histopathological data.

---

## 🎯 Motivation

- Lung cancer remains the **leading cause of cancer-related deaths worldwide** (~19% in 2022).  
- **Non-small cell lung cancer (NSCLC)** represents ~85% of all lung cancers and includes:
  - Adenocarcinoma  
  - Squamous cell carcinoma  
  - Large cell carcinoma  

Accurate subtype classification is essential for **treatment planning**, **prognosis**, and **personalized medicine**.  
Radiomics offers a non-invasive alternative to histopathological diagnosis by extracting quantitative features from standard imaging data and integrating them into predictive ML models.

---

## 🧠 Project Goals

This work is part of a broader research collaboration between:

- 🏥 iKnowHealth S.A.  
- 🧬 AnaBioSi-Data LTD  
- 🏛️ University of Crete  
- 🏛️ University of Cyprus  

The overall project aims to:

1. 📊 Create a radiomics database from CT and histopathological data collected in Greece and Cyprus.  
2. 🧠 Integrate a lung nodule segmentation tool into the **EvoRad PACS** workstation.  
3. 🤖 Develop an ML classification model for NSCLC subtypes using radiomic features.  

**This repository implements Goal #3.**

---

## ⚙️ Pipeline Architecture

The pipeline is fully modular and consists of the following stages:

| Stage | Module | Description |
|-------|--------|-------------|
| 1️⃣ Data Loading & Cleaning | `src/load_data.py` | Loads the dataset, encodes labels, handles missing values |
| 2️⃣ Preprocessing | `src/preprocessing.py` | Variance filtering, correlation removal, statistical feature filtering |
| 3️⃣ Feature Selection | `src/feature_selection.py` | CorrSF, Boruta, RFE (SVM), LASSO, RF importance |
| 4️⃣ Modeling | `src/models.py` | ML classifiers: Random Forest, Logistic Regression (L1), SVM (RBF) |
| 5️⃣ Evaluation | `src/evaluation.py` | GridSearchCV + Stratified K-Fold CV, evaluation metrics |
| 6️⃣ Visualization | `src/visualization.py` | Performance plots across feature selection methods and models |
| 7️⃣ Explainability (future) | — | SHAP & LIME interpretability analysis |

---

## 📁 Project Structure

