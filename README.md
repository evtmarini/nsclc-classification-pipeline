
# NSCLC Classification Pipeline

**Development of ML model for non-small cell lung cancer detection**

---

## Project Overview

Αυτό το repository περιέχει τον πλήρη κώδικα για το machine learning pipeline που αναπτύχθηκε στο πλαίσιο της διπλωματικής εργασίας 
*"Development of ML model for non-small cell lung cancer detection"* στο Μεταπτυχιακό **Βιοπληροφορικής και Νευροπληροφορικής** του **Ιονίου Πανεπιστημίου**.

Στόχος του έργου είναι η ανάπτυξη ενός *ML pipeline* που ταξινομεί υποτύπους **μη-μικροκυτταρικού καρκίνου του πνεύμονα (NSCLC)** με βάση **ραδιομικά χαρακτηριστικά** από αξονικές τομογραφίες. 

---

## Introduction

* Ο καρκίνος του πνεύμονα είναι η **κύρια αιτία θανάτων από καρκίνο** παγκοσμίως (~19% το 2022).
* Το **NSCLC** αντιστοιχεί περίπου στο 85% όλων των περιπτώσεων και περιλαμβάνει:

  * Αδενοκαρκίνωμα
  * Πλακώδες καρκίνωμα
  * Μεγαλοκυτταρικό καρκίνωμα

Η σωστή διάκριση μεταξύ αυτών των υποτύπων είναι κρίσιμη για **στοχευμένες θεραπείες** και **πρόγνωση**.
Η ραδιομική αποτελεί μη επεμβατική προσέγγιση που εξάγει **ποσοτικά χαρακτηριστικά** από ιατρικές εικόνες και τα χρησιμοποιεί σε **μοντέλα πρόβλεψης**.

---


---

## Pipeline Architecture

Το pipeline είναι modular και αποτελείται από τα εξής στάδια:

| Στάδιο                         | Module                     | Περιγραφή                                                                                   |
| ------------------------------ | -------------------------- | ------------------------------------------------------------------------------------------- |
| 1. Data Loading                | `src/load_data.py`         | Φόρτωση δεδομένων, encoding labels, χειρισμός ελλειπών τιμών                                |
| 2. Preprocessing               | `src/preprocessing.py`     | Φιλτράρισμα με βάση τη διακύμανση, αφαίρεση συσχετισμένων χαρακτηριστικών, στατιστικά tests |
| 3. Feature Selection           | `src/feature_selection.py` | CorrSF, Boruta, RFE, LASSO, RF-importance                                                   |
| 4. Modeling                    | `src/models.py`            | Random Forest, Logistic Regression (L1), SVM (RBF)                                          |
| 5. Evaluation                  | `src/evaluation.py`        | GridSearchCV + Stratified K-Fold CV, αξιολόγηση μετρικών                                    |
| 6. Visualization               | `src/visualization.py`     | Γραφήματα επιδόσεων                                                                         |
| 7. Explainability              | —                          | SHAP & LIME για ερμηνευσιμότητα                                                             |

---

## Project Structure

```
nsclc-classification-pipeline/
│
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
│
├── data/
│   └── labeled_radiomics_features.csv
│
├── results/
│   ├── ml_results.csv
│   └── ml_results.png
│
└── main.py
```
