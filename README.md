# Fake News Detection from Social Media Screenshots

**Using OCR, Transformer-Based NLP and Explainable AI**

MSc Computer Science Dissertation Project
**Sujan Khadka · B01046458 · Ulster University**
Supervisor: **Muhammed Sihan Haroon**

---

## Overview

This project investigates the robustness of fake-news classifiers when text is obtained from social-media screenshots.

Most fake-news detection research evaluates models using clean, machine-readable text. In practice, screenshot-based content must first be processed using Optical Character Recognition (OCR), which can introduce character-level errors. This project evaluates how such noise affects both classification performance and model explainability.

Five classification models are compared:

* Logistic Regression
* Linear Support Vector Machine (SVM)
* BERT
* DistilBERT
* RoBERTa

The models are evaluated on the **WELFake** and **ISOT** fake-news datasets using clean text and simulated OCR-style character noise at **5%, 10%, 15%, and 20%** severity levels. SHAP is used to examine model explanations, while a Streamlit application demonstrates screenshot-based classification using Tesseract OCR and DistilBERT.

---

## Repository Contents

| File / Folder        | Description                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| `dissertation.ipynb` | Main notebook containing preprocessing, model training, OCR-noise experiments, evaluation, and SHAP analysis |
| `app.py`             | Streamlit application for screenshot-based fake-news classification                                          |
| `requirements.txt`   | Python dependencies                                                                                          |                                                      

Trained transformer models are not included because of their large file sizes. The notebook saves trained models to Google Drive.

The datasets are also not redistributed in this repository and should be obtained from their original public sources.

---

## Datasets

Two publicly available benchmark datasets are used:

### ISOT Fake News Dataset

Contains real and fake full-length news articles. Real articles originate primarily from Reuters, while fake articles were collected from unreliable news sources.

### WELFake

A larger binary fake-news dataset created by combining several existing news datasets.

The preprocessing pipeline removes duplicate and incomplete records, verifies label encoding, combines article titles and body text, and prepares consistent train, validation, and test sets.

**Important:** The WELFake version used in this project required verification of its label polarity during preprocessing. Users substituting a different copy should verify the meaning of the binary labels before training.

---

## Method Summary

### Classical Models

Logistic Regression and Linear SVM use TF-IDF representations limited to **5,000 features**.

### Transformer Models

BERT, DistilBERT, and RoBERTa are fine-tuned for binary sequence classification using HuggingFace Transformers.

Main training settings:

* Maximum sequence length: **256**
* Batch size: **16**
* Optimiser: **AdamW**
* Loss: **Cross-entropy**
* Initial training: **2 epochs**
* RoBERTa on WELFake: **3 epochs** after incomplete convergence was observed after two

A balanced **20,000-row training subset** is used for transformer fine-tuning to remain within Google Colab GPU constraints.

### OCR Noise Experiment

Noise is applied only at test time using:

* Character substitution
* Character insertion
* Character deletion

Noise levels:

* 0% (clean)
* 5%
* 10%
* 15%
* 20%

Accuracy and F1-score are used to evaluate degradation as input quality decreases.

### Explainability

SHAP is applied to the WELFake DistilBERT model to compare token-level explanations under clean and corrupted input conditions.

---

## Running the Notebook

The notebook was developed for **Google Colab with GPU acceleration**.

1. Open `dissertation.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Download the ISOT and WELFake datasets and upload them to the Colab environment.
4. Mount Google Drive when prompted.
5. Run the notebook cells in order.

The notebook is organised into stages covering:

1. Data preprocessing
2. Classical model training
3. Transformer fine-tuning
4. OCR-noise robustness experiments
5. Performance evaluation
6. SHAP analysis

Saved models can be reloaded in later stages, avoiding unnecessary retraining.

Training all three transformer architectures across the experimental datasets can take several hours depending on GPU availability.

---

## Running the Streamlit Application

The Streamlit demonstrator accepts an uploaded screenshot, extracts its text using **Tesseract OCR**, and classifies the extracted content as real or fake using a fine-tuned DistilBERT model.

### Requirements

* Python 3.9 or later
* Tesseract OCR installed on the system
* Python packages listed in `requirements.txt`
* A trained DistilBERT model stored in a directory named `welfake_distilbert/` beside `app.py`

Tesseract for Windows can be obtained from:

https://github.com/UB-Mannheim/tesseract/wiki

### Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application will open in a web browser. Upload a screenshot to display:

* OCR-extracted text
* Predicted class (`Real` or `Fake`)
* Prediction confidence

On Windows, ensure the Tesseract executable path in `app.py` matches your installation location:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

SHAP explanations are generated separately in the notebook because their computational cost makes real-time integration impractical.

---

## Key Findings

Transformer models achieved the strongest performance on clean WELFake text, particularly BERT and DistilBERT. However, their performance deteriorated substantially as character-level noise increased, while the classical TF-IDF models generally degraded more gradually.

The experiments therefore demonstrate that **high clean-text accuracy does not necessarily imply robustness to noisy screenshot-derived input**.

SHAP analysis also showed that OCR-style corruption can reduce the coherence of model explanations by fragmenting meaningful words into less interpretable token sequences.

Detailed quantitative results and analysis are provided in the dissertation report.

---

## Known Limitations

* OCR noise is simulated rather than produced directly by multiple OCR engines.
* Noise generation is not seeded, so repeated experiments may produce statistically similar but non-identical corrupted samples.
* Models are primarily trained on full-length news articles and do not generalise equally well to short social-media-style text.
* The SHAP analysis uses representative examples rather than a large-scale quantitative evaluation of explanation stability.
* The classifiers identify linguistic patterns associated with fake news; they do not independently verify factual claims.

---

## Ethical Considerations

The benchmark datasets contain publicly available published news content. No private dataset was collected for this project.

All screenshots used for demonstration are **synthetic and use fabricated accounts and content**. Real individuals' social-media posts are not labelled as fake or real, reducing privacy and potential defamation risks.

The application is intended as a **research demonstrator**, not as a production fact-checking system. Predictions should support human judgement rather than replace independent factual verification.

---

## Further Information

Full details of the methodology, experimental results, limitations, ethical considerations, and critical evaluation are provided in the accompanying MSc dissertation report and supporting document.
