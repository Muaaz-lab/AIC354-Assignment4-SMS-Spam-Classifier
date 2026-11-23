# AIC354 Assignment 4 — End-to-End Text ML System

**Deployment link:** Pending publication. Replace this line with the live app URL before submitting the notebook.

This package reviews the provided emotion-classification notebooks and builds a separate text-classification system for SMS spam detection.

## Contents

- `notebooks/Assignment4_End_to_End_Text_ML_System.ipynb` — Task 1 review and the complete Task 2 experiment.
- `data/SMSSpamCollection.tsv` — all 5,574 labeled UCI SMS messages in tab-separated format.
- `data/UCI_SMS_Spam_Attribution.txt` — data source, citation, and license note.
- `train_models.py` — reproducible stratified split, TF-IDF extraction, six-model comparison, and final-model training.
- `build_notebook.py` — regenerates the notebook with the latest saved comparison table.
- `results/model_comparison.csv` — held-out evaluation metrics for all six classifiers.
- `results/experiment_summary.json` — experiment settings, dataset counts, and best-model metrics.
- `results/*.png` — class distribution, comparison chart, and best-model confusion matrix.
- `models/best_sms_spam_model.joblib` — saved TF-IDF and classifier pipeline fitted on all messages.
- `app.py` — Streamlit prediction interface.
- `requirements.txt` — pinned packages matching the training environment.

## Task 1 — Provided Emotion Notebooks

Both notebooks use cleaned English comments, TF-IDF text features, stratified 80/20 train/test splits, six classifiers, held-out metrics, and a best-model prediction example.

- [Binary emotion notebook](https://github.com/haqnawaz99/ml-teaching-notebooks/tree/main/5%20-%20English%20Emotion%20Analysis%20-%20Binary%20Classification) predicts Joy versus Not Joy from an imbalanced sample. Its saved output compares unweighted and class-balanced variations; the balanced Decision Tree has 0.8729 accuracy and 0.8146 Joy F1, compared with 0.6604 accuracy for the always-Not-Joy baseline.
- [Six-emotion notebook](https://github.com/haqnawaz99/ml-teaching-notebooks/tree/main/6%20-%20English%20Emotion%20Analysis%20-%20Multi-class%20Classification) uses 2,400 comments (400 per emotion), six labels, and a 2,000-term TF-IDF vocabulary. The saved 480-row test result selects Linear SVM at 0.7854 accuracy and 0.7805 macro F1.

The source repository includes sample CSVs, not a larger full emotion corpus. To reproduce the instructor's Colab workflow, open each linked notebook, choose **Runtime → Run all**, and upload the matching CSV from its folder when prompted.

## Task 2 — SMS Spam Classification

### Problem and data

The system classifies a short English SMS as `ham` (legitimate) or `spam`. The complete UCI SMS Spam Collection has 5,574 messages: 4,827 ham and 747 spam. The original data is text, so the system extracts word unigram and bigram TF-IDF features. The train-only vectorizer preserves the test set as unseen data.

### Evaluation

The notebook and `train_models.py` use one stratified 80/20 split with random seed 42. The six classifiers are Logistic Regression, Linear SVM, Multinomial Naive Bayes, Decision Tree, Random Forest, and K-Nearest Neighbors. Results include accuracy, balanced accuracy, macro F1, spam precision, spam recall, spam F1, and fitting time. The deployment algorithm is selected by macro F1, with spam recall used to break ties, then refit on all 5,574 messages.

Macro F1 and spam recall are included because ham makes up most of this dataset; accuracy alone can make a model that overlooks spam appear strong.

## Run locally

Use Python 3.14 to match the saved scikit-learn 1.9.0 model. From this folder:

```powershell
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

The notebook can be opened with Jupyter from this folder. Run cells from top to bottom to reproduce the data summary, comparison, and plots. If you open the notebook directly in Google Colab, the setup cell will ask you to upload `Assignment4_Submission.zip` so Colab can access the dataset and training script.

## Cloud deployment

The app is ready for Streamlit Community Cloud, but a live URL is not available until this folder is pushed to a GitHub repository and deployed from an account you control.

1. Create or select a GitHub repository and upload the contents of this folder, including the `models/` directory.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app** and select that repository, branch, and `app.py` entrypoint.
3. In **Advanced settings**, choose Python 3.14 to match the saved model's training environment, then deploy.
4. Copy the public `streamlit.app` URL to the first line of this README and the first cell of the notebook.

Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) explains the repository, entrypoint, and Python-version settings. The trained model can be used with joblib only in a compatible scikit-learn environment; this package pins scikit-learn 1.9.0.

## Limitations

The corpus is old, English-only, and collected from a limited set of sources. Message styles, languages, and spam behavior change over time. The saved model is intended for coursework demonstration and should not be treated as a production spam filter.

