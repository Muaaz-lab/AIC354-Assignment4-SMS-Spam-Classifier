"""Build and execute the self-contained Assignment 4 notebook."""

from pathlib import Path
import csv

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "notebooks" / "Assignment4_End_to_End_Text_ML_System.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
        **Live deployment link:** PENDING - replace with the published Streamlit app URL before submitting.
        """
    ),
    markdown(
        """
        # AIC354 Assignment 4
        ## End-to-End Machine Learning System Using Unstructured Text

        **Student:** [Enter your name and roll number]  
        **Problem:** SMS spam classification (`ham` vs `spam`)  
        **Course:** Machine Learning Fundamentals, Fall 2026

        This notebook first reviews the instructor-provided emotion notebooks, then trains, evaluates, and prepares a deployable text classification system using the complete UCI SMS Spam Collection.
        """
    ),
    markdown(
        """
        ## Task 1 — Review of the provided Emotion Prediction system

        The two instructor notebooks demonstrate the same end-to-end pattern: load labeled comments, clean and lowercase text, make a stratified 80/20 train/test split, convert text to TF-IDF features, compare six classifiers, report held-out scores, save the selected model, and predict a new comment.

        | Notebook | Task and data | Feature/model details | Reported best result |
        | --- | --- | --- | --- |
        | Binary emotion | Predict Joy vs Not Joy from an imbalanced sample | 2,000 TF-IDF features; six algorithms; unweighted and class-balanced runs | Balanced Decision Tree: accuracy 0.8729, Joy F1 0.8146, Joy recall 0.8221. Always predicting Not Joy gives 0.6604 accuracy. |
        | Six-emotion | Predict anger, fear, joy, love, sadness, or surprise; 2,400 rows with 400 per class | 2,000 TF-IDF features; six algorithms; 480-row held-out set | Linear SVM: accuracy 0.7854, macro F1 0.7805. |

        **What the results show:** TF-IDF provides a simple numeric representation of text that works with standard classifiers. The binary notebook also illustrates why class imbalance matters: accuracy alone can hide weak detection of the minority Joy class, while class-balanced training can raise Joy recall. In the six-class notebook, macro F1 gives each emotion equal weight when comparing models.

        **Source notebooks:** [Binary emotion notebook](https://github.com/haqnawaz99/ml-teaching-notebooks/blob/main/5%20-%20English%20Emotion%20Analysis%20-%20Binary%20Classification/English_Emotion_Analysis_BinaryClass.ipynb) · [Six-emotion notebook](https://github.com/haqnawaz99/ml-teaching-notebooks/blob/main/6%20-%20English%20Emotion%20Analysis%20-%20Multi-class%20Classification/English_Emotion_Analysis_MultiClass.ipynb) · [Instructor repository README](https://github.com/haqnawaz99/ml-teaching-notebooks/blob/main/README.md)

        The source repository supplies sample CSVs for these instructional notebooks. To reproduce them in Google Colab, open each linked notebook, select **Runtime → Run all**, and upload its matching CSV (`emotion_binary_sample_data.csv` or `emotion_sample_data.csv`) when prompted.
        """
    ),
    markdown(
        """
        ## Task 2 — Select a machine learning problem

        **Problem statement:** Given an SMS message, predict whether it is legitimate (`ham`) or unsolicited spam (`spam`). This is a binary classification problem using unstructured text.

        **Why it is suitable:** It uses real short-message text, has documented labels, is small enough to compare several classical algorithms on the complete dataset, and supports a simple web demo where a user enters a message.
        """
    ),
    markdown(
        """
        ### Dataset source and structure

        The dataset is the **UCI SMS Spam Collection**, a public corpus of 5,574 labeled messages. The UCI record reports 4,827 ham messages and 747 spam messages (about 86.6% vs 13.4%). Each source row contains a class label and a raw SMS message. The complete source file and its attribution are in `data/SMSSpamCollection.tsv` and `data/UCI_SMS_Spam_Attribution.txt`.

        **Citation:** Almeida, T., & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5CC84](https://doi.org/10.24432/C5CC84). The UCI record lists a CC BY 4.0 license.
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import sys
        import zipfile

        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "train_models.py").exists():
            parent = PROJECT_ROOT.parent
            if (parent / "train_models.py").exists():
                PROJECT_ROOT = parent
            else:
                try:
                    from google.colab import files
                except ImportError as error:
                    raise FileNotFoundError(
                        "Open this notebook from the Assignment4_Submission folder, "
                        "or run it in Colab and upload Assignment4_Submission.zip."
                    ) from error
                print("Upload Assignment4_Submission.zip to load the data and model files.")
                uploaded = files.upload()
                zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
                if not zip_names:
                    raise FileNotFoundError("Please upload the Assignment4_Submission.zip package.")
                with zipfile.ZipFile(zip_names[0]) as package:
                    package.extractall(Path.cwd())
                PROJECT_ROOT = Path.cwd() / "Assignment4_Submission"
        if not (PROJECT_ROOT / "train_models.py").exists():
            raise FileNotFoundError(f"Project files were not found under {PROJECT_ROOT}.")
        os.chdir(PROJECT_ROOT)
        sys.path.insert(0, str(PROJECT_ROOT))

        import pandas as pd
        from IPython.display import Image, display
        from train_models import load_dataset, run_experiment

        data = load_dataset()
        print(f"Project folder: {PROJECT_ROOT}")
        print(f"Dataset shape: {data.shape}")
        """
    ),
    markdown(
        """
        ### Data understanding

        The full dataset has one text input and one binary target. The next output checks the complete row count, label balance, and message-length distribution without displaying any raw phone numbers or message contents.
        """
    ),
    code(
        """
        class_summary = (
            data["label"].value_counts()
            .reindex(["ham", "spam"])
            .rename_axis("Class")
            .to_frame("Messages")
        )
        class_summary["Share (%)"] = class_summary["Messages"] / len(data) * 100
        display(class_summary.round(2))

        message_lengths = data["message"].str.len().describe().rename("Characters")
        display(message_lengths.to_frame().round(2))
        """
    ),
    code(
        """
        display(Image(filename=str(PROJECT_ROOT / "results" / "class_distribution.png")))
        """
    ),
    markdown(
        """
        ### Preprocessing and evaluation design

        The dataset is split once into 80% training and 20% test data, with stratification and random seed 42 so both classes keep approximately the same proportions. TF-IDF is fitted on the training messages only, then applied to the held-out messages. It uses lowercased word unigrams and bigrams, ignores terms appearing in fewer than two training messages, and retains up to 30,000 terms. Stop words are not removed, so terms such as “not” remain available to the classifier.

        The six algorithms are Logistic Regression, Linear SVM, Multinomial Naive Bayes, Decision Tree, Random Forest, and K-Nearest Neighbors. The primary selection metric is **macro F1**, with spam recall as a tie-breaker. The evaluation reports additional metrics because accuracy can look strong when most records belong to the ham class.
        """
    ),
    code(
        """
        experiment = run_experiment()
        comparison = experiment["comparison"]
        display(comparison.round(4))
        """
    ),
    code(
        """
        display(Image(filename=str(PROJECT_ROOT / "results" / "model_comparison.png")))
        display(Image(filename=str(PROJECT_ROOT / "results" / "best_model_confusion_matrix.png")))
        """
    ),
    markdown(
        """
        ### Evaluation and model selection

        The comparison above uses predictions on the held-out test set; those reported scores are not training scores. The chosen algorithm is then refit on all 5,574 labeled messages for the saved deployment pipeline. Because the complete-data pipeline has seen every record, its training performance is not reported as an unbiased evaluation result.
        """
    ),
    code(
        """
        selected = comparison.iloc[0]
        print(f"Selected algorithm: {selected['Model']}")
        print(f"Macro F1: {selected['Macro F1']:.4f}")
        print(f"Spam precision: {selected['Spam precision']:.4f}")
        print(f"Spam recall: {selected['Spam recall']:.4f}")
        print(f"Overall accuracy: {selected['Accuracy']:.4f}")
        print(f"Saved full-data pipeline: {PROJECT_ROOT / 'models' / 'best_sms_spam_model.joblib'}")
        """
    ),
    markdown(
        """
        ### Example application predictions

        The final fitted pipeline contains both the TF-IDF transformation and selected classifier, so the same preprocessing is applied when new text is entered in the web application.
        """
    ),
    code(
        """
        import joblib

        final_model = joblib.load(PROJECT_ROOT / "models" / "best_sms_spam_model.joblib")
        examples = [
            "Congratulations, you have won a free prize. Reply now to claim your reward!",
            "Hi, are we still meeting at the cafe after class?",
        ]
        predictions = final_model.predict(examples)
        display(pd.DataFrame({"Example message": examples, "Predicted class": predictions}))
        """
    ),
    markdown(
        """
        ## Model deployment

        The `app.py` file provides an interactive Streamlit application. The trained pipeline is saved at `models/best_sms_spam_model.joblib`, and the pinned dependencies are in `requirements.txt`.

        **Cloud deployment steps:**
        1. Upload the contents of this project folder, including the `models/` directory, to a GitHub repository.
        2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app using that repository, branch, and `app.py` as the entrypoint.
        3. In Advanced settings, select Python 3.14 to match this model's training environment and the pinned scikit-learn 1.9.0 dependency.
        4. Deploy the app, copy the resulting `streamlit.app` URL, and replace the pending link at the top of this notebook.

        Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) describes the current repository, entrypoint, and Python-version settings. A live URL must be added after publication.
        """
    ),
    markdown(
        """
        ## Limitations and conclusion

        This project applies TF-IDF feature extraction to unstructured SMS text and compares six machine-learning algorithms on the full labeled dataset using a stratified hold-out evaluation. The experiment selects and saves the best algorithm by macro F1, then exposes it through a small web application.

        The dataset is old, English-only, and drawn from a limited collection of sources. Modern spam, other languages, and regional spelling may differ. A mistaken spam label can hide a legitimate message, while a missed spam message can still cause harm; the app is a classroom demonstration and not a production filter.
        """
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {
        "display_name": "Python 3.14 (AIC354 Assignment 4)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.14"},
}

with (ROOT / "results" / "model_comparison.csv").open(encoding="utf-8", newline="") as stream:
    saved_rows = list(csv.DictReader(stream))
result_table = [
    "### Held-out model comparison (run completed)",
    "",
    "The following results were produced by running `train_models.py` on all 5,574 messages with the stratified split described above. `Fit time` excludes TF-IDF fitting, which is shared across all six classifiers.",
    "",
    "| Model | Accuracy | Balanced accuracy | Macro F1 | Spam precision | Spam recall | Spam F1 | Fit time (s) |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in saved_rows:
    result_table.append(
        "| {Model} | {Accuracy} | {Balanced accuracy} | {Macro F1} | {Spam precision} | {Spam recall} | {Spam F1} | {Fit seconds} |".format(
            Model=row["Model"],
            Accuracy=f"{float(row['Accuracy']):.4f}",
            **{
                "Balanced accuracy": f"{float(row['Balanced accuracy']):.4f}",
                "Macro F1": f"{float(row['Macro F1']):.4f}",
                "Spam precision": f"{float(row['Spam precision']):.4f}",
                "Spam recall": f"{float(row['Spam recall']):.4f}",
                "Spam F1": f"{float(row['Spam F1']):.4f}",
                "Fit seconds": f"{float(row['Fit seconds']):.4f}",
            },
        )
    )
result_table.append("")
result_table.append(
    f"**Selected algorithm:** {saved_rows[0]['Model']} (macro F1 {float(saved_rows[0]['Macro F1']):.4f}; spam recall {float(saved_rows[0]['Spam recall']):.4f})."
)
cells.insert(11, markdown("\n".join(result_table)))

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Saved notebook with completed experiment results: {NOTEBOOK_PATH}")

