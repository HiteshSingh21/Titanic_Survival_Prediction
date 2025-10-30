
## 📓 Project Notebook: `Final.ipynb`

This repository's core logic is contained in the Jupyter Notebook `Final.ipynb`. This notebook details the complete process of building and evaluating a deep learning model for sentiment analysis.

The project uses a **fine-tuned BERT model** (from the Hugging Face `transformers` library) to classify the sentiment of tweets (likely from the Sentiment140 dataset) as either positive or negative.

### Notebook Workflow

The notebook is structured to cover the following key steps:

1.  **Data Loading:** The Twitter sentiment dataset is loaded into a pandas DataFrame.
2.  **Data Pre-processing:** (Implicit) Text data is cleaned and tokenized to be suitable for the BERT model.
3.  **Model Fine-Tuning:** A pre-trained BERT model is loaded and fine-tuned on the specific task of sentiment classification.
4.  **Model Evaluation:** The performance of the fine-tuned model is assessed using standard classification metrics. The notebook generates:
      * A **Classification Report**, showing precision, recall, and F1-score for both positive and negative classes.
      * A **Confusion Matrix**, visualized with `seaborn` and `matplotlib`, to show the model's performance in distinguishing between the two classes.

### 🛠️ How to Run

To run this notebook and reproduce the results, you must first install all the required Python libraries.

1.  Ensure you have Python 3.x installed.

2.  Install all dependencies using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

3.  Once the dependencies are installed, you can launch the notebook:

    ```bash
    jupyter notebook Final.ipynb
    ```

-----
