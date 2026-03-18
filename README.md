# Winstars AI DS internship test

## Tasks

- [Task 1: MNIST Classification](task1/README.md) - three classifiers (Random Forest, Feed-Forward NN, CNN) for handwritten digit recognition.
- [Task 2: Animal Text-Image Matching](task2/README.md) - NER + image classification pipeline that checks whether a text description matches an animal in an image.

## Running Notebooks in Google Colab

Any `.ipynb` in this repo can be opened in Google Colab by replacing the GitHub URL domain:

```
https://github.com/BohdanMarko/winstars-ai-ds-internship/blob/master/<path-to-notebook>
->
https://colab.research.google.com/github/BohdanMarko/winstars-ai-ds-internship/blob/master/<path-to-notebook>
```

Or open [colab.research.google.com](https://colab.research.google.com) -> **File -> Open notebook -> GitHub** and paste the repo URL.

Since notebooks import local modules and data, add this cell at the top before running:

```python
!git clone https://github.com/BohdanMarko/winstars-ai-ds-internship.git
%cd winstars-ai-ds-internship/<task-folder>
!pip install -r requirements.txt
```

For GPU-accelerated training (CNN / ResNet), go to **Runtime -> Change runtime type -> T4 GPU**.
