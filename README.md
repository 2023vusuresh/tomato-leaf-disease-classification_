# Tomato Leaf Disease Detection Dashboard

This Streamlit app classifies tomato leaf disease using your trained TensorFlow/Keras model.

## Features

- Upload leaf image
- Take photo using camera
- Predict tomato disease
- Show exact confidence percentage for the predicted disease
- Show top 3 prediction probabilities
- Show all class confidence percentages
- Give disease-wise recommendations
- Basic warning if image does not look like a tomato leaf

## Required Files

Keep these files together in one GitHub repository:

```text
app.py
recommendations.json
requirements.txt
tomato_disease_final_model.keras
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

### Step 1: Create GitHub Repository

Create a new GitHub repository, for example:

```text
tomato-leaf-disease-dashboard
```

Upload these files:

```text
app.py
recommendations.json
requirements.txt
tomato_disease_final_model.keras
```

### Step 2: Open Streamlit Cloud

Go to Streamlit Community Cloud and select:

```text
New app → Select your GitHub repo → Main file path: app.py → Deploy
```

### Step 3: Model File Warning

If your model file is larger than GitHub's normal file limit, use Git LFS.

Install Git LFS:

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add tomato_disease_final_model.keras
git commit -m "Add model using Git LFS"
git push
```

## Important Limitation

The current model was trained only on tomato leaf classes. So the app uses confidence and green-pixel checking to warn when the image may not be a tomato leaf.

For the most accurate "not tomato leaf" detection, retrain the model with an extra class:

```text
Not_Tomato_Leaf
```

This class should include:

- Potato leaves
- Other plant leaves
- Human hand images
- Soil/background images
- Random non-leaf images
- Unclear/blurry images