# MKT3434 - Homework Assignment #1: Machine Learning GUI Enhancements

##  How to Use the GUI

- Run the application with: `python base_gui_for_MKT3434_by_eb.py`
- Select a dataset (e.g., Iris, Digits, Boston Housing).
- Choose missing value handling and scaling options if needed.
- Select a test split ratio (default is 0.2).
- Click **Load Data**.
- Navigate to the **Classical ML** tab.
- Choose a model (e.g., SVR, Naive Bayes).
- Set the parameters.
- Click the **Train** button.
- Results and visualizations appear at the bottom.

---

##  New Features Added

- **Missing value handling** options: Mean Imputation, Interpolation, Forward Fill, Backward Fill.
- **Dynamic loss functions** for regression and classification:
  - Regression: MSE, MAE, Huber
  - Classification: Cross-Entropy, Hinge Loss
- **SVM/SVR kernel selection** and hyperparameters (C, epsilon, degree, kernel).
- **Custom prior** input for Gaussian Naive Bayes.
- **Manual integration** of Boston Housing dataset using `urllib`.
- Result **visualization** and performance metrics shown.

---

##  How to Choose Missing Value Handling

- In the **top panel**, choose a method from the **"Missing Value Handling"** dropdown.
- Options include:
  - Mean Imputation (with SimpleImputer)
  - Interpolation
  - Forward Fill
  - Backward Fill
- Selected method is automatically applied before splitting the data.

---

##  SVM / SVR Configuration

- For **SVR** (Regression):
  - `C`: Regularization parameter.
  - `epsilon`: Defines epsilon-tube within which no penalty is associated.
  - `kernel`: Choose between "linear", "rbf", or "poly".

- For **SVM** (Classification):
  - `C`, `kernel`, and optionally `degree` for polynomial kernel.

---

##  Custom Priors in Gaussian Naive Bayes

- Under the **Naive Bayes** section:
  - Choose "Uniform" or "Custom" from the dropdown.
  - If "Custom" is selected, enter values like: `0.3,0.7` (must match number of classes).
  - These priors are passed to the `GaussianNB` model.

---

##  Using Dynamic Loss Functions

- **Regression Loss** options:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Huber Loss

- **Classification Loss** options:
  - Cross-Entropy
  - Hinge Loss

- Select your preferred loss function before training.
- Loss is calculated after prediction and shown in the Results panel.

---

##  Prepared by

- **Student Name**: Fatih  
- **Student ID**: (Write your ID here)  
- **Course**: MKT3434 Machine Learning  
- **University**: Yildiz Technical University
