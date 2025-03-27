# Part 1: Imports and GUI Initialization
import sys
import numpy as np
import pandas as pd
import urllib.request
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
    QGroupBox, QScrollArea, QTextEdit, QStatusBar, QProgressBar, QCheckBox,
    QMessageBox, QDialog, QLineEdit
)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.layer_config = []

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.create_data_section()
        self.create_loss_function_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def create_visualization(self):
        group = QGroupBox("Results")
        layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        layout.addWidget(self.metrics_text)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.setStatusBar(self.status_bar)

    def create_data_section(self):
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"
        ])

        self.missing_value_combo = QComboBox()
        self.missing_value_combo.addItems([
            "None", "Mean Imputation", "Interpolation", "Forward Fill", "Backward Fill"
        ])

        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)

        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Missing Value Handling:"))
        data_layout.addWidget(self.missing_value_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def load_dataset(self):
        try:
            dataset_name = self.dataset_combo.currentText()

            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
                X, y = data.data, data.target
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
                X, y = data.data, data.target
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
                X, y = data.data, data.target
            elif dataset_name == "Boston Housing Dataset":
                url = "http://lib.stat.cmu.edu/datasets/boston"
                raw = urllib.request.urlopen(url).read().decode("utf-8").split("\n")[22:]
                raw = list(filter(None, raw))
                data = []
                for i in range(0, len(raw), 2):
                    first = list(map(float, raw[i].strip().split()))
                    second = list(map(float, raw[i + 1].strip().split()))
                    row = first + second
                    data.append(row)
                data = np.array(data)
                X = data[:, :-1]
                y = data[:, -1]
            else:
                return

            df = pd.DataFrame(X)
            self.handle_missing_values(df)
            self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                df, y, test_size=self.split_spin.value(), random_state=42
            )
            self.apply_scaling()
            self.status_bar.showMessage(f"{dataset_name} loaded.")
        except Exception as e:
            self.show_error(f"Dataset loading error: {str(e)}")

    def load_custom_data(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV files (*.csv)")
            if file_name:
                df = pd.read_csv(file_name)
                target_col = self.select_target_column(df.columns)
                if target_col:
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    self.handle_missing_values(X)
                    self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                        X, y, test_size=self.split_spin.value(), random_state=42
                    )
                    self.apply_scaling()
                    self.status_bar.showMessage("Custom dataset loaded")
        except Exception as e:
            self.show_error(f"Custom dataset error: {str(e)}")

    def handle_missing_values(self, df):
        method = self.missing_value_combo.currentText()
        if method == "Mean Imputation":
            imputer = SimpleImputer(strategy="mean")
            df[df.columns] = imputer.fit_transform(df)
        elif method == "Interpolation":
            df.interpolate(method="linear", inplace=True)
        elif method == "Forward Fill":
            df.fillna(method="ffill", inplace=True)
        elif method == "Backward Fill":
            df.fillna(method="bfill", inplace=True)

    def apply_scaling(self):
        method = self.scaling_combo.currentText()
        if method == "Standard Scaling":
            scaler = preprocessing.StandardScaler()
        elif method == "Min-Max Scaling":
            scaler = preprocessing.MinMaxScaler()
        elif method == "Robust Scaling":
            scaler = preprocessing.RobustScaler()
        else:
            return
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def select_target_column(self, columns):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        dialog.setLayout(layout)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None

    def create_loss_function_section(self):
        group = QGroupBox("Loss Function")
        layout = QHBoxLayout()

        self.regression_loss_combo = QComboBox()
        self.regression_loss_combo.addItems(["MSE", "MAE", "Huber"])
        layout.addWidget(QLabel("Regression Loss:"))
        layout.addWidget(self.regression_loss_combo)

        self.classification_loss_combo = QComboBox()
        self.classification_loss_combo.addItems(["Cross-Entropy", "Hinge Loss"])
        layout.addWidget(QLabel("Classification Loss:"))
        layout.addWidget(self.classification_loss_combo)

        group.setLayout(layout)
        self.layout.addWidget(group)

    def get_selected_loss(self, task_type="regression"):
        if task_type == "regression":
            return self.regression_loss_combo.currentText()
        else:
            return self.classification_loss_combo.currentText()

    def calculate_loss(self, y_true, y_pred, task_type="regression"):
        loss = self.get_selected_loss(task_type)
        if task_type == "regression":
            if loss == "MSE":
                return mean_squared_error(y_true, y_pred)
            elif loss == "MAE":
                return np.mean(np.abs(y_true - y_pred))
            elif loss == "Huber":
                delta = 1.0
                error = y_true - y_pred
                return np.mean(np.where(np.abs(error) <= delta,
                                        0.5 * error ** 2,
                                        delta * (np.abs(error) - 0.5 * delta)))
        else:
            if loss == "Cross-Entropy":
                eps = 1e-15
                y_pred = np.clip(y_pred, eps, 1 - eps)
                return -np.mean(np.log(y_pred[range(len(y_true)), y_true]))
            elif loss == "Hinge Loss":
                y_true_bin = np.where(y_true == 1, 1, -1)
                y_pred_bin = np.where(y_pred >= 0.5, 1, -1)
                return np.mean(np.maximum(0, 1 - y_true_bin * y_pred_bin))

    def train_model(self, name, param_widgets):
        try:
            X_train = self.X_train
            y_train = self.y_train
            X_test = self.X_test
            y_test = self.y_test

            params = {}
            for key, widget in param_widgets.items():
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    params[key] = widget.value()
                elif isinstance(widget, QCheckBox):
                    params[key] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    params[key] = widget.currentText()

            if name == "Linear Regression":
                model = LinearRegression(**params)
            elif name == "Logistic Regression":
                model = LogisticRegression(**params)
            elif name == "Naive Bayes":
                priors = None
                if self.prior_combo.currentText() == "Custom":
                    try:
                        priors = list(map(float, self.nb_prior_input.text().split(",")))
                    except:
                        self.show_error("Invalid prior format.")
                        return
                model = GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9), priors=priors)
            elif name == "Support Vector Machine":
                model = SVC(**params)
            elif name == "SVR":
                model = SVR(**params)
            elif name == "Decision Tree":
                model = DecisionTreeClassifier(**params)
            elif name == "Random Forest":
                model = RandomForestClassifier(**params)
            elif name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(**params)
            elif name == "K-Means Parameters":
                model = KMeans(**params)
                model.fit(X_train)
                y_pred = model.predict(X_train)
                self.update_visualization(y_pred)
                self.metrics_text.setText("KMeans clustering completed.")
                return
            elif name == "PCA Parameters":
                model = PCA(**params)
                X_pca = model.fit_transform(X_train)
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
                self.canvas.draw()
                self.metrics_text.setText("PCA applied.")
                return
            else:
                raise ValueError("Unknown model name")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.current_model = model
            self.update_visualization(y_pred)
            self.update_metrics(y_pred)

        except Exception as e:
            self.show_error(f"Training error: {str(e)}")

    def update_visualization(self, y_pred):
        self.figure.clear()
        if len(np.unique(self.y_test)) > 10:
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        else:
            if self.X_test.shape[1] > 2:
                X_2d = PCA(n_components=2).fit_transform(self.X_test)
                ax = self.figure.add_subplot(111)
                sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap="viridis")
                self.figure.colorbar(sc)
            else:
                ax = self.figure.add_subplot(111)
                sc = ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap="viridis")
                self.figure.colorbar(sc)
        self.canvas.draw()

    def update_metrics(self, y_pred):
        if len(np.unique(self.y_test)) > 10:
            mse = mean_squared_error(self.y_test, y_pred)
            mae = np.mean(np.abs(self.y_test - y_pred))
            loss_val = self.calculate_loss(self.y_test, y_pred, task_type="regression")
            msg = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nCustom Loss: {loss_val:.4f}"
        else:
            acc = accuracy_score(self.y_test, y_pred)
            conf = confusion_matrix(self.y_test, y_pred)
            loss_val = self.calculate_loss(self.y_test, y_pred, task_type="classification")
            msg = f"Accuracy: {acc:.4f}\nLoss: {loss_val:.4f}\n\nConfusion Matrix:\n{conf}"
        self.metrics_text.setText(msg)

    def create_tabs(self):
        self.tab_widget = QTabWidget()
        scroll = QScrollArea()
        classical_tab = self.create_classical_ml_tab()
        scroll.setWidget(classical_tab)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, "Classical ML")
        self.layout.addWidget(self.tab_widget)

    def create_classical_ml_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Regression Algorithms
        reg_group = QGroupBox("Regression")
        reg_layout = QVBoxLayout()

        linear_group = self.create_algorithm_group("Linear Regression", {
            "fit_intercept": "checkbox"
        })
        svr_group = self.create_algorithm_group("SVR", {
            "C": "double", "epsilon": "double", "kernel": ["linear", "rbf", "poly"]
        })

        reg_layout.addWidget(linear_group)
        reg_layout.addWidget(svr_group)
        reg_group.setLayout(reg_layout)

        # Classification Algorithms
        clf_group = QGroupBox("Classification")
        clf_layout = QVBoxLayout()

        logistic_group = self.create_algorithm_group("Logistic Regression", {
            "C": "double", "max_iter": "int"
        })
        svm_group = self.create_algorithm_group("Support Vector Machine", {
            "C": "double", "kernel": ["linear", "rbf", "poly"], "degree": "int"
        })
        nb_group = self.create_algorithm_group("Naive Bayes", {
            "var_smoothing": "double"
        })
        self.create_naive_bayes_section(clf_layout)

        clf_layout.addWidget(logistic_group)
        clf_layout.addWidget(svm_group)
        clf_layout.addWidget(nb_group)
        clf_group.setLayout(clf_layout)

        # Add both to layout
        layout.addWidget(reg_group)
        layout.addWidget(clf_group)

        return widget

    def create_algorithm_group(self, name, params):
        group = QGroupBox(name)
        layout = QVBoxLayout()
        param_widgets = {}

        for param_name, param_type in params.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(param_name))

            if param_type == "int":
                widget = QSpinBox()
                widget.setMaximum(10000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setMaximum(1e4)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)

            row.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(row)

        btn = QPushButton(f"Train {name}")
        btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(btn)

        group.setLayout(layout)
        return group

    def create_naive_bayes_section(self, layout):
        self.prior_combo = QComboBox()
        self.prior_combo.addItems(["Uniform", "Custom"])
        self.nb_prior_input = QLineEdit()
        self.nb_prior_input.setPlaceholderText("e.g. 0.3,0.7")

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Naive Bayes Priors:"))
        hbox.addWidget(self.prior_combo)
        hbox.addWidget(self.nb_prior_input)

        group = QGroupBox("Naive Bayes Priors")
        group.setLayout(hbox)
        layout.addWidget(group)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())
