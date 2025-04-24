"""train_decision_tree.py
Script ini melakukan training model Decision Tree untuk klasifikasi kelayakan kredit komputer
menggunakan dataset CSV (default: data/dataset_buys_comp.csv).

Usage:
    python train_decision_tree.py --data data/dataset_buys_comp.csv --model_out model_dt.pkl
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt


def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_preprocessor(df: pd.DataFrame):
    feature_cols = df.columns.drop('Buys_Computer')
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if df[c].dtype != 'object']

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipe, cat_cols),
        ('num', num_pipe, num_cols)
    ])
    return preprocessor


def get_model():
    return DecisionTreeClassifier(random_state=42)


def main(args):
    df = load_data(Path(args.data))

    X = df.drop('Buys_Computer', axis=1)
    y = df['Buys_Computer']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = build_preprocessor(df)

    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('model', get_model())
    ])

    param_grid = {
        'model__criterion': ['gini', 'entropy'],
        'model__max_depth': [None, 3, 5, 7, 10],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print('Best params:', grid.best_params_)
    print('Best CV accuracy:', grid.best_score_)

    y_pred = grid.predict(X_test)
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)

    # plot confusion matrix
    ConfusionMatrixDisplay.from_estimator(grid, X_test, y_test)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    if len(y.unique()) == 2:
        y_prob = grid.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print('ROC-AUC:', auc)
        RocCurveDisplay.from_estimator(grid, X_test, y_test)
        plt.tight_layout()
        plt.savefig('roc_curve.png')

    # save model
    joblib.dump(grid.best_estimator_, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset_buys_comp.csv')
    parser.add_argument('--model_out', type=str, default='model_dt.pkl')
    main(parser.parse_args())