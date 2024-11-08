from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json
import pickle
import warnings

warnings.simplefilter("ignore", category=UserWarning)

DEFAULT_DATA_PATH = Path('data.xlsx')
DEFAULT_CLASSIFIER_PATH = Path('classifier.pkl')
EXPECTED_FEATURE_COUNT = 7


def load_data(path: Path) -> pd.DataFrame:
    """
    Load data from a specified file path (supports Excel).
    """
    try:
        return pd.read_excel(path)
    except FileNotFoundError:
        print(f"File '{path}' not found.")
        return None


def manage_classifier(path: Path, classifier: RandomForestClassifier = None) -> RandomForestClassifier:
    """
    Save or load the classifier depending on whether a classifier is provided.
    """
    if classifier:
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def train_classifier(class_column: str) -> None:
    """
    Train the RandomForest classifier on the provided dataset.
    """
    data = load_data(DEFAULT_DATA_PATH)
    if data is not None and class_column in data.columns:
        X = data.drop(class_column, axis=1)
        y = data[class_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27)
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, classifier.predict(X_test))
        print(f"Random Forest Classifier trained successfully! Accuracy: {accuracy * 100:.2f}%")

        manage_classifier(DEFAULT_CLASSIFIER_PATH, classifier)
    else:
        print(f"Class column '{class_column}' not found in dataset or data could not be loaded.")


def predict_class(input_data: list) -> None:
    """
    Predict the class for the provided input data using the trained classifier.
    """
    try:
        classifier = manage_classifier(DEFAULT_CLASSIFIER_PATH)
        prediction = classifier.predict([input_data])[0]
        print(f"Predicted class: {prediction}")
    except FileNotFoundError:
        print("Classifier not found. Please train the classifier first using '--train' command.")


def show_data(cmd_flag: bool, plt_flag: bool, column: str = None) -> None:
    """
    Show data in a pandas DataFrame or as a plot.
    """
    data = load_data(DEFAULT_DATA_PATH)
    if data is not None:
        if cmd_flag:
            if column:
                if column in data.columns:
                    print(data[[column]])
                else:
                    print(f"Column '{column}' not found in data.")
            else:
                print(data)
        if plt_flag:
            if column and column in data.columns:
                plt.plot(data.index, data[column], label=column)
                plt.xlabel('Index')
                plt.ylabel(column)
                plt.title(f'{column} over Index')
                plt.show()
            else:
                print(f"Error: Column '{column}' not found in data.")


def plot_data(columns: tuple) -> None:
    """
    Plot data by two specified column names.
    """
    data = load_data(DEFAULT_DATA_PATH)
    if data is not None and len(columns) == 2:
        x, y = columns
        if x in data.columns and y in data.columns:
            plt.scatter(data[x], data[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f'{x} vs {y}')
            plt.show()
        else:
            print(f"Error: Columns '{x}' and/or '{y}' not found in data.")
    else:
        print("Error: Please provide exactly two valid column names.")
def main():
    """
    Main function to handle command-line arguments and execute the appropriate function.
    """
    parser = argparse.ArgumentParser(description="Train and predict with RandomForest Classifier.")
    parser.add_argument('--train', '-t', action='store_true', help="Train the classifier on the dataset.")
    parser.add_argument('--predict', '-p', type=str, help="Predict class for a sample input row (provide values as a list, e.g., '[5.1, 3.5, 1.4, 0.2, 2.3, 1.5, 0.3]').")
    parser.add_argument('--show', '-s', action='store_true', help="Show dataset in terminal.")
    parser.add_argument('--column', '-c', type=str, help="Show specific column (e.g., '--column Engine Size').")
    parser.add_argument('--plot', '-l', type=str, help="Plot data by providing two column names for the x and y axes (e.g., '--plot Engine Size Horsepower').")

    args = parser.parse_args()

    if args.train:
        train_classifier('Class')

    if args.predict:
        try:
            input_data = json.loads(args.predict)
            if isinstance(input_data, list) and len(input_data) == EXPECTED_FEATURE_COUNT:
                predict_class(input_data)
            else:
                print(f"Error: Expected {EXPECTED_FEATURE_COUNT} features, but got {len(input_data) if isinstance(input_data, list) else 'invalid input format'}.")
        except json.JSONDecodeError:
            print("Error: Failed to decode input data. Ensure it is in proper list format, e.g., '[5.1, 3.5, 1.4, 0.2, 2.3, 1.5, 0.3]'.")
    
    if args.show:
        show_data(True, False, args.column)

    if args.plot:
        columns = args.plot.split()
        plot_data(tuple(columns))


if __name__ == '__main__':
    main()