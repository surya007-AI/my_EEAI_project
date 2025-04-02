<<<<<<< HEAD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

def model_predict(data, df, name):
    """
    Train a logistic regression model and evaluate it.
    """
    X_train = data.X  # TF-IDF embeddings
    y_train = df['innso typology_ticket']  # Target variable (adjust if needed)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average accuracy from cross-validation: {cv_scores.mean()}")

    # Fit the model on the entire dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    # Metrics evaluation
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_train, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_train, y_pred, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(y_train, y_pred)

    # Print metrics
    print(f"Model Evaluation for {name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save the model using joblib
    joblib.dump(model, f'{name}_model.pkl')  # Save the trained model to a file
    print(f"Model saved as {name}_model.pkl")
=======
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

def model_predict(data, df, name):
    """
    Train a logistic regression model and evaluate it.
    """
    X_train = data.X  # TF-IDF embeddings
    y_train = df['innso typology_ticket']  # Target variable (adjust if needed)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average accuracy from cross-validation: {cv_scores.mean()}")

    # Fit the model on the entire dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    # Metrics evaluation
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_train, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_train, y_pred, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(y_train, y_pred)

    # Print metrics
    print(f"Model Evaluation for {name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save the model using joblib
    joblib.dump(model, f'{name}_model.pkl')  # Save the trained model to a file
    print(f"Model saved as {name}_model.pkl")
>>>>>>> 798080e5b70fcfdb441579832e1348b3b1b4f4fc
