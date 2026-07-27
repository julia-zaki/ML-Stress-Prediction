"""
Random Forest model
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

# with random forest we won't overfit

DROP_COLS = ["id", "day_in_study", "stress"]
TARGET = "stress"
def main():
    # 1. Load Data
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")

    # 2. Define Features and Target
    X_train = train_df.drop(columns=DROP_COLS)
    y_train = train_df[TARGET]

    X_val = val_df.drop(columns=DROP_COLS)
    y_val = val_df[TARGET]

    X_test = test_df.drop(columns=DROP_COLS)
    y_test = test_df[TARGET]

    # 3. Find Best parameters using Validation Set (Tuning)
    best_tree_parameters = {}
    best_tree_val_acc = 0

    for depth in [3, 5, 7, 10]:
        for min_leaf in [2, 5, 7, 10, 15]:
            for min_split in [2, 5, 7, 10, 15]:
                dt = DecisionTreeClassifier(max_depth=depth,
                                            min_samples_leaf=min_leaf,
                                            min_samples_split=min_split,
                                            random_state=42)
                dt.fit(X_train, y_train)
                val_acc = dt.score(X_val, y_val)

                if val_acc > best_tree_val_acc:
                    best_tree_val_acc = val_acc
                    best_tree_parameters = {
                        "max_depth": depth,
                        "min_samples_leaf": min_leaf,
                        "min_samples_split": min_split
                    }

    print(f"Best Tree parameters found: {best_tree_parameters}")

    # 4. Apply to Test Set and Report Accuracies
    final_tree = DecisionTreeClassifier(**best_tree_parameters, random_state=42)
    final_tree.fit(X_train, y_train)

    train_tree_acc = final_tree.score(X_train, y_train)
    val_tree_acc = final_tree.score(X_val, y_val)
    test_tree_acc = final_tree.score(X_test, y_test)

    print(f"Training Tree Accuracy: {train_tree_acc:.4f}")
    print(f"Validation Tree Accuracy: {val_tree_acc:.4f}")
    print(f"Test Tree Accuracy: {test_tree_acc:.4f}")

    # Macro F1 Score
    y_test_pred_tree = final_tree.predict(X_test)
    test_tree_f1 = f1_score(y_test, y_test_pred_tree, average='macro')
    print(f"Test Tree Macro F1: {test_tree_f1:.4f}")

    # Confusion Matrix
    labels = sorted(y_test.unique())
    cm_tree = confusion_matrix(y_test, y_test_pred_tree, labels=labels)
    print(f"\nTree Confusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>30s}  {'  '.join(l[:12] for l in labels)}")
    for i, label in enumerate(labels):
        print(f"{label:>30s}  {'     '.join(str(v).rjust(5) for v in cm_tree[i])}")


# --- Same process but for random forest ---
    best_forest_parameters = {}
    best_forest_val_acc = 0

    for n_estimators in [50, 100, 200]:
        for max_depth in [3, 5, 7, 10]:
            for min_split in [2, 5, 7, 10, 15]:
                for min_leaf in [2, 5, 7, 10, 15]:
                    for max_feat in ['sqrt']:
                        rf = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            max_features=max_feat,
                            random_state=42
                        )
                        rf.fit(X_train, y_train)
                        val_acc = rf.score(X_val, y_val)

                        if val_acc > best_forest_val_acc:
                            best_forest_val_acc = val_acc
                            best_forest_parameters = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf,
                                'max_features': max_feat
                            }

    print(f"Best Forest parameters found: {best_forest_parameters}")

    # 4. Apply to Test Set and Report Accuracies
    final_forest = RandomForestClassifier(**best_forest_parameters, random_state=42)
    final_forest.fit(X_train, y_train)

    train_forest_acc = final_forest.score(X_train, y_train)
    val_forest_acc = final_forest.score(X_val, y_val)
    test_forest_acc = final_forest.score(X_test, y_test)

    print(f"Training Forest Accuracy: {train_forest_acc:.4f}")
    print(f"Validation Forest Accuracy: {val_forest_acc:.4f}")
    print(f"Test Forest Accuracy: {test_forest_acc:.4f}")

    # Macro F1 Score. Seeing how well the model does on each class separately
    y_test_pred_forest = final_forest.predict(X_test)
    test_forest_f1 = f1_score(y_test, y_test_pred_forest, average='macro')
    print(f"Test Forest Macro F1: {test_forest_f1:.4f}")

    # Confusion Matrix
    cm_forest = confusion_matrix(y_test, y_test_pred_forest, labels=labels)
    print(f"\nForest Confusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>30s}  {'  '.join(l[:12] for l in labels)}")
    for i, label in enumerate(labels):
        print(f"{label:>30s}  {'     '.join(str(v).rjust(5) for v in cm_forest[i])}")

if __name__ == "__main__":
    main()
