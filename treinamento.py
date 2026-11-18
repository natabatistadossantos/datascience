import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "breast-cancer.data"
SEED = 42
COLUNAS = ["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps",
           "deg-malig", "breast", "breast-quad", "irradiat"]
ATRIBUTOS = COLUNAS[1:]

# 1. carga 
df = pd.read_csv(CSV, header=None, names=COLUNAS, na_values="?",
                 dtype={"deg-malig": str})
print("Dados:", df.shape)
print(df["Class"].value_counts().to_string(), "\n")

# 2. nulos 
modas = {}
for c in ATRIBUTOS:
    if df[c].isnull().any():
        modas[c] = df[c].mode()[0]
        df[c] = df[c].fillna(modas[c])
        print(f"Nulos em '{c}' preenchidos com a moda '{modas[c]}'")

X, y = df[ATRIBUTOS], df["Class"]
categorias = {c: sorted(X[c].unique().tolist()) for c in ATRIBUTOS}

# 3. treino
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=SEED, stratify=y)

def pipe(modelo):
    enc = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False), ATRIBUTOS)])
    return Pipeline([("enc", enc), ("smote", SMOTE(random_state=SEED)),
                     ("norm", MinMaxScaler()), ("modelo", modelo)])

# 4. modelos
candidatos = {
    "DecisionTree": (DecisionTreeClassifier(random_state=SEED), {
        "modelo__criterion": ["gini", "entropy", "log_loss"],
        "modelo__max_depth": [2, 3, 4, 5, 8, None],
        "modelo__min_samples_split": [2, 5, 10, 20],
        "modelo__min_samples_leaf": [1, 3, 5, 10]}),
    "RandomForest": (RandomForestClassifier(random_state=SEED), {
        "modelo__n_estimators": [100, 200, 400],
        "modelo__max_depth": [None, 4, 6, 10],
        "modelo__min_samples_leaf": [1, 3, 5],
        "modelo__criterion": ["gini", "entropy"]}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=SEED), {
        "modelo__n_estimators": [50, 100, 200],
        "modelo__learning_rate": [0.01, 0.05, 0.1],
        "modelo__max_depth": [1, 2, 3]}),
    "KNN": (KNeighborsClassifier(), {
        "modelo__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "modelo__weights": ["uniform", "distance"]}),
    "SVM": (SVC(random_state=SEED), {
        "modelo__C": [0.1, 0.5, 1, 5, 10],
        "modelo__kernel": ["linear", "rbf"],
        "modelo__gamma": ["scale", 0.01, 0.05, 0.1]}),
    "LogisticRegression": (LogisticRegression(max_iter=2000), {
        "modelo__C": [0.01, 0.1, 0.5, 1, 5, 10]}),
}

# 5. hiperparametrizacao
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
buscas = {}
print("\n=== Busca de hiperparametros (acuracia CV, 5 folds) ===")
for nome, (modelo, grade) in candidatos.items():
    n_comb = int(np.prod([len(v) for v in grade.values()]))
    busca = RandomizedSearchCV(pipe(modelo), grade, n_iter=min(15, n_comb),
                               cv=cv, scoring="accuracy", random_state=SEED, n_jobs=-1)
    busca.fit(X_tr, y_tr)
    buscas[nome] = busca
    print(f"{nome:<19} CV={busca.best_score_:.4f}  {busca.best_params_}")


melhor = max(buscas, key=lambda k: buscas[k].best_score_)
print(f"\n>>> Modelo de melhor acuracia (CV): {melhor}")


y_pred = buscas[melhor].predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"Acuracia no teste: {acc:.4f}  (baseline 'sempre no-recurrence': "
      f"{(y_te == 'no-recurrence-events').mean():.4f})")
print(classification_report(y_te, y_pred))
print("Matriz de confusao:\n", confusion_matrix(y_te, y_pred))

# 6. modelo final
final = buscas[melhor].best_estimator_
final.fit(X, y)

joblib.dump({"pipeline": final, "modelo_nome": melhor, "atributos": ATRIBUTOS,
             "categorias": categorias, "modas": modas, "acuracia_cv": buscas[melhor].best_score_,
             "acuracia_teste": acc, "best_params": buscas[melhor].best_params_},
            "modelo_breast.joblib")
print("\nModelo salvo em modelo_breast.joblib")
