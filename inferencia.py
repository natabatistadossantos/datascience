import json
import sys
import joblib
import pandas as pd

m = joblib.load("modelo_breast.joblib")
pipeline, ATRIBUTOS, CATEGORIAS = m["pipeline"], m["atributos"], m["categorias"]

def prever(dados: dict):
    linha = {}
    for c in ATRIBUTOS:
        v = dados.get(c)
        v = None if v is None or str(v).strip() == "" else str(v).strip()
        if v is None:
            v = m["modas"].get(c, CATEGORIAS[c][0])
        elif v not in CATEGORIAS[c]:
            raise ValueError(f"Valor invalido para '{c}': '{v}'. Opcoes: {CATEGORIAS[c]}")
        linha[c] = v
    X = pd.DataFrame([linha])[ATRIBUTOS]
    classe = pipeline.predict(X)[0]
    proba = {c: float(p) for c, p in zip(pipeline.classes_, pipeline.predict_proba(X)[0].round(3))}
    return classe, proba

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dados = json.loads(sys.argv[1])
    else:
        dados = {}
        print("Dados da paciente (Enter = nao sei):")
        for c in ATRIBUTOS:
            dados[c] = input(f"  {c} {CATEGORIAS[c]}: ")
    try:
        classe, proba = prever(dados)
    except ValueError as e:
        sys.exit(f"Erro: {e}")
    print(f"\nModelo: {m['modelo_nome']} (acuracia CV {m['acuracia_cv']:.2%}, teste {m['acuracia_teste']:.2%})")
    print(f"Indicativo (Class): {classe}")
    print("Probabilidades:", proba)
    print("-> " + ("COM recorrencia de eventos" if classe == "recurrence-events"
                   else "SEM recorrencia de eventos"))
