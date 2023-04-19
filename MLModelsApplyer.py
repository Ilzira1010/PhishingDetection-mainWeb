import pickle

def apply(X_test):
    with open('svm.pkl', 'rb') as f:
        model_svm = pickle.load(f)
        y_pred_svm = model_svm.predict(X_test)
        print(y_pred_svm)
        if (y_pred_svm == 1):
            print("Результат SVM: фишинг")
        else:
            print("Результат SVM: не является фишингом")

    with open('neur.pkl', 'rb') as f:
        model_neur = pickle.load(f)
        y_pred_neur = model_neur.predict(X_test)
        print(y_pred_neur)
        if (y_pred_neur == 1):
            print("Результат нейронной сети: фишинг")
        else:
            print("Результат нейронной сети: не является фишингом")
    
    with open('kneighb.pkl', 'rb') as f:
        model_kneighb = pickle.load(f)
        y_pred_kneighb = model_kneighb.predict(X_test)
        print(y_pred_kneighb)
        if (y_pred_kneighb == 1):
            print("Результат метода k-ближайших соседей: фишинг")
        else:
            print("Результат метода k-ближайших соседей: не является фишингом")

    with open('decision.pkl', 'rb') as f:
        model_decision = pickle.load(f)
        y_pred_decision = model_decision.predict(X_test)
        print(y_pred_decision)
        if (y_pred_decision == 1):
            print("Результат метода решающих деревьев: фишинг")
        else:
            print("Результат метода решающих деревьев: не является фишингом")

def apply_to_one_res(X_test):
    y_pred = []
    with open('svm.pkl', 'rb') as f:
        model_svm = pickle.load(f)
        y_pred_svm = model_svm.predict(X_test)
        y_pred.append(y_pred_svm)

    with open('neur.pkl', 'rb') as f:
        model_neur = pickle.load(f)
        y_pred_neur = model_neur.predict(X_test)
        y_pred.append(y_pred_neur)
    
    with open('kneighb.pkl', 'rb') as f:
        model_kneighb = pickle.load(f)
        y_pred_kneighb = model_kneighb.predict(X_test)
        y_pred.append(y_pred_kneighb)

    with open('decision.pkl', 'rb') as f:
        model_decision = pickle.load(f)
        y_pred_decision = model_decision.predict(X_test)
        y_pred.append(y_pred_decision)
    return sum(y_pred) / len(y_pred)
    