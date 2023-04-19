import pickle

def nlp_apply(X_test):
    with open('svm_nlp.pkl', 'rb') as f:
        model_svm = pickle.load(f)
        y_pred_svm = model_svm.predict(X_test)
        if (y_pred_svm == 1):
            print("Результат SVM(по содержанию): фишинг")
        else:
            print("Результат SVM(по содержанию): не является фишингом")

    with open('neur_nlp.pkl', 'rb') as f:
        model_neur = pickle.load(f)
        y_pred_neur = model_neur.predict(X_test)
        if (y_pred_neur == 1):
            print("Результат нейронной сети(по содержанию): фишинг")
        else:
            print("Результат нейронной сети(по содержанию): не является фишингом")
    
    with open('kneighb_nlp.pkl', 'rb') as f:
        model_kneighb = pickle.load(f)
        y_pred_kneighb = model_kneighb.predict(X_test)
        if (y_pred_kneighb == 1):
            print("Результат метода k-ближайших соседей(по содержанию): фишинг")
        else:
            print("Результат метода k-ближайших соседей(по содержанию): не является фишингом")

    with open('decision_nlp.pkl', 'rb') as f:
        model_decision = pickle.load(f)
        y_pred_decision = model_decision.predict(X_test)
        if (y_pred_decision == 1):
            print("Результат метода решающих деревьев(по содержанию): фишинг")
        else:
            print("Результат метода решающих деревьев(по содержанию): не является фишингом")

def nlp_apply_one_result(X_test):
    y_pred = []
    with open('svm_nlp.pkl', 'rb') as f:
        model_svm = pickle.load(f)
        y_pred_svm = model_svm.predict(X_test)
        y_pred.append(y_pred_svm)

    with open('neur_nlp.pkl', 'rb') as f:
        model_neur = pickle.load(f)
        y_pred_neur = model_neur.predict(X_test)
        y_pred.append(y_pred_neur)
    
    with open('kneighb_nlp.pkl', 'rb') as f:
        model_kneighb = pickle.load(f)
        y_pred_kneighb = model_kneighb.predict(X_test)
        y_pred.append(y_pred_kneighb)

    with open('decision_nlp.pkl', 'rb') as f:
        model_decision = pickle.load(f)
        y_pred_decision = model_decision.predict(X_test)
        y_pred.append(y_pred_decision)
        
    return sum(y_pred) / len(y_pred)