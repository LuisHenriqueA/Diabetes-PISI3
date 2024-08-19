import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from enum import Enum

class ClassifierType(Enum):
    KNN = 'K-Nearest Neighbors', lambda: KNeighborsClassifier(n_neighbors=3)
    NAIVE_BAYES = 'Naive Bayes', lambda: GaussianNB()
    RANDOM_FOREST = 'Random Forest', lambda: RandomForestClassifier(random_state=42)

    def __init__(self, description: str, builder: callable):
        self.description = description
        self.builder = builder

    @classmethod
    def values(cls):
        return [x.description for x in ClassifierType]

    @classmethod
    def get(cls, description):
        result = [x for x in ClassifierType if x.description == description]
        return None if len(result) == 0 else result[0]

    def build(self, df: pd.DataFrame, target: str, seed=42, test_size=0.33):
        return Classifier(self.description, self.builder, df, target, seed, test_size)

    def __str__(self) -> str:
        return self.description
class Classifier:
    def __init__(self, description: str, builder: callable, df: pd.DataFrame, target: str, seed=42, test_size=0.33):
        self.description = description
        self.seed = seed
        df, X, y = self.__preprocess(df, target)
        self.df = df
        lbl_enc = LabelEncoder()
        y = lbl_enc.fit_transform(y)
        self.target_classes = lbl_enc.classes_
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        self.sklearn_classifier = builder()
        self.sklearn_classifier.fit(self.X_train, self.y_train)

    def __preprocess(self, df: pd.DataFrame, target: str):
        df = df.copy().dropna()
        return df, df[df.columns.difference([target])], df[target]

    def classify(self):
        y_train_pred = self.sklearn_classifier.predict(self.X_train)
        y_test_pred = self.sklearn_classifier.predict(self.X_test)
        self.__report('TREINO', 'Treino', self.y_train, y_train_pred)
        self.__report('TESTE', 'Teste', self.y_test, y_test_pred)

    def __report(self, title, desc, y_true, y_pred):
        print(f"Relatório de {title}:")
        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy, support, df_report = self.__build_report_df(report)
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Suporte: {support}")
        print(pd.DataFrame(df_report))
        self.__build_confusion_matrix(y_true, y_pred)

    def __build_report_df(self, report):
        df_dict = {
            'classe': [],
            'precision': [],
            'recall': [],
            'f1-score': [],
            'support': [],
        }
        accuracy = 0
        support = 0
        for k, v in report.items():
            if k == 'accuracy':
                accuracy = v
            else:
                df_dict['classe'].append(k)
                df_dict['precision'].append(v['precision'])
                df_dict['recall'].append(v['recall'])
                df_dict['f1-score'].append(v['f1-score'])
                s = int(v['support'])
                df_dict['support'].append(s)
                support = s
        df_report = pd.DataFrame(data=df_dict)
        return accuracy, support, df_report

    def __build_confusion_matrix(self, y_true, y_pred):
        matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Valores Previstos')
        plt.ylabel('Valores Reais')
        plt.title(f'Matriz de Confusão - {self.description}')
        plt.show()

    # def score(self, classificador, X_train, X_test, y_train, y_test):
    #     return classificador.score(X_train, y_train), classificador.score(X_test, y_test)

    # def accuracy(self, y_train, y_train_pred, y_test, y_test_pred):
    #     return accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)

    def __str__(self) -> str:
        return self.description
