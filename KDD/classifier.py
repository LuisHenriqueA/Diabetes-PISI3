import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from imblearn.under_sampling import RandomUnderSampler

class ClassifierType(Enum):
    NAIVE_BAYES = 'Naive Bayes', lambda: GaussianNB()
    KNN = 'K-Nearest Neighbors', lambda: KNeighborsClassifier(n_neighbors=3)
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

    def build(self, df: pd.DataFrame, target: str, balance= False, seed=42, test_size=0.33):
        return Classifier(self.description, self.builder, df, target, seed, test_size, balance)

    def __str__(self) -> str:
        return self.description

class Classifier:
    def __init__(self, description: str, builder: callable, df: pd.DataFrame, target: str, seed=42, test_size=0.33, balance: bool = False):
        self.description = description
        self.seed = seed
        self.balance = balance
        df, X, y = self.__preprocess(df, target)
        self.df = df
        lbl_enc = LabelEncoder()
        y = lbl_enc.fit_transform(y)
        self.target_classes = lbl_enc.classes_
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        
        if self.balance:
            self.__balance_data()

        self.sklearn_classifier = builder()
        self.sklearn_classifier.fit(self.X_train, self.y_train)

    def __balance_data(self):
        sampler = RandomUnderSampler(random_state=self.seed)
        self.X_train, self.y_train = sampler.fit_resample(self.X_train, self.y_train)

    def __preprocess(self, df: pd.DataFrame, target: str):
        df = df.copy().dropna()
        return df, df[df.columns.difference([target])], df[target]

    def classify(self):
        y_train_pred = self.sklearn_classifier.predict(self.X_train)
        y_test_pred = self.sklearn_classifier.predict(self.X_test)
        st.write(f'<h2>{self.description}</h2>', unsafe_allow_html=True)
        c1, _, c2 = st.columns([.49, .02, .49])
        with c1:
            self.__report('TREINO', '<div style="color: red; font-size: 1.5em"></div>', self.y_train, y_train_pred)
        with c2:
            self.__report('TESTE', '<div style="font-family: cursive; font-size: 1.1em"></div>', self.y_test, y_test_pred)
        st.write('')
        #with st.expander('Dados Brutos'):
        #   st.dataframe(self.df)
        #with st.expander('Dados Processados'):
        #    c1, _, c2 = st.columns([.49, .02, .49])
        #    c1.write('<h3>Treino</h3>', unsafe_allow_html=True)
        #    c1.dataframe(self.X_train)
        #    c2.write('<h3>Teste</h3>', unsafe_allow_html=True)
        #    c2.dataframe(self.X_test)

    def __report(self, title, desc, y_true, y_pred):
        st.write(f'<h3>{title}</h3>', unsafe_allow_html=True)
        # self.__build_confusion_matrix(y_true, y_pred)
        self.__build_results(desc, y_true, y_pred)

    def __build_results(self, desc, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy, support, df_report = self.__build_report_df(report)
        st.write(f'''
                    <b>Accuracy: {accuracy:.4%}</b><br/>
                    Suppport: {support:.0f}
                    ''', unsafe_allow_html=True)
        st.write('')
        df_report = df_report.style.format({'precision': '{:.2%}', 'recall': '{:.2%}', 'f1-score': '{:.2%}', 'support': '{:.0f}'})
        st.dataframe(df_report)
        st.write(desc, unsafe_allow_html=True)

    # def __build_confusion_matrix(self, y_true, y_pred):
    #     matrix = confusion_matrix(y_true, y_pred)
    #     fig, ax = plt.subplots()
    #     sns.heatmap(matrix, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    #     ax.set_xlabel('Valores Previstos')
    #     ax.set_ylabel('Valores Reais')
    #     ax.set_title(f'Matriz de Confusão - {self.description}')
    #     st.write('Matriz de Confusão')
    #     st.pyplot(fig)

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

    def __str__(self) -> str:
        return self.description