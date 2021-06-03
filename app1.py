import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
# matplotlib.use('Agg')


# setting title
st.title('Simple streamlit app')

# # putting picture
image = Image.open('octavian.jpg')
st.image(image, use_column_width=True)



            # data = st.file_uploader('Upload dataset:', type=['csv', 'xlsx', 'txt', 'json'])
            # st.success('Data uploades successfully')

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    #
    x = data.data
    y = data.target
    return x, y

def main():

    data = ['Breast Cancer', "Iris", 'Wine']
    option1 = st.sidebar.selectbox('Select dataset:', data)

    activities = ['Background', 'EDA', "Visualisation", 'Model']
    option2 = st.sidebar.selectbox('Select option:', activities)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    if option1 == 'Breast Cancer':
        x, y = get_dataset(option1)
    elif option1 == 'Wine':
        x, y = get_dataset(option1)
    else:
        x, y = get_dataset(option1)


# dealing with EDA
    if option2 == 'EDA':
        st.subheader('Exploratory Data Analysis')
        st.dataframe(x)

        if st.checkbox('Display shape'):
            st.write(x.shape)

        if st.checkbox('Display null values'):
            df = pd.DataFrame(x)
            st.write(df.isnull().sum())
        # if st.checkbox('Display columns'):
        #     df = pd.DataFrame(x)
        #     st.write(df.columns)
        # if st.checkbox('Select multiple columns'):
        #     df = pd.DataFrame(x)
        #     selected_columns = st.multiselect('Select preferred columns:', df.columns)
        #     df1 = x[selected_columns]
        #     st.dataframe(df1)
        if st.checkbox('Display summary'):
            df = pd.DataFrame(x)
            st.write(df.describe())

        if st.checkbox('Display correlation'):
            df = pd.DataFrame(x)
            st.write(df.corr())

# dealing with visualisation part
    elif option2 == 'Visualisation':
        st.subheader('Data Visualisation')

        if st.checkbox('Pairplot'):
            df = pd.DataFrame(x)
            st.write(sns.pairplot(df, diag_kind='kde'))
            st.pyplot()

        if st.checkbox('Heatmap'):
            df = pd.DataFrame(x)
            st.write(sns.heatmap(df.corr(), vmax=1, square=True, annot=True, cmap='viridis'))
            st.pyplot()

# building various models
    elif option2 == 'Model':
        seed = st.sidebar.slider('Seed', 1, 200)
        classifier_name = st.sidebar.selectbox('Select yout model:',('SVM', 'KNN', 'Naive Bayes', 'Random Forest'))

        def add_parameter(name_of_clf):
            params = dict()
            if name_of_clf == 'SVM':
                c = st.sidebar.slider('C', 0.01, 15.0)
                params['C'] = c
            elif name_of_clf == 'KNN':
                k = st.sidebar.slider('K', 1, 15)
                params['K'] = k
            else:
                name_of_clf == 'Random Forest'
                t = st.sidebar.slider('T', 5,100)
                params['T'] = t
            return params

        # calling the function
        params = add_parameter(classifier_name)


        # accessing the classifer
        def get_classifer(name_of_clf, params):
            clf = None
            if name_of_clf == 'SVM':
                clf = SVC(C = params['C'])
            elif name_of_clf == 'KNN':
                clf = KNeighborsClassifier(n_neighbors = params['K'])
            elif name_of_clf == 'Naive Bayes':
                clf = GaussianNB()
            elif name_of_clf == 'Random Forest':
                clf = RandomForestClassifier(n_estimators=params['T'])
            else:
                st.warning('Select your choice of algorithm')
            return clf

        clf = get_classifer(classifier_name, params)


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

        clf.fit(x_train, y_train)

        pred = clf.predict(x_test)

        st.write(pred)

        accuracy = accuracy_score(y_test, pred)

        st.write('classifier: ', classifier_name)
        st.write('Accuracy for your model is: ', accuracy)

# about dataset
    elif option2 == 'Background':
        st.write('This is an interactive webpage for ML projects, feel free to use it. Datasets are already loaded from the open source library. The analysis done here are to demonstrate how I ca present my work to stakeholders in an interactive way by building a webapp for the machine learning algorithm.')

if __name__ == '__main__':
    main()
