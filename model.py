import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Carregar o dataset Iris para obter nomes das classes
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]

# Carregar os modelos
model_rf = joblib.load("modelo_rf.pkl")
model_knn = joblib.load("modelo_knn.pkl")
model_svm = joblib.load("modelo_svm.pkl")

# Mapear os modelos para os nomes
modelos = {
    "Random Forest": model_rf,
    "K-Nearest Neighbors": model_knn,
    "Support Vector Machine": model_svm
}

# Título e descrição do app
st.title("Classificação de Espécies de Iris")
st.write("Escolha um modelo de classificação, visualize dados e faça previsões individuais ou em lote.")

# Opção para selecionar o modelo
modelo_selecionado = st.selectbox("Escolha o modelo", list(modelos.keys()))

# Sidebar para entrada de dados
sepal_length = st.sidebar.slider("Comprimento da Sépala (cm)", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()))
sepal_width = st.sidebar.slider("Largura da Sépala (cm)", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()))
petal_length = st.sidebar.slider("Comprimento da Pétala (cm)", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()))
petal_width = st.sidebar.slider("Largura da Pétala (cm)", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()))

# Botão para realizar a classificação
if st.button("Classificar uma amostra"):
    # Criar DataFrame com os valores inseridos
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    df = pd.DataFrame(data, columns=iris.feature_names)
    
    # Selecionar o modelo escolhido
    model = modelos[modelo_selecionado]
    
    # Fazer a previsão
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    # Exibir a espécie prevista
    st.write(f"**Modelo Selecionado**: {modelo_selecionado}")
    st.write(f"**Espécie Prevista**: {iris.target_names[prediction[0]]}")
    
    # Exibir as probabilidades de cada classe
    st.write("**Probabilidades:**")
    for i, species in enumerate(iris.target_names):
        st.write(f"{species}: {prediction_proba[0][i] * 100:.2f}%")

# Upload de arquivo CSV para classificação em lote
st.subheader("Classificação em Lote com Arquivo CSV")
uploaded_file = st.file_uploader("Faça upload de um arquivo CSV com as características das flores", type="csv")

if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)
    st.write("Amostras carregadas:")
    st.write(df_csv.head())
    # Fazer a previsão para cada linha do arquivo
    predictions_csv = modelos[modelo_selecionado].predict(df_csv)
    df_csv['Previsao'] = [iris.target_names[i] for i in predictions_csv]
    st.write("Resultados das previsões:")
    st.write(df_csv)

# Seção de visualização dos dados
st.subheader("Visualização dos Dados")

# Escolher o tipo de gráfico
plot_tipo = st.selectbox("Escolha o tipo de plot", ["Scatter Plot", "Gráfico de Barras"])

if plot_tipo == "Scatter Plot":
    # Scatter plot entre comprimento e largura das pétalas, colorido por espécie
    st.write("Scatter Plot: Comprimento vs. Largura da Pétala")
    fig, ax = plt.subplots()
    sns.scatterplot(data=iris_df, x="petal length (cm)", y="petal width (cm)", hue="species", ax=ax)
    st.pyplot(fig)

elif plot_tipo == "Gráfico de Barras":
    # Gráfico de barras da quantidade de cada espécie
    st.write("Gráfico de Barras: Quantidade de cada Espécie")
    fig, ax = plt.subplots()
    iris_df['species'].value_counts().plot(kind="bar", color=['blue', 'green', 'red'], ax=ax)
    ax.set_ylabel("Quantidade")
    ax.set_title("Distribuição das Espécies")
    st.pyplot(fig)

