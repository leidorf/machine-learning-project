import streamlit as st
import pandas as pd 
import numpy as np
import joblib
from datetime import datetime, timezone
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    st.sidebar.title('Streamlit ile ML Uygulaması')
    selected_page = st.sidebar.selectbox('Sayfa Seçiniz',["Ana Sayfa","Tahmin Yap","İstatislik Görüntüle"])

    if selected_page == "Ana Sayfa":
        st.title('Deprem Büyüklüğü Tahmini 🌍🔮')
        st.markdown(
            """
            Bu proje makine öğrenmesi uygulamalarının web ortamında streamlit
            kullanılarak yayınlanmasına örnek olarak geliştirilmiştir. Kandilli Rasathanesi (KRDAE)'nden 756 adet deprem verileri çekilmiş
            ve incelenmiştir. Bu veriler kullanılarak makine öğrenmesi modelleri eğitilmiş ve projeye dahil edilmiştir.
            
            """)
        st.info("Tahmin yapmak ve istatistlikleri görüntülemek için sol tarafta bulunan menüyü kullanınız.")

        st.subheader('Geliştirici Bilgileri')
        st.markdown(
            """
            [Github](https://github.com/leidorf)

            [Linkedin](https://www.linkedin.com/in/guraydag/)
            """
        )

    if selected_page == "Tahmin Yap":
        predict()
    
    if selected_page == "İstatislik Görüntüle":
        eda() 


def eda():
    st.title('İstatistlikler')

    data = pd.read_csv('../data/birlestirilmis_veriler.csv')

    st.header("Bütün Veriler")
    st.dataframe(data)

    plt.figure(figsize=(16,16))
    plt.subplot(2,1,1)
    sns.countplot(x='ML',data = data,order = data['ML'].value_counts().index)
    plt.xticks(rotation = 90)
    plt.xlabel("Deprem Büyüklüğü")
    plt.ylabel("Deprem Sayısı")
    st.header("Deprem Büyüklüğüne Göre Yerlerin Sıralaması")
    st.pyplot(fig=plt)

    plt.figure(figsize=(16,16))
    plt.subplot(2,1,1)
    sns.countplot(x='Yer',data = data,order = data['Yer'].value_counts().index)
    plt.xlabel("Yer")
    plt.ylabel("Deprem Sayısı")
    plt.xticks(rotation = 90)
    

    st.header("Deprem Sayısına Göre Yerler")
    st.pyplot(fig=plt)
    

def predict():

    # Markalar ve Modellerin yüklenmesi
    cities = load_data_cities()

    # Kullanıcı arayüzü ve değer alma
    st.title('Merhaba, *Streamlit!* 👨‍💻')
    selected_city = city_index(cities, st.selectbox('Şehir seçiniz', cities))
    selected_depth = st.number_input("Derinlik Giriniz (km):", min_value=0, max_value=100)

    # Şuanki tarih ve saat bilgisini al
    current_datetime = datetime.now(timezone.utc)
    timestamp = int(current_datetime.timestamp())
    st.write("Şuanki Tarih ve Saat:", current_datetime)

    selected_model = st.selectbox('Tahmin Modeli Seçiniz',["Linear Regression","Decision Tree","Elastic Net Regression","Polynomial Regression","Random Forest","Support Vector Machine","KNN","CART","Multiple Linear Regression","PCR Regression"])

    prediction_value = create_prediction_value(selected_depth,selected_city,timestamp)
    prediction_model = load_models(selected_model)
    

    if st.button("Tahmin Yap"):
            result = predict_models(prediction_model,prediction_value)
            if result != None:
                st.success('Tahmin Başarılı')
                st.balloons()
                result_float = float(result)
                st.write("Tahmin Edilen Deprem Büyüklüğü: {:.2f}ML".format(result_float))
            else:
                st.error('Tahmin yaparken hata meydana geldi!')

#Yer verisi yükleme
def load_data_cities():
    cities = pd.read_csv("../data/yer.csv")
    return cities

def city_index(cities,city):
    index = int(cities[cities["Yer"]==city].index.values)
    return index

#Model yükleme
def load_models(modelName):
    if modelName == "Linear Regression":
        dt_model = joblib.load("../1-linear-regression/linear_regression_model.pkl")
        return dt_model

    elif modelName == "Decision Tree":
        dt_model = joblib.load("../2-decision-tree/decision_tree_model.pkl")
        return dt_model

    elif modelName == "Elastic Net Regression":
        linear_model = joblib.load("../3-elastic-net-regression/elasticnet_regression_model.pkl")
        return linear_model

    elif modelName == "Polynomial Regression":
        dt_model = joblib.load("../4-polynomial-regression/polynomial_regression_model.pkl")
        return dt_model

    elif modelName == "Random Forest":
        dt_model = joblib.load("../5-random-forest/random_forest_model.pkl")
        return dt_model

    elif modelName == "Support Vector Machine":  
        rf_model = joblib.load("../6-support-vector-machine/svm_model.pkl")
        return rf_model

    elif modelName == "KNN":  
        rf_model = joblib.load("../7-KNN/knn_model.pkl")
        return rf_model

    elif modelName == "CART":  
        rf_model = joblib.load("../8-CART/cart_model.pkl")
        return rf_model

    elif modelName == "Multiple Linear Regression":  
        rf_model = joblib.load("../9-multiple-linear-regression/linear_regression_model.pkl")
        return rf_model

    elif modelName == "PCR Regression":  
        rf_model = joblib.load("../10-PCR-regression/pcr_regression_model.pkl")
        return rf_model    

    else:
        st.write("Model yüklenirken hata meydana geldi..!")
        return 0

def create_prediction_value(derinlik, yer, timestamp):
    res = pd.DataFrame(data={'Derinlik': [derinlik], 'Yer': [yer], 'Timestamp': [timestamp]})
    return res[['Derinlik', 'Yer', 'Timestamp']] 
    
def predict_models(model,res):
    result = str(float(model.predict(res))).strip('[]')
    return result

if __name__ == "__main__":
    main()