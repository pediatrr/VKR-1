import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st
from scipy import stats
from scipy.stats import chi2_contingency
from menu import menu_with_redirect
# Название страницы
st.set_page_config(page_title="Статистические тесты", layout="wide")

# Перенаправить на app.py если вы не вошли в систему, в противном случае отобразить навигационное меню
menu_with_redirect()

# Загрузить датасет
st.cache_data
df = pd.read_csv(r'C:\Users\krest\OneDrive\Рабочий стол\Новая папка\diabetes_prediction_dataset.csv')

# Приведение названий столбцов к единому виду с помощью функции
def preprocess_column_names(df):
    def clean_name(name):
        cleaned_name = name.strip()
        if cleaned_name[0].islower():
            cleaned_name = cleaned_name.capitalize()
        return cleaned_name
    df.columns = df.columns.map(clean_name)
    return df
preprocess_column_names(df)
# Разделение столбцов на числовые и категориальные(меньше5уникальных значений)
num_col=[]
cat_col=[]
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        if(df[col].nunique()<4):
            cat_col.append(col)
        else:
            num_col.append(col)
    else:
        cat_col.append(col) 

# Создание преобразователя столбцов
transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_col),
        ("num", "passthrough", num_col)
    ]
)
# Преобразование данных
encoded_data = transformer.fit_transform(df)
encoded_df = pd.DataFrame(encoded_data, columns=transformer.get_feature_names_out())        
        
# Функция для выполнения корреляционных тестов Пирсона и Спирмена, отображения результатов и визуализации
def run_correlation_tests(df, test_x_var, test_y_var):
    # Вычисление корреляций и p-value
    pearson_coef, pearson_p = stats.pearsonr(df[test_x_var], df[test_y_var])
    spearman_coef, spearman_p = stats.spearmanr(df[test_x_var], df[test_y_var])
    # Отображение результатов в виде таблицы
    results_df = pd.DataFrame({
        "Тест": ["Пирсон", "Спирмен"],
        "Коэффициент корреляции": [pearson_coef, spearman_coef],
        "P-value": [pearson_p, spearman_p]
    })
    st.dataframe(results_df.style.format({"Коэффициент корреляции": "{:.4f}", "P-value": "{:.4f}"}))
    # Интерпретация результатов 
    st.write("_Интерпретация_")
    if pearson_p < 0.05:  # Проверяем значимость только теста Пирсона
        if pearson_coef > 0.5:
            st.write(f"Существует сильная положительная линейная корреляция между {test_x_var} и {test_y_var}.")
        elif pearson_coef < -0.5:
            st.write(f"Существует сильная отрицательная линейная корреляция между {test_x_var} и {test_y_var}.")
        else:
            st.write(f"Существует слабая или отсутствует линейная корреляция между {test_x_var} и {test_y_var}.")
    else:
        st.write(f"Линейная корреляция между {test_x_var} и {test_y_var} статистически не значима (p-value > 0.05).")

# Визуализация корреляции
def visualize_correlations(df, test_x_var, test_y_var):    
    # Вычисление корреляций и p-value (для визуализации)
    pearson_coef, pearson_p = stats.pearsonr(df[test_x_var], df[test_y_var])
    spearman_coef, spearman_p = stats.spearmanr(df[test_x_var], df[test_y_var])
    test_names = ["Пирсон", "Спирмен"]
    coefs = [pearson_coef, spearman_coef]
    p_values = [pearson_p, spearman_p]
    # График рассеяния
    fig_scatter = px.scatter(df, x=test_x_var, y=test_y_var, 
                              title=f"Корреляция между {test_x_var} и {test_y_var}",
                              trendline="ols"  # Добавляем линию тренда с использованием метода наименьших квадратов (OLS)
                              )
    st.plotly_chart(fig_scatter)
    # Столбчатая диаграмма коэффициентов корреляции
    fig_bar = px.bar(x=test_names, y=coefs, color=test_names, 
                      title="Сравнение тестов",
                      labels={'x': 'Тест', 'y': 'Коэффициент корреляции'})
    fig_bar.update_traces(text=p_values, textposition='outside', texttemplate='p = %{text:.4f}')
    st.plotly_chart(fig_bar)    

# Выполняет хи-квадрат тест для двух категориальных переменных, отображает результаты и создает визуализации    
def perform_chi_square_test(df, var1, var2):
    # Создание таблицы сопряженности
    contingency_table = pd.crosstab(df[var1], df[var2])
    st.subheader("таблица сопряженности", help="отображает частоты наблюдений для каждой комбинации категорий двух переменных")
    st.dataframe(contingency_table)
    # Выполнение хи-квадрат теста
    chi2_contingency(contingency_table)
    # Визуализация
    fig_mosaic = px.histogram(
        df, 
        x=var2, 
        color=var1, 
        barmode='group',
        labels={var2: var2, 'count': "Частота"}
    )
    st.plotly_chart(fig_mosaic)
    
st.toast('материалы на этом сайте защищены законом об авторском праве')

tab1, tab2= st.tabs(["тест Пирсона и тест Спирмена", "Хи-квадрат тест"])  

with tab1:
    # Выбор переменных для тестов
    col1, col2 = st.columns(2)  
    with col1:
        test_x_var = st.selectbox("Выберите числовые признаки для оценки корреляции:", num_col, key="test_x")
        test_y_var = st.selectbox("вторая переменная", num_col, key="test_y", label_visibility="hidden")
        if test_x_var and test_y_var:
            run_correlation_tests(df, test_x_var, test_y_var) 
        st.write("""
        **Тест Пирсона:** 
        Измеряет **линейную** зависимость между двумя переменными. 
        Коэффициент корреляции Пирсона принимает значения от -1 до +1, 
        где +1 указывает на идеальную положительную линейную корреляцию, 
        -1 указывает на идеальную отрицательную линейную корреляцию, 
        а 0 указывает на отсутствие линейной корреляции.
        """)
        st.write("""
        **Тест Спирмена:** 
        Измеряет **монотонную** зависимость между двумя переменными. 
        Он основан на рангах значений, а не на самих значениях, 
        поэтому он менее чувствителен к выбросам и может обнаруживать 
        нелинейные зависимости, если они монотонны (т.е. всегда возрастают 
        или всегда убывают).
        """)    
    with col2:    
        if test_x_var and test_y_var:   
            visualize_correlations(df, test_x_var, test_y_var)      
            
with tab2:
    st.write("_статистический тест, используемый для определения значимой связи между двумя категориальными переменными_")
    col1, col2 = st.columns(2)
    with col1:
        # Выбор переменных для теста
        selected_var1 = st.selectbox("Выберите переменные:", cat_col, key="chi2_var1")
        selected_var2 = st.selectbox("вторая категориальная переменная ", cat_col, key="chi2_var2", label_visibility="hidden")
        # Выполнение теста, если выбраны обе переменные
        if selected_var1 and selected_var2:
            perform_chi_square_test(df, selected_var1, selected_var2)    
    with col2:
        st.write("""
                 **Интерпретация таблицы сопряженности:**
        Таблица сопряженности показывает количество наблюдений для каждой комбинации 
        категорий в двух выбранных переменных. Например, если переменные - это 
        "Курение" (Да/Нет) и "Диабет" (Да/Нет), то таблица может показать, сколько 
        человек курят и имеют диабет, сколько курят и не имеют диабета, и так далее.
        """)
        st.write("""        
                  **Как работает хи-квадрат тест**
        Хи-квадрат тест используется для проверки гипотезы о том, что *нет связи* 
        между двумя категориальными переменными. 
        
        * **Нулевая гипотеза:** Тест предполагает, что нет связи между переменными 
          (т.е. курение и диабет независимы друг от друга).
        * **Ожидаемые частоты:** Тест вычисляет ожидаемые частоты для каждой ячейки 
          таблицы, предполагая, что нулевая гипотеза верна.
        * **Хи-квадрат статистика:** Тест сравнивает наблюдаемые частоты (из таблицы 
          сопряженности) с ожидаемыми частотами. Чем больше разница, тем больше 
          значение хи-квадрат статистики.
        * **P-значение:** Тест вычисляет p-значение, которое представляет вероятность 
          получить наблюдаемые результаты, если нулевая гипотеза верна.
        """)
        st.write("""
                    **Интерпретация p-значения:**
        * **Малое p-значение (обычно < 0.05):** Отвергаем нулевую гипотезу и делаем 
          вывод, что существует статистически значимая связь между переменными.
        * **Большое p-значение (>= 0.05):** Не отвергаем нулевую гипотезу и делаем 
          вывод, что нет достаточных доказательств связи между переменными.
        """)        

    
            
