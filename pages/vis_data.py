import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st   
from menu import menu_with_redirect
# Название страницы
st.set_page_config(layout="wide")
# Перенаправить на app.py если вы не вошли в систему, в противном случае отобразить навигационное меню
menu_with_redirect()

# Загрузить датасет
st.cache_data
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Приведение названий столбцов к единому виду с помощью функции
def preprocess_column_names(df):
    def clean_name(name):
        # Удаление пробелов в начале и конце названий
        cleaned_name = name.strip()
        # Сделать первую букву заглавной, если она в нижнем регистре
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

# Функция для создания матрицы корреляции 
def create_correlation_heatmap(df):
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, aspect="auto")
    fig_corr.update_xaxes(side="top")
    fig_corr.update_layout(
        width=1200,  
        height=600  
    )
    return fig_corr

# Функция для создания точечной диаграммы 2д
def create_scatter_plot(df, x_col, y_col):
    fig_scatter = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color="Diabetes", # Раскрашиваем точки по переменной 'Diabetes'
        width=1000,  
        height=600, 
        labels={x_col: x_col, y_col: y_col}, # Настройка меток осей
    )
    fig_scatter.update_traces(marker=dict(size=15))  # Увеличиваем размер точек
    return fig_scatter

# Создайте трехмерную точечную диаграмму
def create_3d_bubble_chart(df, x_col, y_col, z_col, size_col, color_col):
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df[z_col],
            mode='markers',
            marker=dict(
                size=df[size_col],
                color=df[color_col],
                colorbar=dict(title=color_col),
                sizemode='diameter'  
            )
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        return fig

st.toast('материалы на этом сайте защищены законом об авторском праве')

tab1, tab2, tab3, tab4 = st.tabs(["Описание признаков", "Тепловая карта корреляции", "Графики рассеяния 2D", "Графики рассеяния 3D"])  

with tab1:
    st.title("***Что влияет на риск диабета?***")
    st.caption("визуализация в дальнейших вкладках")
    with st.expander("Возраст"):
        st.write("""
            Возраст является важным фактором для прогнозирования риска развития диабета. 
            С возрастом риск развития диабета у людей возрастает. Отчасти это связано 
            с такими факторами, как снижение физической активности, изменение уровня 
            гормонов и более высокая вероятность развития других заболеваний, 
            которые могут способствовать развитию диабета.
            """)
    with st.expander("Пол"):
        st.write("""
            Пол может влиять на риск развития диабета, хотя эффект может быть разным. 
            Например, женщины с гестационным диабетом в анамнезе (диабет во время 
            беременности) имеют более высокий риск развития диабета 2 типа в дальнейшей жизни. 
            Кроме того, некоторые исследования показали, что мужчины могут иметь несколько 
            более высокий риск развития диабета по сравнению с женщинами.
            """)    
    with st.expander("Индекс массы тела"):
        st.write("""
            ИМТ - это показатель жировых отложений, основанный на росте и весе человека. 
            Он обычно используется в качестве показателя общего веса и может быть 
            полезен для прогнозирования риска развития диабета. Более высокий ИМТ 
            связан с большей вероятностью развития диабета 2 типа. Избыток жира в 
            организме, особенно в области талии, может привести к резистентности 
            к инсулину и ухудшить способность организма регулировать уровень 
            сахара в крови.
            """)    
    with st.expander("Гипертония"):
        st.write("""
            Гипертония, или высокое кровяное давление,- это заболевание, которое часто сочетается  
            с сахарным диабетом. Эти два заболевания имеют общие факторы риска и могут способствовать 
            развитию друг друга. Наличие гипертонии увеличивает риск развития диабета 2 типа и наоборот. 
            Оба состояния могут оказывать пагубное воздействие на сердечно-сосудистую систему.  
            """)    
    with st.expander("Болезни сердца"):
        st.write("""
            Болезни сердца, включая такие состояния, как ишемическая болезнь сердца 
            и сердечная недостаточность, связаны с повышенным риском развития диабета. 
            Взаимосвязь между болезнями сердца и диабетом является двунаправленной, 
            что означает, что наличие одного заболевания увеличивает риск развития 
            другого. Это объясняется тем, что у них много общих факторов риска, 
            таких как ожирение, высокое кровяное давление и высокий уровень холестерина. 
        """)
    with st.expander("История курения"):
        st.write("""
            Курение является модифицируемым фактором риска развития диабета. Было 
            установлено, что курение сигарет повышает риск развития диабета 2 типа. 
            Курение может способствовать развитию резистентности к инсулину и 
            нарушать метаболизм глюкозы. Отказ от курения может значительно 
            снизить риск развития диабета и его осложнений.
        """)
    with st.expander("Уровень HbA1c"):
        st.write("""
            HbA1c (гликированный гемоглобин) - это показатель среднего уровня глюкозы 
            в крови за последние 2-3 месяца. Он предоставляет информацию о 
            долгосрочном контроле уровня сахара в крови. Более высокие уровни HbA1c 
            указывают на ухудшение контроля гликемии и связаны с повышенным риском 
            развития сахарного диабета и его осложнений.
        """)
    with st.expander("Уровень глюкозы в крови"):
        st.write("""
            Уровень глюкозы в крови отражает количество глюкозы (сахара), присутствующей 
            в крови в данный момент времени. Повышенный уровень глюкозы в крови, 
            особенно натощак или после употребления углеводов, может указывать 
            на нарушение регуляции уровня глюкозы и повышать риск развития диабета. 
            Регулярный контроль уровня глюкозы в крови важен для диагностики 
            и лечения сахарного диабета.
        """)
        
with tab2:
    # Матрица корреляции 
    st.header("Тепловая карта корреляции", help="статистическая взаимосвязь двух или более случайных величин")
    st.info("_Помните, корреляция не означает причинно-следственную связь._")
    # Создание анимации
    fig_heat = create_correlation_heatmap(encoded_df)
    st.plotly_chart(fig_heat)
    
with tab3:
    # Графики рассеяния
    st.subheader("График рассеяния")
    selected_x_var = st.selectbox("Выберите переменную для оси X:", df.columns, key="x_var")
    selected_y_var = st.selectbox("Выберите переменную для оси Y:", df.columns, key="y_var")
    scatter_fig = create_scatter_plot(df, selected_x_var, selected_y_var)
    st.plotly_chart(scatter_fig)
    
with tab4:        
    st.header("3D-график рассеяния")
    st.caption("размер пузырьков отражает возраст")
    # Создаем два столбца
    col1, col2 = st.columns(2)
    with col1:
        # определение для осей X, Y и Z.
        selected_x1_var = st.selectbox("X ось:", df.columns, key="x1_var", index=0)
        selected_y1_var = st.selectbox("Y ось:", df.columns, key="y1_var", index=1)
        selected_z1_var = st.selectbox("Z ось:", df.columns, key="z1_var", index=2)
        # Выпадающий список для выбора цвета
        color_options = df.columns.tolist()
        selected_color_var = st.selectbox("Цвет точек:", color_options, key="color_var", index=color_options.index("Diabetes"))
    with col2:    
        # Создайте трехмерную точечную диаграмму
        bubble_chart_fig = create_3d_bubble_chart(df, selected_x1_var, selected_y1_var, selected_z1_var, "Age", selected_color_var)
        st.plotly_chart(bubble_chart_fig, use_container_width=True)