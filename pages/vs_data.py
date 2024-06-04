import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st   
from plotly.subplots import make_subplots
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
           
# Функция для отображения дублирующихся строк, если они есть.
def show_duplicates():
    with st.expander("Проверить наличие повторяющихся строк"):
        duplicate_rows = df[df.duplicated()]
        num_duplicates = len(duplicate_rows)
        if num_duplicates > 0:
            st.write(f"найдено _:red[{num_duplicates}]_ дубликатов")
            st.dataframe(duplicate_rows)
        else:
            st.write("Дубликатов нет.")
# Функция для отображения описательной статистики
def show_data_info():
    with st.expander("Показать статистическую информацию о датасете"):
        st.write("по числовым столбцам")
        st.dataframe(df.describe(include='number').T)
        st.write("по категориальным столбцам")
        st.dataframe(df.describe(include='object'))     
# Функция для отображения первых строк, количества строк и столбцов        
def show_basic_data_info(df):
    with st.expander("Основные сведения о датасете"):
        st.dataframe(df.head())
        row_count, col_count = df.shape
        st.write(f"всего строк _:red[{row_count}]_, всего столбцов _:red[{col_count}]_")    
        
# Функция для отображения точечной взаимосвязи        
def display_splom(df):
    # Создание SPLOM с помощью Plotly
    fig = go.Figure(data=go.Splom(
        dimensions=[
                dict(label='Age', values=df['Age']),
                dict(label='Gender', values=df['Gender']),
                dict(label='BMI', values=df['Bmi']),
                dict(label='Hypertension', values=df['Hypertension'].astype('category').cat.codes),
                dict(label='HeartDisease', values=df['Heart_disease'].astype('category').cat.codes),
                dict(label='SmokingHistory', values=df['Smoking_history']),
                dict(label='HbA1c', values=df['HbA1c_level']),
                dict(label='BloodGlucose', values=df['Blood_glucose_level'])
            ],
        showupperhalf=False, 
        text=df['Diabetes'], 
        marker=dict(
            color=df['Diabetes'].astype('category').cat.codes,
            size=5,
            colorscale='Bluered',
            showscale=True, 
            line_color='rgb(230,230,230)', 
            line_width=0.5,
        )
    ))
    fig.update_layout(
            title='Взаимосвязь факторов риска диабета',
            width=1200,
            height=900,
            dragmode='select'
        )
    st.plotly_chart(fig)
               
                                  
# Функции для создания графиков       
# Скрипичная диаграмма        
def create_violin_plot(df, num_col, cat_col):
    try:
        fig_violin = px.violin(df, x=num_col, y=cat_col, color=cat_col,)
        fig_violin.update_traces(meanline_visible=True)
        return fig_violin
    except ValueError as e:
        st.error(f"Ошибка при создании скрипичной диаграммы: {e}")
        return None
# тепловая карта распределений
def create_heatmap(df, num_col, cat_col):
    try:
        fig_heatmap = px.density_heatmap(df, x=num_col, y=cat_col, marginal_x="box")
        return fig_heatmap
    except ValueError as e:
        st.error(f"Ошибка при создании тепловой карты: {e}")
        return None
# гистограмма распределений
def create_histogram(df, num_col, cat_col):
    try:
        fig_hist = px.histogram(df, x=num_col, color=cat_col, marginal="rug")
        return fig_hist
    except ValueError as e:
        st.error(f"Ошибка при создании гистограммы: {e}")
        return None
# Кумулятивное распределение
def create_ecdf(df, num_col, cat_col):
    try:
        fig_ecdf = px.ecdf(df, x=num_col, color=cat_col)
        return fig_ecdf 
    except ValueError as e:
        st.error(f"Ошибка при создании кумулятивного распределения: {e}")
        return None
# График плотности 
def create_density_contour(df, num_col, cat_col):
    try:
        fig_density_contour = px.density_contour(df, x=num_col, y=cat_col)
        return fig_density_contour
    except ValueError as e:
        st.error(f"Ошибка при создании графика плотности: {e}")
        return None
# Точечная диаграмма (с jitter для категориальной переменной)
def create_scatter_plot(df, num_col, cat_col):
    try:
        fig_scatter = px.scatter(df, x=selected_num_column, y=selected_cat_column, color=selected_cat_column)
        fig_scatter.update_traces(marker=dict(size=8))  # Увеличить размер точек
        fig_scatter.update_xaxes(showgrid=False)
        fig_scatter.update_yaxes(showgrid=False)
        return fig_scatter
    except ValueError as e:
        st.error(f"Ошибка при создании точечной диаграммы: {e}")
        return None 
    
# Словарь графиков     
plot_types = {
    "скрипичный график": create_violin_plot,
    "тепловая карта": create_heatmap,
    "гистограмма": create_histogram,
    "кумулятивное распределение": create_ecdf,
    "график плотности": create_density_contour,
    "точечная диаграмма": create_scatter_plot
}    
      
def create_num_visualization(df, var):
    try:
        # Cтандартное отклонение
        std = df[var].std()
        st.write(f"_Стандартное отклонение_ :red[{std:.3f}]")
        # Визуализация (гистограмма и ящик с усами)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Гистограмма", "Ящик с усами"))
        fig.add_trace(go.Histogram(x=df[var], nbinsx=33), row=1, col=1)
        fig.add_trace(go.Box(x=df[var]), row=1, col=2)
        # Вертикальные линии (минимум, максимум, медиана)
        min_val, max_val, median_val = df[var].min(), df[var].max(), df[var].median()
        for value, text in zip([min_val, max_val, median_val],
                              ["min {:.2f}".format(min_val), "max {:.2f}".format(max_val), "медиана {:.2f}".format(median_val)]):
            for col in [1, 2]:
                fig.add_vline(x=value, line_width=2, line_dash="longdash", line_color="red",
                              annotation_text=text, annotation_position="top right", row=1, col=col)
        fig['layout'].update(height=600, width=800, title_text=f"Распределение {var}", showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Ошибка при создании визуализации для числовой переменной '{var}': {e}")
        return None

def create_cat_visualization(df, var):
    try:
        # Уникальные значения
        unique_values = df[var].nunique()
        unique_values_str = ", ".join(df[var].unique().astype(str))
        st.write(f"_Уникальные значениия_ {unique_values}: :red[{unique_values_str}]")
        # Визуализация (гистограмма и круговая диаграмма)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]],
                            subplot_titles=("Гистограмма", "Круговая диаграмма"))
        fig.add_trace(go.Histogram(x=df[var]), row=1, col=1)
        fig.add_trace(go.Pie(labels=df[var].value_counts().index, values=df[var].value_counts()), row=1, col=2)
        fig['layout'].update(height=600, width=800, title_text=f"Распределение {var}", showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Ошибка при создании визуализации для категориальной переменной '{var}': {e}")
        return None
    
st.toast('материалы на этом сайте защищены законом об авторском праве')
    
tab1, tab2,  tab3 = st.tabs([":blue[Основная информация о наборе данных]",":blue[Распределение признаков]", ":blue[Сравнение распределений признаков]"])    

with tab1:
    show_basic_data_info(df)        
    show_duplicates()               
    show_data_info()
    display_splom(df)
      
# Выбор признака для визуализации каждого столбца       
with tab2:
    multiselect_vis = st.multiselect("_**Выберите признак для анализа распределения**_", df.columns, key="col1")
    if multiselect_vis:
        for var in multiselect_vis:
            try:
                # Количество пропущенных значений
                null_count = df[var].isnull().sum()
                null_percent = null_count / len(df)
                st.write(f"_Пропущенные значения_ {var} :red[{null_count} ({null_percent:.2%})]")
                if var in num_col:
                    fig = create_num_visualization(df, var)
                elif var in cat_col:
                    fig = create_cat_visualization(df, var)
                else:
                    raise ValueError(f"Неизвестный тип переменной: {var}")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка при обработке переменной '{var}': {e}")
    else:
        st.write(":red[_не выбрано_]")  
            
with tab3:
    # Выбор числового столбца
    selected_num_column = st.selectbox("_Выберите числовой столбец_", num_col, key='num_col')
    # Выбор категориального столбца
    selected_cat_column = st.selectbox("_Выберите категориальный столбец_", cat_col, key='cat_col')
    if selected_num_column and selected_cat_column:
    # Визуализация распределения числового столбца по категориям
        st.write(f"_Выберите график для визуализации распределения :blue[{selected_num_column}] по :blue[{selected_cat_column}]_") 
        # Чекбоксы для выбора графиков
        for plot_type, plot_function in plot_types.items():
            if st.checkbox(f"{plot_type}", key=plot_type):
                fig = plot_function(df, selected_num_column, selected_cat_column)
                if fig is not None:  
                    st.plotly_chart(fig, use_container_width=True)
   

                

                
                

        


