import streamlit as st

page1_path = r'C:\Users\krest\OneDrive\Рабочий стол\Новая папка\pages\org_data.py'
page2_path = r'C:\Users\krest\OneDrive\Рабочий стол\Новая папка\pages\vs_data.py'
page3_path = r'C:\Users\krest\OneDrive\Рабочий стол\Новая папка\pages\vis_data.py'
#page4_path = r'C:\Users\krest\OneDrive\Рабочий стол\Новая папка\pages\your_data.py'
page_path = r'C:\Users\krest\OneDrive\Рабочий стол\Новая папка\pages\instr.py'

# отобразит меню в зависимости от роли аутентифицированного пользователя.
def authenticated_menu():
    st.sidebar.page_link("app.py", label="МедТехИнтерфейс главная")
    st.sidebar.page_link(page_path, label="Инструкция пользователям")
    if st.session_state.role in ["Организатор"]:      
        st.sidebar.page_link(page1_path, label="Статистический анализ")
        st.sidebar.page_link(page2_path, label="Распределение данных")
        st.sidebar.page_link(page3_path, label="Взаимосвязи данных")  
    elif st.session_state.role in ["Врач"]:
        st.sidebar.page_link(page2_path, label="Распределение данных")
        st.sidebar.page_link(page3_path, label="Взаимосвязи данных")  
    else:
        #st.sidebar.page_link(page4_path, label="Ваши данные")
        st.sidebar.page_link(page3_path, label="Взаимосвязи данных") 
# Показать навигационное меню для пользователей без выбранной роли        
def unauthenticated_menu():
    st.sidebar.page_link("app.py", label="МедТехИнтерфейс главная")
    st.sidebar.page_link(page_path, label="Инструкция пользователям")    
#вызовет правильную вспомогательную функцию для отображения меню в зависимости от того, вошел пользователь в систему или нет        
def menu():
    if "role" not in st.session_state or st.session_state.role is None:
        unauthenticated_menu()
        return
    authenticated_menu()
# проверяет, вошел ли пользователь в систему, затем либо перенаправляет его на главную страницу, либо отображает меню.
def menu_with_redirect():
    if "role" not in st.session_state or st.session_state.role is None:
        st.switch_page("app.py")
    menu()
    
    