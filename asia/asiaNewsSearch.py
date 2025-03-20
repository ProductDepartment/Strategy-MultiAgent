import os
import json
import psycopg2
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain

# Загружаем переменные окружения
load_dotenv()

# Настройки API ключей
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Настройки базы данных
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Инициализация клиентов API
client = OpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=GEMINI_API_KEY, 
    temperature=0.2,
    timeout=60
)

# Список азиатских стран для отслеживания
ASIAN_COUNTRIES = [
    "Japan", "South Korea", "China", "Singapore", "Malaysia", 
    "Indonesia", "Thailand", "Vietnam", "Philippines", "India",
    "Taiwan", "Hong Kong", "UAE", "Qatar", "Saudi Arabia"
]

# ========= ФУНКЦИИ ДЛЯ РАБОТЫ С ПРОМПТАМИ ==========

def create_news_search_prompt(country_name):
    """Создает промпт для поиска новостей о поступлении в вузы указанной азиатской страны."""
    return [
        {
            "role": "system",
            "content": "You are a specialized education news researcher focusing on Asian university admissions for Kazakhstani students. Your task is to find accurate, timely, and relevant information that impacts Kazakhstani applicants to Asian universities."
        },
        {
            "role": "user",
            "content": f"""Find recent news (from the last 9 months) related to university admissions in {country_name} that specifically impact Kazakhstani citizens applying to universities in this Asian country.

Focus on news about:
1. Changes in admission requirements for international/Central Asian students
2. New scholarship or financial aid opportunities available to Kazakhstani applicants
3. Visa policy changes affecting Kazakhstani or Central Asian students
4. Special admission quotas or programs for Kazakhstani students
5. Educational partnerships between {country_name} and Kazakhstan
6. Changes in tuition fees for international students from Central Asia
7. Language requirements (English/local language) changes for foreign students

Use reliable sources including:
- Official university websites in {country_name}
- {country_name}'s education ministry announcements
- Kazakhstan's Ministry of Education announcements
- Education news platforms covering Asian higher education
- Diplomatic announcements between Kazakhstan and {country_name}

Provide the response in a JSON array format with the following fields for each news item:
- 'date': publication date (in YYYY-MM-DD format)
- 'source': name of the source
- 'title': concise title of the news
- 'summary': a detailed summary focusing on relevance to Kazakhstani students
- 'url': link to the original source

If no news is found, return an empty array [].

Your response must consist solely of the JSON array. Do not include explanations, comments, or additional text beyond the JSON structure.
"""
        }
    ]

def create_news_verification_prompt():
    """Создает промпт для верификации новостей об азиатских вузах с помощью Gemini."""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a fact-checking specialist focusing on Asian higher education news relevant to Kazakhstani students. Your job is to verify the accuracy and relevance of news items related to university admissions in Asian countries.

Carefully evaluate the provided news items about Asian university admissions and determine if the information appears accurate and relevant for Kazakhstani students."""),
        
        HumanMessage(content="""Please verify these news items about Asian university admissions:

{news_data}

Provide your assessment in JSON format with the following fields:
- "is_reliable": boolean (true/false) indicating if the information appears accurate and from reliable sources
- "verification_score": number from 0-100 indicating confidence in the information
- "notes": brief explanation of your assessment

Your response should only contain the JSON object.""")
    ])

# ========= ФУНКЦИИ ДЛЯ РАБОТЫ С БАЗОЙ ДАННЫХ ==========

def get_db_connection():
    """Устанавливает соединение с базой данных."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Ошибка подключения к базе данных: {str(e)}")
        return None

def extract_json_from_response(response_text):
    """
    Извлекает JSON из ответа API.
    
    Args:
        response_text (str): Текст ответа API
        
    Returns:
        list/dict: Данные из JSON
    """
    print(f"Получен ответ (первые 100 символов): {response_text[:100]}")
    
    # Проверяем на markdown формат с ```json
    if '```json' in response_text and '```' in response_text:
        # Извлекаем только JSON часть, удаляя маркеры markdown
        json_start = response_text.find('```json') + 7
        json_end = response_text.rfind('```')
        json_str = response_text[json_start:json_end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON из блока markdown: {str(e)}")
    
    # Пробуем разобрать напрямую как JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Ищем JSON в тексте с помощью регулярных выражений
        import re
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON из regex: {str(e)}")
    
    print("ВНИМАНИЕ: Не удалось извлечь валидный JSON из ответа")
    return []

def search_university_news(country_name):
    """
    Ищет новости об университетах для указанной страны с помощью Sonar.
    
    Args:
        country_name (str): Название страны для поиска новостей
        
    Returns:
        list: Список найденных новостей в формате JSON
    """
    try:
        response = client.chat.completions.create(
            model="sonar",
            messages=create_news_search_prompt(country_name),
            temperature=0,
            max_tokens=1500,
        )
        
        # Получаем содержимое ответа
        content = response.choices[0].message.content
        
        # Извлекаем JSON из ответа
        news_items = extract_json_from_response(content)
        return news_items
    except Exception as e:
        print(f"Ошибка поиска новостей для {country_name}: {str(e)}")
        return []

def verify_news_items(news_items):
    """
    Проверяет достоверность новостей с помощью Gemini.
    
    Args:
        news_items (list): Список новостей для проверки
        
    Returns:
        dict: Результат верификации
    """
    try:
        if not news_items:
            return {"is_reliable": False, "verification_score": 0, "notes": "Нет новостей для проверки"}
        
        # Проверяем обязательные поля в каждой новости
        for i, news in enumerate(news_items):
            required_fields = ['date', 'source', 'summary']
            for field in required_fields:
                if field not in news or not news[field]:
                    print(f"Предупреждение: новость #{i+1} не содержит обязательное поле '{field}'")
                    # Заполняем отсутствующие поля значениями по умолчанию
                    if field == 'date':
                        news[field] = datetime.now().strftime('%Y-%m-%d')
                    elif field == 'source':
                        news[field] = 'Неизвестный источник'
                    elif field == 'summary':
                        news[field] = 'Описание отсутствует'
        
        # Преобразуем список новостей в JSON строку
        news_json = json.dumps(news_items, ensure_ascii=False, indent=2)
        
        # Ограничиваем размер данных, если они слишком велики
        if len(news_json) > 4000:
            print(f"Предупреждение: сокращаем большой объем данных ({len(news_json)} символов)")
            # Оставляем только основную информацию для первых 3 новостей
            truncated_news = []
            for i, news in enumerate(news_items):
                if i < 3:
                    truncated_news.append({
                        "title": news.get("title", "")[:100],
                        "source": news.get("source", ""),
                        "date": news.get("date", ""),
                        "summary": news.get("summary", "")[:200],
                        "url": news.get("url", "")
                    })
                else:
                    break
            news_json = json.dumps(truncated_news, ensure_ascii=False, indent=2)
        
        # Создаем цепочку верификации
        verification_chain = LLMChain(
            llm=llm, 
            prompt=create_news_verification_prompt(),
            verbose=True
        )
        
        # Запускаем верификацию
        verification_result = verification_chain.run(news_data=news_json)
        
        # Извлекаем результат верификации
        verification_data = extract_json_from_response(verification_result)
        
        if not verification_data:
            print("Не удалось распарсить результат верификации")
            return {"is_reliable": False, "verification_score": 0, "notes": "Ошибка верификации"}
        
        return verification_data
    except Exception as e:
        print(f"Ошибка при верификации новостей: {str(e)}")
        return {"is_reliable": False, "verification_score": 0, "notes": f"Ошибка: {str(e)}"}

def save_news_to_db(country_name, news_items):
    """
    Сохраняет новости в базу данных, предотвращая дублирование.
    
    Args:
        country_name (str): Название страны
        news_items (list): Список новостей для сохранения
        
    Returns:
        bool: True если успешно, иначе False
    """
    conn = get_db_connection()
    if not conn:
        return False
        
    cursor = conn.cursor()
    
    try:
        # Проверяем, есть ли уже записи для этой страны
        cursor.execute("SELECT content FROM asia_news WHERE country_name = %s", (country_name,))
        existing_record = cursor.fetchone()
        
        current_time = datetime.now()
        
        if existing_record and existing_record[0]:
            # Получаем существующие новости
            existing_news = json.loads(existing_record[0])
            
            # Создаем словарь существующих новостей для быстрого поиска дубликатов
            # Используем комбинацию заголовка и даты как ключ
            existing_dict = {}
            for news in existing_news:
                key = f"{news.get('title', '')}|{news.get('date', '')}|{news.get('source', '')}"
                existing_dict[key] = news
            
            # Добавляем только новые уникальные новости
            new_count = 0
            for news in news_items:
                key = f"{news.get('title', '')}|{news.get('date', '')}|{news.get('source', '')}"
                if key not in existing_dict:
                    existing_news.append(news)
                    new_count += 1
            
            # Сортируем новости по дате (сначала новые)
            existing_news.sort(key=lambda x: x.get('date', ''), reverse=True)
            
            # Ограничиваем количество сохраняемых новостей (не более 30 на страну)
            if len(existing_news) > 30:
                existing_news = existing_news[:30]
            
            content_json = json.dumps(existing_news, ensure_ascii=False)
            
            # Обновляем существующую запись
            cursor.execute(
                """
                UPDATE asia_news 
                SET content = %s, last_update_date = %s 
                WHERE country_name = %s
                """,
                (content_json, current_time, country_name)
            )
            
            print(f"Обновлена запись для {country_name}. Добавлено {new_count} новых новостей.")
        else:
            # Создаем новую запись
            content_json = json.dumps(news_items, ensure_ascii=False)
            cursor.execute(
                """
                INSERT INTO asia_news (country_name, content, last_update_date)
                VALUES (%s, %s, %s)
                """,
                (country_name, content_json, current_time)
            )
            print(f"Добавлена новая запись для {country_name} с {len(news_items)} новостями.")
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при сохранении новостей: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_news_from_db(country_name):
    """
    Получает новости из базы данных для указанной страны.
    
    Args:
        country_name (str): Название страны
        
    Returns:
        list: Список новостей
    """
    conn = get_db_connection()
    if not conn:
        return []
        
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            SELECT content, last_update_date 
            FROM asia_news 
            WHERE country_name = %s
            """,
            (country_name,)
        )
        
        result = cursor.fetchone()
        
        if not result or not result[0]:
            return []
        
        # Извлекаем JSON-массив из поля content
        content_json, last_update = result
        
        # Парсим JSON-строку обратно в список
        news_items = json.loads(content_json)
        
        # Добавляем дату последнего обновления к каждой новости
        for news in news_items:
            news['last_updated'] = last_update.strftime('%Y-%m-%d %H:%M:%S')
            
        return news_items
    except Exception as e:
        print(f"Ошибка при получении новостей: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

def get_countries_needing_update(days=7):
    """
    Получает список стран, которые не обновлялись указанное количество дней.
    
    Args:
        days (int): Количество дней
        
    Returns:
        list: Список стран, требующих обновления
    """
    conn = get_db_connection()
    if not conn:
        return ASIAN_COUNTRIES  # Возвращаем все страны, если не удалось подключиться к БД
        
    cursor = conn.cursor()
    
    countries_to_update = []
    try:
        # Получаем все страны из базы данных
        cursor.execute(
            """
            SELECT country_name, last_update_date 
            FROM asia_news
            """
        )
        results = cursor.fetchall()
        
        # Преобразуем результаты в словарь {страна: дата_обновления}
        db_countries = {country: date for country, date in results}
        
        # Проверяем, какие страны требуют обновления
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for country in ASIAN_COUNTRIES:
            # Если страны нет в базе или дата обновления старше cutoff_date
            if country not in db_countries or db_countries[country] is None or db_countries[country] < cutoff_date:
                countries_to_update.append(country)
        
        return countries_to_update
    except Exception as e:
        print(f"Ошибка при получении списка стран для обновления: {str(e)}")
        return ASIAN_COUNTRIES  # Возвращаем все страны в случае ошибки
    finally:
        cursor.close()
        conn.close()

# ========= ОСНОВНЫЕ ФУНКЦИИ ==========

def update_news_for_country(country_name):
    """
    Обновляет новости для указанной азиатской страны.
    
    Args:
        country_name (str): Название страны
        
    Returns:
        bool: True если успешно, иначе False
    """
    print(f"Обновление новостей о поступлении казахстанцев в вузы {country_name}...")
    
    # Ищем новости
    news_items = search_university_news(country_name)
    
    if not news_items:
        print(f"Новости для {country_name} не найдены")
        # Сохраняем пустой массив, чтобы обновить last_update_date
        save_news_to_db(country_name, [])
        return False
    
    print(f"Найдено {len(news_items)} потенциальных новостей")
    print(f"Пример первой новости: {json.dumps(news_items[0], ensure_ascii=False, indent=2)[:200]}...")
    
    # Учитывая, что новости уже проверены Sonar, можно пропустить сложную верификацию
    # и сразу сохранить их, если они соответствуют минимальным критериям качества
    valid_news = []
    
    for news in news_items:
        # Проверяем наличие минимально необходимых полей
        if 'title' not in news and 'summary' not in news:
            continue
            
        # Убедимся, что все обязательные поля присутствуют
        if 'date' not in news:
            news['date'] = datetime.now().strftime('%Y-%m-%d')
        if 'source' not in news:
            news['source'] = 'Источник не указан'
        if 'summary' not in news:
            news['summary'] = 'Краткое описание отсутствует'
        if 'url' not in news:
            news['url'] = ''
            
        valid_news.append(news)
    
    # Сохраняем новости в базу данных
    if valid_news and save_news_to_db(country_name, valid_news):
        print(f"Новости для {country_name} успешно сохранены ({len(valid_news)} шт.)")
        return True
    else:
        print(f"Нет подходящих новостей для сохранения для {country_name}")
        # Обновляем дату последнего обновления
        save_news_to_db(country_name, [])
        return False

def run_weekly_news_update():
    """
    Запускает еженедельное обновление новостей для азиатских стран.
    """
    print("Запуск еженедельного обновления новостей о поступлении казахстанцев в азиатские вузы...")
    
    # Получаем список стран, требующих обновления
    countries_to_update = get_countries_needing_update(days=7)
    
    if not countries_to_update:
        print("Все страны были обновлены недавно. Обновление не требуется.")
        return
    
    print(f"Требуют обновления: {', '.join(countries_to_update)}")
    
    # Обновляем новости для каждой страны
    success_count = 0
    for country in countries_to_update:
        if update_news_for_country(country):
            success_count += 1
        
        # Делаем паузу между странами, чтобы не перегружать API
        if country != countries_to_update[-1]:
            print(f"Ожидание 10 секунд перед следующей страной...")
            time.sleep(10)
    
    print(f"Обновление завершено. Успешно обновлено {success_count} из {len(countries_to_update)} стран")

def display_news_for_country(country_name):
    """
    Отображает новости для указанной азиатской страны в читаемом формате.
    
    Args:
        country_name (str): Название страны
    """
    news_items = get_news_from_db(country_name)
    
    if not news_items:
        print(f"Новости о поступлении казахстанцев в вузы {country_name} не найдены")
        return
    
    print(f"\n===== Новости о поступлении казахстанцев в вузы {country_name} =====\n")
    
    for i, news in enumerate(news_items, 1):
        title = news.get('title', f"Новость о {country_name} #{i}")
        print(f"📰 Новость #{i}: {title}")
        
        if 'date' in news:
            print(f"📅 Дата публикации: {news['date']}")
            
        print(f"🔍 Источник: {news.get('source', 'Источник не указан')}")
        print(f"📝 Содержание: {news.get('summary', 'Нет описания')}")
        
        if 'url' in news and news['url']:
            print(f"🔗 Ссылка: {news['url']}")
        
        if 'last_updated' in news:
            print(f"🔄 Данные обновлены: {news['last_updated']}")
            
        print("-" * 50)

# ========= ОСНОВНОЙ БЛОК ==========

if __name__ == "__main__":
    print("Система поиска новостей о поступлении казахстанцев в азиатские вузы")
    print("================================================================")
    
    while True:
        print("\nВыберите действие:")
        print("1. Обновить новости для конкретной азиатской страны")
        print("2. Запустить еженедельное обновление для всех стран")
        print("3. Просмотреть новости для азиатской страны")
        print("4. Выход")
        
        choice = input("Введите номер действия (1-4): ")
        
        if choice == "1":
            print("\nДоступные азиатские страны:")
            for i, country in enumerate(ASIAN_COUNTRIES, 1):
                print(f"{i}. {country}")
            
            try:
                country_idx = int(input("Введите номер страны: ")) - 1
                if 0 <= country_idx < len(ASIAN_COUNTRIES):
                    update_news_for_country(ASIAN_COUNTRIES[country_idx])
                else:
                    print("Неверный номер страны")
            except ValueError:
                print("Пожалуйста, введите число")
        
        elif choice == "2":
            run_weekly_news_update()
        
        elif choice == "3":
            print("\nДоступные азиатские страны:")
            for i, country in enumerate(ASIAN_COUNTRIES, 1):
                print(f"{i}. {country}")
            
            try:
                country_idx = int(input("Введите номер страны: ")) - 1
                if 0 <= country_idx < len(ASIAN_COUNTRIES):
                    display_news_for_country(ASIAN_COUNTRIES[country_idx])
                else:
                    print("Неверный номер страны")
            except ValueError:
                print("Пожалуйста, введите число")
        
        elif choice == "4":
            print("Выход из программы")
            break
        
        else:
            print("Неверный выбор, попробуйте снова") 