import os
import json
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
import time  # Добавляем импорт модуля time для создания задержек

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
    timeout=100  # Устанавливаем таймаут в 25 секунд
)

# Список стран Азии (кроме Казахстана)
ASIAN_COUNTRIES = [
    "Japan", "South Korea", "China", "Singapore", "Malaysia",
    "Indonesia", "Thailand", "Vietnam", "Philippines", "India",
    "Taiwan", "Hong Kong", "UAE", "Qatar", "Saudi Arabia"
]


# =========== ФУНКЦИИ ДЛЯ РАБОТЫ С ПРОМПТАМИ ===========

def create_scholarship_search_prompt(country_name):
    """Создает промпт для поиска стипендий в указанной стране для граждан Казахстана."""
    return [
        {
            "role": "system",
            "content": "You are a scholarship specialist who helps students from Kazakhstan find funding opportunities for studying abroad. Your task is to provide detailed, accurate information about scholarships, grants, and tuition fee discounts available to Kazakhstani citizens at universities in Asian countries."
        },
        {
            "role": "user",
            "content": f"""Find detailed information about scholarships, grants, and tuition fee discounts available at universities in {country_name} that are open to Kazakhstani citizens.

Focus on opportunities that:
1. Are specifically available to international students from Kazakhstan or Central Asia
2. Are general international scholarships that Kazakhstani citizens are eligible for
3. Have application deadlines in the future or are recurring annually

For each scholarship/grant opportunity, provide the following details in a structured format:
- Scholarship name (full official name)
- Country (the country where the university is located)
- Amount (scholarship value - full coverage, percentage, or exact amount)
- Requirements (academic requirements, language proficiency, other eligibility criteria)
- Application deadline (in YYYY-MM-DD format, use "rolling" for continuous applications)
- Next deadline (the upcoming deadline date if it's an annual scholarship in YYYY-MM-DD format)
- Application website (direct URL to the application portal)
- Additional information (application process, eligibility details, etc.)

Return the information in a JSON array format with each scholarship as a separate object containing the above fields.

If no specific scholarships are found for Kazakhstani students in this country, return information about general international scholarships that Kazakhstani citizens would be eligible for.

For deadlines:
1. Use the strict YYYY-MM-DD format for all dates
2. Include only deadlines that are confirmed from official sources
3. For rolling admissions, use "rolling" instead of a specific date
4. If a deadline has passed, provide the next expected deadline based on the annual cycle
5. If a deadline is unknown, use null but try to avoid this when possible

Your response must be ONLY a valid JSON array without any additional text, explanations, or comments.
"""
        }
    ]


def create_scholarship_verification_prompt():
    """Создает промпт для верификации информации о стипендиях с помощью Gemini."""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a scholarship verification specialist focused on accuracy assessment. Your task is to evaluate scholarship information found for Kazakhstani students and determine if it's accurate, current, and relevant.

Review the provided scholarship details carefully and flag any potential inaccuracies, outdated information, or misleading claims. Focus on verification of:
1. Scholarship existence and eligibility for Kazakhstani citizens
2. Accuracy of financial amounts and coverage
3. Correctness of requirements and application procedures
4. Current relevance (not expired or discontinued programs)

Provide your assessment as a JSON with a verification score (0-100) and specific notes on accuracy."""),

        HumanMessage(content="""
Scholarship information to verify:
{scholarship_data}

Based on your knowledge and the information provided, evaluate the accuracy and reliability of this scholarship information. Consider factors like:
- Does this scholarship exist and match known programs?
- Are the details internally consistent?
- Are the requirements and eligibility criteria reasonable?
- Is there anything suspicious or questionable in the information?

Return your verification in the following format:
```json
{
  "verification_score": [0-100 score],
  "is_reliable": [true/false],
  "notes": ["specific notes about accuracy issues or confirmation of validity"]
}
```

Your response should contain ONLY the JSON verification result without additional text.
""")
    ])


# =========== ФУНКЦИИ ДЛЯ РАБОТЫ С API ===========

def search_scholarships(country_name):
    """
    Ищет информацию о стипендиях для указанной страны с помощью Sonar.

    Args:
        country_name (str): Название страны для поиска стипендий

    Returns:
        list: Список найденных стипендий в формате JSON
    """
    try:
        response = client.chat.completions.create(
            model="sonar",
            messages=create_scholarship_search_prompt(country_name),
            temperature=0,
            max_tokens=1500,
        )

        # Получаем содержимое ответа
        content = response.choices[0].message.content

        # Извлекаем JSON из ответа
        scholarships_data = extract_json_from_response(content)

        # Стандартизируем ключи для всех стипендий
        standardized_data = []
        for scholarship in scholarships_data:
            # Преобразуем ключи к стандартному формату
            standardized_scholarship = standardize_scholarship_keys(scholarship)
            standardized_data.append(standardized_scholarship)

        return standardized_data
    except Exception as e:
        print(f"Error searching scholarships for {country_name}: {str(e)}")
        return []


def standardize_scholarship_keys(scholarship):
    """
    Преобразует ключи стипендии в стандартный формат.

    Args:
        scholarship (dict): Исходные данные о стипендии

    Returns:
        dict: Стипендия со стандартизированными ключами
    """
    key_mapping = {
        "Scholarship name": "scholarship_name",
        "country": "country",
        "Amount": "amount",
        "Requirements": "requirements",
        "Additional information": "additional_info"
    }

    # Создаем новый словарь со стандартизированными ключами
    standardized = {}

    for original_key, value in scholarship.items():
        # Проверяем, есть ли точное соответствие в карте ключей
        if original_key in key_mapping:
            standardized_key = key_mapping[original_key]
        else:
            # Пытаемся преобразовать ключ: приводим к нижнему регистру и заменяем пробелы на подчеркивания
            standardized_key = original_key.lower().replace(" ", "_")

        standardized[standardized_key] = value

    return standardized


def verify_scholarship_data(scholarship_data):
    """
    Проверяет достоверность информации о стипендиях с помощью Gemini.

    Args:
        scholarship_data (dict): Информация о стипендии для проверки

    Returns:
        dict: Результат верификации с оценкой достоверности
    """
    try:
        # Конвертируем данные в JSON-строку для передачи в промпт
        scholarship_json = json.dumps(scholarship_data, ensure_ascii=False, indent=2)

        # Ограничиваем размер данных, если они слишком велики
        if len(scholarship_json) > 2000:
            print(f"Warning: Truncating large scholarship data ({len(scholarship_json)} chars)")
            # Оставляем только основную информацию
            truncated_data = {
                "scholarship_name": scholarship_data.get("scholarship_name", "Unknown"),
                "country": scholarship_data.get("country", ""),
                "amount": scholarship_data.get("amount", "")[:200] if scholarship_data.get("amount") else "",
                "requirements": scholarship_data.get("requirements", "")[:300] if scholarship_data.get(
                    "requirements") else ""
            }
            scholarship_json = json.dumps(truncated_data, ensure_ascii=False, indent=2)

        print(f"Verifying scholarship: {scholarship_data.get('scholarship_name', 'Unknown')}")

        # Создаем цепочку верификации
        verification_chain = LLMChain(
            llm=llm,
            prompt=create_scholarship_verification_prompt(),
            verbose=True  # Включаем подробный вывод для отладки
        )

        # Выполняем запрос к API (таймаут уже установлен в объекте llm)
        verification_result = verification_chain.run(scholarship_data=scholarship_json)

        # Извлекаем JSON из ответа
        verification_data = extract_json_from_response(verification_result)

        if not verification_data:
            print("Failed to parse verification result")
            print(f"Raw verification result: {verification_result}")
            return {
                "verification_score": 0,
                "is_reliable": False,
                "notes": ["Failed to parse verification result"]
            }

        return verification_data
    except Exception as e:
        print(f"Error during scholarship verification: {str(e)}")
        # В случае ошибки возвращаем объект с низкой оценкой и информацией об ошибке
        return {
            "verification_score": 0,
            "is_reliable": False,
            "notes": [f"Verification error: {str(e)}"]
        }


def extract_json_from_response(response_text):
    """
    Извлекает JSON из ответа API, который может содержать маркеры Markdown.

    Args:
        response_text (str): Текст ответа API

    Returns:
        list/dict: Данные из JSON
    """
    print(f"Raw response (first 100 chars): {response_text[:100]}")

    # Проверяем на markdown формат с ```json
    if response_text.startswith('```json') and '```' in response_text[7:]:
        # Извлекаем только JSON часть, удаляя маркеры markdown
        json_str = response_text.replace('```json', '', 1)
        json_str = json_str.rsplit('```', 1)[0].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from markdown block: {str(e)}")

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
                print(f"Error parsing JSON from regex match: {str(e)}")

    print("WARNING: Could not extract valid JSON from response")
    print(f"Full response: {response_text}")
    return []


# =========== ФУНКЦИИ ДЛЯ РАБОТЫ С БАЗОЙ ДАННЫХ ===========

def get_db_connection():
    """Устанавливает соединение с базой данных."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return None


def create_scholarships_table_if_not_exists():
    """Создает таблицу для хранения информации о стипендиях, если она не существует."""
    conn = None
    cursor = None

    try:
        # Подключаемся к базе данных
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()

        # Создаем таблицу, если она не существует
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS asia_scholarships (
                asia_sch_id SERIAL PRIMARY KEY,
                scholarship_name TEXT NOT NULL,
                country TEXT NOT NULL,
                amount TEXT,
                requirements TEXT,
                additional_info TEXT,
                application_deadline TEXT,
                application_website TEXT,
                next_deadline DATE,
                last_update_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

        # Проверяем и добавляем отсутствующие столбцы
        check_and_add_db_columns()

        print("Таблица asia_scholarships готова к использованию")
        return True

    except Exception as e:
        print(f"Ошибка при создании таблицы: {str(e)}")
        if conn:
            conn.rollback()
        return False

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def save_scholarship(scholarship_data):
    """
    Сохраняет информацию о стипендии в базу данных.

    Args:
        scholarship_data (dict): Данные о стипендии

    Returns:
        bool: True если сохранение прошло успешно, иначе False
    """
    # Проверяем наличие обязательных полей
    if not all(key in scholarship_data for key in ["scholarship_name", "country"]):
        print("Missing required fields in scholarship data")
        return False

    conn = get_db_connection()
    if not conn:
        return False

    cursor = conn.cursor()

    try:
        # Проверяем, существует ли уже такая стипендия
        cursor.execute(
            """
            SELECT asia_sch_id FROM asia_scholarships 
            WHERE scholarship_name = %s AND country = %s
            """,
            (scholarship_data["scholarship_name"], scholarship_data["country"])
        )
        existing_scholarship = cursor.fetchone()

        # Подготавливаем данные для вставки/обновления
        sch_data = {
            'scholarship_name': scholarship_data["scholarship_name"],
            'country': scholarship_data["country"],
            'amount': scholarship_data.get("amount", ""),
            'requirements': scholarship_data.get("requirements", ""),
            'additional_info': scholarship_data.get("additional_info", ""),
            'last_update_date': datetime.now()
        }

        if existing_scholarship:
            # Обновляем существующую запись
            update_query = """
                UPDATE asia_scholarships SET 
                    scholarship_name = %(scholarship_name)s,
                    country = %(country)s,
                    amount = %(amount)s,
                    requirements = %(requirements)s,
                    additional_info = %(additional_info)s,
                    last_update_date = %(last_update_date)s
                WHERE asia_sch_id = %(asia_sch_id)s
            """
            # Используем именованный параметр для asia_sch_id
            cursor.execute(update_query, {**sch_data, "asia_sch_id": existing_scholarship[0]})
            print(f"Scholarship '{scholarship_data['scholarship_name']}' for {scholarship_data['country']} updated.")
        else:
            # Добавляем новую запись
            insert_query = """
                INSERT INTO asia_scholarships (
                    scholarship_name, country, amount, requirements, additional_info, last_update_date
                ) VALUES (
                    %(scholarship_name)s, %(country)s, %(amount)s, %(requirements)s,
                    %(additional_info)s, %(last_update_date)s
                )
            """
            cursor.execute(insert_query, sch_data)
            print(f"New scholarship '{scholarship_data['scholarship_name']}' for {scholarship_data['country']} added.")

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error saving scholarship data: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()


def get_scholarships_for_country(country):
    """
    Получает информацию о стипендиях для указанной страны.

    Args:
        country (str): Название страны

    Returns:
        list: Список стипендий
    """
    print(f"Поиск стипендий в {country}...")

    try:
        # Запрос к API
        response = client.chat.completions.create(
            model="sonar",
            messages=create_scholarship_search_prompt(country),
            temperature=0.2,
            max_tokens=2048,
        )

        # Получаем текст ответа
        response_text = response.choices[0].message.content

        # Выводим для отладки первые символы ответа
        print(f"Ответ API (первые 100 символов): {response_text[:100]}")

        # Извлекаем JSON из ответа
        scholarships = extract_json_from_response(response_text)

        # Проверяем формат стипендий
        if isinstance(scholarships, list) and len(scholarships) > 0:
            print(f"Найдено {len(scholarships)} стипендий")

            # Выводим структуру первой стипендии для отладки
            first = scholarships[0]
            print(f"Структура первой стипендии: {', '.join(first.keys())}")

            # Нормализуем ключи для соответствия формату БД
            normalized_scholarships = normalize_scholarship_keys(scholarships)
            print("Ключи нормализованы")

            # Проверяем структуру нормализованной стипендии
            if normalized_scholarships:
                print(f"Структура после нормализации: {', '.join(normalized_scholarships[0].keys())}")

            # Проверяем наличие полей дедлайна
            for s in normalized_scholarships:
                # Убедимся, что поля дедлайнов существуют
                if 'application_deadline' not in s and 'deadline' in s:
                    s['application_deadline'] = s['deadline']
                elif 'application_deadline' not in s:
                    s['application_deadline'] = None

                # Убедимся, что поле next_deadline существует
                if 'next_deadline' not in s:
                    s['next_deadline'] = s.get('application_deadline', None)

                # Убедимся, что поле application_website существует
                if 'application_website' not in s and 'application_url' in s:
                    s['application_website'] = s['application_url']
                elif 'application_website' not in s:
                    s['application_website'] = None

            return normalized_scholarships
        else:
            print("Ошибка: Ответ не содержит список стипендий")
            print(f"Полученные данные: {scholarships}")
            return []

    except Exception as e:
        print(f"Ошибка при получении стипендий для {country}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def get_scholarships_to_update():
    """
    Получает список стипендий, которые нужно обновить (последнее обновление более 2 недель назад).

    Returns:
        list: Список стипендий для обновления
    """
    conn = get_db_connection()
    if not conn:
        return []

    cursor = conn.cursor()

    try:
        # Получаем стипендии, обновленные более 2 недель назад
        two_weeks_ago = datetime.now() - timedelta(days=14)
        cursor.execute(
            """
            SELECT scholarship_name, country 
            FROM asia_scholarships 
            WHERE last_update_date < %s OR last_update_date IS NULL
            """,
            (two_weeks_ago,)
        )

        scholarships_to_update = []
        for row in cursor.fetchall():
            scholarships_to_update.append({
                "scholarship_name": row[0],
                "country": row[1]
            })

        return scholarships_to_update
    except Exception as e:
        print(f"Error getting scholarships to update: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()


# =========== ОСНОВНЫЕ ФУНКЦИИ ПОИСКА И ОБНОВЛЕНИЯ ===========

def normalize_scholarship_keys(scholarships):
    """
    Нормализует ключи стипендий для соответствия формату БД.

    Args:
        scholarships (list): Список стипендий с ненормализованными ключами

    Returns:
        list: Список стипендий с нормализованными ключами
    """
    key_mapping = {
        "scholarship name": "scholarship_name",
        "Scholarship name": "scholarship_name",
        "scholarship_name": "scholarship_name",
        "name": "scholarship_name",
        "Name": "scholarship_name",

        "country": "country",
        "Country": "country",

        "amount": "amount",
        "Amount": "amount",
        "value": "amount",
        "Value": "amount",

        "requirements": "requirements",
        "Requirements": "requirements",
        "eligibility": "requirements",
        "Eligibility": "requirements",

        "application deadline": "application_deadline",
        "Application deadline": "application_deadline",
        "deadline": "application_deadline",
        "Deadline": "application_deadline",

        "next deadline": "next_deadline",
        "Next deadline": "next_deadline",
        "next_deadline": "next_deadline",

        "application website": "application_website",
        "Application website": "application_website",
        "website": "application_website",
        "Website": "application_website",
        "url": "application_website",
        "URL": "application_website",

        "additional information": "additional_info",
        "Additional information": "additional_info",
        "additional info": "additional_info",
        "Additional info": "additional_info",
        "info": "additional_info",
        "Info": "additional_info"
    }

    normalized_scholarships = []

    for scholarship in scholarships:
        normalized = {}

        # Нормализуем каждый ключ
        for key, value in scholarship.items():
            # Приводим ключ к нижнему регистру для упрощения сопоставления
            key_lower = key.lower()

            # Ищем соответствие в mapping или используем исходный ключ
            for source_key, target_key in key_mapping.items():
                if source_key.lower() == key_lower:
                    normalized[target_key] = value
                    break
            else:
                # Если не нашли соответствия, используем исходный ключ
                normalized[key] = value

        # Проверяем наличие обязательных полей
        if "scholarship_name" not in normalized:
            for key in scholarship.keys():
                if "name" in key.lower() or "scholarship" in key.lower():
                    normalized["scholarship_name"] = scholarship[key]
                    break

        if "country" not in normalized:
            normalized["country"] = scholarship.get("Country", scholarship.get("country", ""))

        normalized_scholarships.append(normalized)

    return normalized_scholarships


def search_new_scholarships(country_name):
    """
    Ищет новые стипендии для указанной страны и сохраняет их в базу данных.

    Args:
        country_name (str): Название страны

    Returns:
        int: Количество новых стипендий
    """
    print(f"Searching for new scholarships in {country_name}...")

    # Получаем существующие стипендии из БД
    existing_scholarships = get_scholarships_for_country(country_name)
    print(f"Found {len(existing_scholarships)} existing scholarships in database")

    # Список имен существующих стипендий для проверки дубликатов
    existing_names = []
    for s in existing_scholarships:
        # Проверяем, является ли s словарем и содержит ли нужный ключ
        if isinstance(s, dict):
            if "scholarship_name" in s:
                existing_names.append(s["scholarship_name"])
            # Проверяем альтернативные ключи
            elif "name" in s:
                existing_names.append(s["name"])
            elif "Scholarship name" in s:
                existing_names.append(s["Scholarship name"])

    print(f"Existing scholarship names: {existing_names}")

    # Получаем новые стипендии через API
    api_scholarships = get_scholarships_for_country(country_name)

    if not api_scholarships:
        print("No scholarships found or error occurred")
        return 0

    # Отфильтровываем только новые стипендии
    new_scholarships = []
    for scholarship in api_scholarships:
        # Проверяем наличие ключа scholarship_name
        if "scholarship_name" in scholarship:
            name = scholarship["scholarship_name"]
        elif "name" in scholarship:
            name = scholarship["name"]
        elif "Scholarship name" in scholarship:
            name = scholarship["Scholarship name"]
        else:
            print(f"Warning: Scholarship has no name field. Keys: {scholarship.keys()}")
            continue

        # Проверяем, есть ли уже такая стипендия в БД
        if name not in existing_names:
            new_scholarships.append(scholarship)
            print(f"New scholarship found: {name}")
        else:
            print(f"Scholarship already exists: {name}")

    # Сохраняем новые стипендии в БД
    if new_scholarships:
        saved_count = save_scholarships_to_db(new_scholarships, country_name)
        print(f"Saved {saved_count} new scholarships to database")
        return saved_count
    else:
        print("No new scholarships found")
        return 0


def update_existing_scholarships():
    """
    Обновляет информацию о существующих стипендиях.

    Returns:
        int: Количество обновленных стипендий
    """
    conn = get_db_connection()
    if not conn:
        return 0

    cursor = conn.cursor()
    updated_count = 0

    try:
        # Получаем стипендии для обновления
        # (которые не обновлялись более 2 недель)
        two_weeks_ago = datetime.now() - timedelta(days=14)
        cursor.execute(
            """
            SELECT asia_sch_id, scholarship_name, country 
            FROM asia_scholarships 
            WHERE last_update_date < %s OR last_update_date IS NULL
            """,
            (two_weeks_ago,)
        )

        scholarships_to_update = cursor.fetchall()

        for i, (sch_id, name, country) in enumerate(scholarships_to_update):
            print(f"Updating scholarship: {name}")

            # Ищем актуальную информацию о стипендии
            search_result = search_scholarships(country)

            # Ищем совпадение по имени стипендии
            for scholarship in search_result:
                if standardize_scholarship_keys(scholarship).get("scholarship_name", "").lower() == name.lower():
                    # Нашли совпадение, верифицируем информацию
                    standardized = standardize_scholarship_keys(scholarship)
                    verification = verify_scholarship_data(standardized)

                    if verification["is_reliable"] and verification["verification_score"] >= 70:
                        # Обновляем информацию в базе данных
                        if update_scholarship(sch_id, standardized):
                            updated_count += 1
                            print(f"Updated scholarship: {name}")
                    else:
                        print(f"Updated information not reliable: {name}")
                    break

            # Добавляем задержку в 4 секунды между запросами к Gemini API
            # Но только если это не последний элемент списка
            if i < len(scholarships_to_update) - 1:
                print("Waiting 4 seconds before next verification...")
                time.sleep(4)

        return updated_count
    except Exception as e:
        conn.rollback()
        print(f"Error updating scholarships: {str(e)}")
        return 0
    finally:
        cursor.close()
        conn.close()


def run_weekly_search():
    """
    Запускает еженедельный поиск новых стипендий для всех стран.
    """
    print("Starting weekly search for new scholarships...")

    # Создаем таблицу, если она не существует
    if not create_scholarships_table_if_not_exists():
        print("Error creating table, aborting search")
        return

    # Проходим по каждой стране и ищем новые стипендии
    total_new = 0
    for country in ASIAN_COUNTRIES:
        new_count = search_new_scholarships(country)
        total_new += new_count

    print(f"Weekly search completed. Total new scholarships added: {total_new}")


def run_biweekly_update():
    """
    Запускает двухнедельное обновление существующих стипендий.
    """
    print("Starting biweekly update of existing scholarships...")

    # Создаем таблицу, если она не существует
    if not create_scholarships_table_if_not_exists():
        print("Error creating table, aborting update")
        return

    # Обновляем существующие стипендии
    update_count = update_existing_scholarships()

    print(f"Biweekly update completed. Total scholarships updated: {update_count}")


def display_scholarships_for_country(country_name):
    """
    Отображает все стипендии для указанной страны в читаемом формате.

    Args:
        country_name (str): Название страны
    """
    scholarships = get_scholarships_for_country(country_name)

    if not scholarships:
        print(f"No scholarships found for {country_name}")
        return

    print(f"\n===== Scholarships for {country_name} =====\n")

    for i, sch in enumerate(scholarships, 1):
        print(f"Scholarship #{i}: {sch['scholarship_name']}")
        print(f"Country: {sch['country']}")
        print(f"Amount: {sch['amount']}")
        print(f"Requirements: {sch['requirements']}")
        print(f"Additional Info: {sch['additional_info']}")
        print(f"Last Updated: {sch['last_update_date']}")
        print("=" * 50)


def extract_next_deadline(scholarship_data):
    """
    Извлекает ближайший дедлайн из данных о стипендии.

    Args:
        scholarship_data (dict): Данные о стипендии

    Returns:
        str: Ближайший дедлайн в формате YYYY-MM-DD или None
    """
    today = datetime.now().date()
    print(f"Извлечение дедлайна для {scholarship_data.get('scholarship_name', 'неизвестной стипендии')}")

    # Проверяем поле next_deadline
    if "next_deadline" in scholarship_data and scholarship_data["next_deadline"]:
        deadline_str = scholarship_data["next_deadline"]
        print(f"Найдено поле next_deadline: {deadline_str}")

        if deadline_str and deadline_str.lower() != "rolling" and deadline_str.lower() != "n/a" and deadline_str.lower() != "null":
            try:
                # Проверяем, что это валидная дата
                datetime.strptime(deadline_str, "%Y-%m-%d")
                return deadline_str
            except ValueError:
                print(f"Неверный формат даты next_deadline: {deadline_str}")

    # Проверяем application_deadline или deadline
    deadline_field = None
    deadline_value = None

    if "application_deadline" in scholarship_data and scholarship_data["application_deadline"]:
        deadline_field = "application_deadline"
        deadline_value = scholarship_data["application_deadline"]
    elif "deadline" in scholarship_data and scholarship_data["deadline"]:
        deadline_field = "deadline"
        deadline_value = scholarship_data["deadline"]

    if deadline_field and deadline_value:
        print(f"Найдено поле {deadline_field}: {deadline_value}")

        if deadline_value.lower() != "rolling" and deadline_value.lower() != "n/a" and deadline_value.lower() != "null":
            try:
                # Проверяем, что это валидная дата
                date_obj = datetime.strptime(deadline_value, "%Y-%m-%d").date()

                # Если дедлайн прошел, предполагаем следующий год для ежегодных стипендий
                if date_obj < today:
                    next_year_date = date_obj.replace(year=today.year + 1)
                    print(f"Дедлайн уже прошел, устанавливаем на следующий год: {next_year_date}")
                    return next_year_date.strftime("%Y-%m-%d")

                return deadline_value
            except ValueError:
                print(f"Неверный формат даты {deadline_field}: {deadline_value}")

    # Если не нашли дедлайн, возвращаем None
    print("Дедлайн не найден")
    return None


def create_or_update_scholarship(conn, scholarship_data):
    """
    Создает новую запись о стипендии или обновляет существующую.

    Args:
        conn: Соединение с БД
        scholarship_data (dict): Данные о стипендии

    Returns:
        bool: True если успешно, иначе False
    """
    try:
        cursor = conn.cursor()

        # Основные поля
        name = scholarship_data.get("scholarship_name", "")
        country = scholarship_data.get("country", "")

        if not name or not country:
            print(f"Ошибка: отсутствует название стипендии или страна: {scholarship_data}")
            return False

        # Дедлайн и вебсайт
        application_deadline = scholarship_data.get("application_deadline", None)
        application_website = scholarship_data.get("application_website", None)

        # Преобразуем next_deadline в объект date для БД если есть
        next_deadline = None
        next_deadline_str = scholarship_data.get("next_deadline")
        if next_deadline_str and isinstance(next_deadline_str, str):
            try:
                if next_deadline_str.lower() not in ["rolling", "null", "n/a", "none", "varies"]:
                    next_deadline = datetime.strptime(next_deadline_str, "%Y-%m-%d").date()
            except ValueError:
                print(f"Ошибка формата даты: {next_deadline_str}")

        # Проверяем, существует ли запись
        cursor.execute(
            "SELECT asia_sch_id FROM asia_scholarships WHERE scholarship_name = %s AND country = %s",
            (name, country)
        )
        existing = cursor.fetchone()

        if existing:
            # Обновляем существующую запись - упрощенный вариант
            cursor.execute("""
                UPDATE asia_scholarships SET
                amount = %s,
                requirements = %s,
                additional_info = %s,
                application_deadline = %s,
                application_website = %s,
                next_deadline = %s,
                last_update_date = %s
                WHERE scholarship_name = %s AND country = %s
            """, (
                scholarship_data.get("amount", ""),
                scholarship_data.get("requirements", ""),
                scholarship_data.get("additional_info", ""),
                application_deadline,
                application_website,
                next_deadline,
                datetime.now(),
                name,
                country
            ))
            print(f"Обновлена стипендия: {name}")
        else:
            # Создаем новую запись - упрощенный вариант
            cursor.execute("""
                INSERT INTO asia_scholarships (
                    scholarship_name, country, amount, requirements, 
                    additional_info, application_deadline, application_website,
                    next_deadline, last_update_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                name,
                country,
                scholarship_data.get("amount", ""),
                scholarship_data.get("requirements", ""),
                scholarship_data.get("additional_info", ""),
                application_deadline,
                application_website,
                next_deadline,
                datetime.now()
            ))
            print(f"Добавлена новая стипендия: {name}")

        conn.commit()
        print(f"Успешно сохранена стипендия. Дедлайн: {application_deadline}, Сайт: {application_website}")
        return True

    except Exception as e:
        print(f"Ошибка при сохранении стипендии {name}: {e}")
        conn.rollback()
        return False


def get_upcoming_scholarship_deadlines(days=30):
    """
    Получает список стипендий с дедлайнами в ближайшие N дней.

    Args:
        days (int): Количество дней для поиска

    Returns:
        list: Список словарей с информацией о ближайших дедлайнах
    """
    conn = None
    cursor = None

    try:
        # Подключаемся к базе данных
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()

        # Проверяем наличие столбца next_deadline
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'asia_scholarships' AND column_name = 'next_deadline'
            );
        """)
        has_next_deadline = cursor.fetchone()[0]

        if not has_next_deadline:
            print("Столбец next_deadline отсутствует в таблице asia_scholarships")
            print("Выполняется: ALTER TABLE asia_scholarships ADD COLUMN next_deadline DATE;")
            cursor.execute("""
                ALTER TABLE asia_scholarships 
                ADD COLUMN next_deadline DATE;
            """)
            conn.commit()
            print("Столбец next_deadline добавлен")

            # Обновляем next_deadline для существующих записей
            cursor.execute("""
                UPDATE asia_scholarships
                SET next_deadline = 
                    CASE 
                        WHEN application_deadline ~ '^\d{4}-\d{2}-\d{2}$' THEN 
                            CASE 
                                WHEN TO_DATE(application_deadline, 'YYYY-MM-DD') < CURRENT_DATE 
                                THEN TO_DATE(application_deadline, 'YYYY-MM-DD') + INTERVAL '1 year'
                                ELSE TO_DATE(application_deadline, 'YYYY-MM-DD')
                            END
                        ELSE NULL
                    END
                WHERE application_deadline IS NOT NULL AND application_deadline != 'rolling'
            """)
            conn.commit()
            print("Столбец next_deadline обновлен для существующих записей")

        # Получаем текущую дату и дату через N дней
        today = datetime.now().date()
        future_date = today + timedelta(days=days)

        # Запрашиваем стипендии с ближайшими дедлайнами
        cursor.execute("""
            SELECT 
                scholarship_name,
                country,
                amount,
                requirements,
                application_deadline,
                next_deadline, 
                additional_info,
                application_website
            FROM 
                asia_scholarships
            WHERE 
                next_deadline IS NOT NULL 
                AND next_deadline BETWEEN %s AND %s
            ORDER BY 
                next_deadline ASC
        """, (today, future_date))

        rows = cursor.fetchall()

        # Форматируем результаты
        result = []
        for row in rows:
            deadline_info = {
                "scholarship_name": row[0],
                "country": row[1],
                "amount": row[2],
                "requirements": row[3],
                "application_deadline": row[4],
                "deadline_date": row[5].strftime("%Y-%m-%d") if row[5] else None,
                "days_remaining": (row[5] - today).days if row[5] else None,
                "additional_info": row[6],
                "application_website": row[7]
            }
            result.append(deadline_info)

        return result

    except Exception as e:
        print(f"Ошибка при получении дедлайнов стипендий: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_tb(e.__traceback__)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def parse_scholarship_data(scholarship_json):
    """
    Парсит JSON с данными о стипендии в структуру для базы данных.

    Args:
        scholarship_json (dict): JSON с данными о стипендии

    Returns:
        dict: Структурированные данные
    """
    # Упрощенный вариант, сосредоточенный на дедлайнах и вебсайте
    scholarship = {
        "scholarship_name": scholarship_json.get("scholarship_name", ""),
        "country": scholarship_json.get("country", ""),
        "amount": scholarship_json.get("amount", ""),
        "requirements": scholarship_json.get("requirements", ""),
        "additional_info": scholarship_json.get("additional_info",
                                                scholarship_json.get("additional_information", "")),
    }

    # Добавляем дедлайн заявки если есть
    if "application_deadline" in scholarship_json:
        scholarship["application_deadline"] = scholarship_json["application_deadline"]
    elif "deadline" in scholarship_json:
        scholarship["application_deadline"] = scholarship_json["deadline"]
    else:
        scholarship["application_deadline"] = None

    # Добавляем ссылку на сайт заявки если есть
    if "application_website" in scholarship_json:
        scholarship["application_website"] = scholarship_json["application_website"]
    elif "website" in scholarship_json:
        scholarship["application_website"] = scholarship_json["website"]
    elif "url" in scholarship_json:
        scholarship["application_website"] = scholarship_json["url"]
    else:
        scholarship["application_website"] = None

    # Добавляем next_deadline если есть
    if "next_deadline" in scholarship_json:
        scholarship["next_deadline"] = scholarship_json["next_deadline"]

    return scholarship


def save_scholarships_to_db(scholarships, country):
    """
    Сохраняет информацию о стипендиях в базу данных.

    Args:
        scholarships (list): Список стипендий для сохранения
        country (str): Страна стипендий

    Returns:
        int: Количество сохраненных стипендий
    """
    conn = None
    saved_count = 0

    # Проверка валидности входных данных
    if not scholarships:
        print("Предупреждение: Пустой список стипендий для сохранения")
        return 0

    # Вывод отладочной информации
    print(f"Получено {len(scholarships)} стипендий для сохранения из {country}")

    # Проверяем структуру первой стипендии
    if len(scholarships) > 0:
        first_scholarship = scholarships[0]
        print(f"Пример стипендии: {json.dumps(first_scholarship, ensure_ascii=False, indent=2)}")

        # Проверяем наличие необходимых полей
        if "scholarship_name" not in first_scholarship or "country" not in first_scholarship:
            print("Ошибка: В данных о стипендиях отсутствуют обязательные поля")
            return 0

        # Проверка полей дедлайнов
        if "application_deadline" in first_scholarship:
            print(f"application_deadline: {first_scholarship['application_deadline']}")
        else:
            print("Поле application_deadline отсутствует")

        if "next_deadline" in first_scholarship:
            print(f"next_deadline: {first_scholarship['next_deadline']}")
        else:
            print("Поле next_deadline отсутствует")

    try:
        # Подключаемся к базе данных
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )

        # Обрабатываем каждую стипендию
        for scholarship in scholarships:
            # Парсим данные
            parsed_data = parse_scholarship_data(scholarship)

            # Сохраняем стипендию
            if create_or_update_scholarship(conn, parsed_data):
                saved_count += 1

        print(f"Успешно сохранено {saved_count} из {len(scholarships)} стипендий")
        return saved_count

    except Exception as e:
        print(f"Ошибка при сохранении стипендий: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_tb(e.__traceback__)
        return saved_count
    finally:
        if conn:
            conn.close()


def test_save_scholarship():
    """
    Тестовая функция для проверки сохранения стипендии с дедлайном и вебсайтом.
    """
    # Создаем тестовую стипендию с явными полями дедлайна и вебсайта
    test_data = {
        "scholarship_name": "Test Japan Scholarship 2024",
        "country": "Japan",
        "amount": "JPY 1,500,000 per year",
        "requirements": "GPA 3.5+, JLPT N2, Bachelor's degree",
        "application_deadline": "2024-09-30",  # Формат YYYY-MM-DD
        "next_deadline": "2024-09-30",
        "application_website": "https://example.jp/scholarships",
        "additional_info": "For engineering and science students"
    }

    print("\n=== ТЕСТИРОВАНИЕ СОХРАНЕНИЯ СТИПЕНДИИ ===\n")
    print(f"Тестовая стипендия: {test_data['scholarship_name']}")
    print(f"Дедлайн: {test_data['application_deadline']}")
    print(f"Вебсайт: {test_data['application_website']}\n")

    # Создаем таблицу, если она не существует
    create_scholarships_table_if_not_exists()

    # Получаем соединение с БД
    conn = get_db_connection()
    if not conn:
        print("Ошибка соединения с БД")
        return

    # Пробуем сохранить стипендию напрямую через create_or_update_scholarship
    try:
        result = create_or_update_scholarship(conn, test_data)
        if result:
            print("\nСтипендия успешно сохранена напрямую!")

            # Проверяем, что было сохранено
            cursor = conn.cursor()
            cursor.execute("""
                SELECT scholarship_name, application_deadline, application_website, next_deadline
                FROM asia_scholarships
                WHERE scholarship_name = %s AND country = %s
            """, (test_data["scholarship_name"], test_data["country"]))

            row = cursor.fetchone()
            if row:
                print("\nСохраненные данные:")
                print(f"Название: {row[0]}")
                print(f"Дедлайн заявки: {row[1]}")
                print(f"Сайт заявки: {row[2]}")
                print(f"Следующий дедлайн: {row[3]}")
            else:
                print("Запись не найдена в БД после сохранения!")

            cursor.close()
        else:
            print("Ошибка при сохранении тестовой стипендии")
    except Exception as e:
        print(f"Исключение при тестировании: {str(e)}")
    finally:
        conn.close()

    print("\n=== ЗАВЕРШЕНИЕ ТЕСТА ===\n")


def get_stored_scholarships_for_country(country_name):
    """
    Получает список всех стипендий для указанной страны из базы данных.

    Args:
        country_name (str): Название страны

    Returns:
        list: Список стипендий для указанной страны
    """
    conn = get_db_connection()
    if not conn:
        return []

    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT scholarship_name, country, amount, requirements, 
                   additional_info, application_deadline, next_deadline, 
                   application_website, last_update_date 
            FROM asia_scholarships 
            WHERE country = %s
            """,
            (country_name,)
        )

        scholarships = []
        for row in cursor.fetchall():
            scholarship = {
                "scholarship_name": row[0],
                "country": row[1],
                "amount": row[2],
                "requirements": row[3],
                "additional_info": row[4],
                "application_deadline": row[5],
                "next_deadline": row[6].strftime('%Y-%m-%d') if row[6] else None,
                "application_website": row[7],
                "last_update_date": row[8].strftime('%Y-%m-%d %H:%M:%S') if row[8] else None
            }
            scholarships.append(scholarship)

        return scholarships

    except Exception as e:
        print(f"Error getting scholarships for {country_name}: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()


def check_and_add_db_columns():
    """
    Проверяет и добавляет отсутствующие столбцы, необходимые для хранения дедлайнов и вебсайтов.
    """
    conn = None
    cursor = None

    try:
        # Подключаемся к базе данных
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()

        # Получаем список существующих столбцов
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'asia_scholarships'
        """)
        existing_columns = [row[0] for row in cursor.fetchall()]

        # Проверяем и добавляем столбец application_deadline
        if 'application_deadline' not in existing_columns:
            cursor.execute("""
                ALTER TABLE asia_scholarships 
                ADD COLUMN application_deadline TEXT
            """)
            conn.commit()
            print("Добавлен столбец application_deadline")

        # Проверяем и добавляем столбец application_website
        if 'application_website' not in existing_columns:
            cursor.execute("""
                ALTER TABLE asia_scholarships 
                ADD COLUMN application_website TEXT
            """)
            conn.commit()
            print("Добавлен столбец application_website")

        # Проверяем и добавляем столбец next_deadline
        if 'next_deadline' not in existing_columns:
            cursor.execute("""
                ALTER TABLE asia_scholarships 
                ADD COLUMN next_deadline DATE
            """)
            conn.commit()
            print("Добавлен столбец next_deadline")

        return True
    except Exception as e:
        print(f"Ошибка при проверке/добавлении столбцов: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    print("Asian Scholarships Search & Update Tool")
    print("======================================")

    # Создаем таблицу, если она не существует
    create_scholarships_table_if_not_exists()

    while True:
        print("\nSelect an action:")
        print("1. Search for new scholarships in a specific country")
        print("2. Run weekly search for all countries")
        print("3. Update existing scholarships")
        print("4. Display scholarships for a country")
        print("5. Show upcoming scholarship deadlines")
        print("6. Exit")
        print("7. Run test scholarship save")
        print("8. Check table structure")

        choice = input("Enter your choice (1-8): ")

        if choice == "1":
            print("\nAvailable countries:")
            for i, country in enumerate(ASIAN_COUNTRIES, 1):
                print(f"{i}. {country}")

            try:
                country_idx = int(input("Enter country number: ")) - 1
                if 0 <= country_idx < len(ASIAN_COUNTRIES):
                    search_new_scholarships(ASIAN_COUNTRIES[country_idx])
                else:
                    print("Invalid country number")
            except ValueError:
                print("Please enter a valid number")

        elif choice == "2":
            run_weekly_search()

        elif choice == "3":
            run_biweekly_update()

        elif choice == "4":
            print("\nAvailable countries:")
            for i, country in enumerate(ASIAN_COUNTRIES, 1):
                print(f"{i}. {country}")

            try:
                country_idx = int(input("Enter country number: ")) - 1
                if 0 <= country_idx < len(ASIAN_COUNTRIES):
                    display_scholarships_for_country(ASIAN_COUNTRIES[country_idx])
                else:
                    print("Invalid country number")
            except ValueError:
                print("Please enter a valid number")

        elif choice == "5":
            # Показать ближайшие дедлайны стипендий
            days = input("Show deadlines for how many days ahead? (default 30): ")
            try:
                days_num = int(days) if days.strip() else 30
                deadlines = get_upcoming_scholarship_deadlines(days_num)

                if not deadlines:
                    print(f"No upcoming scholarship deadlines in the next {days_num} days")
                else:
                    print(f"\n=== Upcoming Scholarship Deadlines (next {days_num} days) ===\n")
                    for i, d in enumerate(deadlines, 1):
                        print(f"{i}. {d['scholarship_name']}")
                        print(f"   Country: {d['country']}")
                        print(f"   Amount: {d['amount']}")
                        print(f"   Deadline: {d['deadline_date']}")
                        print(f"   Days remaining: {d['days_remaining']}")
                        print(f"   Requirements: {d['requirements'][:100]}..." if len(
                            d['requirements']) > 100 else f"   Requirements: {d['requirements']}")
                        print()
            except ValueError:
                print("Please enter a valid number of days")

        elif choice == "6":
            print("Exiting program")
            break

        elif choice == "7":
            test_save_scholarship()

        elif choice == "8":
            check_and_add_db_columns()

        else:
            print("Invalid choice, please try again")