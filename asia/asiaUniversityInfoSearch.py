import os
import json
import time
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain

# Загружаем переменные окружения
load_dotenv()
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Инициализация клиентов API
client = OpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0.2)

# Список азиатских стран для валидации вузов
ASIAN_COUNTRIES = [
    "Japan", "South Korea", "China", "Singapore", "Malaysia",
    "Indonesia", "Thailand", "Vietnam", "Philippines", "India",
    "Taiwan", "Hong Kong", "UAE", "Qatar", "Saudi Arabia",
    "Israel", "Turkey", "Kazakhstan", "Uzbekistan", "Pakistan"
]

# Популярные азиатские вузы для быстрой проверки
COMMON_ASIAN_UNIVERSITIES = {
    "China": ["Tsinghua University", "Peking University", "Fudan University", "Shanghai Jiao Tong University"],
    "Japan": ["University of Tokyo", "Kyoto University", "Osaka University", "Tohoku University"],
    "South Korea": ["Seoul National University", "KAIST", "Korea University", "Yonsei University"],
    "Singapore": ["National University of Singapore", "Nanyang Technological University"],
    "Hong Kong": ["University of Hong Kong", "Hong Kong University of Science and Technology"],
    "India": ["Indian Institute of Technology Bombay", "Indian Institute of Science"]
}

def create_asia_university_info_prompt(university_name, major):
    """
    Создает специализированный промпт для поиска информации о азиатском вузе
    с форматом, соответствующим структуре базы данных.
    """
    return [
        {
            "role": "system",
            "content": """You are a specialized higher education consultant with expertise in Asian university admissions. 
            Your knowledge focuses on admission requirements, application processes, and program details for 
            universities across East Asia, Southeast Asia, South Asia, and the Middle East.
            
            You will provide detailed, accurate information about Asian universities, following a strict format 
            that matches the database structure. Always use the local currency of the university's country for financial information.
            """
        },
        {
            "role": "user",
            "content": f"""Provide detailed information about {university_name} for the {major} program.
            
            Return your response (no comments or explanations) in a strict JSON format with the following fields:
            
            {{
                "university_name": "Full official name of the university",
                "major": "{major}",
                "country": "Country where the university is located",
                "location": "City or specific region within the country",
                "acceptance_rate": "Acceptance rate percentage (number only, without % sign), if there is no specific data, it finds the general",
                "qs_ranking": "Current QS World University Ranking position (number only). if there is no qs rating, determine the world ranking based on other resources",
                "application_fee": "Application fee amount with the local currency of the university's country for the international students",
                "official_website": "Main university website URL",
                "tuition_fees": "Annual tuition fee amount with the local currency of the university's country",
                "food_housing_cost": "Monthly or annual living costs with the local currency of the university's country",
                "indirect_costs": "Any additional costs students should know about, in the local currency of the university's country",
                "application_url": "Direct URL to application portal or admissions page",
                "admission_requirements": "Key requirements for international students. List of required documents and letters  e.g., 'High school transcript', 'Recommendation letters (number)', 'Personal statement', 'Passport copy', 'Financial statement', provide detailed information",
                "typical_scores": {{
                    "TOEFL": "Minimum or typical TOEFL score",
                    "IELTS": "Minimum or typical IELTS score",
                    "SAT": "Minimum or typical SAT score",
                    "GRE": "Typical GRE score (if required)",
                    "GMAT": "Typical GMAT score (if required)",
                    "HSK": "Typical HSK level for Chinese universities (if required)",
                    "JLPT": "Typical JLPT level for Japanese universities (if required)",
                    "TOPIK": "Typical TOPIK level for Korean universities (if required)"
                }},
                "application_deadlines": "Specific date or period (e.g., 'November 1, 2024', 'October 1, 2024 - January 15, 2025', 'Year-round') for the next appointment"       
               
            }}
            
            Ensure all fields are filled with accurate data. For numeric fields (acceptance_rate, qs_ranking), 
            provide only numbers. For financial fields (application_fee, tuition_fees, food_housing_cost, indirect_costs), provide strings with the amount 
            and local currency code, e.g., '100000 KRW'.  and always use the local currency of the university's country (e.g., KRW for South Korea, JPY for Japan, CNY for China).
            
            For test scores, provide the minimum required or typical scores that successful applicants achieve.
            Only include relevant tests for the specific university and major - omit those that are not applicable.
            
            For financial information, always use the official local currency of the university's country:
            - South Korea: KRW (Korean Won)
            - Japan: JPY (Japanese Yen)
            - China: CNY (Chinese Yuan)
            - Singapore: SGD (Singapore Dollar)
            - India: INR (Indian Rupee)
            - Hong Kong: HKD (Hong Kong Dollar)
            - And so on for other Asian countries
            
            Your response must be a single, valid JSON object with all these fields.
            """
        }
    ]

def create_asia_university_verification_prompt():
    """Создает промпт для верификации информации об азиатском вузе."""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an Asian higher education expert specializing in validating university information.
        Your task is to analyze multiple sources about an Asian university and determine which is most accurate.
        
        Answer directly without explanations, following the exact format requested. 
        Do not provide any commentary or reasoning.
        
        Consider these aspects when evaluating information about Asian universities:
        1. Consistency with official university policies
        2. Recognition of unique aspects of Asian educational systems
        3. Accurate reflection of international student requirements
        4. Current and up-to-date information (especially for competitive programs)
        5. Proper terminology used for the specific country's educational system"""),

        HumanMessage(content="""
        Compare these responses about an Asian university's admission requirements:
        
        Response 1:
        {first_response}
        
        Response 2:
        {second_response}
        
        Return only "Response 1" or "Response 2" to indicate which is better, 
        or "Contradiction" if there are significant conflicts requiring further verification.
        Do not provide any explanation or reasoning.
        """)
    ])

def is_asian_university(university_name):
    """Проверяет, является ли университет азиатским."""
    # Проверка по списку популярных азиатских вузов
    for country, universities in COMMON_ASIAN_UNIVERSITIES.items():
        if any(uni.lower() in university_name.lower() for uni in universities):
            return True

    # Проверка на наличие названия азиатской страны в названии университета
    if any(country.lower() in university_name.lower() for country in ASIAN_COUNTRIES):
        return True

    # Дополнительная проверка с помощью API для неоднозначных случаев
    try:
        validation_prompt = [
            {
                "role": "system",
                "content": "You are a geography expert specializing in university locations. Answer directly without explanations or comments."
            },
            {
                "role": "user",
                "content": f"Is {university_name} located in Asia? Answer with only 'Yes' or 'No'. No explanation."
            }
        ]

        response = client.chat.completions.create(
            model="sonar-pro",
            messages=validation_prompt,
            temperature=0,
            max_tokens=10
        )

        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception as e:
        print(f"Ошибка при проверке расположения университета: {str(e)}")
        # В случае ошибки предполагаем, что это может быть азиатский вуз
        return True

def get_asia_university_info(university_name, major):
    """Получает информацию о азиатском университете."""
    # Проверяем, является ли университет азиатским
    if not is_asian_university(university_name):
        print(f"Предупреждение: {university_name} может не быть азиатским университетом")

    print(f"Поиск информации о {university_name} ({major}) в Азии...")

    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=create_asia_university_info_prompt(university_name, major),
            temperature=0,
            max_tokens=1500,
        )
        return response
    except Exception as e:
        print(f"Ошибка при получении информации о вузе: {str(e)}")
        # Пауза перед повторной попыткой
        time.sleep(2)
        try:
            # Повторная попытка с уменьшенным объемом запроса
            simplified_prompt = [
                {
                    "role": "system",
                    "content": "You are an Asian university admissions expert."
                },
                {
                    "role": "user",
                    "content": f"""Provide basic information about {university_name}'s {major} program as a 
                    JSON object with fields: university_name, major, country, location, tuition_fees, 
                    official_website, application_url."""
                }
            ]

            response = client.chat.completions.create(
                model="sonar-pro",
                messages=simplified_prompt,
                temperature=0,
                max_tokens=1000
            )

            return response
        except Exception as retry_error:
            print(f"Повторная ошибка: {str(retry_error)}")
            # Возвращаем базовую информацию об ошибке
            return {"error": f"Не удалось получить информацию о {university_name}: {str(e)}"}

def parse_university_data(json_text, university_name, major):
    """
    Парсит JSON с данными об университете в структуру для базы данных.

    Args:
        json_text (str): JSON-строка с данными
        university_name (str): Название университета (для случая ошибки)
        major (str): Специальность (для случая ошибки)

    Returns:
        dict: Структурированные данные для сохранения в БД
    """
    try:
        # Проверяем, не пустая ли строка
        if not json_text or json_text.strip() == "":
            raise ValueError("Получена пустая строка вместо JSON")

        # Пытаемся обнаружить начало JSON-объекта
        if not json_text.strip().startswith("{"):
            print(f"Предупреждение: строка не начинается с '{{'. Пытаемся найти JSON объект...")
            json_text = extract_json_from_text(json_text)

        # Парсим JSON
        data = json.loads(json_text)

        # Проверяем соответствие валюты стране
        country = data.get("country", "")

        # Словарь правильных валют для стран
        currency_map = {
            "South Korea": "KRW",
            "Japan": "JPY",
            "China": "CNY",
            "Singapore": "SGD",
            "India": "INR",
            "Hong Kong": "HKD",
            "Malaysia": "MYR",
            "Thailand": "THB",
            "Taiwan": "TWD",
            "Indonesia": "IDR",
            "Vietnam": "VND",
            "Philippines": "PHP"
        }

        # Проверяем валюту в финансовых полях
        expected_currency = currency_map.get(country)
        if expected_currency:
            for field in ["application_fee", "tuition_fees", "food_housing_cost"]:
                value = data.get(field, "")
                # Если поле содержит другую валюту, выводим предупреждение
                if value and expected_currency not in value and any(curr in value for curr in currency_map.values()):
                    wrong_currency = next((curr for curr in currency_map.values() if curr in value), None)
                    print(f"Предупреждение: для {field} используется неправильная валюта {wrong_currency} вместо {expected_currency}")
                    # Можно также исправить значение, но это требует дополнительной проверки

        # Создаем структуру для базы данных
        university_data = {
            "university_name": data.get("university_name", university_name),
            "major": data.get("major", major),
            "country": data.get("country", ""),
            "location": data.get("location", ""),
            "acceptance_rate": data.get("acceptance_rate", "0"),
            "qs_ranking": data.get("qs_ranking", ""),
            "application_fee": data.get("application_fee", ""),
            "official_website": data.get("official_website", ""),
            "tuition_fees": data.get("tuition_fees", ""),
            "food_housing_cost": data.get("food_housing_cost", ""),
            "indirect_costs": data.get("indirect_costs", ""),
            "application_url": data.get("application_url", ""),
            "last_update_date": datetime.now().isoformat(),
            "typical_scores": data.get("typical_scores", ""),
            "admission_requirements": data.get("admission_requirements", ""),
            "application_deadlines": data.get("application_deadlines", ""),
        }

        # university_data["typical_scores"] = data.get("typical_scores", {})
        # university_data["admission_requirements"] = data.get("admission_requirements", [])
        # university_data["application_deadlines"] = data.get("application_deadlines", {})

        return university_data
    except Exception as e:
        print(f"Ошибка при обработке данных: {str(e)}")
        print(f"Проблемная строка (первые 100 символов): {json_text}")

        # Создаем базовую структуру данных при ошибке
        return {
            "university_name": university_name,
            "major": major,
            "country": "",
            "location": "",
            "acceptance_rate": 0,
            "qs_ranking": 0,
            "application_fee": "",
            "official_website": "",
            "tuition_fees": "",
            "food_housing_cost": "",
            "indirect_costs": "",
            "application_url": "",
            "last_update_date": datetime.now().isoformat(),
            "parsing_error": f"Не удалось распарсить данные: {str(e)}"
        }

def parse_numeric_value(value):
    """
    Извлекает числовое значение из строки.

    Args:
        value (str): Строка с числом и возможным текстом

    Returns:
        int: Извлеченное число или 0
    """
    if not value:
        return 0

    # Если это уже число, просто возвращаем его
    if isinstance(value, (int, float)):
        return int(value)

    # Пытаемся извлечь число из строки
    import re
    numeric_match = re.search(r'(\d+)', str(value))
    if numeric_match:
        return int(numeric_match.group(1))

    return 0

def extract_json_from_text(text):
    """
    Извлекает JSON из текстового ответа API, даже если он обернут в markdown или другой текст.

    Args:
        text (str): Текст ответа, возможно содержащий JSON

    Returns:
        str: Извлеченный JSON или пустая строка при ошибке
    """
    import re

    # Проверяем наличие JSON в блоке markdown
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    json_match = re.search(json_pattern, text)

    if json_match:
        json_str = json_match.group(1).strip()
        try:
            # Проверяем, является ли извлеченная строка валидным JSON
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            print(f"Извлеченная строка не является валидным JSON: {json_str[:100]}...")

    # Попробуем найти начало и конец JSON в тексте
    try:
        start_idx = text.find('{')
        if start_idx >= 0:
            # Ищем соответствующую закрывающую скобку
            brackets = 0
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brackets += 1
                elif text[i] == '}':
                    brackets -= 1
                    if brackets == 0:
                        # Нашли конец JSON-объекта
                        json_str = text[start_idx:i+1]
                        try:
                            # Проверяем валидность
                            json.loads(json_str)
                            return json_str
                        except json.JSONDecodeError:
                            print(f"Найденный JSON-объект невалиден: {json_str}")
                        break
    except Exception as e:
        print(f"Ошибка при поиске JSON в тексте: {str(e)}")

    # Если все не удалось, возвращаем пустую строку
    print(f"Не удалось извлечь JSON из ответа. Первые 200 символов: {text}")
    return "{}"

def get_verified_asia_university_info(university_name, major):
    """Получает проверенную информацию об азиатском университете через двойную верификацию."""
    # Получаем первые два ответа
    print(f"Запрос первичной информации о {university_name}...")
    first_response = get_asia_university_info(university_name, major)

    # Небольшая пауза между запросами
    time.sleep(2)

    print(f"Запрос вторичной информации для верификации...")
    second_response = get_asia_university_info(university_name, major)

    # Извлекаем контент из ответов
    first_content = first_response.choices[0].message.content if hasattr(first_response, 'choices') else str(first_response)
    second_content = second_response.choices[0].message.content if hasattr(second_response, 'choices') else str(second_response)

    # Логируем первые символы ответов для отладки
    print(f"Первый ответ (первые 100 символов): {first_content}")
    print(f"Второй ответ (первые 100 символов): {second_content}")

    # Создаем цепочку для сравнения ответов
    comparison_chain = LLMChain(llm=llm, prompt=create_asia_university_verification_prompt())

    # Запускаем сравнение
    print("Верификация полученной информации...")
    comparison_result = comparison_chain.run(
        first_response=first_content,
        second_response=second_content
    )

    # Проверяем результат сравнения
    if "contradiction" in comparison_result.lower():
        print("Обнаружены противоречия, запрашиваем дополнительную информацию...")
        # Получаем третий ответ для разрешения противоречий
        time.sleep(2)
        third_response = get_asia_university_info(university_name, major)
        third_content = third_response.choices[0].message.content if hasattr(third_response, 'choices') else str(third_response)

        # Создаем шаблон для определения наиболее достоверного ответа
        verification_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert on Asian higher education systems.
            Analyze these three responses about an Asian university and determine which is most accurate.
            Consider specifically Asian educational contexts, terminology, and requirements."""),

            HumanMessage(content="""
            Response 1:
            {first_response}

            Response 2:
            {second_response}

            Response 3:
            {third_response}

            Which response provides the most accurate information about this Asian university?
            Return only "Response 1", "Response 2", or "Response 3".
            """)
        ])

        # Создаем цепочку для финальной верификации
        verification_chain = LLMChain(llm=llm, prompt=verification_template)

        # Запускаем верификацию
        verification_result = verification_chain.run(
            first_response=first_content,
            second_response=second_content,
            third_response=third_content
        )

        # Определяем, какой ответ выбран как наиболее достоверный
        chosen_content = ""
        if "response 1" in verification_result.lower():
            print("Выбран первый ответ как наиболее достоверный")
            chosen_content = first_content
        elif "response 2" in verification_result.lower():
            print("Выбран второй ответ как наиболее достоверный")
            chosen_content = second_content
        else:
            print("Выбран третий ответ как наиболее достоверный")
            chosen_content = third_content

        # Извлекаем JSON из выбранного ответа
        return extract_json_from_text(chosen_content)
    else:
        # Если противоречий нет, возвращаем более полный ответ
        chosen_content = ""
        if "response 2" in comparison_result.lower():
            print("Выбран второй ответ как более полный")
            chosen_content = second_content
        else:
            print("Выбран первый ответ как более полный")
            chosen_content = first_content

        # Извлекаем JSON из выбранного ответа
        return extract_json_from_text(chosen_content)

def get_asia_university_data_for_db(university_name, major, save_to_db=False):
    """
    Функция-обертка для использования в базе данных азиатских вузов.
    Возвращает данные в структуре, соответствующей таблице asia_universities.

    Args:
        university_name (str): Название университета
        major (str): Специальность
        save_to_db (bool): Флаг автоматического сохранения в БД (по умолчанию False)

    Returns:
        dict/str: Структурированные данные о вузе или JSON-строка
    """
    # Проверяем, является ли университет азиатским
    if not is_asian_university(university_name):
        print(f"Предупреждение: {university_name} не является азиатским университетом")

    # Получаем верифицированную информацию
    result = get_verified_asia_university_info(university_name, major)

    university_data = {}

    # Убедимся, что результат - строка
    if isinstance(result, str):
        # Парсим результат в структуру для базы данных
        university_data = parse_university_data(result, university_name, major)
    # На всякий случай, если результат не строка
    elif hasattr(result, 'choices') and hasattr(result.choices[0].message, 'content'):
        content = result.choices[0].message.content
        university_data = parse_university_data(content, university_name, major)
    else:
        # Создаем базовую структуру данных при ошибке
        university_data = {
            "university_name": university_name,
            "major": major,
            "country": "",
            "location": "",
            "acceptance_rate": 0,
            "qs_ranking": 0,
            "application_fee": "",
            "official_website": "",
            "tuition_fees": "",
            "food_housing_cost": "",
            "indirect_costs": "",
            "application_url": "",
            "last_update_date": datetime.now().isoformat(),
            "error": f"Неизвестный формат ответа: {type(result)}"
        }

    # Если нужно сохранить в БД
    if save_to_db:
        save_university_to_db(university_data)

    # Возвращаем данные в JSON-формате
    return json.dumps(university_data, ensure_ascii=False)

def save_university_to_db(university_data):
    """
    Сохраняет информацию об университете в базу данных asia_universities.

    Args:
        university_data (dict): Словарь с данными об университете

    Returns:
        bool: True если успешно, иначе False
    """
    # Загружаем настройки БД из переменных окружения
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")

    try:
        # Подключаемся к базе данных
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()

        # Преобразуем структурированные данные в JSON-строки
        typical_scores_json = json.dumps(university_data.get("typical_scores", {}), ensure_ascii=False)
        admission_requirements_json = json.dumps(university_data.get("admission_requirements", []), ensure_ascii=False)
        application_deadlines_json = json.dumps(university_data.get("application_deadlines", {}), ensure_ascii=False)


        # Преобразуем данные в строки для SQL-запроса
        university_name = university_data["university_name"]
        major = university_data["major"]

        # Проверяем, существует ли запись для этого университета и специальности
        cursor.execute(
            """
            SELECT uni_id FROM asia_universities 
            WHERE university_name = %s AND major = %s
            """,
            (university_name, major)
        )
        existing_record = cursor.fetchone()

        if existing_record:
            # Проверяем, существует ли столбец typical_scores в таблице
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='asia_universities' AND column_name='typical_scores'
            """)

            column_exists = cursor.fetchone() is not None

            # Формируем запрос в зависимости от наличия столбца
            if column_exists:
                update_query = """
                    UPDATE asia_universities SET 
                    country = %s,
                    location = %s,
                    acceptance_rate = %s,
                    qs_ranking = %s,
                    application_fee = %s,
                    official_website = %s,
                    tuition_fees = %s,
                    food_housing_cost = %s,
                    indirect_costs = %s,
                    application_url = %s,
                    last_update_date = %s,
                    typical_scores = %s,
                    admission_requirements = %s,
                    application_deadlines = %s
                    WHERE university_name = %s AND major = %s
                """
                query_params = (
                    university_data.get("country", ""),
                    university_data.get("location", ""),
                    university_data.get("acceptance_rate", 0),
                    university_data.get("qs_ranking", 0),
                    university_data.get("application_fee", ""),
                    university_data.get("official_website", ""),
                    university_data.get("tuition_fees", ""),
                    university_data.get("food_housing_cost", ""),
                    university_data.get("indirect_costs", ""),
                    university_data.get("application_url", ""),
                    datetime.now(),
                    # university_data.get("typical_scores", ""),
                    # university_data.get("admission_requirements", ""),
                    # university_data.get("application_deadlines", ""),
                    typical_scores_json,
                    admission_requirements_json,
                    application_deadlines_json,
                    university_name,
                    major
                )
                print("--------------------------------------------------")
                print("--------------------------------------------------")
                print(f"typical_scores_json: {typical_scores_json}")
                print(f"admission_requirements_json: {admission_requirements_json}")
                print(f"application_deadlines_json: {application_deadlines_json}")

            # Выполняем запрос
            cursor.execute(update_query, query_params)
            print(f"Обновлена запись для {university_name} ({major})...")
        else:
            # Создаем новую запись
            print(f"Добавление новой записи для {university_name} ({major})...")
            cursor.execute(
                """
                INSERT INTO asia_universities 
                (university_name, major, country, location, acceptance_rate, qs_ranking, 
                application_fee, official_website, tuition_fees, food_housing_cost, 
                indirect_costs, application_url, last_update_date, typical_scores, admission_requirements,
                application_deadlines)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
                """,
                (
                    university_name,
                    major,
                    university_data.get("country", ""),
                    university_data.get("location", ""),
                    university_data.get("acceptance_rate", 0),
                    university_data.get("qs_ranking", 0),
                    university_data.get("application_fee", ""),
                    university_data.get("official_website", ""),
                    university_data.get("tuition_fees", ""),
                    university_data.get("food_housing_cost", ""),
                    university_data.get("indirect_costs", ""),
                    university_data.get("application_url", ""),
                    datetime.now(),
                    typical_scores_json,
                    admission_requirements_json,
                    application_deadlines_json,
                )
            )
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            print(f"typical_scores_json: {typical_scores_json}")
            print(f"admission_requirements_json: {admission_requirements_json}")
            print(f"application_deadlines_json: {application_deadlines_json}")

        # Фиксируем изменения
        conn.commit()
        print("Данные успешно сохранены в базу данных!")
        return True

    except Exception as e:
        print(f"Ошибка при сохранении данных в БД: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
        return False

    finally:
        # Закрываем соединение
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

def get_university_majors(university_name):
    """
    Получает список всех специальностей, предлагаемых университетом.
    
    Args:
        university_name (str): Название университета
        
    Returns:
        list: Список названий специальностей
    """
    print(f"Поиск всех специальностей в {university_name}...")
    
    majors_prompt = [
        {
            "role": "system",
            "content": """You are an educational consultant specializing in Asian university programs. 
            Your task is to provide a complete list of majors (only bachelor program). The courses must be taught in English.
            
            Follow the instructions exactly. Do not provide any explanations, notes, or comments.
            Return only the requested JSON array format. No introductory text, no conclusions."""
        },
        {
            "role": "user",
            "content": f"""Provide a complete list of all academic programs (majors) available at {university_name}.
            
            Include undergraduate, graduate, and doctoral programs.
            Return the result as a JSON array containing the major names in English.
            
            Required response format:
            [
                "Computer Science ",
                "Business Administration",
                "Mechanical Engineering",
                ...and so on
            ]
            
            Important: Return ONLY the JSON array. No explanations, no text before or after the array, 
            no markdown formatting. The response must start with '[' and end with ']'."""
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=majors_prompt,
            temperature=0,
            max_tokens=1500,
        )
        
        content = response.choices[0].message.content
        
        # Извлекаем JSON-массив из ответа
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        
        if json_match:
            majors_json = json_match.group(0)
            try:
                majors_list = json.loads(majors_json)
                print(f"Найдено {len(majors_list)} специальностей")
                print(majors_list)
                majors_list = majors_list[:10]
                print(f"Найдено {len(majors_list)} специальностей")
                print(majors_list)
                return majors_list
            except json.JSONDecodeError as e:
                print(f"Ошибка при разборе JSON со специальностями: {str(e)}")
                return []
        else:
            print("Не удалось извлечь список специальностей из ответа")
            
            # Попытка найти списки специальностей в любом формате
            lines = content.strip().split('\n')
            potential_majors = []
            
            for line in lines:
                # Ищем строки, которые похожи на названия специальностей
                line = line.strip()
                if line and not line.startswith('{') and not line.startswith('[') and not line.startswith('"[') and len(line) > 3:
                    # Удаляем маркеры списков, кавычки и т.д.
                    clean_line = re.sub(r'^["\'\-\*\d\.\s]+', '', line).strip()
                    if clean_line:
                        potential_majors.append(clean_line)
            
            if potential_majors:
                print(f"Извлечено {len(potential_majors)} потенциальных специальностей из текста")
                return potential_majors
            
            return []
            
    except Exception as e:
        print(f"Ошибка при получении списка специальностей: {str(e)}")
        return []

def populate_university_with_all_majors(university_name, save_to_db=True):
    """
    Заполняет базу данных информацией обо всех специальностях университета.
    
    Args:
        university_name (str): Название университета
        save_to_db (bool): Флаг сохранения в БД (по умолчанию True)
        
    Returns:
        dict: Результаты операции с информацией об успешно добавленных специальностях
    """
    print(f"Начинаем заполнение БД информацией о всех специальностях в {university_name}...")
    
    # Получаем список всех специальностей
    majors = get_university_majors(university_name)
    
    if not majors:
        print("Не удалось получить список специальностей. Операция прервана.")
        return {"error": "Не удалось получить список специальностей", "university": university_name}
    
    results = {
        "university": university_name,
        "total_majors": len(majors),
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "failures": []
    }
    
    print(f"Найдено {len(majors)} специальностей. Начинаем обработку...")
    
    # Обрабатываем каждую специальность
    for i, major in enumerate(majors):
        print(f"\nОбработка {i+1}/{len(majors)}: {major}")
        try:
            # Получаем и сохраняем информацию о специальности
            result = get_asia_university_data_for_db(university_name, major, save_to_db)
            
            # Проверяем успешность операции
            if "error" in result.lower():
                print(f"Ошибка при обработке специальности {major}: {result}")
                results["failed"] += 1
                results["failures"].append({"major": major, "error": result})
            else:
                print(f"Успешно обработана специальность: {major}")
                results["successful"] += 1
            
            results["processed"] += 1
            
            # Делаем паузу между запросами, чтобы не перегружать API
            if i < len(majors) - 1:
                delay = 3  # 3 секунды между запросами
                print(f"Пауза {delay} секунд перед следующим запросом...")
                time.sleep(delay)
                
        except Exception as e:
            print(f"Непредвиденная ошибка при обработке специальности {major}: {str(e)}")
            results["failed"] += 1
            results["failures"].append({"major": major, "error": str(e)})
            results["processed"] += 1
    
    print(f"\nОбработка завершена. Успешно: {results['successful']}, Ошибок: {results['failed']}")
    return results

def clean_and_group_majors(majors_list):
    """
    Очищает и группирует список специальностей для улучшения качества данных.
    
    Args:
        majors_list (list): Исходный список специальностей
        
    Returns:
        list: Очищенный и сгруппированный список специальностей
    """
    # Очищаем названия специальностей
    cleaned_majors = []
    for major in majors_list:
        # Удаляем лишние символы и нормализуем строку
        if isinstance(major, str):
            major = major.strip()
            # Удаляем номера, маркеры и т.д.
            major = re.sub(r'^[\d\.\-\*]+\s*', '', major)
            # Удаляем скобки с пояснениями типа "Bachelor's", "Master's" и т.д.
            major = re.sub(r'\s*\([^)]*\)', '', major)
            major = re.sub(r'\s*\[[^\]]*\]', '', major)
            
            # Проверяем, что строка не пустая и имеет разумную длину
            if major and 3 < len(major) < 100:
                cleaned_majors.append(major)
    
    # Группируем похожие специальности
    from collections import defaultdict
    grouped_majors = defaultdict(list)
    
    for major in cleaned_majors:
        # Создаем ключ по первым 3-4 словам (или меньше, если слов меньше)
        words = major.split()
        key_words = words[:min(4, len(words))]
        key = ' '.join(key_words).lower()
        
        grouped_majors[key].append(major)
    
    # Выбираем по одному представителю из каждой группы
    result_majors = []
    for group in grouped_majors.values():
        # Выбираем самое короткое название или то, которое не содержит уровень обучения
        selected = min(group, key=len)
        result_majors.append(selected)
    
    return result_majors

if __name__ == "__main__":
    print("=== Система поиска информации об азиатских вузах ===")
    
    # Выбор режима работы
    print("\nВыберите режим работы:")
    print("1. Информация по одной специальности")
    print("2. Заполнить БД всеми специальностями университета")
    
    mode = input("\nВведите номер режима (1 или 2): ")
    
    if mode == "1":
        # Оригинальный функционал
        university = input("Введите название азиатского университета: ")
        major = input("Введите специальность: ")
        
        save_option = input("Сохранить результаты в базу данных? (y/n): ").lower()
        save_to_db = save_option in ('y', 'yes', 'да')
        
        result = get_asia_university_data_for_db(university, major, save_to_db)
        
        # Выводим результат
        print("\n=== Результат поиска ===")
        
        # Пытаемся отформатировать JSON для лучшей читаемости
        try:
            parsed = json.loads(result)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except:
            print(result)
    
    elif mode == "2":
        # Новый функционал - заполнение БД всеми специальностями
        university = input("Введите название азиатского университета: ")
        
        save_option = input("Сохранить результаты в базу данных? (y/n): ").lower()
        save_to_db = save_option in ('y', 'yes', 'да')
        
        print(f"\nНачинаем заполнение базы данных информацией о {university}...")
        results = populate_university_with_all_majors(university, save_to_db)
        
        # Выводим общую статистику результатов
        print("\n=== Статистика заполнения БД ===")
        print(f"Университет: {results['university']}")
        print(f"Всего найдено специальностей: {results['total_majors']}")
        print(f"Обработано: {results['processed']}")
        print(f"Успешно: {results['successful']}")
        print(f"С ошибками: {results['failed']}")
        
        if results['failed'] > 0:
            print("\nСписок ошибок:")
            for failure in results['failures']:
                print(f"- {failure['major']}: {failure['error']}")
    
    else:
        print("Неверный режим работы. Пожалуйста, выберите 1 или 2.")