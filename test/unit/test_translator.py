from src.translator import translate_content, query_llm_robust, get_translation, client
from openai import AzureOpenAI
from mock import patch

def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

def test_llm_normal_response():
    pass

def test_llm_gibberish_response():
    pass

#tests for query_llm function written in colab:
def test_query_llm():
    for test in complete_eval_set:
        is_english, translated_content = translate_content(test["post"])
        assert is_english == test["expected_answer"][0]
        assert translated_content == test["expected_answer"][1]

#tests for get_translation function written in colab:
def test_get_translation():
    for test in translation_eval_set:
        translated_content = get_translation(test["post"])
        assert translated_content == test["expected_answer"]


# the following is for tests for querry_llm_robust function written in colab:
# All mock tests should return (True, post) as they trigger error checks.
@patch.object(client.chat.completions, 'create')
def test_unexpected_language(mocker):
    # we mock the model's response to return a random message
    mocker.return_value.choices[0].message.content = "I don't understand your request"
    """Test 1: When API returns an unexpected response"""
    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    print("Result", result)
    assert result == (True, "Hier ist dein erstes Beispiel.")

@patch.object(client.chat.completions, 'create')
def test_empty_response(mocker):
    """Test 2: When API returns empty response"""
    mocker.return_value.choices[0].message.content = ""

    result = query_llm_robust("Bonjour le monde!")
    assert result == (True, "Bonjour le monde!")

@patch.object(client.chat.completions, 'create')
def test_api_error(mocker):
    """Test 3:When API throws an error"""
    mocker.side_effect = Exception("API Error: Service Unavailable")

    result = query_llm_robust("Mabuhay")
    assert result == (True, "Mabuhay")

@patch.object(client.chat.completions, 'create')
def test_azure_filter(mocker):
    """Test 4: When content gets filtered by Azure's content management policy"""
    mocker.side_effect = Exception(
        "Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt "
        "triggering Azure OpenAI's content management policy.', 'code': 'content_filter'}}"
    )

    result = query_llm_robust("This is not an English text.")
    assert result == (True, "This is not an English text.")

@patch.object(client.chat.completions, 'create')
def test_malformed_response(mocker):
    """Test 5: When API returns malformed response"""
    class MockResponse:
        def __init__(self):
            self.choices = []
    mocker.return_value = MockResponse()

    result = query_llm_robust("こんにちは")
    assert result == (True, "こんにちは")


translation_eval_set = [
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": "Here is your first example."
    },
    {
        "post": "Hello!",
        "expected_answer": "Hello!"
    },
    {
        "post": "Hola!",
        "expected_answer": "Hello!"
    },
    {
        "post": "Grazie mille!",
        "expected_answer": "Thank you very much!"
    },
    {
        "post": "Der Kaffee ist zu heiß.",
        "expected_answer": "The coffee is too hot."
    },
    {
        "post": "Je ne sais pas pourquoi le ciel est bleu, mais c'est magnifique.",
        "expected_answer": "I don't know why the sky is blue, but it's magnificent."
    },
    {
        "post": "新しい仕事は楽しいですが、少し大変です。毎日残業しています。",
        "expected_answer": "My new job is fun, but a bit challenging. I'm working overtime every day."
    },
    {
        "post": "Tijdens mijn laatste vakantie in Amsterdam heb ik veel interessante mensen ontmoet. We hebben samen de stad verkend, in gezellige cafés gezeten en zijn naar verschillende musea geweest. Het was een onvergetelijke ervaring die ik nooit zal vergeten.",
        "expected_answer": "During my last vacation in Amsterdam, I met many interesting people. We explored the city together, sat in cozy cafes, and went to various museums. It was an unforgettable experience that I will never forget."
    },
    {
        "post": """La receta de mi abuela es la mejor del mundo. Ella siempre decía que el secreto está en cocinar con amor y paciencia.

Primero, preparas la masa con harina, huevos y un poco de leche. Después, dejas reposar la mezcla durante una hora. Mientras tanto, preparas el relleno con carne picada, cebolla, ajo y especias.

Lo más importante es el toque final: una pizca de orégano fresco del jardín y un chorrito de aceite de oliva virgen extra.""",
        "expected_answer": """My grandmother's recipe is the best in the world. She always said that the secret is to cook with love and patience.

First, you prepare the dough with flour, eggs, and a little milk. Then, you let the mixture rest for an hour. Meanwhile, you prepare the filling with ground meat, onion, garlic, and spices.

The most important thing is the final touch: a pinch of fresh oregano from the garden and a drizzle of extra virgin olive oil."""
    },
    {
        "post": "Python est un langage de programmation polyvalent qui prend en charge plusieurs paradigmes de programmation, notamment la programmation procédurale, orientée objet et fonctionnelle.",
        "expected_answer": "Python is a versatile programming language that supports several programming paradigms, including procedural, object-oriented, and functional programming."
    },
    {
        "post": """- Где находится ближайшая станция метро?
- Идите прямо две улицы, потом поверните налево.
- Спасибо большое!
- Пожалуйста! Хорошего дня!""",
        "expected_answer": """- Where is the nearest metro station?
- Go straight for two blocks, then turn left.
- Thank you very much!
- You're welcome! Have a nice day!"""
    },
    {
        "post": """Mina morgonrutiner:
        1. Vakna klockan 6
        2. Drick en kopp kaffe
        3. Gå på en kort promenad
        4. Duscha och klä på mig
        5. Äta frukost
        6. Åka till jobbet""",
        "expected_answer": """My morning routines:
        1. Wake up at 6
        2. Drink a cup of coffee
        3. Go for a short walk
        4. Shower and get dressed
        5. Eat breakfast
        6. Go to work"""
    }
]

complete_eval_set = [
    # Non-english posts
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": (False, "This is your first example.")
    },
    {
        "post": "Hola!",
        "expected_answer": (False, "Hello!")
    },
    {
        "post": "Grazie mille!",
        "expected_answer": (False, "Thank you very much!")
    },
    {
        "post": "Der Kaffee ist zu heiß.",
        "expected_answer": (False, "The coffee is too hot.")
    },
    {
        "post": "Je ne sais pas pourquoi le ciel est bleu, mais c'est magnifique.",
        "expected_answer": (False, "I don't know why the sky is blue, but it's magnificent.")
    },
    {
        "post": "新しい仕事は楽しいですが、少し大変です。毎日残業しています。",
        "expected_answer": (False, "My new job is fun, but a bit challenging. I'm working overtime every day.")
    },
    {
        "post": "Tijdens mijn laatste vakantie in Amsterdam heb ik veel interessante mensen ontmoet. We hebben samen de stad verkend, in gezellige cafés gezeten en zijn naar verschillende musea geweest. Het was een onvergetelijke ervaring die ik nooit zal vergeten.",
        "expected_answer": (False, "During my last vacation in Amsterdam, I met many interesting people. We explored the city together, sat in cozy cafes, and went to various museums. It was an unforgettable experience that I will never forget.")
    },
    {
        "post": """La receta de mi abuela es la mejor del mundo. Ella siempre decía que el secreto está en cocinar con amor y paciencia.

Primero, preparas la masa con harina, huevos y un poco de leche. Después, dejas reposar la mezcla durante una hora. Mientras tanto, preparas el relleno con carne picada, cebolla, ajo y especias.

Lo más importante es el toque final: una pizca de orégano fresco del jardín y un chorrito de aceite de oliva virgen extra.""",
        "expected_answer": (False, """My grandmother's recipe is the best in the world. She always said that the secret is to cook with love and patience.

First, you prepare the dough with flour, eggs, and a little milk. Then, you let the mixture rest for an hour. Meanwhile, you prepare the filling with ground meat, onion, garlic, and spices.

The most important thing is the final touch: a pinch of fresh oregano from the garden and a drizzle of extra virgin olive oil.""")
    },
    {
        "post": "Python est un langage de programmation polyvalent qui prend en charge plusieurs paradigmes de programmation, notamment la programmation procédurale, orientée objet et fonctionnelle.",
        "expected_answer": (False, "Python is a versatile programming language that supports several programming paradigms, including procedural, object-oriented, and functional programming.")
    },
    {
        "post": "Антибиотиктер бактериялык инфекцияларды дарылоо үчүн колдонулган дарылардын бир түрү болуп саналат. Алар бактерияларды өлтүрүү же алардын көбөйүшүнө жол бербөө аркылуу организмдин иммундук системасына инфекция менен күрөшүүгө мүмкүндүк берет. Антибиотиктер көбүнчө таблеткалар, капсулалар же суюк эритмелер түрүндө оозеки кабыл алынат, же кээде венага киргизилет. Алар вирустук инфекцияларга каршы эффективдүү эмес жана аларды туура эмес колдонуу антибиотиктерге туруктуулукту алып келиши мүмкүн.",
        "expected_answer": (False, "Antibiotics are a type of medicine used to treat bacterial infections. They enable the body's immune system to fight infection by killing bacteria or preventing them from multiplying. Antibiotics are usually taken orally as tablets, capsules, or liquid solutions, or sometimes intravenously. They are not effective against viral infections and their misuse can lead to antibiotic resistance.")
    },
    {
        "post": """- Где находится ближайшая станция метро?
- Идите прямо две улицы, потом поверните налево.
- Спасибо большое!
- Пожалуйста! Хорошего дня!""",
        "expected_answer": (False, """- Where is the nearest metro station?
- Go straight for two blocks, then turn left.
- Thank you very much!
- You're welcome! Have a nice day!""")
    },
    {
        "post": """Mina morgonrutiner:
        1. Vakna klockan 6
        2. Drick en kopp kaffe
        3. Gå på en kort promenad
        4. Duscha och klä på mig
        5. Äta frukost
        6. Åka till jobbet""",
        "expected_answer": (False, """My morning routines:
        1. Wake up at 6
        2. Drink a cup of coffee
        3. Go for a short walk
        4. Shower and get dressed
        5. Eat breakfast
        6. Go to work""")
    },
    {
      "post": "오늘은 정말 좋은 날이에요.",
      "expected_answer": (False, "Today is really a good day.")
    },
    {
      "post": "أنا الآن في مرحلة تعلم كيفية البرمجة وإنشاء المواقع الإلكترونية.",
      "expected_answer": (False, "I am now in the process of learning how to program and build websites.")
    },
    {
      "post": "Magandang umaga! Ako si Jullia.",
      "expected_answer": (False, "Good morning! I am Jullia.")
    },
    {
        "post": "O'zbekistonning qadimiy shaharlarida hunarmandchilik an'analari avloddan avlodga o'tib kelmoqda. Kulolchilik, kashtachilik va zardo'zlik san'ati hamon rivojlanmoqda.",
        "expected_answer": (False, "In Uzbekistan's ancient cities, craft traditions are passed down from generation to generation. The arts of pottery, embroidery, and gold embroidery continue to develop.")
    },
    {
      "post": "Ko Reddit he whakahiato purongo a Amerika, he reanga ihirangi, he whatunga hapori huinga.",
      "expected_answer": (False, "Reddit is an American news aggregator, content rating, and forum social network.")
    },
    {
      "post": "McDonald's ilithibitisha kwa TODAY.com kuwa Spicy Chicken McNuggets zake zinapatikana kwa muda mfupi katika migahawa inayoshiriki kote Marekani kuanzia Novemba 4. Menyu tayari iko kwenye tovuti ya mnyororo na inakuja katika 6-, 10-, 20- na 40 - ukubwa wa vipande.",
      "expected_answer": (False, "McDonald's confirmed to TODAY.com that its Spicy Chicken McNuggets are available for a limited time at participating restaurants across the United States starting November 4. The menu is already on the chain's website and comes in 6-, 10-, 20- and 40-piece sizes.")
    },

    # English posts
    {
        "post": "Hello world!",
        "expected_answer": (True, "Hello world!")
    },
    {
        "post": "This is not in English.",
        "expected_answer": (True, "This is not in English.")
    },
    {
        "post": "Flipflops.",
        "expected_answer": (True, "Flipflops.")
    },
    {
        "post": "OMG!",
        "expected_answer": (True, "OMG!")
    },
    {
        "post": "The coffee is delicious.",
        "expected_answer": (True, "The coffee is delicious.")
    },
    {
        "post": "Despite the heavy rain, we decided to go for a walk in the park.",
        "expected_answer": (True, "Despite the heavy rain, we decided to go for a walk in the park.")
    },
    {
        "post": "The latest machine learning models utilize transformer architecture with multi-head self-attention mechanisms.",
        "expected_answer": (True, "The latest machine learning models utilize transformer architecture with multi-head self-attention mechanisms.")
    },
    {
        "post": """Shopping List:
1. Fresh vegetables
2. Whole grain bread
3. Organic milk
4. Free-range eggs
5. Dark chocolate""",
        "expected_answer": (True, """Shopping List:
1. Fresh vegetables
2. Whole grain bread
3. Organic milk
4. Free-range eggs
5. Dark chocolate""")
    },
    {
        "post": """"Where are you going?"
"To the library. Want to come?"
"Sure, I need to return some books anyway."
"Great! Let's meet in 10 minutes.""""",
        "expected_answer": (True, """"Where are you going?"
"To the library. Want to come?"
"Sure, I need to return some books anyway."
"Great! Let's meet in 10 minutes.""""")
    },
    {
        "post": "The old bookstore on the corner of Main Street has been there for over fifty years. Its wooden shelves reach from floor to ceiling, packed with countless stories waiting to be discovered. The smell of aged paper and leather bindings fills the air, while sunlight streams through the dusty windows, creating dancing patterns on the worn hardwood floor.",
        "expected_answer": (True, "The old bookstore on the corner of Main Street has been there for over fifty years. Its wooden shelves reach from floor to ceiling, packed with countless stories waiting to be discovered. The smell of aged paper and leather bindings fills the air, while sunlight streams through the dusty windows, creating dancing patterns on the worn hardwood floor.")
    },
    {
        "post": """Classic Chocolate Chip Cookies
Ingredients:
- 2 cups all-purpose flour
- 1 cup butter, softened
- 3/4 cup sugar
- 2 large eggs
- 1 tsp vanilla extract
- 1 cup chocolate chips

Instructions:
1. Preheat oven to 350°F
2. Mix butter and sugar until creamy
3. Add eggs and vanilla
4. Stir in flour and chocolate chips
5. Bake for 12-15 minutes""",
        "expected_answer": (True, """Classic Chocolate Chip Cookies
Ingredients:
- 2 cups all-purpose flour
- 1 cup butter, softened
- 3/4 cup sugar
- 2 large eggs
- 1 tsp vanilla extract
- 1 cup chocolate chips

Instructions:
1. Preheat oven to 350°F
2. Mix butter and sugar until creamy
3. Add eggs and vanilla
4. Stir in flour and chocolate chips
5. Bake for 12-15 minutes""")
    },
    {
        "post": "Breaking News: Scientists have discovered a new species of deep-sea creature living near hydrothermal vents in the Pacific Ocean. The previously unknown organism displays unique adaptations to extreme pressure and temperature conditions. Research teams are currently studying its potential applications in biotechnology.",
        "expected_answer": (True, "Breaking News: Scientists have discovered a new species of deep-sea creature living near hydrothermal vents in the Pacific Ocean. The previously unknown organism displays unique adaptations to extreme pressure and temperature conditions. Research teams are currently studying its potential applications in biotechnology.")
    },
    {
        "post": """Subject: Project Update Meeting - Tomorrow at 2 PM

Dear Team,

I hope this email finds you well. I'm writing to confirm our project status meeting scheduled for tomorrow at 2 PM in Conference Room A. Please bring your weekly progress reports and any questions you may have.

Best regards,
Sarah""",
        "expected_answer": (True, """Subject: Project Update Meeting - Tomorrow at 2 PM

Dear Team,

I hope this email finds you well. I'm writing to confirm our project status meeting scheduled for tomorrow at 2 PM in Conference Room A. Please bring your weekly progress reports and any questions you may have.

Best regards,
Sarah""")
    },
    {
        "post": """Autumn Leaves
Golden and crimson,
Dancing in the gentle breeze,
Nature's farewell dance.""",
        "expected_answer": (True, """Autumn Leaves
Golden and crimson,
Dancing in the gentle breeze,
Nature's farewell dance.""")
    },
    {
        "post": "The study's findings suggest a strong correlation between sleep patterns and cognitive performance. Participants who maintained regular sleep schedules demonstrated significantly improved memory retention and problem-solving capabilities compared to the control group.",
        "expected_answer": (True, "The study's findings suggest a strong correlation between sleep patterns and cognitive performance. Participants who maintained regular sleep schedules demonstrated significantly improved memory retention and problem-solving capabilities compared to the control group.")
    },
    {
        "post": "OMG! 😍 Just tried the new cafe downtown and their avocado toast is AMAZING! #foodie #brunchgoals #weekendvibes Can't wait to go back tomorrow! 🥑✨",
        "expected_answer": (True, "OMG! 😍 Just tried the new cafe downtown and their avocado toast is AMAZING! #foodie #brunchgoals #weekendvibes Can't wait to go back tomorrow! 🥑✨")
    },
    {
        "post": """How to Reset Your Device:
1. Power off the device completely
2. Wait for 30 seconds
3. Press and hold the power button for 10 seconds
4. Release when you see the logo
5. Wait for system restart
Note: If problems persist, contact technical support.""",
        "expected_answer": (True, """How to Reset Your Device:
1. Power off the device completely
2. Wait for 30 seconds
3. Press and hold the power button for 10 seconds
4. Release when you see the logo
5. Wait for system restart
Note: If problems persist, contact technical support.""")
    },

    # Unintelligible or malformed posts
    {
        "post": "j#k$l@m&n^p*q",
        "expected_answer": (True, "j#k$l@m&n^p*q")
    },
    {
        "post": "Hello みんな 안녕 مرحبا",
        "expected_answer": (False, "Hello everyone, Hello, Hello.")
    },
    {
        "post": "Th1s 1s br0k3n t3xt w1th numb3r5",
        "expected_answer": (True, "Th1s 1s br0k3n t3xt w1th numb3r5")
    },
    {
        "post": "   ⌘⌥⇧⌃   ☆★☆★   ",
        "expected_answer": (True, "   ⌘⌥⇧⌃   ☆★☆★   ")
    },
    {
        "post": "<div>Hello</div> こんにちは <span>World</span>",
        "expected_answer": (True, "<div>Hello</div> こんにちは <span>World</span>")
    },
    {
        "post": "!!!???...,,,...???!!!",
        "expected_answer": (True, "!!!???...,,,...???!!!")
    },
    {
        "post": "🌟💫✨🌟💫✨🌟💫✨🌟💫✨🌟💫✨",
        "expected_answer": (True, "🌟💫✨🌟💫✨🌟💫✨🌟💫✨🌟💫✨")
    },
    {
        "post": "123",
        "expected_answer": (True, "123")
    }
]

language_detection_eval_set = [
        {
        "post": "Mabuhay!",
        "expected_answer": "Filipino"
    },
    {
        "post": "おはようございます",
        "expected_answer": "Japanese"
    },
    {
        "post": "The quick brown fox jumps over the lazy dog.",
        "expected_answer": "English"
    },
    {
        "post": "Dans la bibliothèque municipale de notre quartier, les étudiants se réunissent souvent pour réviser leurs cours et préparer leurs examens.",
        "expected_answer": "French"
    },
    {
        "post": "Der alte Mann sitzt auf der Bank im Park. Er füttert die Tauben mit Brotkrumen und lächelt dabei. Die Sonne scheint warm auf sein Gesicht.",
        "expected_answer": "German"
    },
    {
        "post": """El cambio climático es uno de los mayores desafíos de nuestro tiempo. Los científicos han observado cambios significativos en los patrones climáticos globales. Estos cambios afectan a todos los aspectos de nuestra vida.""",
        "expected_answer": "Spanish"
    },
    {
        "post": """Η ελληνική κουζίνα είναι διάσημη σε όλο τον κόσμο. Χρησιμοποιεί φρέσκα υλικά και μπαχαρικά που δίνουν μοναδική γεύση στα πιάτα.

Η μεσογειακή διατροφή, που περιλαμβάνει πολλά λαχανικά, ελαιόλαδο και ψάρια, θεωρείται από τις πιο υγιεινές παγκοσμίως.

Κάθε περιοχή της Ελλάδας έχει τις δικές της μοναδικές συνταγές και παραδόσεις.""",
        "expected_answer": "Greek"
    },
    {
        "post": """Kunstmatige intelligentie (AI) is een vakgebied binnen de informatica dat zich bezighoudt met het creëren van intelligente machines. Machine learning is een subset van AI die zich richt op het ontwikkelen van systemen die kunnen leren van data.""",
        "expected_answer": "Dutch"
    },
    {
        "post": """- Quanto costa questo libro?
- Costa venti euro.
- Mi sembra un po' caro.
- C'è uno sconto del 20% questa settimana.""",
        "expected_answer": "Italian"
    },
    {
        "post": """Список покупок:
1. Хлеб
2. Молоко
3. Яйца
4. Сыр
5. Фрукты""",
        "expected_answer": "Russian"
    }
]
