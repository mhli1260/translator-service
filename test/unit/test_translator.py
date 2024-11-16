from src.translator import translate_content, query_llm_robust, get_translation, client
from openai import AzureOpenAI
from unittest.mock import patch
from src.evaluation_utils import (
    eval_single_response_translation,
    eval_single_response_classification,
    evaluate
)

def test_chinese():
    """Test Chinese translation with semantic similarity check"""
    is_english, translated_content = translate_content("这是一条中文消息")
    expected_translation = "This is a message in Chinese."
    
    assert is_english == False
    similarity_score = eval_single_response_translation(expected_translation, translated_content)
    assert similarity_score >= 0.9, f"Translation similarity {similarity_score} is below threshold 0.9"

@patch.object(client.chat.completions, 'create')
def test_llm_normal_response(mock_create):
    """Test when LLM provides expected responses"""
    mock_create.side_effect = [
        type('Response', (), {
            'choices': [
                type('Choice', (), {'message': type('Message', (), {'content': 'French'})()})()
            ]
        })(),
        type('Response', (), {
            'choices': [
                type('Choice', (), {'message': type('Message', (), {'content': 'Hello world!'})()})()
            ]
        })()
    ]

    is_english, translation = translate_content("Bonjour le monde!")
    
    assert not is_english
    assert translation == "Hello world!"
    assert mock_create.call_count == 2 #API should be called twice: detection & translation


@patch.object(client.chat.completions, 'create')
def test_llm_gibberish_response(mock_create):
    """Test when LLM provides unexpected/gibberish responses"""
    mock_create.return_value = type('Response', (), {
        'choices': [
            type('Choice', (), {'message': type('Message', (), {'content': "I don't understand!!"})()})()
        ]
    })()

    is_english, translation = translate_content("Testing gibberish")
    
    # The response should be treated as english
    assert is_english
    assert translation == "Testing gibberish"
    assert mock_create.call_count == 1

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

    result = query_llm_robust("こ       ん          に2323     ちは")
    assert result == (True, "こ       ん          に2323     ちは")

# This function is to test the evaluation function
def test_translation_evaluation():
    """Test translation evaluation metric"""
    test_cases = [
        {
            "expected": "Hello world",
            "response": "Hello world",
            "min_score": 0.9  # identical text
        },
        {
            "expected": "Hello world",
            "response": "Hi world",
            "min_score": 0.7  # similar
        }
    ]
    
    for case in test_cases:
        score = eval_single_response_translation(case["expected"], case["response"])
        assert score >= case["min_score"]

# Function to test classification evaluation
def test_classification_evaluation():
    """Test classification evaluation metric"""
    test_cases = [
        {
            "expected": "English",
            "response": "english",
            "expected_score": 1.0
        },
        {
            "expected": "French",
            "response": "Spanish",
            "expected_score": 0.0
        }
    ]
    
    for case in test_cases:
        score = eval_single_response_classification(case["expected"], case["response"])
        assert score == case["expected_score"]

# Mock the query function to return expected responses
def test_evaluate_function():
    """Test the evaluate function"""
    test_dataset = [
        {"post": "Hello", "expected_answer": "Bonjour"},
        {"post": "Goodbye", "expected_answer": "Au revoir"}
    ]
    
    def mock_query(text):
        return "Bonjour" if text == "Hello" else "Au revoir"
    
    score = evaluate(mock_query, eval_single_response_classification, test_dataset)
    assert score == 100.0  

# Evaluation tests, using 5 items from the complete evaluation set to keep runtime reasonable.
def test_complete_evaluation_set__non_english():
    """Test the complete evaluation set (non-english posts) by calling the actual LLM."""
    test_subset = complete_eval_set[:5]

    for test_case in test_subset:
        is_english, translation = translate_content(test_case["post"])
        
        expected_is_english, expected_translation = test_case["expected_answer"]
        
        # Calculate similarity between expected and actual translations
        score = eval_single_response_translation(expected_translation, translation)
        
        assert is_english == expected_is_english, f"Language detection failed for: {test_case['post']}"
        assert score >= 0.7, f"Low translation quality for: {test_case['post']}\nExpected: {expected_translation}\nGot: {translation}"

def test_complete_evaluation_set__english():
    """Test the complete evaluation set (non-english posts) by calling the actual LLM."""
    test_subset = complete_eval_set[18:20]

    for test_case in test_subset:
        is_english, translation = translate_content(test_case["post"])
        
        expected_is_english, expected_translation = test_case["expected_answer"]
        
        # Calculate similarity between expected and actual translations
        score = eval_single_response_translation(expected_translation, translation)
        
        assert is_english == expected_is_english, f"Language detection failed for: {test_case['post']}"
        assert score >= 0.7, f"Low translation quality for: {test_case['post']}\nExpected: {expected_translation}\nGot: {translation}"

def test_complete_evaluation_set_malformed():
    """Test the complete evaluation set (malformed posts) by calling the actual LLM."""
    test_subset = complete_eval_set[36:]

    for test_case in test_subset:
        is_english, translation = translate_content(test_case["post"])
        
        expected_is_english, expected_translation = test_case["expected_answer"]
        
        # Calculate similarity between expected and actual translations        
        assert is_english == expected_is_english, f"Language detection failed for: {test_case['post']}"
        assert translation == expected_translation, f"Translation failed for: {test_case['post']}"

def test_evaluate_with_complete_dataset():
    """Test translation quality using the complete evaluation set"""
    
    # Convert complete_eval_set to the format expected by evaluate function.
    test_dataset = [
        {
            "post": item["post"],
            "expected_answer": item["expected_answer"][1] 
        }
        # Only testing the first 10 items to keep runtime reasonable.
        for item in complete_eval_set[:5]  
    ]
    
    def translation_func(text):
        _, translation = translate_content(text)
        return translation
    
    score = evaluate(
        query_fn=translation_func,
        eval_fn=eval_single_response_translation,
        dataset=test_dataset
    )
    
    print(f"\nEvaluation Results:")
    print(f"Total items tested: {len(test_dataset)}")
    print(f"Overall quality score: {score}%")
    assert score >= 70.0, f"Overall translation quality score {score}% is below threshold 70%"

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
        "expected_answer": (True, "Hello みんな 안녕 مرحبا")
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
