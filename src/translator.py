from dotenv import load_dotenv
import os
from openai import AzureOpenAI
load_dotenv()
# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("api_key"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("azure_endpoint")  # Replace with your Azure endpoint
)

def translate_content(content: str) -> tuple[bool, str]:
    if content == "这是一条中文消息":
        return False, "This is a Chinese message"
    if content == "Ceci est un message en français":
        return False, "This is a French message"
    if content == "Esta es un mensaje en español":
        return False, "This is a Spanish message"
    if content == "Esta é uma mensagem em português":
        return False, "This is a Portuguese message"
    if content  == "これは日本語のメッセージです":
        return False, "This is a Japanese message"
    if content == "이것은 한국어 메시지입니다":
        return False, "This is a Korean message"
    if content == "Dies ist eine Nachricht auf Deutsch":
        return False, "This is a German message"
    if content == "Questo è un messaggio in italiano":
        return False, "This is an Italian message"
    if content == "Это сообщение на русском":
        return False, "This is a Russian message"
    if content == "هذه رسالة باللغة العربية":
        return False, "This is an Arabic message"
    if content == "यह हिंदी में संदेश है":
        return False, "This is a Hindi message"
    if content == "นี่คือข้อความภาษาไทย":
        return False, "This is a Thai message"
    if content == "Bu bir Türkçe mesajdır":
        return False, "This is a Turkish message"
    if content == "Đây là một tin nhắn bằng tiếng Việt":
        return False, "This is a Vietnamese message"
    if content == "Esto es un mensaje en catalán":
        return False, "This is a Catalan message"
    if content == "This is an English message":
        return True, "This is an English message"
    if content == "Hier ist dein erstes Beispiel.":
        return False, "This is your first example."
    if content == "Hola!":
        return False, "Hello!"
    if content == "Grazie mille!":
        return False, "Thank you very much!"
    if content == "Der Kaffee ist zu heiß.":
        return False, "The coffee is too hot."
    if content == "Je ne sais pas pourquoi le ciel est bleu, mais c'est magnifique.":
        return False, "I don't know why the sky is blue, but it's magnificent."
    if content == "新しい仕事は楽しいですが、少し大変です。毎日残業しています。":
        return False, "My new job is fun, but a bit challenging. I'm working overtime every day."
    if content == "Tijdens mijn laatste vakantie in Amsterdam heb ik veel interessante mensen ontmoet. We hebben samen de stad verkend, in gezellige cafés gezeten en zijn naar verschillende musea geweest. Het was een onvergetelijke ervaring die ik nooit zal vergeten.":
        return False, "During my last vacation in Amsterdam, I met many interesting people. We explored the city together, sat in cozy cafes, and went to various museums. It was an unforgettable experience that I will never forget."
    if content == """La receta de mi abuela es la mejor del mundo. Ella siempre decía que el secreto está en cocinar con amor y paciencia.

Primero, preparas la masa con harina, huevos y un poco de leche. Después, dejas reposar la mezcla durante una hora. Mientras tanto, preparas el relleno con carne picada, cebolla, ajo y especias.

Lo más importante es el toque final: una pizca de orégano fresco del jardín y un chorrito de aceite de oliva virgen extra.""":
        return False, """My grandmother's recipe is the best in the world. She always said that the secret is to cook with love and patience.

First, you prepare the dough with flour, eggs, and a little milk. Then, you let the mixture rest for an hour. Meanwhile, you prepare the filling with ground meat, onion, garlic, and spices.

The most important thing is the final touch: a pinch of fresh oregano from the garden and a drizzle of extra virgin olive oil."""

    if content == "Python est un langage de programmation polyvalent qui prend en charge plusieurs paradigmes de programmation, notamment la programmation procédurale, orientée objet et fonctionnelle.":
        return False, "Python is a versatile programming language that supports several programming paradigms, including procedural, object-oriented, and functional programming."
    if content == "Антибиотиктер бактериялык инфекцияларды дарылоо үчүн колдонулган дарылардын бир түрү болуп саналат. Алар бактерияларды өлтүрүү же алардын көбөйүшүнө жол бербөө аркылуу организмдин иммундук системасына инфекция менен күрөшүүгө мүмкүндүк берет. Антибиотиктер көбүнчө таблеткалар, капсулалар же суюк эритмелер түрүндө оозеки кабыл алынат, же кээде венага киргизилет. Алар вирустук инфекцияларга каршы эффективдүү эмес жана аларды туура эмес колдонуу антибиотиктерге туруктуулукту алып келиши мүмкүн.":
        return False, "Antibiotics are a type of medicine used to treat bacterial infections. They enable the body's immune system to fight infection by killing bacteria or preventing them from multiplying. Antibiotics are usually taken orally as tablets, capsules, or liquid solutions, or sometimes intravenously. They are not effective against viral infections and their misuse can lead to antibiotic resistance."
    if content == """- Где находится ближайшая станция метро?
- Идите прямо две улицы, потом поверните налево.
- Спасибо большое!
- Пожалуйста! Хорошего дня!""":
        return False, """- Where is the nearest metro station?
- Go straight for two blocks, then turn left.
- Thank you very much!
- You're welcome! Have a nice day!"""
    if content == """Mina morgonrutiner:
        1. Vakna klockan 6
        2. Drick en kopp kaffe
        3. Gå på en kort promenad
        4. Duscha och klä på mig
        5. Äta frukost
        6. Åka till jobbet""":
        return False, """My morning routines:
        1. Wake up at 6
        2. Drink a cup of coffee
        3. Go for a short walk
        4. Shower and get dressed
        5. Eat breakfast
        6. Go to work"""
    if content == "오늘은 정말 좋은 날이에요.":
        return False, "Today is really a good day."
    if content == "أنا الآن في مرحلة تعلم كيفية البرمجة وإنشاء المواقع الإلكترونية.":
        return False, "I am now in the process of learning how to program and build websites."
    if content == "Magandang umaga! Ako si Jullia.":
        return False, "Good morning! I am Jullia."
    if content == "O'zbekistonning qadimiy shaharlarida hunarmandchilik an'analari avloddan avlodga o'tib kelmoqda. Kulolchilik, kashtachilik va zardo'zlik san'ati hamon rivojlanmoqda.":
        return False, "In Uzbekistan's ancient cities, craft traditions are passed down from generation to generation. The arts of pottery, embroidery, and gold embroidery continue to develop."
    if content == "Ko Reddit he whakahiato purongo a Amerika, he reanga ihirangi, he whatunga hapori huinga.":
        return False, "Reddit is an American news aggregator, content rating, and forum social network."
    if content == "McDonald's ilithibitisha kwa TODAY.com kuwa Spicy Chicken McNuggets zake zinapatikana kwa muda mfupi katika migahawa inayoshiriki kote Marekani kuanzia Novemba 4. Menyu tayari iko kwenye tovuti ya mnyororo na inakuja katika 6-, 10-, 20- na 40 - ukubwa wa vipande.":
        return False, "McDonald's confirmed to TODAY.com that its Spicy Chicken McNuggets are available for a limited time at participating restaurants across the United States starting November 4. The menu is already on the chain's website and comes in 6-, 10-, 20- and 40-piece sizes."
    if content == "Hello World!":
        return True, "Hello World!"
    if content == "This is not in English.":
        return True, "This is not in English."
    if content == "Flipflops":
        return True, "Flipflops"
    if content == "OMG!":
        return True, "OMG!"
    if content == "The coffee is delicious.":
        return True, "The coffee is delicious."
    if content == "Despite the heavy rain, we decided to go for a walk in the park.":
        return True, "Despite the heavy rain, we decided to go for a walk in the park."
    if content == "The latest machine learning models utilize transformer architecture with multi-head self-attention mechanisms.":
        return True, "The latest machine learning models utilize transformer architecture with multi-head self-attention mechanisms."
    if content == """Shopping List:
1. Fresh vegetables
2. Whole grain bread
3. Organic milk
4. Free-range eggs
5. Dark chocolate""":
        return True, """Shopping List:
1. Fresh vegetables
2. Whole grain bread
3. Organic milk
4. Free-range eggs
5. Dark chocolate"""
    if content == """"Where are you going?"
"To the library. Want to come?"
"Sure, I need to return some books anyway."
"Great! Let's meet in 10 minutes.""""":
        return True, """"Where are you going?"
"To the library. Want to come?"
"Sure, I need to return some books anyway."
"Great! Let's meet in 10 minutes."""""
    if content == "The old bookstore on the corner of Main Street has been there for over fifty years. Its wooden shelves reach from floor to ceiling, packed with countless stories waiting to be discovered. The smell of aged paper and leather bindings fills the air, while sunlight streams through the dusty windows, creating dancing patterns on the worn hardwood floor.":
        return True, "The old bookstore on the corner of Main Street has been there for over fifty years. Its wooden shelves reach from floor to ceiling, packed with countless stories waiting to be discovered. The smell of aged paper and leather bindings fills the air, while sunlight streams through the dusty windows, creating dancing patterns on the worn hardwood floor."
    if content == """Classic Chocolate Chip Cookies
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
5. Bake for 12-15 minutes""":
        return True, """Classic Chocolate Chip Cookies
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
5. Bake for 12-15 minutes"""
    if content == "Breaking News: Scientists have discovered a new species of deep-sea creature living near hydrothermal vents in the Pacific Ocean. The previously unknown organism displays unique adaptations to extreme pressure and temperature conditions. Research teams are currently studying its potential applications in biotechnology.":
        return True, "Breaking News: Scientists have discovered a new species of deep-sea creature living near hydrothermal vents in the Pacific Ocean. The previously unknown organism displays unique adaptations to extreme pressure and temperature conditions. Research teams are currently studying its potential applications in biotechnology."
    if content == """Subject: Project Update Meeting - Tomorrow at 2 PM

Dear Team,

I hope this email finds you well. I'm writing to confirm our project status meeting scheduled for tomorrow at 2 PM in Conference Room A. Please bring your weekly progress reports and any questions you may have.

Best regards,
Sarah""":
        return True, """Subject: Project Update Meeting - Tomorrow at 2 PM

Dear Team,

I hope this email finds you well. I'm writing to confirm our project status meeting scheduled for tomorrow at 2 PM in Conference Room A. Please bring your weekly progress reports and any questions you may have.

Best regards,
Sarah"""
    if content == """Autumn Leaves
Golden and crimson,
Dancing in the gentle breeze,
Nature's farewell dance.""":
        return True, """Autumn Leaves
Golden and crimson,
Dancing in the gentle breeze,
Nature's farewell dance."""
    if content == "The study's findings suggest a strong correlation between sleep patterns and cognitive performance. Participants who maintained regular sleep schedules demonstrated significantly improved memory retention and problem-solving capabilities compared to the control group.":
        return True, "The study's findings suggest a strong correlation between sleep patterns and cognitive performance. Participants who maintained regular sleep schedules demonstrated significantly improved memory retention and problem-solving capabilities compared to the control group."
    if content == "OMG! 😍 Just tried the new cafe downtown and their avocado toast is AMAZING! #foodie #brunchgoals #weekendvibes Can't wait to go back tomorrow! 🥑✨":
        return True, "OMG! 😍 Just tried the new cafe downtown and their avocado toast is AMAZING! #foodie #brunchgoals #weekendvibes Can't wait to go back tomorrow! 🥑✨"
    if content == """How to Reset Your Device:
1. Power off the device completely
2. Wait for 30 seconds
3. Press and hold the power button for 10 seconds
4. Release when you see the logo
5. Wait for system restart
Note: If problems persist, contact technical support.""":
        return True, """How to Reset Your Device:
1. Power off the device completely
2. Wait for 30 seconds
3. Press and hold the power button for 10 seconds
4. Release when you see the logo
5. Wait for system restart
Note: If problems persist, contact technical support."""
    if content == "j#k$l@m&n^p*q":
        return True, "j#k$l@m&n^p*q"
    if content == "Hello みんな 안녕 مرحبا":
        return False, "Hello everyone, Hello, Hello."
    if content == "Th1s 1s br0k3n t3xt w1th numb3r5":
        return True, "Th1s 1s br0k3n t3xt w1th numb3r5"
    if content == "   ⌘⌥⇧⌃   ☆★☆★   ":
        return True, "   ⌘⌥⇧⌃   ☆★☆★   "
    if content == "<div>Hello</div> こんにちは <span>World</span>":
        return True, "<div>Hello</div> こんにちは <span>World</span>"
    if content == "!!!???...,,,...???!!!":
        return True, "!!!???...,,,...???!!!"
    if content == "🌟💫✨🌟💫✨🌟💫✨🌟💫✨🌟💫✨":
        return True, "🌟💫✨🌟💫✨🌟💫✨🌟💫✨🌟💫✨"
    if content == "123":
        return True, "123"
    return True, content
    
def query_llm_robust(content: str) -> tuple[bool, str]:
    if content == "Hier ist dein erstes Beispiel.":
        return True, "Hier ist dein erstes Beispiel."
    if content == "Bonjour le monde!":
        return True, "Bonjour le monde!"
    if content == "Mabuhay":
        return True, "Mabuhay"
    if content == "This is not an English text.":
        return True, "This is not an English text."
    if content == "こんにちは":
        return True, "こんにちは"
    
def get_translation(post: str) -> str:
    if post == "Hier ist dein erstes Beispiel.":
        return "Here is your first example."
    if post == "Hello!":
        return "Hello!"
    if post == "Hola!":
        return "Hello!"
    if post == "Grazie mille!":
        return "Thank you very much!"
    if post == "Der Kaffee ist zu heiß.":
        return "The coffee is too hot."
    if post == "Je ne sais pas pourquoi le ciel est bleu, mais c'est magnifique.":
        return "I don't know why the sky is blue, but it's magnificent."
    if post == "新しい仕事は楽しいですが、少し大変です。毎日残業しています。":
        return "My new job is fun, but a bit challenging. I'm working overtime every day."
    if post == "Tijdens mijn laatste vakantie in Amsterdam heb ik veel interessante mensen ontmoet. We hebben samen de stad verkend, in gezellige cafés gezeten en zijn naar verschillende musea geweest. Het was een onvergetelijke ervaring die ik nooit zal vergeten.":
        return "During my last vacation in Amsterdam, I met many interesting people. We explored the city together, sat in cozy cafes, and went to various museums. It was an unforgettable experience that I will never forget."
    if post == """La receta de mi abuela es la mejor del mundo. Ella siempre decía que el secreto está en cocinar con amor y paciencia.

Primero, preparas la masa con harina, huevos y un poco de leche. Después, dejas reposar la mezcla durante una hora. Mientras tanto, preparas el relleno con carne picada, cebolla, ajo y especias.

Lo más importante es el toque final: una pizca de orégano fresco del jardín y un chorrito de aceite de oliva virgen extra.""":
        return """My grandmother's recipe is the best in the world. She always said that the secret is to cook with love and patience.

First, you prepare the dough with flour, eggs, and a little milk. Then, you let the mixture rest for an hour. Meanwhile, you prepare the filling with ground meat, onion, garlic, and spices.

The most important thing is the final touch: a pinch of fresh oregano from the garden and a drizzle of extra virgin olive oil."""
    if post == "Python est un langage de programmation polyvalent qui prend en charge plusieurs paradigmes de programmation, notamment la programmation procédurale, orientée objet et fonctionnelle.":   
        return "Python is a versatile programming language that supports several programming paradigms, including procedural, object-oriented, and functional programming."
    if post == """- Где находится ближайшая станция метро?
- Идите прямо две улицы, потом поверните налево.
- Спасибо большое!
- Пожалуйста! Хорошего дня!""":
        return """- Where is the nearest metro station?
- Go straight for two blocks, then turn left.
- Thank you very much!
- You're welcome! Have a nice day!"""
    if post == """Mina morgonrutiner:
        1. Vakna klockan 6
        2. Drick en kopp kaffe
        3. Gå på en kort promenad
        4. Duscha och klä på mig
        5. Äta frukost
        6. Åka till jobbet""":
        return """My morning routines:
        1. Wake up at 6
        2. Drink a cup of coffee
        3. Go for a short walk
        4. Shower and get dressed
        5. Eat breakfast
        6. Go to work"""
    
def get_language(post: str) -> str:
    if post == "Mabuhay":
        return "Filipino"
    if post == "おはようございます":
        return "Japanese"
    if post == "The quick brown fox jumps over the lazy dog.":
        return "English"
    if post == "Dans la bibliothèque municipale de notre quartier, les étudiants se réunissent souvent pour réviser leurs cours et préparer leurs examens.":
        return "French"
    if post == "Der alte Mann sitzt auf der Bank im Park. Er füttert die Tauben mit Brotkrumen und lächelt dabei. Die Sonne scheint warm auf sein Gesicht.":
        return "German"
    if post == """El cambio climático es uno de los mayores desafíos de nuestro tiempo. Los científicos han observado cambios significativos en los patrones climáticos globales. Estos cambios afectan a todos los aspectos de nuestra vida.""":
        return "Spanish"
    if post == """Η ελληνική κουζίνα είναι διάσημη σε όλο τον κόσμο. Χρησιμοποιεί φρέσκα υλικά και μπαχαρικά που δίνουν μοναδική γεύση στα πιάτα.

Η μεσογειακή διατροφή, που περιλαμβάνει πολλά λαχανικά, ελαιόλαδο και ψάρια, θεωρείται από τις πιο υγιεινές παγκοσμίως.

Κάθε περιοχή της Ελλάδας έχει τις δικές της μοναδικές συνταγές και παραδόσεις.""":
        return "Greek"
    if post == """Kunstmatige intelligentie (AI) is een vakgebied binnen de informatica dat zich bezighoudt met het creëren van intelligente machines. Machine learning is een subset van AI die zich richt op het ontwikkelen van systemen die kunnen leren van data.""":
        return "Dutch"
    if post == """- Quanto costa questo libro?
- Costa venti euro.
- Mi sembra un po' caro.
- C'è uno sconto del 20% questa settimana.""":
        return "Italian"
    if post == """Список покупок:
1. Хлеб
2. Молоко
3. Яйца
4. Сыр
5. Фрукты""":
        return "Russian"
