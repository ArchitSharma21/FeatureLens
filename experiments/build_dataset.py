from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'

PAIRS: dict[str, list[tuple[str, str]]] = {
    'code': [
        ('Write a Python function that returns the larger of two integers.', 'In Python, create a function that chooses the maximum of two integer inputs.'),
        ('Explain what a recursive function does in a program.', 'Describe recursion in the context of a programming function.'),
        ('Show a JavaScript loop that prints numbers from one to five.', 'Give JavaScript code for iterating over the integers 1 through 5 and logging them.'),
        ('What is the purpose of a hash map in software?', 'Explain why programmers use hash maps or dictionaries.'),
        ('Write SQL that selects every row from a table named users.', 'Provide a SQL query to retrieve all records in the users table.'),
        ('How do I catch an exception in Python?', 'Show the Python syntax for handling an exception with try and except.'),
        ('Explain the difference between a list and a tuple in Python.', 'Compare Python lists with tuples.'),
        ('Write a function that checks whether a string is a palindrome.', 'Create code that tests if text reads the same forwards and backwards.'),
        ('What does git rebase do?', 'Describe the effect of rebasing a Git branch.'),
        ('How can a REST API return JSON?', 'Explain how a web API endpoint sends a JSON response.'),
        ('Write pseudocode for binary search.', 'Describe the binary-search algorithm as pseudocode.'),
        ('What is a class constructor used for?', 'Explain the role of a constructor when creating an object.'),
        ('Give a regular expression that matches a basic email-like string.', 'Show a regex pattern for a simple email address format.'),
        ('Why is time complexity important when comparing algorithms?', 'Explain why Big-O runtime matters in algorithm analysis.'),
        ('How do unit tests help a codebase?', 'Describe the purpose of automated unit testing in software.'),
        ('Write Python that sorts a list of dictionaries by an age key.', 'Show how to order Python dictionaries in a list according to their age field.'),
    ],
    'mathematics': [
        ('Find the derivative of x cubed plus two x.', 'Differentiate the function x^3 + 2x.'),
        ('Solve the equation 3x + 5 = 20.', 'Find x when three x plus five equals twenty.'),
        ('What is the integral of cosine x?', 'Compute an antiderivative of cos(x).'),
        ('Factor the polynomial x squared minus nine.', 'Rewrite x^2 - 9 as a product of factors.'),
        ('Calculate the mean of 4, 7, 9, and 10.', 'What is the arithmetic average of 4, 7, 9, 10?'),
        ('Explain the Pythagorean theorem.', 'State the relationship among the sides of a right triangle.'),
        ('What is the determinant of a two by two matrix?', 'Give the formula for the determinant of a 2x2 matrix.'),
        ('Simplify two to the third power times two squared.', 'Reduce 2^3 multiplied by 2^2 using exponent rules.'),
        ('Convert one half into a percentage.', 'Express 1/2 as a percent.'),
        ('What is the probability of heads on a fair coin?', 'For a fair coin, calculate the chance of flipping heads.'),
        ('Solve x squared equals sixteen.', 'Find the real values of x satisfying x^2 = 16.'),
        ('What is the slope between points (1,2) and (3,6)?', 'Calculate the gradient of the line through (1,2) and (3,6).'),
        ('Explain what a prime number is.', 'Define a prime integer.'),
        ('Compute the dot product of vectors (1,2) and (3,4).', 'Find (1,2) · (3,4).'),
        ('What is log base ten of one thousand?', 'Evaluate log_10(1000).'),
        ('A circle has radius three. What is its area?', 'Calculate the area of a circle whose radius is 3.'),
    ],
    'positive_sentiment': [
        ('The film was delightful, clever, and beautifully acted.', 'I found the movie charming, smart, and wonderfully performed.'),
        ('This restaurant served an excellent meal and the staff were kind.', 'The food was fantastic and the service team was genuinely friendly.'),
        ('I am very pleased with how reliable this laptop has been.', 'This laptop has worked dependably and I am extremely satisfied with it.'),
        ('The concert was energetic and unforgettable.', 'I had an amazing time at the lively, memorable concert.'),
        ('Her explanation was clear and genuinely helpful.', 'She explained the topic in a useful and easy-to-understand way.'),
        ('The hotel room was spotless and comfortable.', 'Our room was exceptionally clean and pleasant to stay in.'),
        ('I loved the book from beginning to end.', 'The novel kept me delighted all the way through.'),
        ('The new update makes the app much easier to use.', 'After the update, the application feels significantly more convenient.'),
        ('Their customer support solved my problem quickly.', 'Support handled the issue fast and left me very happy.'),
        ('The hike had spectacular views and perfect weather.', 'We enjoyed gorgeous scenery and wonderful conditions on the hike.'),
        ('The presentation was engaging and well organized.', 'I enjoyed the talk because it was compelling and structured clearly.'),
        ('This camera takes sharp photos and feels great to use.', 'The camera produces crisp images and has a satisfying design.'),
        ('The workshop exceeded my expectations.', 'I was impressed because the workshop was even better than I expected.'),
        ('Dinner turned out wonderfully and everyone enjoyed it.', 'The evening meal was a success and all of us had a great time.'),
        ('The museum exhibition was fascinating.', 'I thought the exhibition was deeply interesting and rewarding.'),
        ('The train journey was smooth and relaxing.', 'The trip by train felt easy, calm, and comfortable.'),
    ],
    'negative_sentiment': [
        ('The film was tedious, confusing, and badly acted.', 'I found the movie boring, incoherent, and poorly performed.'),
        ('This restaurant served cold food and the staff were rude.', 'The meal arrived cold and the service team behaved unpleasantly.'),
        ('I am disappointed by how unreliable this laptop has been.', 'This laptop keeps failing and I am extremely dissatisfied with it.'),
        ('The concert was chaotic and forgettable.', 'I had a miserable time at the disorganized, dull concert.'),
        ('Her explanation was unclear and unhelpful.', 'She explained the topic in a confusing way that did not help me.'),
        ('The hotel room was dirty and uncomfortable.', 'Our room was unpleasant, unclean, and difficult to relax in.'),
        ('I regretted reading the book.', 'The novel was a frustrating waste of my time.'),
        ('The new update makes the app harder to use.', 'After the update, the application feels significantly more awkward.'),
        ('Their customer support ignored my problem.', 'Support failed to resolve the issue and left me angry.'),
        ('The hike had awful weather and disappointing views.', 'We dealt with terrible conditions and underwhelming scenery on the hike.'),
        ('The presentation was dull and poorly organized.', 'I disliked the talk because it was tedious and structured badly.'),
        ('This camera takes blurry photos and feels cheap.', 'The camera produces soft images and has a flimsy design.'),
        ('The workshop fell far below my expectations.', 'I was disappointed because the workshop was much worse than I expected.'),
        ('Dinner went badly and nobody enjoyed it.', 'The evening meal was a failure and all of us had a poor time.'),
        ('The museum exhibition was painfully boring.', 'I thought the exhibition was dull and unrewarding.'),
        ('The train journey was stressful and uncomfortable.', 'The trip by train felt frustrating, noisy, and unpleasant.'),
    ],
    'german_language': [
        ('Guten Tag, wie geht es Ihnen heute?', 'Hallo, wie geht es dir heute?'),
        ('Ich möchte einen Tisch für zwei Personen reservieren.', 'Kann ich bitte einen Tisch für zwei reservieren?'),
        ('Die Bibliothek schließt um achtzehn Uhr.', 'Die Bibliothek ist bis achtzehn Uhr geöffnet.'),
        ('Dieses Buch erzählt eine sehr interessante Geschichte.', 'Die Geschichte in diesem Buch ist wirklich interessant.'),
        ('Wir fahren morgen früh mit dem Zug.', 'Morgen früh werden wir mit dem Zug reisen.'),
        ('Können Sie mir den Weg zum Bahnhof zeigen?', 'Wie komme ich zum Bahnhof?'),
        ('Ich lerne gern neue Sprachen.', 'Neue Sprachen zu lernen macht mir viel Spaß.'),
        ('Das Wetter am Meer ist wunderschön.', 'Am Meer ist das Wetter heute sehr schön.'),
        ('Sie bereitet das Abendessen in der Küche zu.', 'In der Küche kocht sie gerade das Abendessen.'),
        ('Wir haben am Wochenende ein Museum besucht.', 'Am Wochenende waren wir in einem Museum.'),
        ('Die Besprechung beginnt genau um neun Uhr.', 'Das Treffen fängt pünktlich um neun Uhr an.'),
        ('Mein Computer funktioniert nicht mehr richtig.', 'Mit meinem Computer stimmt etwas nicht; er läuft nicht richtig.'),
        ('Diese Stadt hat viele historische Gebäude.', 'In dieser Stadt gibt es zahlreiche historische Bauwerke.'),
        ('Wir müssen Brot und Gemüse kaufen.', 'Wir sollten noch Brot sowie Gemüse einkaufen.'),
        ('Ich suche eine Wohnung in der Nähe der Universität.', 'Ich möchte eine Unterkunft nahe der Universität finden.'),
        ('Vielen Dank für Ihre Hilfe.', 'Ich danke Ihnen herzlich für Ihre Unterstützung.'),
    ],
    'factual_entities': [
        ('Tell me about Marie Curie and her scientific work.', 'Summarize the scientific contributions of Marie Curie.'),
        ('What is notable about the city of Kyoto?', 'Give a short factual overview of Kyoto.'),
        ('Explain the role of the Nile in ancient Egypt.', 'Describe why the Nile mattered to ancient Egyptian civilization.'),
        ('Who was Ada Lovelace?', 'Provide a concise factual description of Ada Lovelace.'),
        ('What is Mount Everest?', 'Give basic factual information about Mount Everest.'),
        ('Describe the planet Saturn.', 'Provide several factual details about Saturn.'),
        ('What is the Great Barrier Reef?', 'Give an overview of the Great Barrier Reef.'),
        ('Tell me about Ludwig van Beethoven.', 'Summarize who Beethoven was and why he is remembered.'),
        ('What is the Amazon River?', 'Provide factual information about the Amazon River.'),
        ('Explain what the Rosetta Stone is.', 'Describe the Rosetta Stone and its historical importance.'),
        ('Tell me about the element gold.', 'Give a factual overview of the chemical element gold.'),
        ('What was the Renaissance?', 'Summarize the historical period known as the Renaissance.'),
        ('Describe the city of Hamburg.', 'Give several basic facts about Hamburg, Germany.'),
        ('Who was Alan Turing?', 'Provide a factual summary of Alan Turing’s life and work.'),
        ('What is the Pacific Ocean?', 'Give a concise factual description of the Pacific Ocean.'),
        ('Explain what DNA is.', 'Provide a factual description of DNA and its biological role.'),
    ],
    'uncertainty': [
        ('I have not given enough information to know which box contains the key.', 'From the details provided, the location of the key cannot be determined.'),
        ('The evidence is incomplete, so the cause remains uncertain.', 'There is insufficient evidence to identify the cause with confidence.'),
        ('I do not know which route they chose because the text never says.', 'The passage does not specify the route, so the answer is unknown.'),
        ('Without the missing measurement, the result cannot be calculated.', 'The calculation is underdetermined because a required value is absent.'),
        ('The source does not state when the event happened.', 'The event date is not provided by the available source.'),
        ('Several explanations fit the observations, so no single one is established.', 'The observations support multiple possibilities and do not settle on one explanation.'),
        ('There is not enough context to identify who the pronoun refers to.', 'The pronoun’s referent is ambiguous given the limited context.'),
        ('The sample is too small to draw a reliable conclusion.', 'A confident conclusion would be unjustified because the sample size is inadequate.'),
        ('I cannot verify that claim from the information available.', 'The available information is insufficient to confirm the claim.'),
        ('The instructions omit the final step, so the intended outcome is unclear.', 'Because the last instruction is missing, the desired result cannot be known.'),
        ('We have two plausible answers and no evidence that distinguishes them.', 'Both answers remain possible because there is no discriminating evidence.'),
        ('The report gives a range but not an exact value.', 'Only an interval is reported, so the precise value is unspecified.'),
        ('The image is too blurry to read the number confidently.', 'The number cannot be identified reliably because the image lacks clarity.'),
        ('No forecast was provided, so tomorrow’s value is unknown.', 'The future value cannot be stated because there is no forecast information.'),
        ('The experiment was not repeated, so the finding remains tentative.', 'Without replication, the result should be treated as uncertain.'),
        ('The text names several candidates but never identifies the winner.', 'A winner cannot be determined because the passage lists candidates without a result.'),
    ],
}

CAUSAL_TASKS = [
    {'concept': 'positive_sentiment', 'prompt': 'Sentiment: The meal was wonderful and the staff were kind. Label:', 'target': ' positive'},
    {'concept': 'positive_sentiment', 'prompt': 'Sentiment: I loved the thoughtful, beautifully written novel. Label:', 'target': ' positive'},
    {'concept': 'positive_sentiment', 'prompt': 'Sentiment: The service was fast and extremely helpful. Label:', 'target': ' positive'},
    {'concept': 'positive_sentiment', 'prompt': 'Sentiment: The journey was comfortable and enjoyable. Label:', 'target': ' positive'},
    {'concept': 'negative_sentiment', 'prompt': 'Sentiment: The meal was awful and the staff were rude. Label:', 'target': ' negative'},
    {'concept': 'negative_sentiment', 'prompt': 'Sentiment: I hated the tedious, badly written novel. Label:', 'target': ' negative'},
    {'concept': 'negative_sentiment', 'prompt': 'Sentiment: The service was slow and completely unhelpful. Label:', 'target': ' negative'},
    {'concept': 'negative_sentiment', 'prompt': 'Sentiment: The journey was stressful and unpleasant. Label:', 'target': ' negative'},
    {'concept': 'mathematics', 'prompt': '2 + 3 =', 'target': ' 5'},
    {'concept': 'mathematics', 'prompt': '7 - 4 =', 'target': ' 3'},
    {'concept': 'mathematics', 'prompt': '6 * 2 =', 'target': ' 12'},
    {'concept': 'mathematics', 'prompt': 'The square root of 81 is', 'target': ' 9'},
    {'concept': 'code', 'prompt': 'Python function declaration keyword:', 'target': ' def'},
    {'concept': 'code', 'prompt': 'In Python, an exception handler begins with the keyword', 'target': ' except'},
    {'concept': 'code', 'prompt': 'SQL keyword used to retrieve rows:', 'target': ' SELECT'},
    {'concept': 'code', 'prompt': 'A Python conditional branch commonly starts with', 'target': ' if'},
    {'concept': 'german_language', 'prompt': "Translate 'hello' into German:", 'target': ' hallo'},
    {'concept': 'german_language', 'prompt': "Translate 'thank you' into German:", 'target': ' danke'},
    {'concept': 'german_language', 'prompt': "Translate 'yes' into German:", 'target': ' ja'},
    {'concept': 'german_language', 'prompt': "Translate 'good evening' into German:", 'target': ' guten Abend'},
    {'concept': 'factual_entities', 'prompt': 'The scientist associated with radium research, Marie', 'target': ' Curie'},
    {'concept': 'factual_entities', 'prompt': 'The composer of the Fifth Symphony, Ludwig van', 'target': ' Beethoven'},
    {'concept': 'factual_entities', 'prompt': 'The computer scientist known for the Turing machine, Alan', 'target': ' Turing'},
    {'concept': 'factual_entities', 'prompt': 'The Japanese city famous for many historic temples,', 'target': ' Kyoto'},
    {'concept': 'uncertainty', 'prompt': 'The evidence is insufficient. The answer is', 'target': ' unknown'},
    {'concept': 'uncertainty', 'prompt': 'The passage never states the value, so it is', 'target': ' unknown'},
    {'concept': 'uncertainty', 'prompt': 'There is not enough information to determine the result. It remains', 'target': ' uncertain'},
    {'concept': 'uncertainty', 'prompt': 'No reliable conclusion can be drawn; the outcome is', 'target': ' unclear'},
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = DATA_DIR / 'prompts.jsonl'
    with prompt_path.open('w', encoding='utf-8') as handle:
        sample_id = 0
        for concept, pairs in PAIRS.items():
            for pair_num, (a, b) in enumerate(pairs):
                pair_id = f'{concept}-{pair_num:02d}'
                for variant, text in enumerate((a, b)):
                    record = {
                        'id': sample_id,
                        'concept': concept,
                        'pair_id': pair_id,
                        'variant': variant,
                        'text': text,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    sample_id += 1

    causal_path = DATA_DIR / 'causal_tasks.jsonl'
    with causal_path.open('w', encoding='utf-8') as handle:
        for idx, task in enumerate(CAUSAL_TASKS):
            record = {'id': idx, **task}
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'Wrote {sample_id} discovery prompts to {prompt_path}')
    print(f'Wrote {len(CAUSAL_TASKS)} causal tasks to {causal_path}')


if __name__ == '__main__':
    main()
