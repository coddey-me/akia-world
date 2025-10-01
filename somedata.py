import csv
import random

# Total number of samples to generate
NUM_SAMPLES = 20000

# Define the core components: intents, templates, domains, and entities
# This structure allows for easy expansion. Just add new items to the lists.

INTENTS = {
    'search_news': {
        'templates': [
            "latest news about {topic}", "breaking update on {topic}",
            "what's the latest on {topic}", "trending topics in {topic}",
            "give me news about {topic}", "updates regarding {topic}"
        ],
        'domains': ['general', 'politics', 'technology', 'finance', 'health', 'science']
    },
    'find_product': {
        'templates': [
            "buy {product} online", "affordable {product}", "best {product} under {price}",
            "where can I find {product}", "show me deals for {product}",
            "I need to purchase {product}"
        ],
        'domains': ['ecommerce', 'retail', 'technology', 'lifestyle', 'automotive']
    },
    'learn_concept': {
        'templates': [
            "explain {concept}", "what is {concept}", "define {concept} in simple terms",
            "tell me more about {concept}", "how does {concept} work?",
            "can you explain the theory of {concept}"
        ],
        'domains': ['education', 'science', 'technology', 'history', 'philosophy']
    },
    'compare': {
        'templates': [
            "compare {item_a} and {item_b}", "{item_a} vs {item_b} performance",
            "pros and cons of {item_a} vs {item_b}", "what's the difference between {item_a} and {item_b}",
            "which is better: {item_a} or {item_b}"
        ],
        'domains': ['research', 'technology', 'business', 'finance', 'lifestyle']
    },
    'how_to': {
        'templates': [
            "how to {action}", "steps to {action}", "how can I {action}",
            "tutorial on how to {action}", "guide for {action}"
        ],
        'domains': ['lifestyle', 'technology', 'education', 'home_improvement', 'health']
    },
    'get_weather': {
        'templates': [
            "what's the weather in {city}", "will it rain in {city} tomorrow",
            "forecast for {city} this weekend", "temperature in {city} right now"
        ],
        'domains': ['weather', 'travel']
    },
    'set_reminder': {
        'templates': [
            "remind me to {task} at {time}", "set a reminder for {task}",
            "don't let me forget to {task}"
        ],
        'domains': ['productivity', 'personal', 'business']
    },
    'get_directions': {
        'templates': [
            "directions to {location}", "how do I get to {location}",
            "navigate to {location}", "show me the route to {location}"
        ],
        'domains': ['travel', 'maps', 'local']
    },
    'task_request': {
        'templates': [
            "generate a {document} for me", "draft an email to {person}",
            "write a summary of this {content_type}", "create a {creative_work}"
        ],
        'domains': ['business', 'creative', 'productivity', 'marketing']
    },
    'find_tool': {
        'templates': [
            "find a tool for {purpose}", "best software for {purpose}",
            "apps to help with {purpose}", "what tool can I use to {purpose}"
        ],
        'domains': ['technology', 'business', 'finance', 'design', 'productivity']
    }
}

ENTITIES = {
    '{topic}': ['AI', 'climate change', 'the stock market', 'local elections', 'space exploration', 'quantum physics', 'mental health', 'renewable energy'],
    '{product}': ['a smartphone', 'running shoes', 'a gaming laptop', 'organic coffee', 'a 4k monitor', 'a used car', 'a designer handbag', 'a smart watch'],
    '{price}': ['$500', '$1000', 'a budget', 'a premium'],
    '{concept}': ['quantum computing', 'blockchain technology', 'natural selection', 'the industrial revolution', 'general relativity', 'supply and demand', 'machine learning'],
    '{item_a}': ['Tesla', 'Python', 'solar power', 'renting', 'iOS', 'AWS', 'a traditional bank'],
    '{item_b}': ['BYD', 'Java', 'wind power', 'buying', 'Android', 'Azure', 'a fintech app'],
    '{action}': ['cook pasta', 'build a website', 'train a neural network', 'change a flat tire', 'meditate properly', 'negotiate a salary', 'learn a new language'],
    '{city}': ['Nairobi', 'London', 'Tokyo', 'New York', 'Paris', 'Sydney', 'Cairo', 'Rio de Janeiro'],
    '{task}': ['call my mom', 'buy groceries', 'finish the report', 'take out the trash', 'pay the bills'],
    '{time}': ['5 PM', 'tomorrow morning', 'next Tuesday at 10am', 'an hour'],
    '{location}': ['the nearest hospital', 'the airport', '123 Main Street', 'the city center', 'the closest coffee shop'],
    '{document}': ['marketing campaign', 'business plan', 'quarterly report', 'project proposal'],
    '{person}': ['my boss', 'the HR department', 'a new client', 'the support team'],
    '{content_type}': ['article', 'book', 'research paper', 'meeting transcript'],
    '{creative_work}': ['poem about the sea', 'logo for my startup', 'short story', 'song'],
    '{purpose}': ['design logos', 'edit videos', 'manage personal finance', 'project management', 'learn a language', 'create invoices']
}

def generate_sentence(intent_data):
    """Generates a single sentence by filling a template with random entities."""
    template = random.choice(intent_data['templates'])
    
    # Find all placeholders in the template (e.g., {topic}, {product})
    placeholders = [word for word in template.split() if word.startswith('{') and word.endswith('}')]
    
    sentence = template
    for placeholder in placeholders:
        if placeholder in ENTITIES:
            entity_value = random.choice(ENTITIES[placeholder])
            sentence = sentence.replace(placeholder, entity_value, 1)
            
    return sentence

def main():
    """Main function to generate data and write to CSV."""
    with open('expanded_intent_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'intent', 'domain'])
        
        for _ in range(NUM_SAMPLES):
            # Choose a random intent
            intent_name = random.choice(list(INTENTS.keys()))
            intent_data = INTENTS[intent_name]
            
            # Choose a random domain for that intent
            domain = random.choice(intent_data['domains'])
            
            # Generate the text
            text = generate_sentence(intent_data)
            
            # Write the row to the CSV
            writer.writerow([text, intent_name, domain])
            
    print(f"Successfully generated {NUM_SAMPLES} samples in 'expanded_intent_data.csv'")

if __name__ == '__main__':
    main()
