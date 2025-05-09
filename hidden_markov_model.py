from collections import defaultdict

# Raw training data
train_sentences = [
    "The_DET cat_NOUN sleeps_VERB",
    "A_DET dog_NOUN barks_VERB",
    "The_DET dog_NOUN sleeps_VERB",
    "My_DET dog_NOUN runs_VERB fast_ADV",
    "A_DET cat_NOUN meows_VERB loudly_ADV",
    "Your_DET cat_NOUN runs_VERB",
    "The_DET bird_NOUN sings_VERB sweetly_ADV",
    "A_DET bird_NOUN chirps_VERB"
]

# Initialize count dictionaries
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
tag_counts = defaultdict(int)

# Processing each sentence
for line in train_sentences:
    words_tags = line.strip().split()
    
    previous_tag = 'START'
    for wt in words_tags:
        word, tag = wt.rsplit("_", 1)
        
        # Count transitions
        transition_counts[previous_tag][tag] += 1
        previous_tag = tag

        # Count emissions
        emission_counts[tag][word.lower()] += 1

        # Count tag occurrences
        tag_counts[tag] += 1

# Compute transition probabilities
transition_probs = defaultdict(dict)
for prev_tag in transition_counts:
    total = sum(transition_counts[prev_tag].values())
    for curr_tag in transition_counts[prev_tag]:
        transition_probs[prev_tag][curr_tag] = transition_counts[prev_tag][curr_tag] / total

# Compute emission probabilities
emission_probs = defaultdict(dict)
for tag in emission_counts:
    total = sum(emission_counts[tag].values())
    for word in emission_counts[tag]:
        emission_probs[tag][word] = emission_counts[tag][word] / total

# Output: sample transition and emission probabilities
print("=== Transition Probabilities ===")
for prev_tag in transition_probs:
    for curr_tag in transition_probs[prev_tag]:
        print(f"P({curr_tag} | {prev_tag}) = {transition_probs[prev_tag][curr_tag]:.3f}")

print("\n=== Emission Probabilities ===")
for tag in emission_probs:
    for word in emission_probs[tag]:
        print(f"P({word} | {tag}) = {emission_probs[tag][word]:.3f}")


