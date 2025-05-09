import math
from collections import defaultdict

# Training Data
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

# Initialize count dictionaries for transitions, emissions, and tags
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))
tag_counts = defaultdict(int)

# Processing each sentence in the training data
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

# Define small probability for unknown words
small_prob = 1e-6  # for unknown emissions

# Viterbi Algorithm Implementation
def viterbi(sentence_tokens):
    V = [{}]  # Viterbi matrix: V[i][tag] = (prob, prev_tag)
    path = {}

    # Initialization (t = 0)
    for tag in tag_counts:
        trans_p = transition_probs['START'].get(tag, 0)
        emit_p = emission_probs[tag].get(sentence_tokens[0], small_prob)
        if trans_p > 0:
            V[0][tag] = math.log(trans_p) + math.log(emit_p)
            path[tag] = ['START', tag]

    # Recursion (t = 1 to n-1)
    for t in range(1, len(sentence_tokens)):
        V.append({})
        new_path = {}
        word = sentence_tokens[t]

        for curr_tag in tag_counts:
            max_prob = float('-inf')
            best_prev = None
            emit_p = emission_probs[curr_tag].get(word, small_prob)
            if emit_p == 0:
                continue

            for prev_tag in V[t-1]:
                trans_p = transition_probs[prev_tag].get(curr_tag, 0)
                if trans_p == 0:
                    continue
                prob = V[t-1][prev_tag] + math.log(trans_p) + math.log(emit_p)
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_tag

            if best_prev is not None:
                V[t][curr_tag] = max_prob
                new_path[curr_tag] = path[best_prev] + [curr_tag]

        path = new_path

    # Termination
    if not V[-1]:
        return [], float('-inf')  # No valid path found

    final_tag = max(V[-1], key=lambda k: V[-1][k])
    return path[final_tag][1:], math.exp(V[-1][final_tag])

# TEST SENTENCES
test_sentences = [
    "The cat meows",
    "My dog barks loudly"
]

print("=== Viterbi Tagging Results ===\n")
for sentence in test_sentences:
    tokens = [w.lower() for w in sentence.split()]
    best_tags, prob = viterbi(tokens)
    print(f"Sentence: {sentence}")
    print("Tags   :", " --> ".join(best_tags))  # Changed arrow to '-->'
    print(f"Probability: {prob:.6f}\n")
