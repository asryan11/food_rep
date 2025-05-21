import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizers
model = load_model('food_rep/seq2seq_model.h5')
with open('food_rep/encoder_tokenizer.pkl', 'rb') as f:
    encoder_tokenizer = pickle.load(f)
with open('food_rep/decoder_tokenizer.pkl', 'rb') as f:
    decoder_tokenizer = pickle.load(f)

reverse_decoder_word_index = {v: k for k, v in decoder_tokenizer.word_index.items()}
start_token = 'start'
end_token = 'end'
start_token_index = decoder_tokenizer.word_index[start_token]
end_token_index = decoder_tokenizer.word_index[end_token]

# Beam search decoder function
def decode_sequence_beam_search(input_seq, beam_width=3, max_decoder_seq_length=100):
    states_value = model.predict(input_seq)
    sequences = [[list(), 0.0, states_value]]  # (tokens, score, states)
    
    for _ in range(max_decoder_seq_length):
        all_candidates = []
        for seq, score, states in sequences:
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = start_token_index if not seq else seq[-1]
            output_tokens, h, c = model.predict([target_seq] + states)
            
            for i in np.argsort(output_tokens[0, -1])[-beam_width:]:
                candidate = [seq + [i], score - np.log(output_tokens[0, -1, i]), [h, c]]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]
        if all(seq[-1] == end_token_index for seq, _, _ in sequences):
            break

    best_seq = sequences[0][0]
    decoded_sentence = ' '.join([reverse_decoder_word_index.get(idx, '') 
                                 for idx in best_seq if idx != start_token_index and idx != end_token_index])
    return decoded_sentence.strip()

# Streamlit UI
st.title("Recipe Instruction Generator")

user_input = st.text_input("Enter ingredients (comma-separated):")

if st.button("Generate Recipe"):
    input_seq = encoder_tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=30, padding='post')
    prediction = decode_sequence_beam_search(input_seq, beam_width=5)
    st.subheader("Generated Recipe Instruction:")
    st.write(prediction)
