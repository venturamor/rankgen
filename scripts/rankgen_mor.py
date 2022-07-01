from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers.generation_utils import GenerationMixin
import torch
import random
import json
import pandas as pd
import argparse
from functools import partial
from t5x_embeddings import T5XEmbeddingGenerator


# from transformers import BeamScorer, BeamSearchScorer
# https://huggingface.co/docs/transformers/internal/generation_utils
parser = argparse.ArgumentParser()
parser.add_argument('--retriever_model_path', default='t5x_conversion/t5_xl_all', type=str)
parser.add_argument('--cache_dir', default=None, type=str)

args = parser.parse_args()


def scorer_t5x(t5x_embedder, prefix, suffixes, prefix_vector=None):
    if prefix_vector is None:
        prefix_vector = t5x_embedder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_vectors = t5x_embedder.encode(suffixes, vectors_type="suffix")["embeddings"]
    similarities = torch.matmul(prefix_vector, suffix_vectors.t()).squeeze(dim=0)
    return similarities, prefix_vector, suffix_vectors


def scorer_dull(prefix, suffixes):
    # mauve metric
    # out = mauve.compute_mauve(p_tokens=p_toks, q_tokens=q_toks, device_id=1, max_text_length=1024)
    return 17


def generate_sequence_diff_decoding_methods(model, input_ids, seq_per_method, new_tokens_num):
    # greedy
    outputs_greedy = model.generate(input_ids, max_length=new_tokens_num, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True)

    # beam search
    # constraint beam search force_words_ids=True,
    outputs_bs = model.generate(input_ids, num_beams=3, num_return_sequences=seq_per_method,
                                max_length=new_tokens_num,
                                no_repeat_ngram_size=1, remove_invalid_values=True,
                                return_dict_in_generate=True)  # early_stopping=True,
    # sampling
    temperature = 1
    outputs_sampling = model.generate(input_ids, do_sample=True, top_k=0,
                                      max_length=new_tokens_num,
                                      num_return_sequences=seq_per_method,
                                      temperature=temperature,
                                      return_dict_in_generate=True)
    # top-p
    top_p = 0.9
    temperature = 1
    outputs_top_p = model.generate(input_ids, do_sample=True, output_scores=True,
                                   return_dict_in_generate=True,
                                   max_new_tokens=new_tokens_num, top_k=None, top_p=top_p,
                                   num_return_sequences=seq_per_method, temperature=temperature)

    # top-k sampling
    top_k = 100
    outputs_top_k = model.generate(input_ids, do_sample=True, top_k=top_k,
                                   max_length=new_tokens_num,
                                   num_return_sequences=seq_per_method, return_dict_in_generate=True)

    outputs_decoding_methods = [outputs_greedy, outputs_bs, outputs_sampling, outputs_top_p, outputs_top_k]
    return outputs_decoding_methods


def decode_outputs(tokenizer, outputs, method_name, prefix_len, print_flag=False):
    outputs_decoded = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
    if print_flag:
        for i, output in enumerate(outputs_decoded):
            print("{}, seq - {} : {} ".format(method_name, i, output[prefix_len:]))

    return outputs_decoded


def main(scorer):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    prefix = "I love to eat"
    prefix_len = len(prefix)

    input_ids = tokenizer(prefix, return_tensors="pt",
                          max_length=20, truncation=True, padding=True).input_ids
    new_tokens_num = 10 + input_ids.shape[-1]

    # generate
    NUM_BASIC_DECODING_METHODS = 5
    num_samples = 10
    seq_per_method = num_samples / NUM_BASIC_DECODING_METHODS
    if not seq_per_method.is_integer():
        raise ValueError('num_samples must divided by {}'.format(NUM_BASIC_DECODING_METHODS))
    seq_per_method = max(1, int(seq_per_method))

    outputs_decoding_methods = generate_sequence_diff_decoding_methods(model, input_ids, seq_per_method=seq_per_method,
                                                                       new_tokens_num=new_tokens_num)
    methods = ['greedy', 'beam search', 'sampling', 'top-p', 'top-k']
    # decode each outputs
    prefix_vector = None
    all_outputs_decoded = []
    df_decoding_methods = pd.DataFrame(columns=['generated_output', 'method', 'rankgen_score'])

    for method, output in zip(methods, outputs_decoding_methods):
        # decode
        outputs_decoded = decode_outputs(tokenizer, output, method, prefix_len, print_flag=True)
        print('\n')
        for output_decoded in outputs_decoded:
            all_outputs_decoded.append(output_decoded)

    # rank gen encoder score
    similarities, prefix_vector, suffix_vectors = scorer(prefix=prefix, suffixes=all_outputs_decoded, prefix_vector=prefix_vector)
    # similarities = scorer(prefix=prefix, suffixes=all_outputs_decoded)  # dull
    # similarities = [17] * 9
    # update df
    print('similarities: ', similarities)
    k_method = 0
    for i, output_decoded in enumerate(all_outputs_decoded):
        if (i + 1) % seq_per_method == 0:
            k_method += 1
        df_decoding_methods.loc[i] = [output_decoded, methods[k_method], similarities.cpu()[i]]
    # json_df = df_decoding_methods.to_json(orient='columns')
    # print to json file - works only in pycharm
    # parsed = json.loads(json_df)
    # print('save json file - outputs_decoded.json')
    # with open('outputs_decoded.json', 'w') as f:
    #     json.dump(parsed, f)  # parsed

    # print to csv file
    print('save csv file - outputs_decoded.csv')
    df_decoding_methods.to_csv('outputs_decoded.csv', index=False)


    # top_scores, top_indices = torch.topk(scores, k=beam_size)
    # beams = [all_outs[x] for x in top_indices]  # only track the top k beams

    # final_outputs.append([x["text"] for x in beams])
    # final_scores.append(top_scores)
    #
    # return final_outputs, final_scores


if __name__ == '__main__':
    t5x_embedder = T5XEmbeddingGenerator(model_path=args.retriever_model_path, cache_dir=args.cache_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # random.seed(49)
    # random.shuffle(data)
    #
    # random.seed(442)
    # random.shuffle(data)

    folder_name = f"token_bs_t5x"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer_fn = partial(scorer_t5x, t5x_embedder=t5x_embedder)
    # scorer_fn_dull = scorer_dull
    main(scorer=scorer_fn)

    # print("greedy:", outputs_greedy_decoded[0][len(text):])
    # # logits to probabilities
    # probs = nn.functional.softmax(outputs.scores[0], dim=-1)  # next_token_scores
    # sorted_probs = probs.sort(descending=True)
    # tokenizer.decode(outputs[-1][-1], skip_special_tokens=True)

    print('done')
