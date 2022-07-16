import string


def _remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def _lowercase(text):
    return text.lower()

def rerank(all_contexts, num_reranked_contexts, top_n_answers):
    # list where we add contexts in a new order
    reranked_contexts = []
    # keep track of already added contexts
    added_ctx = []
    for index, ctx in enumerate(all_contexts):
        text = _remove_punctuation(_lowercase(ctx.text)).split()  # list of tokens (strings)
        for ans in top_n_answers:
            flag = False
            ans_text = _remove_punctuation(_lowercase(ans.text)).split()  # list of tokens
            for i in range(0, len(text) - len(ans_text) + 1):
                if ans_text == text[i: i + len(ans_text)]:
                    reranked_contexts.append(ctx)
                    flag = True
                    added_ctx.append(index)
                    break
            # if the context contains each word of one of the top answers, then it's been already added to the ordered contexts
            if flag:
                break
    # add the remaining contexts (the ones which have not been added yet
    for index in range(0, len(all_contexts)):
        if index not in added_ctx:
            reranked_contexts.append(all_contexts[index])

    return reranked_contexts[:num_reranked_contexts]

