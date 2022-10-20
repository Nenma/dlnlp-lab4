import requests
import json

key = open('keys.txt', 'r').readline()
url = "https://nlp-translation.p.rapidapi.com/v1/translate"
headers = {
    "X-RapidAPI-Key": key,
    "X-RapidAPI-Host": "nlp-translation.p.rapidapi.com"
}


def get_sentences():
    # read original Romanian sentences
    ro_sentences = []
    with open('ro-sentences.txt', 'r') as sentences:
        for sentence in sentences:
            ro_sentences.append(sentence.strip())

    # read user English translations
    en_sentences = []
    with open('en-sentences.txt', 'r') as sentences:
        for sentence in sentences:
            en_sentences.append(sentence.strip())

    # get translation API English sentences
    machine_sentences = []
    for sentence in ro_sentences:
        querystring = {"text": sentence, "to": "en", "from": "ro"}
        response = requests.request(
            "GET", url, headers=headers, params=querystring)
        response = json.loads(response.text)
        machine_sentences.append(response['translated_text']['en'])

    return ro_sentences, en_sentences, machine_sentences


def get_machine_stats(en_sentences, machine_sentences):
    stats = []
    # for each of the 2 sentences
    for i in range(2):
        # search how many words occur at least twice
        dict = {}
        for word in en_sentences[i].split():
            dict[word] = dict.get(word, 0) + 1
        for word in machine_sentences[i].split():
            dict[word] = dict.get(word, 0) + 1

        correct = len([word for word in dict if dict[word] >= 2])
        precision = correct / len(machine_sentences[i].split())
        recall = correct / len(en_sentences[i].split())
        f_measure = 2 * precision * recall / (precision + recall)

        stats.append((precision, recall, f_measure))

    return stats


def get_bigram_stats(en_sentences, machine_sentences):
    bigram_stats = []
    for i in range(2):
        dict = {}
        en = en_sentences[i].split()
        mc = machine_sentences[i].split()

        for i in range(len(en) - 1):
            dict[(en[i], en[i+1])] = dict.get((en[i], en[i+1]), 0) + 1
        for i in range(len(mc) - 1):
            dict[(mc[i], mc[i+1])] = dict.get((mc[i], mc[i+1]), 0) + 1

        correct = len([bigram for bigram in dict if dict[bigram] >= 2])
        precision = correct / (len(mc) - 1)
        bigram_stats.append(precision)

    return bigram_stats


def get_trigram_stats(en_sentences, machine_sentences):
    trigram_stats = []
    for i in range(2):
        dict = {}
        en = en_sentences[i].split()
        mc = machine_sentences[i].split()

        for i in range(len(en) - 2):
            dict[(en[i], en[i+1], en[i+2])
                 ] = dict.get((en[i], en[i+1], en[i+2]), 0) + 1
        for i in range(len(mc) - 2):
            dict[(mc[i], mc[i+1], mc[i+2])
                 ] = dict.get((mc[i], mc[i+1], mc[i+2]), 0) + 1

        correct = len([bigram for bigram in dict if dict[bigram] >= 2])
        precision = correct / (len(mc) - 2)
        trigram_stats.append(precision)

    return trigram_stats


def get_tetragram_stats(en_sentences, machine_sentences):
    tetragram_stats = []
    for i in range(2):
        dict = {}
        en = en_sentences[i].split()
        mc = machine_sentences[i].split()

        for i in range(len(en) - 3):
            dict[(en[i], en[i+1], en[i+2], en[i+3])
                 ] = dict.get((en[i], en[i+1], en[i+2], en[i+3]), 0) + 1
        for i in range(len(mc) - 3):
            dict[(mc[i], mc[i+1], mc[i+2], mc[i+3])
                 ] = dict.get((mc[i], mc[i+1], mc[i+2], mc[i+3]), 0) + 1

        correct = len([bigram for bigram in dict if dict[bigram] >= 2])
        precision = correct / (len(mc) - 3)
        tetragram_stats.append(precision)

    return tetragram_stats


def get_bleu_score(unigram, en_sentences, machine_sentences):
    bigram = get_bigram_stats(en_sentences, machine_sentences)
    trigram = get_trigram_stats(en_sentences, machine_sentences)
    tetragram = get_tetragram_stats(en_sentences, machine_sentences)

    scores = []
    for i in range(2):
        en = en_sentences[i].split()
        mc = machine_sentences[i].split()

        score = unigram[i] * bigram[i] * trigram[i] * \
            tetragram[i] * min(1, len(mc)/len(en))

        scores.append(score)

    return scores


if __name__ == '__main__':
    ro, en, machine = get_sentences()
    stats = get_machine_stats(en, machine)
    scores = get_bleu_score([stats[0][0], stats[1][0]], en, machine)
    print(scores)
