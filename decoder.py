
#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
import multi_bleu
import bleu

WEIGHT_DISTORTION = 0.5
WEIGHT_LANG_MODEL = 1
WEIGHT_TRANS_MODEL = 0.85
 
FUTURE_COST_LANG = 0.2
FUTURE_COST_TRANS = 0.4
 
 
 
class State:
 
 
    def __init__(self, phrase, words_used, last_index, predecessor, logprob, lm_state, features):
 
        self.phrase = phrase
 
        self.words_used = words_used
 
        self.last_index = last_index
 
        self.predecessor = predecessor
 
        self.logprob = logprob
 
        self.lm_state = lm_state

        self.features = features
 
    def create_new_state(self, phrase, lm_state, phrase_start, phrase_end, logprob, distortion_max, features):
 
        used = [False for _ in range(len(self.words_used))]
 
        for i in range(len(self.words_used)):
 
            if phrase_start <= i < phrase_end:
 
                if self.words_used[i]:
 
                    return False
 
                else:
 
                    used[i] = True
 
            else:
 
                used[i] = self.words_used[i]
 
        i = 0
 
        while i < len(used) and used[i]:
 
            i += 1
 
        if i  + distortion_max < phrase_end:
 
            return False
 
 
        return State(phrase, used, i, self, logprob, lm_state, features)
 
 
 
    def is_equal(self, state):
 
        if self.last_index != state.last_index:
 
            return False
 
        for i in range(len(self.words_used)):
 
            if self.words_used[i] != state.words_used[i]:
 
                return False
 
        if len(self.lm_state) != len(state.lm_state):
 
            return False
 
        if len(self.lm_state) >= 1:
 
            if self.lm_state[-1] != state.lm_state[-1]:
 
                return False
 
        if len(self.lm_state) >= 2:
 
            if self.lm_state[-2] != state.lm_state[-2]:
 
                return False
 
        return True
 
 
    def get_phrase_list(self):
 
        if self.predecessor != None:
 
            l = self.predecessor.get_phrase_list()
 
        else:
 
            l = []
 
        if self.phrase != None:
 
            l.append(self.phrase.english)
 
        return l
 
 
    def add_to_logprob(self, prob):
 
        self.logprob += prob
 
 
    def get_nbest(self,i,winner_feature):
 
        l = self.get_phrase_list()
 
        return str(i)+"|||"+" ".join(l)+"|||"+" ".join(winner_feature)

    def get_sentence(self):

        l = self.get_phrase_list()

        return " ".join(l)
 
 
    def print_state(self):
 
        s = ""
 
        for i in self.words_used:
 
            if i:
 
                s += "o"
 
            else:
 
                s += "."
 
        return "(%s %s (%d)): %f" % (self.lm_state, s, self.last_index, self.logprob)
 
 
def future_cost(words_used, cost_estimates):
 
    cost = 0
 
    s = 0
 
    t = 0
 
    while s < len(words_used):
 
        if not words_used[s]:
 
            while t < len(words_used) - 1 and not words_used[t]:
 
                t += 1
 
            cost += cost_estimates[s][t - 1]
 
            s = t
 
        s += 1
 
        t += 1
 
    return cost
 
def future_cost_of_phrase(phrase):
 
    cost = (phrase.logprob * FUTURE_COST_TRANS)
 
 
    english = phrase.english.split()
 
    #cost_array = [0 for i in english]
 
    for n in range(len(english)):
 
        #cost_array[n] = lm.table[(n,)]
 
        if (english[n],) in lm.table:
 
            cost += (lm.table[(english[n],)].logprob * FUTURE_COST_LANG)
 
    return cost
 
 
optparser = optparse.OptionParser()
#optparser.add_option("-i", "--input", dest="input", default="data/final_prj_data/dev/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-i", "--input", dest="input", default="data/final_prj_data/test/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
#optparser.add_option("-t", "--translation-model", dest="tm", default="data/final_prj_data/large/phrase-table/test-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
#optparser.add_option("-t", "--translation-model", dest="tm", default="data/final_prj_data/large/phrase-table/dev-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/final_prj_data/toy/phrase-table/phrase_table.out", help="File containing translation model (default=data/tm)")
#optparser.add_option("-l", "--language-model", dest="lm", default="data/final_prj_data/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/final_prj_data/lm/en.tiny.3g.arpa", help="File containing ARPA-format language model (default=data/lm)")


optparser.add_option("-a", "--ref0", dest="ref0", default="data/final_prj_data/test/all.cn-en.en0", help="reference")
optparser.add_option("-b", "--ref1", dest="ref1", default="data/final_prj_data/test/all.cn-en.en1", help="reference")
optparser.add_option("-c", "--ref2", dest="ref2", default="data/final_prj_data/test/all.cn-en.en2", help="reference")
optparser.add_option("-e", "--ref3", dest="ref3", default="data/final_prj_data/test/all.cn-en.en3", help="reference")


optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-d", "--distortion-factor", dest="d", default=6, type="int", help="Limit on how far from each other consecutive phrases can start (default=6)")
optparser.add_option("-s", "--stack-size", dest="s", default=2, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]
 
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

ref0 = [line.rstrip('\n') for line in open(opts.ref0)]
ref1 = [line.rstrip('\n') for line in open(opts.ref1)]
ref2 = [line.rstrip('\n') for line in open(opts.ref2)]
ref3 = [line.rstrip('\n') for line in open(opts.ref3)]

references = zip(ref0,ref1,ref2,ref3)

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
    if (word,) not in tm:
 
        #tm[(word,)] = [models.phrase(word, 0.0)]
        tm[(word,)] = [models.phrase(word, 0.0,['0','0','0','0'])]
 
sys.stderr.write("Decoding %s...\n" % (opts.input,))
source_num=0
Bleu_Score =0
Bleu_Score_mlti_ref=0

evaluator=multi_bleu.BLEUEvaluator()

for f in french:
 
 
    cost_estimates = [[0 for i in f] for j in f]
 
    for s in range(len(f)):
 
 
        if (f[s],) in tm:
 
            cost_estimates[s][s] = future_cost_of_phrase(tm[(f[s],)][0])
 
 
        for t in range(s + 1, len(f)):
 
            best_cost = -9999.0
 
            if f[s:t + 1] in tm:
 
                best_cost = future_cost_of_phrase(tm[f[s:t + 1]][0])
 
 
            for i in range(s + 1, t + 1):
 
                if f[i:t + 1] in tm:
 
                    this_cost = cost_estimates[s][i - 1] + future_cost_of_phrase(tm[f[i:t + 1]][0])
 
                    best_cost = max(best_cost, this_cost)
 
 
            cost_estimates[s][t] = best_cost
 
            #for i in range(len(f)):
 
            #    for j in range(len(f)):
 
            #        print cost_estimates[i][j],
 
            #    print
 
            #raw_input()
  
 
    # stacks is an array of dictionaries one longer than the sentance
 
    # the i-th dict of stacks represents the partial decodings of the sentance
 
    #   with i words matched
 
    stacks = [[] for _ in f] + [[]]
 
 
    initial_hypothesis = State(None, [False for _ in f], 0, None, 0, lm.begin(),['0','0','0','0'])
 
    stacks[0].append(initial_hypothesis)
 
  
 
    # iterates over the array of stacks, building them as it goes
 
    for i, stack in enumerate(stacks[:-1]):
 
        #raw_input()
 
        #print "Stack %d" % i
  
 
        # iterates over all the partial decodings in the stack
 
        # only considers a number of them specified in the option -s
 
        # considers them in the order of likelihood
 
        for state in sorted(iter(stack),key=lambda state: -state.logprob)[:opts.s]: # prune
 
 
            # find every phrase in the translation model (tm)
 
            # that can be found in the remaining sentance
 
            # to avoid:
 
            #    can discount phrases that begin inside the distortion model distance
 
            #    don't bother looking at words used in the current state
 
            start = max(state.last_index - opts.d, 0)
 
            end = min(state.last_index + opts.d - 1, len(f))
 
 
            # looking for phrases: iterating the start index inside bounds of distortion factor
 
            for s in xrange(start, end):
 
                # if the word was already used, dont try to use it again
 
                if state.words_used[s]:
 
                    continue
 
 
                # looking for phrases: iterating the end index
 
                for t in xrange(s + 1, end + 1):
 
                    # if we find an already used word, don't move the end index past it
 
                    if state.words_used[t - 1]:
 
                        break
 
 
                    # is the phrase in the tranlation model
 
                    if f[s:t] in tm:
 
                        for phrase in tm[f[s:t]]:
 
                            # creating a new state for every translation of the phrase
 
                            lm_state = state.lm_state
 
                            word_logprob = 0
 
                            for word in phrase.english.split():
 
                                (lm_state, w_logprob) = lm.score(lm_state, word)
 
                                word_logprob += w_logprob
 
                            distortion_logprob = -3 * abs(state.last_index - s - 1)
 
                            new_logprob = state.logprob 
 
                            new_logprob += WEIGHT_TRANS_MODEL * phrase.logprob
 
                            new_logprob += WEIGHT_LANG_MODEL * word_logprob
 
                            new_logprob += WEIGHT_DISTORTION * distortion_logprob
 
                            new_hypothesis = state.create_new_state(phrase, lm_state, s, t, new_logprob, opts.d ,phrase.features)
 
                            if not new_hypothesis:
 
                                continue
 
                            #print "%s + (%d, %d: %s) --> %s" % (state.print_state(), s, t, phrase.english, new_hypothesis.print_state())
 
 
                            #new_hypothesis.add_to_logprob(future_cost(new_hypothesis.words_used, cost_estimates))
 
 
                            position = i + t - s
 
                            inserted = False
 
                            for st in stacks[position]:
 
                                if st.is_equal(new_hypothesis):
 
                                    if new_hypothesis.logprob > st.logprob:
 
                                        stacks[position].remove(st)
 
                                        stacks[position].append(new_hypothesis)
 
                                    inserted = True
 
                                    break
 
                            if not inserted:
 
                                stacks[position].append(new_hypothesis)
 
 
 
 
    best_stack = []
 
    back = 0
 
    while best_stack == []:
 
        back -= 1
 
        best_stack = stacks[back]

    winner = max(iter(best_stack), key=lambda h: h.logprob)

    #winner_feature= winner.features
 
    def extract_english(h):
 
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

    if back != -1:
 
        print "ERROR",

    #print winner.get_sentence()

    winner_sentence=winner.get_sentence()

    stats = list(bleu.bleu_stats(winner_sentence, references[source_num][0]))
    bleu_smooth_score = bleu.smoothed_bleu(stats)

    winner_bleu=evaluator(winner_sentence, list(references[source_num]))

    Bleu_Score_mlti_ref = Bleu_Score_mlti_ref + winner_bleu
    Bleu_Score = Bleu_Score + bleu_smooth_score

    # source sentence number
    source_num = source_num + 1

    # This liine is for producing nbest
    '''
    for item in best_stack:
        print item.get_nbest(str(source_num), item.features)

    '''
print "Averaged bleu score multi ref:"
print Bleu_Score_mlti_ref/(source_num+1)

print "Averaged bleu score :"
print Bleu_Score/(source_num+1)



'''

    if opts.verbose:
 
        def extract_tm_logprob(h):
 
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
 
        tm_logprob = extract_tm_logprob(winner)
 
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
 
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))




'''