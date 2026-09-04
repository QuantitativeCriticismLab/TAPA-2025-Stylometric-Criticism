import re
import numpy as np

from readers import LatinReader
from typing import List
import string
from spacy.attrs import ORTH


class Features(object):
    def __init__(
        self, reader, fileid=None, norm=False, annotations=False, verbose=False
    ):
        self.reader = reader
        try:
            self.reader.model.add_pipe("sentencizer", first=True)
            
            # additional abbreviations not included in spacy sentencizer
            additional_abbreviations = [
            	"I.",
                "II.",
                "III.",
                "IIII.",
                "IV.",
                "V.",
                "VI.",
                "VII.",
                "VIII.",
                "VIIII.",
                "IX.",
                "X.",
                "XI.",
                "XII.",
                "XIII.",
                "XIIII.",
                "XIV.",
                "XV.",
                "XVI.",
                "XVII.",
                "XVIII.",
                "XVIIII.",
                "XIX.",
                "XX.",
                "XXI.",
                "XXII.",
                "XXIII.",
                "XXIIII.",
                "XXIV.",
                "XXV.",
                "XXVI.",
                "XXVII.",
                "XXVIII.",
                "XXVIIII.",
                "XXIX.",
                "XXX.",
                "XXXI.",
                "XXXII.",
                "XXXIII.",
                "XXXIIII.",
                "XXXIV.",
                "XXXV.",
                "XXXVI.",
                "XXXVII.",
                "XXXVIII.",
                "XXXVIIII.",
                "XXXIX.",
                "XXXX.",
                "XL.",
                "XLI.",
                "XLII.",
                "XLIII.",
                "XLIIII.",
                "XLIV.",
                "XLV.",
                "XLVI.",
                "XLVII.",
                "XLVIII.",
                "XLVIIII.",
                "XLIX.",
                "L.",
                "LI.",
                "LII.",
                "LIII.",
                "LIIII.",
                "LIV.",
                "LV.",
                "LVI.",
                "LVII.",
                "LVIII.",
                "LVIIII.",
                "LIX.",
                "LX.",
                "LXI.",
                "LXII.",
                "LXIII.",
                "LXIIII.",
                "LXIV.",
                "LXV.",
                "LXVI.",
                "LXVII.",
                "LXVIII.",
                "LXVIIII.",
                "LXIX.",
                "LXX.",
                "LXXI.",
                "LXXII.",
                "LXXIII.",
                "LXXIIII.",
                "LXXIV.",
                "LXXV.",
                "LXXVI.",
                "LXXVII.",
                "LXXVIII.",
                "LXXVIIII.",
                "LXXIX.",
                "LXXX.",
                "LXXXI.",
                "LXXXII.",
                "LXXXIII.",
                "LXXXIIII.",
                "LXXXIV.",
                "LXXXV.",
                "LXXXVI.",
                "LXXXVII.",
                "LXXXVIII.",
                "LXXXVIIII.",
                "LXXXIX.",
                "LXXXX.",
                "XC.",
                "XCI.",
                "XCII.",
                "XCIII.",
                "XCIIII.",
                "XCIV.",
                "XCV.",
                "XCVI.",
                "XCVII.",
                "XCVIII.",
                "XCVIIII.",
                "XCIX.",
                "C.",
                "CC.",
                "CCC.",
                "CCCC.",
                "CD.",
                "D.",
                "DC.",
                "DCC.",
                "DCCC.",
                "DCCCC.",
                "CM.",
                "M.",
                "MM.",
                "MMM.",
                "MMMM.",
                "Aed.",
                "Aem.",
                "Agr.",
                "Aim.",
                "An.",
                "Ann.",
                "Ant.",
                "Aur.",
                "Cens.",
                "Cir.",  
                "Ex.",
                "Fab.",
                "Fin.",
                "Germ.",
                "Gn.",
                "In.",    
                "Kl.",
                "Leg.",
                "Lib.",           
                "Max.",        
                "Men.",
                "Min.",               
                "Ob.",             
                "Op.",             
                "Pl.",           
                "Pom.",
                "Pont.",
                "Pr.",         
                "Pup.",
                "Quinct.",        
                "Quir.",
                "Rom.",
                "Sal.",
                "Scrib.",
                "Sec.",           
                "Seq.",
                "Serv.",
                "Sp.",
                "Ter.",
                "Tib.",
                "Tr.",
                "Ver.",
                "Ajm.", 
                "Aprjl.", 
                "A.V.C.", 
                "Avg.", 
                "Cjr.", 
                "Februar.", 
                "Fjn.", 
                "J.", 
                "Jan.", 
                "Jd.", 
                "Jmp.", 
                "Jmpp.", 
                "Jmppp.", 
                "Jn.", 
                "Jul.", 
                "Jun.", 
                "Ljb.", 
                "Maj.", 
                "Mjn.", 
                "Nou.", 
                "Pavl.", 
                "Plvr.", 
                "Prjd.", 
                "Pvp.", 
                "Qvinct.", 
                "Qujnct.", 
                "Qvjnct.", 
                "Qvint.", 
                "Qujnt.", 
                "Qvjnt.", 
                "Qvir.", 
                "Qujr.", 
                "Qvjr.", 
                "Scrjb.", 
                "Seru.", 
                "Svff.", 
                "Tj.", 
                "Trjb.", 
                "U.", 
                "Uer.", 
                "Uol.", 
                "Uop.", 
                "Uu.",
            ]
            
            # -que exceptions not in spacy
            additional_exceptions = [
				"qualiscunque",
				"quandocunque",
				"quantuluscunque",
				"quantumcunque",
				"quantuscunque",
				"quocunque",
				"quomodocumque",
				"quotacunque",
				"quotcunque",
				"quotienscunque",
				"ubicunque",
				"undecunque",
				"utcunque",
				"utercunque",
				"utrinque",
        	]

				
            # -que exceptions in spacy but needed for orthographic expansion
            # # quisque / quique            
            additional_exceptions += [
            	"quisque",
            	"quidque",
            	"quicque",
            	"quodque",
            	"cuiusque",
            	"cuique",
            	"quemque",
            	"quamque",
            	"quoque",
            	"quaque",
            	"quique",
            	"quaeque",
            	"quorumque",
            	"quarumque",
            	"quibusque",
            	"quosque",
            	"quasque",
		    "cuiusquemodi"
            ]   
            
        	# uterque
            additional_exceptions += [
            	"uterque",
            	"utraque",
            	"utrumque",
            	"utriusque",
            	"utrique",
            	"utramque",
            	"utroque",
            	"utraque",
            	"utraeque",
            	"utrorumque",
           	 	"utrarumque",
            	"utrisque",
            	"utrosque",
            	"utrasque",
        	]
        	
        	# quiscumque
            additional_exceptions += [
            	"quicumque",
            	"quidcumque",
            	"quodcumque",
            	"cuiuscumque",
           	 	"cuicumque",
            	"quemcumque",
            	"quamcumque",
            	"quocumque",
            	"quacumque",
            	"quaecumque",
            	"quorumcumque",
            	"quarumcumque",
           	 	"quibuscumque",
            	"quoscumque",
            	"quascumque",
            	"quicunque",
            	"quidcunque",
            	"quodcunque",
            	"cuiuscunque",
            	"cuicunque",
            	"quemcunque",
            	"quamcunque",
            	"quocunque",
            	"quacunque",
            	"quaecunque",
            	"quorumcunque",
            	"quarumcunque",
            	"quibuscunque",
            	"quoscunque",
            	"quascunque",
        ]

        	# unuscumque
            additional_exceptions += [
            	"unusquisque",
           		"unaquaeque",
            	"unumquodque",
           		"unumquidque",
            	"uniuscuiusque",
            	"unicuique",
            	"unumquemque",
            	"unamquamque",
            	"unoquoque",
            	"unaquaque",
        ]

       		# plerusque
            additional_exceptions += [
            	"plerusque",
            	"pleraque",
            	"plerumque",
            	"plerique",
            	"pleraeque",
            	"pleroque",
            	"pleramque",
            	"plerorumque",
            	"plerarumque",
            	"plerisque",
            	"plerosque",
            	"plerasque",
        ]
		
			# misc
#            additional_exceptions += [
#            	"absque",
#           		 "abusque",
#            	"adaeque",
#            	"adusque",
#            	"aeque",
#           	"antique",
#           	"atque",
#            	"circumundique",
#            	"conseque",
#            	"cumque",
#            	"cunque",
#            	"denique",
#            	"deque",
#            	"donique",
#            	"hucusque",
#            	"inique",
#            	"inseque",
#            	"itaque",
#            	"longinque",
#            	"namque",
#            	"neque",
#            	"oblique",
#            	"peraeque",
#            	"praecoque",
#            	"propinque",
#            	"qualiscumque",
#            	"quandocumque",
#            	"quandoque",
#            	"quantuluscumque",
#            	"quantumcumque",
#            	"quantuscumque",
#            	"quinque",
#            	"quocumque",
#            	"quomodocumque",
#            	"quomque",
#            	"quotacumque",
#            	"quotcumque",
#            	"quotienscumque",
#            	"quotiensque",
#            	"quotusquisque",
#            	"quousque",
#            	"relinque",
#            	"simulatque",
#            	"torque",
#            	"ubicumque",
#            	"ubique",
#            	"undecumque",
#            	"undique",
#            	"usque",
#            	"usquequaque",
#            	"utcumque",
#            	"utercumque",
#            	"utique",
#            	"utrimque",
#            	"utrique",
#            	"utriusque",
#            	"utrobique",
#            	"utrubique",
#        ]
        
        # NEW VERSION (09/09/2025)
			# misc
            additional_exceptions += [
            	"absque",
           		"abusque",
            	"adaeque",
            	"adusque",
            	"aeque",
            	"antique",
            	"atque",
            	"circumundique",
            	"conseque",
            	"cumque",
            	"cunque",
            	"denique",
            	"deque",
            	"donique",
            	"dumque",
            	"enimque",
            	"ergoque",
            	"etenimque",
            	"etiamque",
            	"hucusque",
            	"inique",
            	"inseque",
            	"itaque",
            	"longinque",
            	"namque",
            	"neque",
            	"oblique",
            	"peraeque",
            	"postquamque",
            	"praecoque",
            	"priusquamque",
            	"propinque",
            	"qualiscumque",
            	"quandocumque",
            	"quandoque",
            	"quantuluscumque",
            	"quantumcumque",
            	"quantuscumque",
            	"quiaque",
            	"quinque",
            	"quocumque",
            	"quodsique",
            	"quomodocumque",
            	"quomque",
            	"quoniamque",
            	"quotacumque",
            	"quotcumque",
            	"quotienscumque",
            	"quotiensque",
            	"quotusquisque",
            	"quousque",
            	"relinque",
            	"sedque",
            	"setque",
            	"tamquamque",
            	"simulatque",
            	"siueque",
            	"torque",
            	"ubicumque",
            	"ubique",
            	"undecumque",
            	"undique",
            	"usque",
            	"usquequaque",
            	"utcumque",
            	"utercumque",
            	"utique",
            	"utrimque",
            	"utrique",
            	"utriusque",
            	"utrobique",
            	"utrubique",
        ]        
        
            # add u/v and i/j forms          
            additional_exceptions_ij = [word.replace("i", "j") for word in additional_exceptions]	
            additional_exceptions += additional_exceptions_ij
            additional_exceptions_uv = [word.replace("u", "v") for word in additional_exceptions]
            additional_exceptions += additional_exceptions_uv    
            
            # add capitalized forms
            capitalized_exceptions = [word.capitalize() for word in additional_exceptions]
            additional_exceptions += capitalized_exceptions           

            exceptions = additional_abbreviations + additional_exceptions

            for abbr in exceptions:
                self.reader.model.tokenizer.add_special_case(abbr, [{ORTH: abbr}])
        except:
            pass

        self.reader.model.disable_pipes(
            "senter",
            "normer",
            "tok2vec",
            "tagger",
            "morphologizer",
            "trainable_lemmatizer",
            "parser",
            "lookup_lemmatizer",
            "ner",
        )

        self.fileid = fileid
        self.doc = "\n\n".join(list(self.reader.docs(self.fileid)))
        self.sents = list(self.reader.sents(self.fileid))
        
        #self.words = [
        #    word.text
        #    for word in list(self.reader.tokens(self.fileid))
        #    if word.text not in string.punctuation
        #]
        
        # NEW VERSION (04/26/2025) TO ADDRESS WHITESPACE COUNTING ISSUE
        #self.words = [
        #    tok.text
        #    for tok in self.reader.tokens(self.fileid)
        #    if (not tok.is_space) and (tok.text not in string.punctuation)
        #]

		# NEW VERSION (06/19/2025)
        ENCLITICS = {"que", "qve"}

        self.words = [
   		    tok.text
    	    for tok in self.reader.tokens(self.fileid)
            if not tok.is_space and tok.text not in string.punctuation and tok.text not in ENCLITICS
	    ]
        
        print(self.words)
        self.plaintext = " ".join(self.words)
        self.tagged_sents = list(
            self.reader.tagged_sents(
                self.fileid, plaintext=False, special_letters=False
            )
        )
        self.relative_clause_sents = list(
            self.reader.relative_clauses(self.fileid, sent_level=True)
        )
        self.relative_clauses = list(
            self.reader.relative_clauses(self.fileid, sent_level=False, plaintext=True)
        )
        self.flat_tagged_sents = self._flatten_list(self.tagged_sents)
        self.norm = norm
        self.verbose = verbose

    def _flatten_list(self, l):
        return [item for subl in l for item in subl]

    @property
    # sentence count
    def sentence_count(self):
        if self.verbose:
            print(self.tagged_sents)
        return len(self.sents)

    @property
    # word count
    def word_count(self):
        if self.verbose:
            print(self.words)
        return len(self.words)

    @property
    # character count
    def char_count(self):
        return len(" ".join(self.sents))

    @property
    # count of sentences containing a relative clause
    def _relative_clause_sent_count(self):
        feature_count = [item for item in self.relative_clause_sents if item]
        return len(feature_count)

    @property
    # fraction of sentences containing a relative clause
    def fraction_sentence_relative(self):
        def sentence_count(self):
            if self.verbose:
           	    print(self.tagged_sents)
            return len(self.sents)
        feature_count = self._relative_clause_sent_count
        #return self.norm_feature(feature_count, "sent") if self.norm else feature_count
        feature_count_normed = feature_count / sentence_count(self)
        return feature_count_normed

    #@property
    # mean sentence length in words
    #def sentence_length(self):
    #    sents_pp = [
    #        [token.text for token in sent if not token.is_punct and not token.is_space]
    #        for sent in self.sents
    #    ]
    #    sent_lengths = [len(sent) for sent in sents_pp]

    #    return np.mean(sent_lengths)
    
    # NEW VERSION 6/19/2025
    @property
    # mean sentence length in words
    def sentence_length(self):
        ENCLITICS = {"que", "qve"}
        sents_pp = [
            [token.text for token in sent if not token.is_punct and not token.is_space and not token.text.lower() in ENCLITICS]
            for sent in self.sents
        ]
        sent_lengths = [len(sent) for sent in sents_pp]

        return np.mean(sent_lengths)

    @property
    # mean relative clause length in words
    def relative_clause_length(self):
        relative_clause_lengths = [
            len(relclause.split()) for relclause in self.relative_clauses if relclause
        ]
        return np.mean(relative_clause_lengths)

    @property
    # count of question marks
    def interrogatives(self):
        feature_count = self._word_feature_count(words=["?"])
        return self.norm_feature(feature_count, "sent") if self.norm else feature_count
    
    # general function for counting words (with orthographic normalization)
    def _word_feature_count(self, words=None):
        def expand_words(words):
            words_u = [word.replace("u", "v") for word in words]
            words_v = [word.replace("v", "u") for word in words]
            words_i = [word.replace("i", "j") for word in words]
            words_j = [word.replace("j", "i") for word in words]
            words_ui = [word.replace("u", "v").replace("i", "j") for word in words]
            words_uj = [word.replace("u", "v").replace("j", "i") for word in words]
            words_vi = [word.replace("v", "u").replace("i", "j") for word in words]
            words_vj = [word.replace("v", "u").replace("j", "i") for word in words]
            words = set(
                list(words)
                + words_u
                + words_v
                + words_i
                + words_j
                + words_ui
                + words_uj
                + words_vi
                + words_vj
            )
            return words

        words = expand_words(words)
        features = [item for item in self.flat_tagged_sents if item[0].lower() in words]
        if self.verbose:
            print(features)
        return len(features)

    # general function for counting regex pattern
    def _regex_word_feature_count(self, pattern=None):
        features = re.findall(pattern, self.plaintext, flags=re.IGNORECASE)
        if self.verbose:
            print(features)
        return len(features)

    # functions for working with pos tags (not used for core feature set)

    def _pos_feature_count(self, tags=None):
        _pos_features = [item for item in self.flat_tagged_sents if item[1] in tags]
        if self.verbose:
            print(_pos_features)
        return len(_pos_features)

    def _word_pos_feature_count(self, taggeds=None):
        features = [
            item
            for item in self.flat_tagged_sents
            if (item[0].lower(), item[1]) in taggeds
        ]
        if self.verbose:
            print(features)
        return len(features)
    
    def _regex_pos_feature_count(self, pattern=None):
        features = []
        pattern_comp = re.compile(pattern)
        for word, pos in self.flat_tagged_sents:
            match = pattern_comp.findall(pos)
            if match:
                features.extend((word, match))
        if self.verbose:
            print(features)
        return len(features)

    def _regex_doc_search(self, pattern=None):
        features = re.findall(pattern, self.doc, flags=re.IGNORECASE)
        if self.verbose:
            print(features)
        return len(features)
    
    # normalization by characters, words, or sentences
    def norm_feature(self, feature_count, basis="word"):
        if basis == "word":
            return feature_count / self.word_count
        elif basis == "char":
            return feature_count / self.char_count
        elif basis == "sent":
            return feature_count / self.sent_count
        else:
            print("Norm basis not recognized; returning raw count.")
            return feature_count

class LatinFeatures(Features):
    def add_enclitic_forms(self, wordset, enclitic):
        enc_wordset = [
            word + enclitic
            for word in wordset
            if not word.endswith("que")
            and not word.endswith("qve")
            and not word.endswith("ne")
            and not word.endswith("ue")
            and not word.endswith("ve")
        ]
        return set(list(wordset) + enc_wordset)

    @property
    # personal pronouns (including reflexives)
    def personal(self):
        # words = "ego egomet egoque egoqve me med mei meimet meique meiqve meme memet mepte meque meqve mi michi mihi mihimet mihique mihiqve mimet mique miqve mis nobis nobismet nobisque nobisqve nos nosmet nosque nosqve nostri nostrique nostriqve nostrum nostrumque nostrvm nostrvmqve te ted temet tepte teque teqve tete tibi tibimet tibique tibiqve tis tu tui tuimet tuique tuque tute tutemet tutimet tv tvi tvimet tviqve tvqve tvte tvtemet tvtimet uestri uestrique uestrum uestrumque uobis uobismet uobisque uos uosmet uosque uostri uostrique uostrum uostrumque vestri vestrique vestriqve vestrum vestrumque vestrvm vestrvmqve vobis vobismet vobisque vobisqve vos vosmet vosque vosqve vostri vostrique vostriqve vostrum vostrumque vostrvm vostrvmqve"

        #words = "ego egomet me med mei meimet meme memet mepte mi mihi mihimet mimet mis nobis nobismet nos nosmet nostri nostrum te ted temet tepte tete tibi tibimet tis tu tui tuimet tute tutemet tutimet uestri uestrum uobis uobismet uos uosmet uostri uostrum michi meipse metipse meipsum metipsum memetipsum meipso memetipso metipsa meipsam metipsi nosipsi nosipsos nosmetipsos nobismetipsis tuipse teipsum teipso temetipsum temetipso uobisipsis uobismetipsis uosmetipsos metipsos"

        # NEW VERSION (9/9/2025)
        words = "ego egomet me med mei meimet meme memet mepte mi mihi mihimet mimet mis nobis nobismet nos nosmet nostri nostrum te ted temet tepte tete tibi tibimet tis tu tui tuimet tute tutemet tutimet uestri uestrum uobis uobismet uos uosmet uostri uostrum michi meipse metipse meipsum metipsum memetipsum meipso memetipso metipsa meipsam metipsi nosipsi nosipsos nosmetipsos nobismetipsis tuipse teipsum teipso temetipsum temetipso uobisipsis uobismetipsis uosmetipsos metipsos se sese sibi sui semet sepse secum suipsius seipse seipsum seipso seipsis seipsos seipsa seipsam seipsammet seipsas sibiipsi sibiipsis semetipse semetipsum semetipso semetipsa semetipsam semetipsis semetipsos sibimetipsi sibimetipsos sibimetipsis meipsae metipsae meipsis metipsis memetipsis nosipsis temetipsis uosmetipsis seipsae sibiipsae semetipsae sibimetipsae"
        
        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")
        
        words -= {"tene", "sene"}
    
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # demonstrative pronouns
    def demonstrative(self):
        # words = "ea eabus eae eaeque eaeqve eam eamque eamqve eaque eaqve earum earumque earvm earvmqve eas easque easqve ei eique eiqve eis eisque eisqve eius eiusce eiusque eivs eivsqve eo eopte eoque eoqve eorum eorumque eorvm eorvmqve eos eosque eosqve eum eumque evm evmqve aec aecce anc ic oc ocque uic uius ujusce unc ha hac hacce hacine hacmet hacne hacque hacqve hae haec haecce haece haecin haecine haeccine haecque haecqve haeque haeqve hanc hancce hancin hancine hancmet hancque hancqve haque haqve harum harumque harunc harvm harvmqve harvnce harvncharunce has hasce hascine hasmet hasque hasqve hec hecce hecne hecque hi hibus hibvs hic hicc hicce hiccine hice hicin hicine hicque hicqve hique hiqve his hisce hisque hisqve ho hoc hocc hocce hoccine hoce hocine hocmet hocne hocqve hoque hoqve horum horumque horunc horunce horvm horvmqve horvnc horvnce hos hosce hoscin hoscine hosmet hosque hosqve huic huicce huicmet huicque huius huiusce huiuscene hujusmet huiusque hunc huncce hunce huncine huncmet huncque hvic hvicqve hvivs hvivsce hvivsqve hvnc hvnce hvncqve i ibus id idque idqve ii iique iiqve iis iisque iisqve illa illac illacque illacqve illae illaec illaecque illaecqve illaeque illaeqve illam illamque illamqve illanc illancque illancqve illaque illaqve illarum illarumque illarvm illarvmqve illas illasque illasqve ille illeque illeqve illi illic illicque illicqve illique illiqve illis illisque illisqve illius illiusce illiusque illivs illivsce illivsqve illo illoc illocque illocqve illoque illoqve illorum illorumque illorvm illorvmqve illos illosque illosqve illuc illucque illud illudque illum illumque illunc illuncque illvc illvcqve illvd illvdqve illvm illvmqve illvnc illvncqve is isce isque isqve istac istae istaec istaece istanc istic istice isticine isticne istisce istiusce istoc istocine istoscine istuc istucine istunc istvc istvnc olle olli ollos ollosque ollud ollum ollus ollvd ollvm ollvs ste stoc ista istacine istam istamcine istamque istamqve istaque istaqve istarum istarumque istarvm istarvmqve istas istasque istasqve iste isteque isteqve isti istique istiqve istis istisque istisqve istius istiusque istivs istivsqve isto istoque istoqve istorum istorumque istorvm istorvmqve istos istosque istosqve istud istudque istum istumque istvd istvdqve istvm istvmqve"

        #words = "ea eabus eae eam earum eas ei eis eius eiusce eo eopte eorum eos eum eiusmodi ic oc uic uius uiuisce unc anc aec aecce hec hecce ha hac hacce hacine hae haec haecce haece haecin haecine haeccine hanc hancce hancin hancine harum harunc harunce haruncharunce has hasce hascine hi hibus hic hicc hicce hice hicin hicine hiccine his hisce ho hoc hocc hocce hoccine hoce hocine horum horunc horunce hos hosce hoscin hoscine huic huiucce huius huiusce huiuscene hunc hunce huncce huncine huiusmodi huiuscemodi i ibus id ii iis illa illac illae illaec illam illanc illarum illas ille illi illic illis illius illiusce illo illoc illorum illos illuc illud illum illunc illiusmodi is isce istac istaec istaece istanc istic istice istisce istiusce istoc istuc istunc isticine isticne istucine istoscine istamcine istocine istocine istacine olle olli ollos ollud ollum ollus ste stoc ista istam istarum istas iste isti istis istius isto istorum istos istud istum istiusmodi istimodi hancmet hasmet hacmet hocmet hosmet huncmet huiusmet huicmet"

        # NEW VERSION (6/19/2025)
        words = "ea eabus eae eam earum eas ei eis eius eiusce eo eopte eorum eos eum eiusmodi ic oc uic uius uiuisce unc anc aec aecce hec hecce ha hac hacce hacine hae haec haecce haece haecin haecine haeccine hanc hancce hancin hancine harum harunc harunce haruncharunce has hasce hascine hi hibus hic hicc hicce hice hicin hicine hiccine his hisce ho hoc hocc hocce hoccine hoce hocine horum horunc horunce hos hosce hoscin hoscine huic huiucce huius huiusce huiuscene hunc hunce huncce huncine huiusmodi huiuscemodi i ibus id ii iis illa illac illae illaec illam illanc illarum illas ille illi illic illis illius illiusce illo illoc illorum illos illuc illud illum illunc illiusmodi is istae isce istac istaec istaece istanc istic istice istisce istiusce istoc istuc istunc isticine isticne istucine istoscine istamcine istocine istocine istacine olle olli ollos ollud ollum ollus ste stoc ista istam istarum istas iste isti istis istius isto istorum istos istud istum istiusmodi istimodi hancmet hasmet hacmet hocmet hosmet huncmet huiusmet huicmet"

        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")

        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # quidam
    def quidam(self):
        # words = "cuidam cuidamque cuiusdam cuiusdamque cvidam cvidamqve cvivsdam cvivsdamqve quadam quadamque quaedam quaedamque quandam quandamque quarundam quarundamque quasdam quasdamque quendam quendamque quibusdam quibusdamque quidam quidamque quiddam quiddamque quodam quodamque quoddam quoddamque quorundam quorundamque quosdam quosdamque qvadam qvadamqve qvaedam qvaedamqve qvandam qvandamqve qvarvndam qvarvndamqve qvasdam qvasdamqve qvendam qvendamqve qvibvsdam qvibvsdamqve qvidam qvidamqve qviddam qviddamqve qvodam qvodamqve qvoddam qvoddamqve qvorvndam qvorvndamqve qvosdam qvosdamqve"

        words = "cuidam cuiusdam quadam quaedam quandam quarundam quasdam quendam quibusdam quidam quiddam quodam quoddam quorundam quosdam quedam quesdam quoiusdam quoidam quisdam queisdam quemdam quamdam quorumdam quarumdam cuiusdammodi"

        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")

        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # third-person reflexive pronouns
    def reflexive(self):
        # words = "se seque seqve sese seseque seseqve sibi sibique sibiqve sui suique svi sviqve"

        #words = "se sese sibi sui semet sepse secum suipsius seipse seipsum seipso seipsis seipsos seipsa seipsam seipsammet seipsas sibiipsi sibiipsis semetipse semetipsum semetipso semetipsa semetipsam semetipsis semetipsos sibimetipsi sibimetipsos sibimetipsis"
	    
	    # NEW VERSION (9/9/2025)
        words = "se sese sibi sui semet sepse secum suipsius seipse seipsum seipso seipsis seipsos seipsa seipsam seipsammet seipsas sibiipsi sibiipsis semetipse semetipsum semetipso semetipsa semetipsam semetipsis semetipsos sibimetipsi sibimetipsos sibimetipsis seipsae sibiipsae semetipsae sibimetipsae"
	    
        words = set(words.split(" "))
        
        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")
        
        words -= {"tene", "sene"}
        
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # iste
    def iste(self):
        # words = "ista istae istaec istaece istam istamcine istamque istamqve istanc istac istacine istaque istaqve istarum istarumque istarvm istarvmqve istas istasque istasqve iste isteque isteqve isti istic istice isticine isticne istique istiqve istis istisce istisque istisqve istius istiusce istiusque istivs istivsqve isto istoc istocine istoque istoqve istorum istorumque istorvm istorvmqve istos istoscine istosque istosqve istuc istucine istud istudque istum istumque istvd istvdqve istvm istvmqve istunc ste stoc"

        words = "ista istae istaec istaece istam istamcine istanc istac istacine istarum istas iste isti istic istice isticine isticne istis istisce istius istiusce isto istoc istocine istorum istos istoscine istuc istucine istud istum istunc ste stoc istiusmodi istimodi"

        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")

        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # alius
    def alius(self):
        # words = "alia aliae aliaeque aliaeqve aliam aliamque aliamqve aliaque aliaqve aliarum aliarumque aliarvm aliarvmqve alias aliasque aliasqve alid alii aliique aliiqve aliis aliisque aliisqve alio alioque alioqve aliorum aliorumque aliorvm aliorvmqve alios aliosque aliosqve aliud aliudque alium aliumque alius aliusque alivd alivdqve alivm alivmqve alivs alivsqve"

        words = "alia aliae aliam aliarum alias alid alii aliis alio aliorum alios aliud alium alius"

        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")

        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # ipse
    def ipse(self):
        # words = "eampse eapse eopse eumpse ipsa ipsam ipsamque ipsamqve ipsaque ipsaqve ipsarum ipsarumque ipsarvm ipsarvmqve ipsas ipsasque ipsasqve ipse ipseque ipseqve ipsi ipsique ipsiqve ipsius ipsiusque ipsivs ipsivsqve ipso ipsoque ipsoqve ipsorum ipsorumque ipsorvm ipsorvmqve ipsos ipsosque ipsosqve ipsum ipsumque ipsus ipsvm ipsvmqve"

        #words = "eampse eapse eopse eumpse idipsum ipsa ipsam ipsamet ipsammet ipsaque ipsarum ipsas ipse ipsemet ipsi ipsimet ipsius ipso ipsorum ipsos ipsud ipsum ipsummet ipsus meipse metipse meipsum metipsum memetipsum meipso memetipso metipsa meipsam metipsi nosipsi nosipsos nosmetipsos nobismetipsis tuipse teipsum teipso temetipsum temetipso uobisipsis uobismetipsis uosmetipsos metipsos suipsius seipse seipsum seipso seipsis seipsos seipsa seipsam seipsammet seipsas sibiipsi sibiipsis semetipse semetipsum semetipso semetipsa semetipsam semetipsis semetipsos sibimetipsi sibimetipsos sibimetipsis"

        # NEW VERSION (6/19/2025)
        #words = "ipsis ipsae eampse eapse eopse eumpse idipsum ipsa ipsam ipsamet ipsammet ipsaque ipsarum ipsas ipse ipsemet ipsi ipsimet ipsius ipso ipsorum ipsos ipsud ipsum ipsummet ipsus meipse metipse meipsum metipsum memetipsum meipso memetipso metipsa meipsam metipsi nosipsi nosipsos nosmetipsos nobismetipsis tuipse teipsum teipso temetipsum temetipso uobisipsis uobismetipsis uosmetipsos metipsos suipsius seipse seipsum seipso seipsis seipsos seipsa seipsam seipsammet seipsas sibiipsi sibiipsis semetipse semetipsum semetipso semetipsa semetipsam semetipsis semetipsos sibimetipsi sibimetipsos sibimetipsis"
        
        # NEW VERSION (9/9/2025)
        words = "eampse eapse eopse eumpse idipsum ipsa ipsis ipsae ipsam ipsamet ipsammet ipsarum ipsas ipse ipsemet ipsi ipsimet ipsius ipso ipsorum ipsos ipsud ipsum ipsummet ipsus meipse metipse meipsum metipsum memetipsum meipso memetipso metipsa meipsam metipsi nosipsi nosipsos nosmetipsos nobismetipsis tuipse teipsum teipso temetipsum temetipso uobisipsis uobismetipsis uosmetipsos metipsos suipsius seipse seipsum seipso seipsis seipsos seipsa seipsam seipsammet seipsas sibiipsi sibiipsis semetipse semetipsum semetipso semetipsa semetipsam semetipsis semetipsos sibimetipsi sibimetipsos sibimetipsis eaepsae eisipsis ipsaemet ipsismet meipsae metipsae meipsis metipsis memetipsis nosipsis temetipsis uosmetipsis seipsae sibiipsae semetipsae sibimetipsae"
        
        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")

        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # idem
    def idem(self):
        # words = "eadem eademque eademqve eaedem eaedemque eaedemqve eandem eandemque eandemqve earumdem earundemque earvmdem earvndemqve easdem easdemque easdemqve eedem eidem eidemque eidemqve eisdem eisdemque eisdemqve eiusdem eiusdemque eivsdem eivsdemqve eodem eodemque eodemqve eorundem eorundemque eorvndem eorvndemqve eosdem eosdemque eosdemqve eundem eundemque evndem evndemqve idem idemque idemqve iidem iidemque iidemqve iisdem iisdemque iisdemqve isdem isdemque isdemqve”

        words = "eadem eandem eamdem earumdem earundem easdem eedem eaedem eiusdem eodem eorundem eorumdem eosdem eundem eumdem idem iidem iisdem isdem eidem eisdem eiusdemmodi "

        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")

        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # priusquam or prius quam
    def priusquam(self):
        feature_count = self._regex_word_feature_count(pattern=r"pr[ij][uv]s\s?q[uv]am")
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # antequam or ante quam
    def antequam(self):
        feature_count = self._regex_word_feature_count(pattern=r"ante\s?q[uv]am")
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # quominus or quo minus
    def quominus(self):
        feature_count = self._regex_word_feature_count(pattern=r"q[uv]o\s?m[ij]n[uv]s")
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # dum
    def dum(self):
        words = {
            "dum",
            "dummodo",
			"dumtaxat", 
			"dummodom",
        }
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

# NEW VERSION (09/09/2025)
    @property
    # dum
    def dum(self):
        words = {"dum", "dumque", "dummodo", "dumtaxat", "dummodom"}
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # quin
    def quin(self):
        words = {"quin"}
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # ut without any disambiguation 
    def ut(self):
        words = {"ut", "utei", "utque"}  # check utque
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

#    @property
    # si + compounds (i.e., common conditional markers)
#    def si(self):
#        words = {
#           "si",
#            "nisi",
#            "quodsi",
#            "sin",
#            "siue",
#            "seu",
#            "ni",
#            "etsi",
#            "etiamsi",
#            "tametsi",
#        }       
        
#        feature_count = self._word_feature_count(words=words)
#        return self.norm_feature(feature_count, "word") if self.norm else feature_count
    
    # NEW VERSION (09/09/2025)
    @property
    # si + compounds (i.e., common conditional markers)
    def si(self):
        words = "si nisi quodsi sin siue seu ni etsi etiamsi tametsi siquidem"

        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")
        
        words -= {"sine", "niue", "nive"}        
        
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # prepositions excluding cum
    def preposition(self):
        # words = "a ab abque abqve abs absque absqve ad adque adqve aduersus aduersusque adversvs adversvsqve aforis ante anteque anteqve apud apudque aput aputque apvd apvdqve apvt apvtqve aque aqve cata circa circaque circaqve circum circumque circvm circvmqve cis cisque cisqve citra citraque citraqve contra contraque contraqve coram coramque coramqve de deforis deforisque deque deqve e eque eqve erga ergaque ergaqve ex exque exqve extra extraque extraqve in infra infraque infraqve inque inqve inter interque interqve intra intraque intraqve iuxta iuxtaque ivxta ivxtaqve ob obque obqve penes penesque penesqve per perque perqve post postque postqve prae praeque praeqve praeter praeterque praeterqve pro propter propterque propterqve proque proqve sine sineque sineqve sub subque super superque supra supraque svb svbqve svper svperqve svpra svpraqve tenus tenusque tenvs tenvsqve trans transque transqve ultra ultraque vltra vltraqve"

        words = "a ab abs ad aduersus aforis ante apud aput cata circa circum cis citra contra coram de deforis e erga ex extra in infra inter intra iuxta ob penes per post prae praeter pro propter sine sub super supra tenus trans ultra"

		# NEW VERSION (09/09/2025)
        words = "a ab abs absque abusque ad aduersus adusque aforis ante apud aput cata circa circum cis citra contra coram de deforis e erga ex extra in infra inter intra iuxta ob penes per post prae praeter pro propter sine sub super supra tenus trans ultra"

        words = set(words.split(" "))

        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")

        words -= {"ane", "aue", "ave", "eque", "eqve", "eue", "eve", "exue", "exve", "inque", "inqve", "interne", "prone", "superne"}
        
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # question marks
    def interrogative(self):
        # TODO: Check sentences instead? Better for verbose.
        words = {"?"}
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # regular superlative endings (-issim-)
    def superlative(self):
        feature_count = self._regex_word_feature_count(
            pattern=r"\b\w+?[ij]ss[ij]m\w+?\b"
        )
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # atque followed by a consonant / “unelided atque"
    def atque_consonant(self):
        feature_count = self._regex_word_feature_count(
            pattern=r"\batq[uv]e\s[bcdfgjklmnpqrstvwxyz]\w+?\b"
        )
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

#    @property
#    # selected gerund and gerundive endings (-and-, -end-)
#    def gerundive(self):
#        feature_count = self._regex_word_feature_count(
#            pattern=(
#                r"\b"
#                r"(?!quandam|quendam|quando|qvandam|qvendam|qvando|tendo|tendam|tendas|tendis|tende|tendi|uendo|vendo|uendam|vendam|uendas|vendas|uendis|vendis|uende|vende|uendi|vendi|defendo|defendam|defendas|defendis|defende|defendi|incendo|incendam|incendas|incendis|incende|incendi|prehendo|prehendam|prehendas|prehendis|prehende|prehendi|comprehendo|comprehendam|comprehendas|comprehendis|comprehende|comprehendi|ostendo|ostendam|ostendas|ostendis|ostende|ostendi|intendo|intendam|intendas|intendis|intende|intendi|contendo|contendam|contendas|contendis|contende|contendi|offendo|offendam|offendas|offendis|offende|offendi|succendo|succendam|succendas|succendis|succende|succendi|ascendo|ascendam|ascendas|ascendis|ascende|ascendi|extendo|extendam|extendas|extendis|extende|extendi|attendo|attendam|attendas|attendis|attende|attendi|protendo|protendam|protendas|protendis|protende|protendi|pretendo|pretendam|pretendas|pretendis|pretende|pretendi|dependo|dependam|dependas|dependis|depende|dependi|pendo|pendam|pendas|pendis|pende|pendi|pando|pandam|pandas|pandis|pande|pandi|expando|expandam|expandas|expandis|expande|expandi|blandus|blandvs|blandum|blandvm|blandi|blando|blanda|blandam|blandae|blandos|blandas|blandorum|blandorvm|blandarum|blandarvm|blandis|tendjs|tendj|uendjs|vendjs|uendj|vendj|defendjs|defendj|jncendo|jncendam|jncendas|jncendjs|jncende|jncendj|prehendjs|prehendj|comprehendjs|comprehendj|ostendjs|ostendj|jntendo|jntendam|jntendas|jntendjs|jntende|jntendj|contendjs|contendj|offendjs|offendj|succendjs|succendj|ascendjs|ascendj|extendjs|extendj|attendjs|attendj|protendjs|protendj|pretendjs|pretendj|dependjs|dependj|pendjs|pendj|pandjs|pandj|expandjs|expandj|blandj|blandjs|apprehendo|apprehendam|apprehendas|apprehendis|apprehendjs|apprehende|apprehendi|apprehend|deprehendo|deprehendam|deprehendas|deprehendis|deprehendjs|deprehende|deprehend|deprehendj|accendo|accendam|accendas|accendis|accendjs|accende|accendi|accendj|subtendo|subtendam|subtendas|subtendis|subtendjs|subtende|subtendi|subtendj|svbtendo|svbtendam|svbtendas|svbtendis|svbtendjs|svbtende|svibtendi|svibtendj|descendo|descendam|descendas|descendis|descendjs|descende|descendi|descendj|escendo|escendam|escendas|escendis|escendjs|escende|escendi|escendj|uenda|uende|venda|vende|mendum|mendvm|mendi|mendj|mendo|menda|mendorum|mendorvm|mendis|mendjs|pependi|pependj|jmpendo|jmpendam|jmpendas|jmpendjs|jmpende|jmpendj|jmpendo|jmpendam|jmpendas|jmpendjs|jmpende|jmpendj|expendo|expendam|expendas|expendis|expendjs|expende|expendi|expendj)"
#                r"(\w+?"
#                r"(?:andum|andus|andorum|andarum|andam|andvm|andvs|andorvm|andarvm|endum|endus|endorum|endarum|endam|endvm|endvs|endorvm|endarvm|endam|ando|endo|anda|enda|andi|endi|andj|endj|andae|endae|andos|endos|andas|endas|andis|endis|andjs|endjs|ande|ende))"
#                r"\b"
#            )
#        )  # match gerund endings w. negative lookahead to exclude 'quandam' etc.
#        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # selected gerund and gerundive endings (-and-, -end-)
    # NEW VERSION 09/09/2025
    def gerundive(self):
        ENC = r"(?:ne|ue|ve)?"
        feature_count = self._regex_word_feature_count(
            pattern=(
                r"\b"
                r"(?!quandam|quendam|quando|qvandam|qvendam|qvando|tendo|tendam|tendas|tendis|tende|tendi|uendo|vendo|uendam|vendam|uendas|vendas|uendis|vendis|uende|vende|uendi|vendi|defendo|defendam|defendas|defendis|defende|defendi|incendo|incendam|incendas|incendis|incende|incendi|prehendo|prehendam|prehendas|prehendis|prehende|prehendi|comprehendo|comprehendam|comprehendas|comprehendis|comprehende|comprehendi|ostendo|ostendam|ostendas|ostendis|ostende|ostendi|intendo|intendam|intendas|intendis|intende|intendi|contendo|contendam|contendas|contendis|contende|contendi|offendo|offendam|offendas|offendis|offende|offendi|succendo|succendam|succendas|succendis|succende|succendi|ascendo|ascendam|ascendas|ascendis|ascende|ascendi|extendo|extendam|extendas|extendis|extende|extendi|attendo|attendam|attendas|attendis|attende|attendi|protendo|protendam|protendas|protendis|protende|protendi|pretendo|pretendam|pretendas|pretendis|pretende|pretendi|dependo|dependam|dependas|dependis|depende|dependi|pendo|pendam|pendas|pendis|pende|pendi|pando|pandam|pandas|pandis|pande|pandi|expando|expandam|expandas|expandis|expande|expandi|blandus|blandvs|blandum|blandvm|blandi|blando|blanda|blandam|blandae|blandos|blandas|blandorum|blandorvm|blandarum|blandarvm|blandis|tendjs|tendj|uendjs|vendjs|uendj|vendj|defendjs|defendj|jncendo|jncendam|jncendas|jncendjs|jncende|jncendj|prehendjs|prehendj|comprehendjs|comprehendj|ostendjs|ostendj|jntendo|jntendam|jntendas|jntendjs|jntende|jntendj|contendjs|contendj|offendjs|offendj|succendjs|succendj|ascendjs|ascendj|extendjs|extendj|attendjs|attendj|protendjs|protendj|pretendjs|pretendj|dependjs|dependj|pendjs|pendj|pandjs|pandj|expandjs|expandj|blandj|blandjs|apprehendo|apprehendam|apprehendas|apprehendis|apprehendjs|apprehende|apprehendi|apprehend|deprehendo|deprehendam|deprehendas|deprehendis|deprehendjs|deprehende|deprehend|deprehendj|accendo|accendam|accendas|accendis|accendjs|accende|accendi|accendj|subtendo|subtendam|subtendas|subtendis|subtendjs|subtende|subtendi|subtendj|svbtendo|svbtendam|svbtendas|svbtendis|svbtendjs|svbtende|svibtendi|svibtendj|descendo|descendam|descendas|descendis|descendjs|descende|descendi|descendj|escendo|escendam|escendas|escendis|escendjs|escende|escendi|escendj|uenda|uende|venda|vende|mendum|mendvm|mendi|mendj|mendo|menda|mendorum|mendorvm|mendis|mendjs|pependi|pependj|jmpendo|jmpendam|jmpendas|jmpendjs|jmpende|jmpendj|jmpendo|jmpendam|jmpendas|jmpendjs|jmpende|jmpendj|expendo|expendam|expendas|expendis|expendjs|expende|expendi|expendj)" + ENC +
                r"(\w+?"
               r"(?:andum|andus|andorum|andarum|andam|andvm|andvs|andorvm|andarvm|endum|endus|endorum|endarum|endam|endvm|endvs|endorvm|endarvm|endam|ando|endo|anda|enda|andi|endi|andj|endj|andae|endae|andos|endos|andas|endas|andis|endis|andjs|endjs|ande|ende))" + ENC +
                r"\b"
            )
        )  # match gerund endings w. negative lookahead to exclude 'quandam' etc.
        return self.norm_feature(feature_count, "word") if self.norm else feature_count


    @property
    # cum excluding ablative endings
    def cum_clause(self):
        feature_count = self._regex_word_feature_count(
            pattern=r"(\bcum\s\w+?(?<![a|e|i|o|u|is|ibus|ebus|obus|ubus|v|ibvs|ebvs|obvs|vbvs|j|js|jbus|jbvs])\b)"
        )
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # conjunctions excluding cum
    def conjunction(self):
        #words = "ac antequam anteqvam ast at atqui atqve atqvi aut autem avt avtem donec dum dummodo dumque dvm dvmmodo dvmqve enim ergo et etenim etiam etiamque etiamqve etiamtum etiamtunc etiamtvm etiamtvnc igitur igitvr nam namqve nanque nanqve nec necdum necdvm nempe neqve neu neue nev neve postquam postqvam priusquam privsqvam quamquam quamuis quamvis quanquam que quia quiaque quin quippe quocirca quominus quoniam qvamqvam qvamvis qvanqvam qve qvia qviaqve qvin qvippe qvocirca qvominvs qvoniam sed set simul simvl tametsi tamquam tamqvam uel uerumtamen ueruntamen utrumnam vel verumtamen veruntamen vervmtamen vervntamen vtrvmnam"

        #words = "ac antequam ast at atqui aut autem donec dum dummodo dummodom enim ergo et etenim etiam igitur nam nanque nec necdum nempe neu neue postquam priusquam quamquam quamuis quin quippe quocirca quominus quoniam sed set simul tametsi tamquam uel utrumnam que atque itaque namque neque quia etiamtum etiamtunc quanquam si etsi etiamsi nisi quodsi sin siue seu ni"
        
        # NEW VERSION (6/19/2025)
        #words = "ac antequam ast at atqui aut autem donec dum dummodo dummodom enim ergo et etenim etiam igitur nam nanque nec necdum nempe neu neue postquam priusquam quamquam quamuis quin quippe quocirca quominus quoniam sed set simul tametsi tamquam uel uerumtamen utrumnam que atque itaque namque neque quia etiamtum etiamtunc quanquam si etsi etiamsi nisi quodsi sin siue seu ni"

		# NEW VERSION (09/09/2025)
        words = "ac an annon antequam ast at atque atqui aut autem donec dum dummodo dummodom dumque enim enimque ergo ergoque et etenim etenimque etiam etiamque igitur itaque nam namque nec necdum nempe neque neu neue postquam postquamque priusquam priusquamque quamquam quamuis quiaque quin quippe quocirca quodsique quominus quoniam quoniamque sed sedque set setque siquidem simul simulac simulatque tametsi tamquam tamquamque uel uerumtamen utrumnam quandoquidem que quia quippequod etiamtum etiamtunc quanquam si etsi etiamsi nisi quodsi sin siue siueque seu seuque ni"		

        words = set(words.split(" "))
        
        words = self.add_enclitic_forms(words, "ne")
        words = self.add_enclitic_forms(words, "ve")
        words = self.add_enclitic_forms(words, "ue")
        
        words -= {"sine", "niue", "nive", "anne"}
    
        feature_count = self._word_feature_count(words=words)
        return self.norm_feature(feature_count, "word") if self.norm else feature_count

    @property
    # O interjection or address
    def o_interjection(self):
        feature_count = self._regex_word_feature_count(
            # pattern=r"(\bo \w+?(e|i|j|a|u|ae|es|um|us|v|vm|vs)\b)"
            pattern=r"\b[oO] \w+\b"
        )
        return self.norm_feature(feature_count, "word") if self.norm else feature_count
