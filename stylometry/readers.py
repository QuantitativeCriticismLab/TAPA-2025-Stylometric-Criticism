import re
import spacy
from typing import Callable, Iterator, Union

from cltkreaders.lat import (
    LatinPlaintextCorpusReader,
)  # serving as a generic plaintext reader


class LatinReader(LatinPlaintextCorpusReader):
    def __init__(self, root, nlp="la_core_web_lg"):

        RELATIVES = {
            "qui",
            "cuius",
            "cui",
            "quem",
            "quo",
            "quae",
            "quam",
            "qua",
            "quod",
            "quorum",
            "quibus",
            "quos",
            "quarum",
            "quas",
            "quei",
            "quej",
			"qvei",
			"qvej",
			"quoius",
			"quojus",
			"qvoivs",
			"qvojvs",
			"quoi",
			"quoj",
			"qvoi",
			"qvoj",
			"quoiei",
			"quojej",
			"qvoiei",
			"qvojej",
			"queis",
			"quejs",
			"qveis",
			"qvejs",
			"qvi",
			"cvivs",
			"cvi",
			"qvem",
			"qvo",
			"qvae",
			"qvam",
			"qva",
			"qvod",
			"qvorvm",
			"qvibvs",
			"qvos",
			"qvarvm",
			"qvas",
			"quj",
			"cujus",
			"cuj",
			"qujbus",
			"qvj",
			"cvjvs",
			"cvj",
			"qvjbvs"
        }
        
        relative_group = "|".join(RELATIVES)
        self.pattern = rf"(?i)\b(({relative_group})\b.+?)[,:\.!;]"

        self.nlp = nlp
        self.model = spacy.load(nlp)
        # self.model.add_pipe("sentencizer", first=True)
        # self.model.disable_pipes(
        #     "senter",
        #     "normer",
        #     "tok2vec",
        #     "tagger",
        #     "morphologizer",
        #     "trainable_lemmatizer",
        #     "parser",
        #     "lookup_lemmatizer",
        #     "ner",
        # )

        LatinPlaintextCorpusReader.__init__(
            self,
            root,
        )

    def tagged_sents(
        self, fileids=None, plaintext=True, special_letters=False
    ):  # TODO: Deal with special_letters; not needed for Latin
        for sent in self.tokenized_sents(fileids):
            # for sent in sents:
            tagged_sent = [(item[0], item[2]) for item in sent]
            if plaintext:
                tagged_sent = ["/".join(item) for item in sent]
            yield tagged_sent

    def relative_clauses(self, fileids=None, sent_level=False, plaintext=True):

        # TODO: Define at class level

        for sent in self.sents(fileids):
            pp = lambda x: " ".join(x.split("\n"))
            sent_relative_clauses = re.findall(self.pattern, pp(sent.text))
            if not sent_relative_clauses:
                if sent_level:
                    yield None
            else:
                if sent_level:
                    yield [
                        sent_relative_clause[0]
                        for sent_relative_clause in sent_relative_clauses
                    ]
                else:
                    for sent_relative_clause in sent_relative_clauses:
                        yield sent_relative_clause[0]

    def tokenized_paras(
        self,
        fileids: Union[list, str] = None,
        unline: bool = True,
        preprocess: Callable = None,
    ) -> Iterator[list]:
        for para in self.paras(fileids):
            tokenized_para = []
            if unline:
                para = " ".join(para.split()).strip()
            sents = self.sent_tokenizer.tokenize(para)
            for sent in sents:
                if preprocess:
                    sent = preprocess(sent.text)
                tokens_ = [token for token in self.word_tokenizer.tokenize(sent)]
                words = [token.text for token in tokens_]
                lemmas = [token.lemma_ for token in tokens_]
                postags = [token.pos_ for token in tokens_]
                tokenized_para.append(list(zip(words, lemmas, postags)))
            yield tokenized_para
