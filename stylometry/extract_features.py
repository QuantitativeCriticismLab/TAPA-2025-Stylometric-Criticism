from collections import defaultdict
import pandas as pd

from readers import LatinReader
from features import LatinFeatures

from tqdm import tqdm

# NORM = False

# List features to check here
raw_features = [
    "word_count",
    "sentence_count",
    "sentence_length",
    "fraction_sentence_relative",
    "relative_clause_length",
]
normed_features = "alius antequam atque_consonant conjunction cum_clause demonstrative dum gerundive idem interrogative ipse iste o_interjection personal preposition priusquam quidam quin quominus reflexive si superlative ut".split()

if __name__ == "__main__":
    readers = [LatinReader]
    feature_sets = [LatinFeatures]
    #root_base = "data/texts/TAPA_verse_texts_preprocessed/"
    #root_base = "data/texts/TAPA_prose_texts_preprocessed/"

    def find_second_camelcase_position(s: str) -> int:
        for i, c in enumerate(s):
            if c.isupper() and i > 0:
                return i
        return -1

    def get_reader_language(reader):
        return reader.__name__[
            : find_second_camelcase_position(reader.__name__)
        ].lower()

    normed_features_data = defaultdict(list)
    unnormed_features_data = defaultdict(list)

    for reader, feature_set in tqdm(zip(readers, feature_sets)):
        root = f"{root_base}{get_reader_language(reader)}"
        READER = reader(root)

        for fileid in tqdm(READER.fileids()):
            RAW_FEATURES = feature_set(READER, fileid, norm=False, annotations=False)
            for feature in raw_features:
                normed_features_data[fileid].append(
                    {feature: getattr(RAW_FEATURES, feature)}
                )
                unnormed_features_data[fileid].append(
                    {feature: getattr(RAW_FEATURES, feature)}
                )

            NORMED_FEATURES = feature_set(READER, fileid, norm=True, annotations=False)
            for feature in normed_features:
                normed_features_data[fileid].append(
                    {feature: getattr(NORMED_FEATURES, feature)}
                )

            UNNORMED_FEATURES = feature_set(
                READER, fileid, norm=False, annotations=False
            )
            for feature in normed_features:
                unnormed_features_data[fileid].append(
                    {feature: getattr(UNNORMED_FEATURES, feature)}
                )

    index, normed_data = zip(*normed_features_data.items())
    normed_data = [
        {k: v for d in i for k, v in d.items()} for i in normed_data
    ]  # cf. https://stackoverflow.com/a/69492942

    index, unnormed_data = zip(*unnormed_features_data.items())
    unnormed_data = [
        {k: v for d in i for k, v in d.items()} for i in unnormed_data
    ]  # cf. https://stackoverflow.com/a/69492942

    # Python, create timestamp for filename
    import datetime

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    normed_outfile = f"data/output/features/{timestamp}-normed"
    unnormed_outfile = f"data/output/features/{timestamp}-raw"
    normed_pickle_outfile = normed_outfile + ".pickle"
    normed_csv_outfile = normed_outfile + ".csv"
    unnormed_pickle_outfile = unnormed_outfile + ".pickle"
    unnormed_csv_outfile = unnormed_outfile + ".csv"

    df = pd.DataFrame(normed_data, index=index)
    df.to_pickle(normed_pickle_outfile)
    df.to_csv(normed_csv_outfile)

    df = pd.DataFrame(unnormed_data, index=index)
    df.to_pickle(unnormed_pickle_outfile)
    df.to_csv(unnormed_csv_outfile)
