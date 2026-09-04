from readers import LatinReader
from features import LatinFeatures

# List features to check here
features = [
    "fraction_sentence_relative",
    "relative_clause_length"
]

if __name__ == "__main__":
    readers = [LatinReader]
    feature_sets = [LatinFeatures]
    root_base = "data/texts/word_count_testing/"

    for reader, feature_set in zip(readers, feature_sets):
        root = f"{root_base}{reader.__name__.split('Reader')[0].lower()}"
        READER = reader(root)

        for fileid in READER.fileids():
            print(fileid)
            print("-----")
            print(f"File: {fileid}")
            FEATURES = feature_set(READER, fileid, norm=False, verbose=True)
            for feature in features:
                print(f"{feature}: {getattr(FEATURES, feature)}")
            print("-----")
            print()
