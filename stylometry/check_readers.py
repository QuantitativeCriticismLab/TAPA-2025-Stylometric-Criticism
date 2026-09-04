from readers import LatinReader

if __name__ == "__main__":
    #text = "latin_tesserae_cicero.letters_to_atticus.part.1.txt"
    text = "gerundives_synthetic_file.txt"
    CR = LatinReader(root="data/texts/latin")
    rcs = CR.relative_clauses(text, sent_level=False)
    for _ in range(10):
        print(next(rcs))
