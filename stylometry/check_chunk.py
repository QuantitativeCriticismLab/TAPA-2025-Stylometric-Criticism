from readers import LatinReader

if __name__ == "__main__":
    text = "latin_tesserae_cicero.letters_to_atticus.part.1.txt"
    CR = LatinReader(root="data/texts/latin")
    chunks = CR.chunks(text, chunk_size=100)
    for i, chunk in enumerate(chunks):
        print("-----")
        print(f"{text}")
        print(f"Chunk {i}")
        print(f"{len(chunk)} words")
        print("-----")
        print(f"{chunk}")
        print("-----")
