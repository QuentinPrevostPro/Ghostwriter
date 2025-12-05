def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks