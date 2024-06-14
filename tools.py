import hashlib

def hash_file(fileName):
    h = hashlib.sha1()
    with open(fileName, "rb") as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h.update(chunk)
    return h.hexdigest()

def compare_files(*files):

    hash1 = hash_file(files[0])
    hash2 = hash_file(files[1])

    return hash1 == hash2

def split_text(text, max_words=256, overlap=10):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words-overlap)]

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()