from bs4 import BeautifulSoup

with open("./corpus/english.txt", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

chapters = []

for ch in soup.find_all("chapter"):
    cid = ch.get("id")
    name = ch.get("name")
    text = "\n\n".join(p.get_text(strip=True) for p in ch.find_all("p"))

    chapters.append({
        "id": cid,
        "name": name,
        "text": text,
    })