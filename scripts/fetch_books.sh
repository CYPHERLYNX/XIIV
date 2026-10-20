#!/bin/bash
# Fetch public-domain books from Project Gutenberg via curl.
# Adult prose: the antidote to TinyStories.
cd ~/workspace/ai-tools/corsac
mkdir -p data_books/raw
# id:title — mix of fiction classics + non-fiction
IDS="1342 84 1661 74 98 43 120 16 11 2600 1080 55 36 1184 45 175 244 30360 1497 1400 203 820 100 1597 25525"
for id in $IDS; do
  out="data_books/raw/pg${id}.txt"
  if [ -s "$out" ]; then echo "have $id"; continue; fi
  echo "fetch $id ..."
  curl -sL --max-time 60 -o "$out" "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt" || \
  curl -sL --max-time 60 -o "$out" "https://www.gutenberg.org/files/${id}/${id}-0.txt"
  sleep 1
done
ls -la data_books/raw/ | tail -3
du -sh data_books/raw/
