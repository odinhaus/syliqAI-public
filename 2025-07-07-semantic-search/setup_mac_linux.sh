# Linux/macOS setup script (setup.sh)
# Make it executable: chmod +x setup.sh
# Run it: ./setup.sh

: <<'SETUP.SH'
#!/bin/bash

set -e

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Starting Qdrant via Docker..."
docker run -d --name qdrant-local -p 6333:6333 -p 6334:6334 qdrant/qdrant

echo "⬇️ Preloading Instructor-XL embedding model..."
python3 -c "from InstructorEmbedding import INSTRUCTOR; INSTRUCTOR('hkunlp/instructor-xl')"

echo "🎉 Setup complete. You can now run semantic_search_demo.py"
SETUP.SH
