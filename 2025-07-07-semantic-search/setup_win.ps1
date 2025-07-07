# Windows setup script (setup.ps1)
# Run this from PowerShell (as Administrator if needed)

<### PowerShell Script: setup.ps1 ###>

Write-Host "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "🚀 Starting Qdrant via Docker..."
docker run -d --name qdrant-local -p 6333:6333 -p 6334:6334 qdrant/qdrant

Write-Host "⬇️ Preloading Instructor-XL embedding model..."
python -c "from InstructorEmbedding import INSTRUCTOR; INSTRUCTOR('hkunlp/instructor-xl')"

Write-Host "🎉 Setup complete. You can now run semantic_search_demo.py"
