cat > cosmic-net/backend/tests/conftest.py << 'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
EOFs