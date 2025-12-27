import os
import sys

# Ensure project root, libs, and pods are importable in tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIBS = os.path.join(ROOT, "libs")
PODS = os.path.join(ROOT, "pods")
for p in (ROOT, LIBS, PODS):
    if p not in sys.path:
        sys.path.insert(0, p)
