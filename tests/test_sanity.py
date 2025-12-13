def test_readme_exists():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Advanced-Active-Inference-Agent" in content
