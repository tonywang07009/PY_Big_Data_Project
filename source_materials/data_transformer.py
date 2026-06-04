from markitdown import MarkItDown  
md = MarkItDown()
test_result = md.convert('group_six.pdf')
print(test_result.text_content)

with open("group_six.md", "w", encoding="utf-8") as f:  
    f.write(test_result.markdown)  