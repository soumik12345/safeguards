import os

import pymupdf4llm
import weave
from firerequests import FireRequests


@weave.op()
def get_markdown_from_pdf_url(url: str) -> str:
    FireRequests().download(url, "temp.pdf", show_progress=False)
    markdown = pymupdf4llm.to_markdown("temp.pdf", show_progress=False)
    os.remove("temp.pdf")
    return markdown
