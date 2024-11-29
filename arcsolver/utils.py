"""an xml parsing util"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_utils.ipynb.

# %% auto 0
__all__ = ['parse_from_xml']

# %% ../nbs/04_utils.ipynb 3
import re

# %% ../nbs/04_utils.ipynb 4
class TagNotFoundError(Exception):
    """Raised when the specified tag is not found in the XML."""
    pass


class NoContentError(Exception):
    """Raised when the specified tag is found but contains no content."""
    pass


class MultipleTagsError(Exception):
    """
    Raised when multiple instances of the specified tag are found in the XML.
    """
    pass

# %% ../nbs/04_utils.ipynb 5
def parse_from_xml(input_text: str, tag_name: str) -> str:
    """
    Parse content from an XML-like string for a specific tag.

    Args:
        input_text (str): The input text containing XML-like tags.
        tag_name (str): The name of the tag to parse.

    Returns:
        str: The content within the specified tag.

    Raises:
        TagNotFoundError: If the specified tag is not found in the text.
        MultipleTagsError: If multiple instances of the tag are found.
        NoContentError: If the tag is present but contains no content.
    """
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, input_text, re.DOTALL)

    if not matches:
        raise TagNotFoundError(f"Tag '{tag_name}' not found in the text.")

    if len(matches) > 1:
        raise MultipleTagsError(
            f"Multiple instances of tag '{tag_name}' found in the text."
        )

    content = matches[0].strip()
    content = content.replace("```python", "").replace("```", "").strip()

    if not content:
        raise NoContentError(
            f"Tag '{tag_name}' is present but contains no content."
        )

    return content

